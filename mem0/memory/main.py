import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import os
import uuid
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryTier, MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    TRUST_SCORING_PROMPT,
    get_conflict_aware_update_memory_messages,
    get_update_memory_messages,
)
from mem0.exceptions import ValidationError as Mem0ValidationError
from mem0.memory.base import MemoryBase
from mem0.memory.cleanup import (
    apply_temporal_decay,
    build_compaction_prompt,
    compute_expires_at,
    compute_memory_entropy,
    compute_tier_ttl,
    is_gc_eligible,
    is_memory_expired,
)
from mem0.memory.setup import mem0_dir, setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    extract_json,
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    process_telemetry_filters,
    remove_code_blocks,
)
from mem0.utils.factory import (
    EmbedderFactory,
    GraphStoreFactory,
    LlmFactory,
    VectorStoreFactory,
    RerankerFactory,
)

# Suppress SWIG deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

# Initialize logger early for util functions
logger = logging.getLogger(__name__)


def _safe_deepcopy_config(config):
    """Safely deepcopy config, falling back to JSON serialization for non-serializable objects."""
    try:
        return deepcopy(config)
    except Exception as e:
        logger.debug(f"Deepcopy failed, using JSON serialization: {e}")
        
        config_class = type(config)
        
        if hasattr(config, "model_dump"):
            try:
                clone_dict = config.model_dump(mode="json")
            except Exception:
                clone_dict = {k: v for k, v in config.__dict__.items()}
        elif hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict
            clone_dict = asdict(config)
        else:
            clone_dict = {k: v for k, v in config.__dict__.items()}
        
        sensitive_tokens = ("auth", "credential", "password", "token", "secret", "key", "connection_class")
        for field_name in list(clone_dict.keys()):
            if any(token in field_name.lower() for token in sensitive_tokens):
                clone_dict[field_name] = None
        
        try:
            return config_class(**clone_dict)
        except Exception as reconstruction_error:
            logger.warning(
                f"Failed to reconstruct config: {reconstruction_error}. "
                f"Telemetry may be affected."
            )
            raise


def _build_filters_and_metadata(
    *,  # Enforce keyword-only arguments
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    actor_id: Optional[str] = None,  # For query-time filtering
    input_metadata: Optional[Dict[str, Any]] = None,
    input_filters: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Constructs metadata for storage and filters for querying based on session and actor identifiers.

    This helper supports multiple session identifiers (`user_id`, `agent_id`, and/or `run_id`)
    for flexible session scoping and optionally narrows queries to a specific `actor_id`. It returns two dicts:

    1. `base_metadata_template`: Used as a template for metadata when storing new memories.
       It includes all provided session identifier(s) and any `input_metadata`.
    2. `effective_query_filters`: Used for querying existing memories. It includes all
       provided session identifier(s), any `input_filters`, and a resolved actor
       identifier for targeted filtering if specified by any actor-related inputs.

    Actor filtering precedence: explicit `actor_id` arg → `filters["actor_id"]`
    This resolved actor ID is used for querying but is not added to `base_metadata_template`,
    as the actor for storage is typically derived from message content at a later stage.

    Args:
        user_id (Optional[str]): User identifier, for session scoping.
        agent_id (Optional[str]): Agent identifier, for session scoping.
        run_id (Optional[str]): Run identifier, for session scoping.
        actor_id (Optional[str]): Explicit actor identifier, used as a potential source for
            actor-specific filtering. See actor resolution precedence in the main description.
        input_metadata (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session identifiers for the storage metadata template. Defaults to an empty dict.
        input_filters (Optional[Dict[str, Any]]): Base dictionary to be augmented with
            session and actor identifiers for query filters. Defaults to an empty dict.

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing:
            - base_metadata_template (Dict[str, Any]): Metadata template for storing memories,
              scoped to the provided session(s).
            - effective_query_filters (Dict[str, Any]): Filters for querying memories,
              scoped to the provided session(s) and potentially a resolved actor.
    """

    base_metadata_template = deepcopy(input_metadata) if input_metadata else {}
    effective_query_filters = deepcopy(input_filters) if input_filters else {}

    # ---------- add all provided session ids ----------
    session_ids_provided = []

    if user_id:
        base_metadata_template["user_id"] = user_id
        effective_query_filters["user_id"] = user_id
        session_ids_provided.append("user_id")

    if agent_id:
        base_metadata_template["agent_id"] = agent_id
        effective_query_filters["agent_id"] = agent_id
        session_ids_provided.append("agent_id")

    if run_id:
        base_metadata_template["run_id"] = run_id
        effective_query_filters["run_id"] = run_id
        session_ids_provided.append("run_id")

    if not session_ids_provided:
        raise Mem0ValidationError(
            message="At least one of 'user_id', 'agent_id', or 'run_id' must be provided.",
            error_code="VALIDATION_001",
            details={"provided_ids": {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}},
            suggestion="Please provide at least one identifier to scope the memory operation."
        )

    # ---------- optional actor filter ----------
    resolved_actor_id = actor_id or effective_query_filters.get("actor_id")
    if resolved_actor_id:
        effective_query_filters["actor_id"] = resolved_actor_id

    return base_metadata_template, effective_query_filters


setup_config()
logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        
        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider, 
                config.reranker.config
            )

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None
        # Create telemetry config manually to avoid deepcopy issues with thread locks
        telemetry_config_dict = {}
        if hasattr(self.config.vector_store.config, 'model_dump'):
            # For pydantic models
            telemetry_config_dict = self.config.vector_store.config.model_dump()
        else:
            # For other objects, manually copy common attributes
            for attr in ['host', 'port', 'path', 'api_key', 'index_name', 'dimension', 'metric']:
                if hasattr(self.config.vector_store.config, attr):
                    telemetry_config_dict[attr] = getattr(self.config.vector_store.config, attr)

        # Override collection name for telemetry
        telemetry_config_dict['collection_name'] = "mem0migrations"

        # Set path for file-based vector stores
        telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            telemetry_config_dict['path'] = os.path.join(mem0_dir, provider_path)
            os.makedirs(telemetry_config_dict['path'], exist_ok=True)

        # Create the config object using the same class as the original
        telemetry_config = self.config.vector_store.config.__class__(**telemetry_config_dict)
        self._telemetry_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, telemetry_config
        )
        capture_event("mem0.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False
        
        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.
            
        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories.",
                error_code="VALIDATION_002",
                details={"provided_type": memory_type, "valid_type": MemoryType.PROCEDURAL.value},
                suggestion=f"Use '{MemoryType.PROCEDURAL.value}' to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=processed_metadata, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, processed_metadata, effective_filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, effective_filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format: {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = self.embedding_model.embed(msg_content, "add")
                mem_id = self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)

        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            # Determine if this should use agent memory extraction based on agent_id presence
            # and role types in messages
            is_agent_memory = self._should_use_agent_memory_extraction(messages, metadata)
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            response = remove_code_blocks(response)
            if not response.strip():
                new_retrieved_facts = []
            else:
                try:
                    # First try direct JSON parsing
                    new_retrieved_facts = json.loads(response)["facts"]
                except json.JSONDecodeError:
                    # Try extracting JSON from response using built-in function
                    extracted_json = extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        # ── Trust Scoring: evaluate importance of each extracted fact ──
        trust_scores_map = {}
        if new_retrieved_facts and self.config.trust_scoring.enabled:
            trust_scores_map = self._score_facts(new_retrieved_facts)

        retrieved_old_memory = []
        new_message_embeddings = {}
        # Search for existing memories using the provided session identifiers
        # Use all available session identifiers for accurate memory retrieval
        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_filters["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            search_filters["run_id"] = filters["run_id"]
        for new_mem in new_retrieved_facts:
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=search_filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            # Use conflict-aware prompt when conflict resolution is enabled
            if self.config.conflict_resolution.enabled:
                function_calling_prompt = get_conflict_aware_update_memory_messages(
                    retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
                )
            else:
                function_calling_prompt = get_update_memory_messages(
                    retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
                )

            try:
                response: str = self.llm.generate_response(
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        logger.info("Skipping memory entry because of empty `text` field.")
                        continue

                    event_type = resp.get("event")

                    # ── Attach trust score and tier to metadata for ADD ──
                    action_metadata = deepcopy(metadata)
                    if self.config.trust_scoring.enabled and event_type in ("ADD", "CONFLICT"):
                        fact_score = trust_scores_map.get(action_text, 0.5)
                        action_metadata["trust_score"] = fact_score
                        if fact_score < self.config.trust_scoring.archive_threshold:
                            action_metadata["memory_tier"] = MemoryTier.ARCHIVED.value
                            logger.info(f"Low trust score ({fact_score:.2f}), archiving memory: {action_text[:60]}")

                    if event_type == "ADD":
                        memory_id = self._create_memory(
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=action_metadata,
                        )
                        returned_memories.append({
                            "id": memory_id,
                            "memory": action_text,
                            "event": event_type,
                            "trust_score": action_metadata.get("trust_score"),
                            "memory_tier": action_metadata.get("memory_tier"),
                        })
                    elif event_type == "UPDATE":
                        self._update_memory(
                            memory_id=temp_uuid_mapping[resp.get("id")],
                            data=action_text,
                            existing_embeddings=new_message_embeddings,
                            metadata=deepcopy(metadata),
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": action_text,
                                "event": event_type,
                            }
                        )
                    elif event_type == "CONFLICT":
                        # Handle memory conflicts
                        conflicting_memory_id = temp_uuid_mapping.get(resp.get("id"))
                        conflict_action = self.config.conflict_resolution.contradiction_action

                        if conflict_action == "auto_resolve" and conflicting_memory_id:
                            # Keep newer fact, demote older
                            existing_memory = self.vector_store.get(vector_id=conflicting_memory_id)
                            if existing_memory:
                                old_payload = deepcopy(existing_memory.payload)
                                old_payload["memory_tier"] = MemoryTier.ARCHIVED.value
                                old_payload["conflict_superseded_by"] = action_text
                                old_payload["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
                                self.vector_store.update(
                                    vector_id=conflicting_memory_id,
                                    vector=None,
                                    payload=old_payload,
                                )
                            # Add the new fact
                            memory_id = self._create_memory(
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=action_metadata,
                            )
                            returned_memories.append({
                                "id": memory_id,
                                "memory": action_text,
                                "event": "CONFLICT_RESOLVED",
                                "previous_memory": resp.get("old_memory"),
                                "conflict_type": resp.get("conflict_type"),
                                "superseded_id": conflicting_memory_id,
                            })
                        else:
                            # Flag mode: add the new fact with conflict metadata
                            action_metadata["conflict_with"] = conflicting_memory_id
                            action_metadata["conflict_type"] = resp.get("conflict_type", "unknown")
                            action_metadata["conflict_resolved"] = False
                            memory_id = self._create_memory(
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=action_metadata,
                            )
                            returned_memories.append({
                                "id": memory_id,
                                "memory": action_text,
                                "event": "CONFLICT_FLAGGED",
                                "previous_memory": resp.get("old_memory"),
                                "conflict_type": resp.get("conflict_type"),
                                "conflict_with": conflicting_memory_id,
                            })
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Update only the session identifiers, keep content the same
                            existing_memory = self.vector_store.get(vector_id=memory_id)
                            updated_metadata = deepcopy(existing_memory.payload)
                            if metadata.get("agent_id"):
                                updated_metadata["agent_id"] = metadata["agent_id"]
                            if metadata.get("run_id"):
                                updated_metadata["run_id"] = metadata["run_id"]
                            updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                            self.vector_store.update(
                                vector_id=memory_id,
                                vector=None,  # Keep same embeddings
                                payload=updated_metadata,
                            )
                            logger.info(f"Updated session IDs for memory {memory_id}")
                        else:
                            logger.info("NOOP for Memory.")
                except Exception as e:
                    logger.error(f"Error processing memory action: {resp}, Error: {e}")
        except Exception as e:
            logger.error(f"Error iterating new_memories_with_actions: {e}")

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"},
        )
        return returned_memories

    def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        Retrieve a memory by ID.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id",
                                  "trust_score", "memory_tier", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
            trust_score=memory.payload.get("trust_score"),
            memory_tier=memory.payload.get("memory_tier"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

        Args:
            user_id (str, optional): user id
            agent_id (str, optional): agent id
            run_id (str, optional): run id
            filters (dict, optional): Additional custom key-value filters to apply to the search.
                These are merged with the ID-based scoping filters. For example,
                `filters={"actor_id": "some_user"}`.
            limit (int, optional): The maximum number of memories to return. Defaults to 100.

        Returns:
            dict: A dictionary containing a list of memories under the "results" key,
                  and potentially "relations" if graph store is enabled. For API v1.0,
                  it might return a direct list (see deprecation warning).
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"}
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, effective_filters, limit)
            future_graph_entities = (
                executor.submit(self.graph.get_all, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories_result = future_memories.result()
            graph_entities_result = future_graph_entities.result() if future_graph_entities else None

        if self.enable_graph:
            return {"results": all_memories_result, "relations": graph_entities_result}

        return {"results": all_memories_result}

    def _get_all_from_vector_store(self, filters, limit):
        memories_result = self.vector_store.list(filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id",
                                  "trust_score", "memory_tier", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                trust_score=mem.payload.get("trust_score"),
                memory_tier=mem.payload.get("memory_tier"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be specified.")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "sync",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, effective_filters, limit, threshold)
            future_graph_entities = (
                executor.submit(self.graph.search, query, effective_filters, limit) if self.enable_graph else None
            )

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            original_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.
        
        Args:
            metadata_filters: Enhanced metadata filters with operators
            
        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}
        
        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}
            
            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte", 
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }
                
                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result
        
        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))
        
        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.
        
        Args:
            filters: Dictionary of filters to check
            
        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False
            
        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id",
                                  "expires_at", "access_count", "last_accessed_at",
                                  "trust_score", "memory_tier",
                                  *promoted_payload_keys}

        ttl_cfg = self.config.cleanup.ttl
        gc_cfg = self.config.cleanup.garbage_collection
        expired_ids = []

        original_memories = []
        for mem in memories:
            # ── TTL: skip & mark expired memories ──
            if ttl_cfg.enabled and is_memory_expired(mem.payload):
                if ttl_cfg.auto_purge_on_search:
                    expired_ids.append(mem.id)
                continue

            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
                trust_score=mem.payload.get("trust_score"),
                memory_tier=mem.payload.get("memory_tier"),
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        # ── Temporal Decay ──
        decay_cfg = self.config.cleanup.temporal_decay
        if decay_cfg.enabled and original_memories:
            apply_temporal_decay(original_memories, decay_cfg.decay_rate, decay_cfg.time_field)

        # ── Async TTL purge of expired entries discovered above ──
        if expired_ids:
            for eid in expired_ids:
                try:
                    self._delete_memory(eid)
                    logger.info(f"TTL purge: deleted expired memory {eid}")
                except Exception as e:
                    logger.warning(f"TTL purge failed for {eid}: {e}")

        # ── GC: track access on returned memories ──
        if gc_cfg.enabled:
            self._track_access(original_memories)

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "sync"})
        # delete all vector memories and reset the collections
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)
        self.vector_store.reset()

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # ── Hierarchical Memory: assign tier if not already set ──
        hier_cfg = self.config.hierarchical_memory
        if hier_cfg.enabled and "memory_tier" not in metadata:
            if metadata.get("run_id"):
                metadata["memory_tier"] = MemoryTier.WORKING.value
            elif metadata.get("agent_id"):
                metadata["memory_tier"] = MemoryTier.SESSION.value
            else:
                metadata["memory_tier"] = MemoryTier.LONG_TERM.value

        # ── TTL: stamp expiration if configured and not already set ──
        ttl_cfg = self.config.cleanup.ttl
        if ttl_cfg.enabled and "expires_at" not in metadata and ttl_cfg.default_ttl_seconds is not None:
            metadata["expires_at"] = compute_expires_at(ttl_cfg.default_ttl_seconds)

        # ── Proactive Forgetting: apply tier-based TTL ──
        pf_cfg = self.config.cleanup.proactive_forgetting
        if pf_cfg.enabled and "expires_at" not in metadata:
            tier = metadata.get("memory_tier")
            if tier:
                tier_expires = compute_tier_ttl(tier, pf_cfg.tier_ttl_seconds)
                if tier_expires:
                    metadata["expires_at"] = tier_expires

        # ── GC: initialise access tracking fields ──
        gc_cfg = self.config.cleanup.garbage_collection
        if gc_cfg.enabled:
            metadata.setdefault("access_count", 0)
            metadata.setdefault("last_accessed_at", metadata["created_at"])

        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        self.db.add_history(
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
            procedural_memory = remove_code_blocks(procedural_memory)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")

        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload.get("data", "")
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )
        return memory_id

    def _score_facts(self, facts: list) -> Dict[str, float]:
        """
        Use LLM to score each extracted fact for trust/importance.

        Args:
            facts: List of fact strings to score.

        Returns:
            Dict mapping fact text → trust score (0.0-1.0).
        """
        trust_cfg = self.config.trust_scoring
        prompt_template = trust_cfg.scoring_prompt or TRUST_SCORING_PROMPT
        facts_text = "\n".join(f"- {f}" for f in facts)
        prompt = prompt_template.format(facts=facts_text)

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are an expert memory evaluator."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            response = remove_code_blocks(response)
            scored = json.loads(response).get("scored_facts", [])
            return {item["text"]: float(item["score"]) for item in scored if "text" in item and "score" in item}
        except Exception as e:
            logger.warning(f"Trust scoring failed, using default scores: {e}")
            return {f: 0.5 for f in facts}

    # ── Hierarchical Memory & Forgetting public API ────────────────

    def promote_session_memories(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Promote high-value working/session memories to long-term storage.

        Summarizes session memories via LLM and promotes those exceeding
        the promotion threshold to long-term tier.

        Args:
            user_id, agent_id, run_id: Scope the promotion.
            limit: Max memories to process.

        Returns:
            dict: ``{"promoted": <int>, "archived": <int>, "summary_ids": [...]}``
        """
        hier_cfg = self.config.hierarchical_memory
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id,
        )

        memories_result = self.vector_store.list(filters=effective_filters, limit=limit)
        actual_memories = memories_result[0] if (isinstance(memories_result, (list, tuple))
                                                 and memories_result
                                                 and isinstance(memories_result[0], (list, tuple))) else memories_result

        # Filter to working and session tier memories
        eligible = [
            m for m in actual_memories
            if m.payload.get("memory_tier") in (MemoryTier.WORKING.value, MemoryTier.SESSION.value)
        ]

        if not eligible:
            return {"promoted": 0, "archived": 0, "summary_ids": []}

        # Build summaries using compaction infrastructure
        batch_dicts = [{"id": m.id, "memory": m.payload.get("data", "")} for m in eligible]
        compaction_prompt = build_compaction_prompt(batch_dicts)

        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are a knowledge curator."},
                    {"role": "user", "content": compaction_prompt},
                ],
                response_format={"type": "json_object"},
            )
            response = remove_code_blocks(response)
            try:
                summaries = json.loads(response).get("summaries", [])
            except json.JSONDecodeError:
                summaries = json.loads(extract_json(response)).get("summaries", [])
        except Exception as e:
            logger.error(f"promote_session_memories: LLM summarization failed: {e}")
            return {"promoted": 0, "archived": 0, "summary_ids": []}

        # Score the summaries for trust
        trust_scores = {}
        if self.config.trust_scoring.enabled and summaries:
            trust_scores = self._score_facts(summaries)

        promoted = 0
        archived = 0
        summary_ids = []
        base_metadata = deepcopy(effective_filters)

        for summary_text in summaries:
            if not summary_text or not summary_text.strip():
                continue
            score = trust_scores.get(summary_text, 0.5)
            meta = deepcopy(base_metadata)
            meta["trust_score"] = score

            if score >= hier_cfg.promotion_threshold:
                meta["memory_tier"] = MemoryTier.LONG_TERM.value
                promoted += 1
            else:
                meta["memory_tier"] = MemoryTier.ARCHIVED.value
                archived += 1

            emb = self.embedding_model.embed(summary_text, "add")
            mid = self._create_memory(summary_text, {summary_text: emb}, metadata=meta)
            summary_ids.append(mid)

        # Delete original working/session memories
        for mem in eligible:
            try:
                self._delete_memory(mem.id)
            except Exception as e:
                logger.warning(f"promote_session_memories: failed to delete {mem.id}: {e}")

        logger.info(f"promote_session_memories: promoted={promoted}, archived={archived}")
        return {"promoted": promoted, "archived": archived, "summary_ids": summary_ids}

    def entropy_report(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """
        Compute and return the memory entropy metrics.

        Entropy measures the noise ratio of recently accumulated memories.
        High entropy indicates many low-quality memories are being stored.

        Args:
            user_id, agent_id, run_id: Scope the analysis.
            limit: Max memories to scan.

        Returns:
            dict: ``{"entropy": float, "total_in_window": int, "low_trust_count": int,
                     "avg_trust_score": float, "needs_cleanup": bool}``
        """
        pf_cfg = self.config.cleanup.proactive_forgetting
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id,
        )

        memories_result = self.vector_store.list(filters=effective_filters, limit=limit)
        actual_memories = memories_result[0] if (isinstance(memories_result, (list, tuple))
                                                 and memories_result
                                                 and isinstance(memories_result[0], (list, tuple))) else memories_result

        memory_dicts = [
            {
                "created_at": m.payload.get("created_at"),
                "updated_at": m.payload.get("updated_at"),
                "trust_score": m.payload.get("trust_score", 0.5),
            }
            for m in actual_memories
        ]

        report = compute_memory_entropy(memory_dicts, pf_cfg.entropy_window_hours)
        report["needs_cleanup"] = report["entropy"] >= pf_cfg.entropy_threshold
        return report

    # ── Cleanup public API ─────────────────────────────────────────

    def purge_expired(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """
        Scan and delete all memories whose TTL has expired.

        Args:
            user_id, agent_id, run_id: Scope the scan to a specific session.
            limit: Maximum number of memories to scan per call.

        Returns:
            dict: ``{"deleted": <int>, "scanned": <int>}``
        """
        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id
        )
        memories_result = self.vector_store.list(filters=effective_filters, limit=limit)
        actual_memories = memories_result[0] if (isinstance(memories_result, (list, tuple))
                                                 and memories_result
                                                 and isinstance(memories_result[0], (list, tuple))) else memories_result

        deleted = 0
        for mem in actual_memories:
            if is_memory_expired(mem.payload):
                try:
                    self._delete_memory(mem.id)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"purge_expired: failed to delete {mem.id}: {e}")

        logger.info(f"purge_expired: deleted {deleted}/{len(actual_memories)} memories")
        return {"deleted": deleted, "scanned": len(actual_memories)}

    def compact_memories(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        batch_size: Optional[int] = None,
        preserve_recent_hours: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Summarise many fine-grained memories into dense knowledge blocks using an LLM.

        Memories newer than ``preserve_recent_hours`` are left untouched.
        Older memories are grouped into batches, summarised, and the originals deleted.

        Args:
            user_id, agent_id, run_id: Scope the compaction to a specific session.
            limit: Max memories to fetch for compaction.
            batch_size: Memories per LLM summarisation call. Falls back to config.
            preserve_recent_hours: Skip memories newer than this. Falls back to config.
            prompt: Custom LLM prompt. Falls back to config, then default.

        Returns:
            dict: ``{"compacted": <int>, "created": <int>, "summary_ids": [...]}``
        """
        compact_cfg = self.config.cleanup.compaction
        batch_size = batch_size or compact_cfg.summary_batch_size
        preserve_recent_hours = preserve_recent_hours if preserve_recent_hours is not None else compact_cfg.preserve_recent_hours
        prompt_template = prompt or compact_cfg.summary_prompt

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id,
        )

        memories_result = self.vector_store.list(filters=effective_filters, limit=limit)
        actual_memories = memories_result[0] if (isinstance(memories_result, (list, tuple))
                                                 and memories_result
                                                 and isinstance(memories_result[0], (list, tuple))) else memories_result

        # Filter out recent memories
        cutoff = datetime.now(pytz.timezone("US/Pacific")) - timedelta(hours=preserve_recent_hours)
        eligible = []
        for mem in actual_memories:
            ts_str = mem.payload.get("updated_at") or mem.payload.get("created_at")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = pytz.timezone("US/Pacific").localize(ts)
                    if ts < cutoff:
                        eligible.append(mem)
                except (ValueError, TypeError):
                    eligible.append(mem)
            else:
                eligible.append(mem)

        if not eligible:
            return {"compacted": 0, "created": 0, "summary_ids": []}

        # Process in batches
        compacted_total = 0
        summary_ids = []
        base_metadata = deepcopy(effective_filters)

        for i in range(0, len(eligible), batch_size):
            batch = eligible[i : i + batch_size]
            batch_dicts = [
                {"id": m.id, "memory": m.payload.get("data", "")} for m in batch
            ]

            compaction_prompt = build_compaction_prompt(batch_dicts, prompt_template)
            try:
                response = self.llm.generate_response(
                    messages=[
                        {"role": "system", "content": "You are a knowledge curator."},
                        {"role": "user", "content": compaction_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                from mem0.memory.utils import remove_code_blocks, extract_json
                response = remove_code_blocks(response)
                try:
                    summaries = json.loads(response).get("summaries", [])
                except json.JSONDecodeError:
                    summaries = json.loads(extract_json(response)).get("summaries", [])
            except Exception as e:
                logger.error(f"compact_memories: LLM call failed for batch {i}: {e}")
                continue

            # Delete originals
            for mem in batch:
                try:
                    self._delete_memory(mem.id)
                    compacted_total += 1
                except Exception as e:
                    logger.warning(f"compact_memories: failed to delete {mem.id}: {e}")

            # Insert summaries
            for summary_text in summaries:
                if not summary_text or not summary_text.strip():
                    continue
                emb = self.embedding_model.embed(summary_text, "add")
                meta = deepcopy(base_metadata)
                mid = self._create_memory(summary_text, {summary_text: emb}, metadata=meta)
                summary_ids.append(mid)

        logger.info(f"compact_memories: compacted {compacted_total} into {len(summary_ids)} summaries")
        return {"compacted": compacted_total, "created": len(summary_ids), "summary_ids": summary_ids}

    def garbage_collect(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Remove memories that are idle (not accessed recently) and have low access counts.

        Args:
            user_id, agent_id, run_id: Scope the GC scan.
            limit: Max memories to scan. Falls back to config ``batch_size``.

        Returns:
            dict: ``{"deleted": <int>, "scanned": <int>}``
        """
        gc_cfg = self.config.cleanup.garbage_collection
        limit = limit or gc_cfg.batch_size

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id,
        )

        memories_result = self.vector_store.list(filters=effective_filters, limit=limit)
        actual_memories = memories_result[0] if (isinstance(memories_result, (list, tuple))
                                                 and memories_result
                                                 and isinstance(memories_result[0], (list, tuple))) else memories_result

        deleted = 0
        for mem in actual_memories:
            if is_gc_eligible(mem.payload, gc_cfg.min_idle_days, gc_cfg.min_access_count):
                try:
                    self._delete_memory(mem.id)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"garbage_collect: failed to delete {mem.id}: {e}")

        logger.info(f"garbage_collect: deleted {deleted}/{len(actual_memories)} memories")
        return {"deleted": deleted, "scanned": len(actual_memories)}

    def _track_access(self, memories: list):
        """
        Increment ``access_count`` and update ``last_accessed_at`` for each memory
        returned from a search. Runs as a best-effort background update.
        """
        now_str = datetime.now(pytz.timezone("US/Pacific")).isoformat()
        for mem in memories:
            try:
                existing = self.vector_store.get(vector_id=mem["id"])
                if existing is None:
                    continue
                payload = deepcopy(existing.payload)
                payload["access_count"] = payload.get("access_count", 0) + 1
                payload["last_accessed_at"] = now_str
                self.vector_store.update(vector_id=mem["id"], payload=payload)
            except Exception as e:
                logger.debug(f"_track_access: failed for {mem.get('id')}: {e}")

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")

        if hasattr(self.db, "connection") and self.db.connection:
            self.db.connection.execute("DROP TABLE IF EXISTS history")
            self.db.connection.close()

        self.db = SQLiteManager(self.config.history_db_path)

        if hasattr(self.vector_store, "reset"):
            self.vector_store = VectorStoreFactory.reset(self.vector_store)
        else:
            logger.warning("Vector store does not support reset. Skipping.")
            self.vector_store.delete_col()
            self.vector_store = VectorStoreFactory.create(
                self.config.vector_store.provider, self.config.vector_store.config
            )
        capture_event("mem0.reset", self, {"sync_type": "sync"})

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version
        
        # Initialize reranker if configured
        self.reranker = None
        if config.reranker:
            self.reranker = RerankerFactory.create(
                config.reranker.provider, 
                config.reranker.config
            )

        self.enable_graph = False

        if self.config.graph_store.config:
            provider = self.config.graph_store.provider
            self.graph = GraphStoreFactory.create(provider, self.config)
            self.enable_graph = True
        else:
            self.graph = None

        telemetry_config = _safe_deepcopy_config(self.config.vector_store.config)
        telemetry_config.collection_name = "mem0migrations"
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            telemetry_config.path = os.path.join(mem0_dir, provider_path)
            os.makedirs(telemetry_config.path, exist_ok=True)
        self._telemetry_vector_store = VectorStoreFactory.create(self.config.vector_store.provider, telemetry_config)

        capture_event("mem0.init", self, {"sync_type": "async"})

    @classmethod
    async def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def _should_use_agent_memory_extraction(self, messages, metadata):
        """Determine whether to use agent memory extraction based on the logic:
        - If agent_id is present and messages contain assistant role -> True
        - Otherwise -> False
        
        Args:
            messages: List of message dictionaries
            metadata: Metadata containing user_id, agent_id, etc.
            
        Returns:
            bool: True if should use agent memory extraction, False for user memory extraction
        """
        # Check if agent_id is present in metadata
        has_agent_id = metadata.get("agent_id") is not None
        
        # Check if there are assistant role messages
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        
        # Use agent memory extraction if agent_id is present and there are assistant messages
        return has_agent_id and has_assistant_messages

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None.
                                         Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
        """
        processed_metadata, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_metadata=metadata
        )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(
                messages, metadata=processed_metadata, prompt=prompt, llm=llm
            )
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        vector_store_task = asyncio.create_task(
            self._add_to_vector_store(messages, processed_metadata, effective_filters, infer)
        )
        graph_task = asyncio.create_task(self._add_to_graph(messages, effective_filters))

        vector_store_result, graph_result = await asyncio.gather(vector_store_task, graph_task)

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _add_to_vector_store(
        self,
        messages: list,
        metadata: dict,
        effective_filters: dict,
        infer: bool,
    ):
        if not infer:
            returned_memories = []
            for message_dict in messages:
                if (
                    not isinstance(message_dict, dict)
                    or message_dict.get("role") is None
                    or message_dict.get("content") is None
                ):
                    logger.warning(f"Skipping invalid message format (async): {message_dict}")
                    continue

                if message_dict["role"] == "system":
                    continue

                per_msg_meta = deepcopy(metadata)
                per_msg_meta["role"] = message_dict["role"]

                actor_name = message_dict.get("name")
                if actor_name:
                    per_msg_meta["actor_id"] = actor_name

                msg_content = message_dict["content"]
                msg_embeddings = await asyncio.to_thread(self.embedding_model.embed, msg_content, "add")
                mem_id = await self._create_memory(msg_content, msg_embeddings, per_msg_meta)

                returned_memories.append(
                    {
                        "id": mem_id,
                        "memory": msg_content,
                        "event": "ADD",
                        "actor_id": actor_name if actor_name else None,
                        "role": message_dict["role"],
                    }
                )
            return returned_memories

        parsed_messages = parse_messages(messages)
        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            # Determine if this should use agent memory extraction based on agent_id presence
            # and role types in messages
            is_agent_memory = self._should_use_agent_memory_extraction(messages, metadata)
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages, is_agent_memory)

        response = await asyncio.to_thread(
            self.llm.generate_response,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )
        try:
            response = remove_code_blocks(response)
            if not response.strip():
                new_retrieved_facts = []
            else:
                try:
                    # First try direct JSON parsing
                    new_retrieved_facts = json.loads(response)["facts"]
                except json.JSONDecodeError:
                    # Try extracting JSON from response using built-in function
                    extracted_json = extract_json(response)
                    new_retrieved_facts = json.loads(extracted_json)["facts"]
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        # ── Trust Scoring: evaluate importance of each extracted fact ──
        trust_scores_map = {}
        if new_retrieved_facts and self.config.trust_scoring.enabled:
            trust_scores_map = await asyncio.to_thread(self._score_facts, new_retrieved_facts)

        if not new_retrieved_facts:
            logger.debug("No new facts retrieved from input. Skipping memory update LLM call.")

        retrieved_old_memory = []
        new_message_embeddings = {}
        # Search for existing memories using the provided session identifiers
        # Use all available session identifiers for accurate memory retrieval
        search_filters = {}
        if effective_filters.get("user_id"):
            search_filters["user_id"] = effective_filters["user_id"]
        if effective_filters.get("agent_id"):
            search_filters["agent_id"] = effective_filters["agent_id"]
        if effective_filters.get("run_id"):
            search_filters["run_id"] = effective_filters["run_id"]

        async def process_fact_for_search(new_mem_content):
            embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem_content, "add")
            new_message_embeddings[new_mem_content] = embeddings
            existing_mems = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem_content,
                vectors=embeddings,
                limit=5,
                filters=search_filters,
            )
            return [{"id": mem.id, "text": mem.payload.get("data", "")} for mem in existing_mems]

        search_tasks = [process_fact_for_search(fact) for fact in new_retrieved_facts]
        search_results_list = await asyncio.gather(*search_tasks)
        for result_group in search_results_list:
            retrieved_old_memory.extend(result_group)

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logger.info(f"Total existing memories: {len(retrieved_old_memory)}")
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        if new_retrieved_facts:
            if self.config.conflict_resolution.enabled:
                function_calling_prompt = get_conflict_aware_update_memory_messages(
                    retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
                )
            else:
                function_calling_prompt = get_update_memory_messages(
                    retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
                )
            try:
                response = await asyncio.to_thread(
                    self.llm.generate_response,
                    messages=[{"role": "user", "content": function_calling_prompt}],
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""
            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM, no memories to extract")
                    new_memories_with_actions = {}
                else:
                    response = remove_code_blocks(response)
                    new_memories_with_actions = json.loads(response)
            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                new_memories_with_actions = {}
        else:
            new_memories_with_actions = {}

        returned_memories = []
        try:
            memory_tasks = []
            for resp in new_memories_with_actions.get("memory", []):
                logger.info(resp)
                try:
                    action_text = resp.get("text")
                    if not action_text:
                        continue
                    event_type = resp.get("event")

                    if event_type == "ADD":
                        action_metadata = deepcopy(metadata)
                        if self.config.trust_scoring.enabled and event_type in ("ADD", "CONFLICT"):
                            fact_score = trust_scores_map.get(action_text, 0.5)
                            action_metadata["trust_score"] = fact_score
                            if fact_score < self.config.trust_scoring.archive_threshold:
                                action_metadata["memory_tier"] = MemoryTier.ARCHIVED.value
                        task = asyncio.create_task(
                            self._create_memory(
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=action_metadata,
                            )
                        )
                        memory_tasks.append((task, resp, "ADD", None))
                    elif event_type == "CONFLICT":
                        action_metadata = deepcopy(metadata)
                        if self.config.trust_scoring.enabled:
                            fact_score = trust_scores_map.get(action_text, 0.5)
                            action_metadata["trust_score"] = fact_score
                            if fact_score < self.config.trust_scoring.archive_threshold:
                                action_metadata["memory_tier"] = MemoryTier.ARCHIVED.value
                        conflicting_memory_id = temp_uuid_mapping.get(resp.get("id"))
                        conflict_action = self.config.conflict_resolution.contradiction_action
                        if conflict_action == "auto_resolve" and conflicting_memory_id:
                            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=conflicting_memory_id)
                            if existing_memory:
                                old_payload = deepcopy(existing_memory.payload)
                                old_payload["memory_tier"] = MemoryTier.ARCHIVED.value
                                old_payload["conflict_superseded_by"] = action_text
                                old_payload["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
                                await asyncio.to_thread(self.vector_store.update, vector_id=conflicting_memory_id, vector=None, payload=old_payload)
                            task = asyncio.create_task(
                                self._create_memory(data=action_text, existing_embeddings=new_message_embeddings, metadata=action_metadata)
                            )
                            memory_tasks.append((task, resp, "CONFLICT_RESOLVED", None))
                        else:
                            action_metadata["conflict_with"] = conflicting_memory_id
                            action_metadata["conflict_type"] = resp.get("conflict_type", "unknown")
                            action_metadata["conflict_resolved"] = False
                            task = asyncio.create_task(
                                self._create_memory(data=action_text, existing_embeddings=new_message_embeddings, metadata=action_metadata)
                            )
                            memory_tasks.append((task, resp, "CONFLICT_FLAGGED", None))
                    elif event_type == "UPDATE":
                        task = asyncio.create_task(
                            self._update_memory(
                                memory_id=temp_uuid_mapping[resp["id"]],
                                data=action_text,
                                existing_embeddings=new_message_embeddings,
                                metadata=deepcopy(metadata),
                            )
                        )
                        memory_tasks.append((task, resp, "UPDATE", temp_uuid_mapping[resp["id"]]))
                    elif event_type == "DELETE":
                        task = asyncio.create_task(self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")]))
                        memory_tasks.append((task, resp, "DELETE", temp_uuid_mapping[resp.get("id")]))
                    elif event_type == "NONE":
                        # Even if content doesn't need updating, update session IDs if provided
                        memory_id = temp_uuid_mapping.get(resp.get("id"))
                        if memory_id and (metadata.get("agent_id") or metadata.get("run_id")):
                            # Create async task to update only the session identifiers
                            async def update_session_ids(mem_id, meta):
                                existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=mem_id)
                                updated_metadata = deepcopy(existing_memory.payload)
                                if meta.get("agent_id"):
                                    updated_metadata["agent_id"] = meta["agent_id"]
                                if meta.get("run_id"):
                                    updated_metadata["run_id"] = meta["run_id"]
                                updated_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

                                await asyncio.to_thread(
                                    self.vector_store.update,
                                    vector_id=mem_id,
                                    vector=None,  # Keep same embeddings
                                    payload=updated_metadata,
                                )
                                logger.info(f"Updated session IDs for memory {mem_id} (async)")

                            task = asyncio.create_task(update_session_ids(memory_id, metadata))
                            memory_tasks.append((task, resp, "NONE", memory_id))
                        else:
                            logger.info("NOOP for Memory (async).")
                except Exception as e:
                    logger.error(f"Error processing memory action (async): {resp}, Error: {e}")

            for task, resp, event_type, mem_id in memory_tasks:
                try:
                    result_id = await task
                    if event_type == "ADD":
                        add_result = {"id": result_id, "memory": resp.get("text"), "event": event_type}
                        if resp.get("text") and trust_scores_map:
                            add_result["trust_score"] = trust_scores_map.get(resp.get("text"))
                        returned_memories.append(add_result)
                    elif event_type in ("CONFLICT_RESOLVED", "CONFLICT_FLAGGED"):
                        conflict_result = {
                            "id": result_id,
                            "memory": resp.get("text"),
                            "event": event_type,
                            "previous_memory": resp.get("old_memory"),
                            "conflict_type": resp.get("conflict_type"),
                        }
                        if event_type == "CONFLICT_RESOLVED":
                            conflict_result["superseded_id"] = temp_uuid_mapping.get(resp.get("id"))
                        else:
                            conflict_result["conflict_with"] = temp_uuid_mapping.get(resp.get("id"))
                        returned_memories.append(conflict_result)
                    elif event_type == "UPDATE":
                        returned_memories.append(
                            {
                                "id": mem_id,
                                "memory": resp.get("text"),
                                "event": event_type,
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        returned_memories.append({"id": mem_id, "memory": resp.get("text"), "event": event_type})
                except Exception as e:
                    logger.error(f"Error awaiting memory task (async): {e}")
        except Exception as e:
            logger.error(f"Error in memory processing loop (async): {e}")

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"},
        )
        return returned_memories

    async def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = await asyncio.to_thread(self.graph.add, data, filters)

        return added_entities

    def _score_facts(self, facts: list) -> Dict[str, float]:
        """
        Use LLM to score each extracted fact for trust/importance.

        Args:
            facts: List of fact strings to score.

        Returns:
            Dict mapping fact text → trust score (0.0-1.0).
        """
        trust_cfg = self.config.trust_scoring
        prompt_template = trust_cfg.scoring_prompt or TRUST_SCORING_PROMPT
        facts_text = "\n".join(f"- {f}" for f in facts)
        prompt = prompt_template.format(facts=facts_text)
        try:
            response = self.llm.generate_response(
                messages=[
                    {"role": "system", "content": "You are an expert memory evaluator."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            response = remove_code_blocks(response)
            scored = json.loads(response).get("scored_facts", [])
            return {item["text"]: float(item["score"]) for item in scored if "text" in item and "score" in item}
        except Exception as e:
            logger.warning(f"Trust scoring failed, using default scores: {e}")
            return {f: 0.5 for f in facts}

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id",
                                  "trust_score", "memory_tier", *promoted_payload_keys}

        result_item = MemoryItem(
            id=memory.id,
            memory=memory.payload.get("data", ""),
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
            trust_score=memory.payload.get("trust_score"),
            memory_tier=memory.payload.get("memory_tier"),
        ).model_dump()

        for key in promoted_payload_keys:
            if key in memory.payload:
                result_item[key] = memory.payload[key]

        additional_metadata = {k: v for k, v in memory.payload.items() if k not in core_and_promoted_keys}
        if additional_metadata:
            result_item["metadata"] = additional_metadata

        return result_item

    async def get_all(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ):
        """
        List all memories.

         Args:
             user_id (str, optional): user id
             agent_id (str, optional): agent id
             run_id (str, optional): run id
             filters (dict, optional): Additional custom key-value filters to apply to the search.
                 These are merged with the ID-based scoping filters. For example,
                 `filters={"actor_id": "some_user"}`.
             limit (int, optional): The maximum number of memories to return. Defaults to 100.

         Returns:
             dict: A dictionary containing a list of memories under the "results" key,
                   and potentially "relations" if graph store is enabled. For API v1.0,
                   it might return a direct list (see deprecation warning).
                   Example for v1.1+: `{"results": [{"id": "...", "memory": "...", ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError(
                "When 'conversation_id' is not provided (classic mode), "
                "at least one of 'user_id', 'agent_id', or 'run_id' must be specified for get_all."
            )

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.get_all", self, {"limit": limit, "keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"}
        )

        vector_store_task = asyncio.create_task(self._get_all_from_vector_store(effective_filters, limit))

        graph_task = None
        if self.enable_graph:
            graph_get_all = getattr(self.graph, "get_all", None)
            if callable(graph_get_all):
                if asyncio.iscoroutinefunction(graph_get_all):
                    graph_task = asyncio.create_task(graph_get_all(effective_filters, limit))
                else:
                    graph_task = asyncio.create_task(asyncio.to_thread(graph_get_all, effective_filters, limit))

        results_dict = {}
        if graph_task:
            vector_store_result, graph_entities_result = await asyncio.gather(vector_store_task, graph_task)
            results_dict.update({"results": vector_store_result, "relations": graph_entities_result})
        else:
            results_dict.update({"results": await vector_store_task})

        return results_dict

    async def _get_all_from_vector_store(self, filters, limit):
        memories_result = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)

        # Handle different vector store return formats by inspecting first element
        if isinstance(memories_result, (tuple, list)) and len(memories_result) > 0:
            first_element = memories_result[0]

            # If first element is a container, unwrap one level
            if isinstance(first_element, (list, tuple)):
                actual_memories = first_element
            else:
                # First element is a memory object, structure is already flat
                actual_memories = memories_result
        else:
            actual_memories = memories_result

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]
        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id",
                                  "trust_score", "memory_tier", *promoted_payload_keys}

        formatted_memories = []
        for mem in actual_memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                trust_score=mem.payload.get("trust_score"),
                memory_tier=mem.payload.get("memory_tier"),
            ).model_dump(exclude={"score"})

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            formatted_memories.append(memory_item_dict)

        return formatted_memories

    async def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """

        _, effective_filters = _build_filters_and_metadata(
            user_id=user_id, agent_id=agent_id, run_id=run_id, input_filters=filters
        )

        if not any(key in effective_filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("at least one of 'user_id', 'agent_id', or 'run_id' must be specified ")

        # Apply enhanced metadata filtering if advanced operators are detected
        if filters and self._has_advanced_operators(filters):
            processed_filters = self._process_metadata_filters(filters)
            effective_filters.update(processed_filters)
        elif filters:
            # Simple filters, merge directly
            effective_filters.update(filters)

        keys, encoded_ids = process_telemetry_filters(effective_filters)
        capture_event(
            "mem0.search",
            self,
            {
                "limit": limit,
                "version": self.api_version,
                "keys": keys,
                "encoded_ids": encoded_ids,
                "sync_type": "async",
                "threshold": threshold,
                "advanced_filters": bool(filters and self._has_advanced_operators(filters)),
            },
        )

        vector_store_task = asyncio.create_task(self._search_vector_store(query, effective_filters, limit, threshold))

        graph_task = None
        if self.enable_graph:
            if hasattr(self.graph.search, "__await__"):  # Check if graph search is async
                graph_task = asyncio.create_task(self.graph.search(query, effective_filters, limit))
            else:
                graph_task = asyncio.create_task(asyncio.to_thread(self.graph.search, query, effective_filters, limit))

        if graph_task:
            original_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
        else:
            original_memories = await vector_store_task
            graph_entities = None

        # Apply reranking if enabled and reranker is available
        if rerank and self.reranker and original_memories:
            try:
                # Run reranking in thread pool to avoid blocking async loop
                reranked_memories = await asyncio.to_thread(
                    self.reranker.rerank, query, original_memories, limit
                )
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        return {"results": original_memories}

    def _process_metadata_filters(self, metadata_filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process enhanced metadata filters and convert them to vector store compatible format.

        Args:
            metadata_filters: Enhanced metadata filters with operators

        Returns:
            Dict of processed filters compatible with vector store
        """
        processed_filters = {}

        def process_condition(key: str, condition: Any) -> Dict[str, Any]:
            if not isinstance(condition, dict):
                # Simple equality: {"key": "value"}
                if condition == "*":
                    # Wildcard: match everything for this field (implementation depends on vector store)
                    return {key: "*"}
                return {key: condition}

            result = {}
            for operator, value in condition.items():
                # Map platform operators to universal format that can be translated by each vector store
                operator_map = {
                    "eq": "eq", "ne": "ne", "gt": "gt", "gte": "gte",
                    "lt": "lt", "lte": "lte", "in": "in", "nin": "nin",
                    "contains": "contains", "icontains": "icontains"
                }

                if operator in operator_map:
                    result[key] = {operator_map[operator]: value}
                else:
                    raise ValueError(f"Unsupported metadata filter operator: {operator}")
            return result

        for key, value in metadata_filters.items():
            if key == "AND":
                # Logical AND: combine multiple conditions
                if not isinstance(value, list):
                    raise ValueError("AND operator requires a list of conditions")
                for condition in value:
                    for sub_key, sub_value in condition.items():
                        processed_filters.update(process_condition(sub_key, sub_value))
            elif key == "OR":
                # Logical OR: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("OR operator requires a non-empty list of conditions")
                # Store OR conditions in a way that vector stores can interpret
                processed_filters["$or"] = []
                for condition in value:
                    or_condition = {}
                    for sub_key, sub_value in condition.items():
                        or_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$or"].append(or_condition)
            elif key == "NOT":
                # Logical NOT: Pass through to vector store for implementation-specific handling
                if not isinstance(value, list) or not value:
                    raise ValueError("NOT operator requires a non-empty list of conditions")
                processed_filters["$not"] = []
                for condition in value:
                    not_condition = {}
                    for sub_key, sub_value in condition.items():
                        not_condition.update(process_condition(sub_key, sub_value))
                    processed_filters["$not"].append(not_condition)
            else:
                processed_filters.update(process_condition(key, value))

        return processed_filters

    def _has_advanced_operators(self, filters: Dict[str, Any]) -> bool:
        """
        Check if filters contain advanced operators that need special processing.

        Args:
            filters: Dictionary of filters to check

        Returns:
            bool: True if advanced operators are detected
        """
        if not isinstance(filters, dict):
            return False

        for key, value in filters.items():
            # Check for platform-style logical operators
            if key in ["AND", "OR", "NOT"]:
                return True
            # Check for comparison operators (without $ prefix for universal compatibility)
            if isinstance(value, dict):
                for op in value.keys():
                    if op in ["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains"]:
                        return True
            # Check for wildcard values
            if value == "*":
                return True
        return False

    async def _search_vector_store(self, query, filters, limit, threshold: Optional[float] = None):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, limit=limit, filters=filters
        )

        promoted_payload_keys = [
            "user_id",
            "agent_id",
            "run_id",
            "actor_id",
            "role",
        ]

        core_and_promoted_keys = {"data", "hash", "created_at", "updated_at", "id",
                                  "expires_at", "access_count", "last_accessed_at",
                                  "trust_score", "memory_tier",
                                  *promoted_payload_keys}

        original_memories = []
        for mem in memories:
            memory_item_dict = MemoryItem(
                id=mem.id,
                memory=mem.payload.get("data", ""),
                hash=mem.payload.get("hash"),
                created_at=mem.payload.get("created_at"),
                updated_at=mem.payload.get("updated_at"),
                score=mem.score,
                trust_score=mem.payload.get("trust_score"),
                memory_tier=mem.payload.get("memory_tier"),
            ).model_dump()

            for key in promoted_payload_keys:
                if key in mem.payload:
                    memory_item_dict[key] = mem.payload[key]

            additional_metadata = {k: v for k, v in mem.payload.items() if k not in core_and_promoted_keys}
            if additional_metadata:
                memory_item_dict["metadata"] = additional_metadata

            if threshold is None or mem.score >= threshold:
                original_memories.append(memory_item_dict)

        return original_memories

    async def update(self, memory_id, data):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (str): New content to update the memory with.

        Returns:
            dict: Success message indicating the memory was updated.

        Example:
            >>> await m.update(memory_id="mem_123", data="Likes to play tennis on weekends")
            {'message': 'Memory updated successfully!'}
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})
        await self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        keys, encoded_ids = process_telemetry_filters(filters)
        capture_event("mem0.delete_all", self, {"keys": keys, "encoded_ids": encoded_ids, "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        if self.enable_graph:
            await asyncio.to_thread(self.graph.delete_all, filters)

        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logger.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # ── Hierarchical Memory: assign tier if not already set ──
        hier_cfg = self.config.hierarchical_memory
        if hier_cfg.enabled and "memory_tier" not in metadata:
            if metadata.get("run_id"):
                metadata["memory_tier"] = MemoryTier.WORKING.value
            elif metadata.get("agent_id"):
                metadata["memory_tier"] = MemoryTier.SESSION.value
            else:
                metadata["memory_tier"] = MemoryTier.LONG_TERM.value

        # ── TTL: stamp expiration if configured and not already set ──
        ttl_cfg = self.config.cleanup.ttl
        if ttl_cfg.enabled and "expires_at" not in metadata and ttl_cfg.default_ttl_seconds is not None:
            metadata["expires_at"] = compute_expires_at(ttl_cfg.default_ttl_seconds)

        # ── Proactive Forgetting: apply tier-based TTL ──
        pf_cfg = self.config.cleanup.proactive_forgetting
        if pf_cfg.enabled and "expires_at" not in metadata:
            tier = metadata.get("memory_tier")
            if tier:
                tier_expires = compute_tier_ttl(tier, pf_cfg.tier_ttl_seconds)
                if tier_expires:
                    metadata["expires_at"] = tier_expires

        # ── GC: initialise access tracking fields ──
        gc_cfg = self.config.cleanup.garbage_collection
        if gc_cfg.enabled:
            metadata.setdefault("access_count", 0)
            metadata.setdefault("last_accessed_at", metadata["created_at"])

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            None,
            data,
            "ADD",
            created_at=metadata.get("created_at"),
            actor_id=metadata.get("actor_id"),
            role=metadata.get("role"),
        )

        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (llm, optional): LLM to use for the procedural memory creation. Defaults to None.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
                procedural_memory = remove_code_blocks(procedural_memory)
        
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            logger.error(f"Error getting memory with ID {memory_id} during update.")
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = deepcopy(metadata) if metadata is not None else {}

        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # Preserve session identifiers from existing memory only if not provided in new metadata
        if "user_id" not in new_metadata and "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" not in new_metadata and "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" not in new_metadata and "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if "actor_id" not in new_metadata and "actor_id" in existing_memory.payload:
            new_metadata["actor_id"] = existing_memory.payload["actor_id"]
        if "role" not in new_metadata and "role" in existing_memory.payload:
            new_metadata["role"] = existing_memory.payload["role"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
            actor_id=new_metadata.get("actor_id"),
            role=new_metadata.get("role"),
        )
        return memory_id

    async def _delete_memory(self, memory_id):
        logger.info(f"Deleting memory with {memory_id=}")
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        prev_value = existing_memory.payload.get("data", "")

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            None,
            "DELETE",
            actor_id=existing_memory.payload.get("actor_id"),
            role=existing_memory.payload.get("role"),
            is_deleted=1,
        )

        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, "client") and hasattr(self.vector_store.client, "close"):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, "connection") and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("mem0.reset", self, {"sync_type": "async"})

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")
