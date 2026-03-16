from typing import List, Dict, Any, Union

import numpy as np

from mem0.reranker.base import BaseReranker
from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.configs.rerankers.onnx import OnnxRerankerConfig

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


class OnnxReranker(BaseReranker):
    """ONNX Runtime based reranker supporting quantized models (q8/q4) for efficient CPU inference."""

    def __init__(self, config: Union[BaseRerankerConfig, OnnxRerankerConfig, Dict]):
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum[onnxruntime] is required for OnnxReranker. "
                "Install with: pip install optimum[onnxruntime] transformers"
            )

        if isinstance(config, dict):
            config = OnnxRerankerConfig(**config)
        elif isinstance(config, BaseRerankerConfig) and not isinstance(config, OnnxRerankerConfig):
            config = OnnxRerankerConfig(
                provider=getattr(config, "provider", "onnx"),
                model=getattr(config, "model", "onnx-community/bge-reranker-v2-m3-ONNX"),
                api_key=getattr(config, "api_key", None),
                top_k=getattr(config, "top_k", None),
            )

        self.config = config

        # Build subfolder for quantized weights (e.g., "onnx/model_q8.onnx")
        file_name = None
        subfolder = None
        if self.config.quantization:
            subfolder = "onnx"
            file_name = f"model_{self.config.quantization}.onnx"

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.config.model,
            file_name=file_name,
            subfolder=subfolder,
        )

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        doc_texts = []
        for doc in documents:
            if "memory" in doc:
                doc_texts.append(doc["memory"])
            elif "text" in doc:
                doc_texts.append(doc["text"])
            elif "content" in doc:
                doc_texts.append(doc["content"])
            else:
                doc_texts.append(str(doc))

        try:
            scores = []

            for i in range(0, len(doc_texts), self.config.batch_size):
                batch_docs = doc_texts[i : i + self.config.batch_size]
                batch_pairs = [[query, doc] for doc in batch_docs]

                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).detach().numpy()

                if batch_scores.ndim == 0:
                    batch_scores = [float(batch_scores)]
                else:
                    batch_scores = batch_scores.tolist()

                scores.extend(batch_scores)

            if self.config.normalize and len(scores) > 1:
                arr = np.array(scores)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                scores = arr.tolist()

            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            final_top_k = top_k or self.config.top_k
            if final_top_k:
                doc_score_pairs = doc_score_pairs[:final_top_k]

            reranked_docs = []
            for doc, score in doc_score_pairs:
                reranked_doc = doc.copy()
                reranked_doc["rerank_score"] = float(score)
                reranked_docs.append(reranked_doc)

            return reranked_docs

        except Exception:
            for doc in documents:
                doc["rerank_score"] = 0.0
            final_top_k = top_k or self.config.top_k
            return documents[:final_top_k] if final_top_k else documents
