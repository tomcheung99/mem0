from typing import Optional
from pydantic import Field

from mem0.configs.rerankers.base import BaseRerankerConfig


class OnnxRerankerConfig(BaseRerankerConfig):
    """
    Configuration class for ONNX Runtime reranker.
    Supports quantized models (e.g., q8) for efficient CPU inference.
    """

    model: Optional[str] = Field(
        default="onnx-community/bge-reranker-v2-m3-ONNX",
        description="The ONNX model to use for reranking",
    )
    quantization: Optional[str] = Field(
        default="q8",
        description="Quantization level to use (e.g., 'q8', 'q4', None for fp32)",
    )
    max_length: int = Field(default=512, description="Maximum token length for tokenization")
    batch_size: int = Field(default=32, description="Batch size for processing documents")
    normalize: bool = Field(default=True, description="Whether to normalize scores to 0-1")
