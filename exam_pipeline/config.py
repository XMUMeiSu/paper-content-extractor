"""Environment-backed settings without secret disclosure."""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineSettings:
    api_key: str = field(default_factory=lambda: os.getenv("DOUBAO_API_KEY", ""), repr=False)
    model: str = field(default_factory=lambda: os.getenv("DOUBAO_MODEL", "doubao-seed-2.0-lite"))
    endpoint: str = field(default_factory=lambda: os.getenv(
        "DOUBAO_RESPONSES_ENDPOINT",
        os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3").rstrip("/") + "/responses"))
    structure_vlm_provider: str = field(
        default_factory=lambda: os.getenv("STRUCTURE_VLM_PROVIDER", "auto"))
    paddleocr_vl_endpoint: str = field(
        default_factory=lambda: os.getenv("PADDLEOCR_VL_ENDPOINT", ""))
    paddleocr_vl_model: str = field(default_factory=lambda: os.getenv(
        "PADDLEOCR_VL_MODEL", "PaddlePaddle/PaddleOCR-VL-1.6"))
    paddleocr_vl_api_key: str = field(
        default_factory=lambda: os.getenv("PADDLEOCR_VL_API_KEY", ""), repr=False)
    tree_llm_endpoint: str = field(default_factory=lambda: os.getenv("TREE_LLM_ENDPOINT", ""))
    tree_llm_model: str = field(default_factory=lambda: os.getenv("TREE_LLM_MODEL", ""))
    tree_llm_api_key: str = field(
        default_factory=lambda: os.getenv("TREE_LLM_API_KEY", ""), repr=False)
    ocr_engine: str = "paddle"

    def validate(self, require_vlm: bool = False) -> List[str]:
        errors = []
        if require_vlm and not self.api_key.strip():
            errors.append("DOUBAO_API_KEY is required")
        if not self.endpoint.startswith(("http://", "https://")):
            errors.append("endpoint must be an HTTP(S) URL")
        provider = self.structure_vlm_provider.strip().lower()
        if provider not in {"auto", "paddleocr-vl", "doubao", "none"}:
            errors.append("structure_vlm_provider is invalid")
        if provider == "paddleocr-vl" and not self.paddleocr_vl_endpoint.startswith(
                ("http://", "https://")):
            errors.append("PADDLEOCR_VL_ENDPOINT must be an HTTP(S) URL")
        if self.tree_llm_endpoint and not self.tree_llm_endpoint.startswith(("http://", "https://")):
            errors.append("TREE_LLM_ENDPOINT must be an HTTP(S) URL")
        if self.tree_llm_endpoint and not self.tree_llm_model.strip():
            errors.append("TREE_LLM_MODEL is required when TREE_LLM_ENDPOINT is configured")
        return errors
