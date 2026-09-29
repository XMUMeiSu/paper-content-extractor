"""Environment-backed settings without secret disclosure."""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineSettings:
    api_key: str = field(default_factory=lambda: os.getenv("DOUBAO_API_KEY", ""), repr=False)
    model: str = field(default_factory=lambda: os.getenv("DOUBAO_MODEL", "doubao-seed-2.1-turbo"))
    endpoint: str = field(default_factory=lambda: os.getenv(
        "DOUBAO_RESPONSES_ENDPOINT",
        os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3").rstrip("/") + "/responses"))

    def validate(self, require_vlm: bool = False) -> List[str]:
        errors = []
        if require_vlm and not self.api_key.strip():
            errors.append("DOUBAO_API_KEY is required")
        if not self.endpoint.startswith(("http://", "https://")):
            errors.append("endpoint must be an HTTP(S) URL")
        return errors
