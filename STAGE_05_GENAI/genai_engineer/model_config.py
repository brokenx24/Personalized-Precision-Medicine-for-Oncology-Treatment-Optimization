"""
GenAI Model Configuration.
Configures backend providers, execution timeouts, temperatures, and seeds.
"""
import os
from pydantic import BaseModel, Field

class GenAIModelConfig(BaseModel):
    provider: str = Field(default="grounded_rule_engine")
    model_name: str = Field(default="qwen2.5_1.5b_lora_oncology")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=8192)
    timeout_seconds: int = Field(default=30, ge=5, le=300)
    retry_count: int = Field(default=3, ge=1, le=5)
    random_seed: int = Field(default=42)
    deterministic_mode: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "GenAIModelConfig":
        return cls(
            provider=os.getenv("GENAI_PROVIDER", "grounded_rule_engine"),
            random_seed=int(os.getenv("STAGE05_SEED", 42))
        )
