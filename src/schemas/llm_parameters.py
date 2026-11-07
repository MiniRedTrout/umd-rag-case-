from typing import Optional

from pydantic import BaseModel, Field, field_validator


class LLMGenerationParams(BaseModel):
    temperature: Optional[float] = Field(
        default=0.0001,
        ge=0.0001,
        le=1.2,
        description="Controls randomness in the output. Higher values like 0.8 produce more random outputs, while lower values like 0.2 make outputs more focused and deterministic.",
    )

    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.01,
        le=1.0,
        description="Nucleus sampling parameter. The model considers the results of tokens with top_p probability mass. 1.0 means no filtering.",
    )

    max_tokens: Optional[int] = Field(
        default=1024, ge=512, le=32768, description="Maximum number of tokens to generate in the response."
    )

    repetition_penalty: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Penalty applied to previously generated tokens. 1.0 means no penalty, >1.0 discourages repetition.",
    )

    @field_validator("max_tokens")
    @classmethod
    def ensure_positive_integer(cls, v):
        if v is not None and not isinstance(v, int):
            raise ValueError(f"Must be an integer, got {type(v)}")
        return v
