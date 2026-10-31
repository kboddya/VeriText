from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")
    mode: str = Field(default="auto", description="Model selection mode: auto, ru, or en")
