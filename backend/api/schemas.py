from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to analyze")
    mode: str = Field(default="auto", description="Model selection mode: auto, ru, or en")
    request_id: Optional[str] = Field(default=None, description="Optional request ID for aborting")


class AbortRequest(BaseModel):
    request_id: str = Field(..., description="ID of the in-flight request to cancel")


class AbortResponse(BaseModel):
    request_id: str
    aborted: bool
    message: str

