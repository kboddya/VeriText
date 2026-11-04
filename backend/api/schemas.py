from __future__ import annotations

from typing import Any, Optional
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


class MarkdownReportRequest(BaseModel):
    data: Optional[dict] = Field(default=None, description="Inspection result payload from score or document endpoints")
    report_type: Optional[str] = Field(default="single", description="Type of report: 'single' for text/document, 'humanize' for humanizer audit")
    filename: Optional[str] = Field(default=None, description="Optional custom filename override")


class HumanizeRequest(BaseModel):
    text: str = Field(default="", description="Input text to humanize")
    mode: str = Field(default="auto", description="Model selection mode: auto, ru, or en")
    strength: str = Field(default="balanced", description="Transformation intensity: light, balanced, aggressive")
    request_id: Optional[str] = Field(default=None, description="Optional request ID for progress tracking and aborting")


class ParagraphDiff(BaseModel):
    index: int = Field(..., description="1-based paragraph index")
    original_text: str = Field(..., description="Original paragraph text before transformation")
    humanized_text: str = Field(..., description="Humanized paragraph text after transformation")
    before_scale: float = Field(..., description="Original continuous AI probability percentage (0-100)")
    after_scale: float = Field(..., description="Humanized continuous AI probability percentage (0-100)")
    delta_scale: float = Field(..., description="AI probability reduction (before_scale - after_scale)")
    before_label: str = Field(default="human", description="'ai' or 'human'")
    after_label: str = Field(default="human", description="'ai' or 'human'")
    before_confidence: dict[str, Any] = Field(default_factory=dict, description="Confidence status grade and risk level before")
    after_confidence: dict[str, Any] = Field(default_factory=dict, description="Confidence status grade and risk level after")
    changed: bool = Field(default=False, description="True if paragraph content was modified")
    stage_applied: str = Field(default="full", description="Stage applied: scrub_and_paraphrase, scrub_only, unchanged, or code")
    is_code: bool = Field(default=False, description="True if paragraph is a code block, script, or markdown table")


class HumanizeSummary(BaseModel):
    word_count_before: int = Field(default=0, description="Original word count")
    word_count_after: int = Field(default=0, description="Humanized word count")
    char_count_before: int = Field(default=0, description="Original character count")
    char_count_after: int = Field(default=0, description="Humanized character count")
    total_paragraphs: int = Field(default=0, description="Total paragraphs in document")
    modified_paragraphs_count: int = Field(default=0, description="Number of paragraphs modified")
    removed_cliches_count: int = Field(default=0, description="Total count of stylistic clichés removed")
    original_ai_paragraphs: int = Field(default=0, description="Count of AI paragraphs before humanization")
    humanized_ai_paragraphs: int = Field(default=0, description="Count of AI paragraphs after humanization")
    generator_provider: str = Field(default="", description="Active generator model or LLM used for humanization")
    external_llm_active: bool = Field(default=False, description="True if an external LLM was active")
    local_model_loaded: bool = Field(default=False, description="True if local PyTorch T5 model weights were loaded into memory")


class HumanizeResponse(BaseModel):
    original_text: str = Field(default="", description="Original full input text")
    humanized_text: str = Field(default="", description="Humanized full text")
    mode: str = Field(default="auto", description="Requested model language mode")
    effective_language: str = Field(default="en", description="Effective language applied")
    strength: str = Field(default="balanced", description="Humanization strength preset")
    request_id: str = Field(default="", description="Active request ID")
    generator_model: str = Field(default="", description="Active generator model or LLM used for humanization")
    external_llm_active: bool = Field(default=False, description="True if an external LLM was active")
    local_model_loaded: bool = Field(default=False, description="True if local PyTorch T5 model weights were loaded into memory")
    before_scale: float = Field(default=0.0, description="Baseline continuous AI probability percentage")
    after_scale: float = Field(default=0.0, description="Transformed continuous AI probability percentage")
    delta_scale: float = Field(default=0.0, description="AI probability reduction")
    before_paragraph_scale: float = Field(default=0.0, description="Paragraph-weighted baseline AI score")
    after_paragraph_scale: float = Field(default=0.0, description="Paragraph-weighted transformed AI score")
    before_confidence: dict[str, Any] = Field(default_factory=dict, description="Baseline confidence status")
    after_confidence: dict[str, Any] = Field(default_factory=dict, description="Transformed confidence status")
    summary: HumanizeSummary = Field(default_factory=HumanizeSummary, description="High-level transformation summary")
    removed_markers: list[dict[str, Any]] = Field(default_factory=list, description="Audited list of removed clichés")
    paragraphs: list[ParagraphDiff] = Field(default_factory=list, description="Granular paragraph transformation diffs")


class HumanizeDocumentResponse(HumanizeResponse):
    filename: str = Field(..., description="Name of the processed document file")
    file_type: str = Field(default="document", description="Detected file extension or type")


