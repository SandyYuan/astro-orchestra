"""Minimal Pydantic models for structured data."""

from pydantic import BaseModel
from typing import Optional


class OrchestratorDecision(BaseModel):
    reasoning: str
    next_agent: Optional[str] = None
    summary: str 