"""
VectorDesk: Action space definitions.
Each task has its own action schema; all inherit from BaseAction.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from environment.state import Priority


class BaseAction(BaseModel):
    """Every action must have a type tag and optional free-text reasoning."""
    action_type: str
    reasoning: Optional[str] = None


# ── Email Actions ──────────────────────────────────────────────────────────────

class EmailTriageAction(BaseAction):
    action_type: str = "email_triage"
    category: str                       # e.g. "billing", "hr", "support", "spam"
    priority: Priority
    reply_draft: str                    # agent-generated reply
    escalate: bool = False              # escalate to human?


# ── Customer Support Actions ───────────────────────────────────────────────────

class SupportAction(BaseAction):
    action_type: str = "support"
    response: str                       # reply to the customer
    resolution_code: str                # e.g. "refund", "replacement", "info"
    policy_citations: List[str] = Field(default_factory=list)   # policies used
    resolved: bool = False


# ── Calendar Actions ───────────────────────────────────────────────────────────

class CalendarAction(BaseAction):
    action_type: str = "calendar"
    proposed_slot: str                  # ISO-like string: "2024-03-15T10:00"
    duration_minutes: int = 60
    attendees: List[str] = Field(default_factory=list)
    conflict_resolution: Optional[str] = None   # how conflicts were handled
    alternatives: List[str] = Field(default_factory=list)


# ── Union type for type-safe dispatch ─────────────────────────────────────────

Action = EmailTriageAction | SupportAction | CalendarAction
