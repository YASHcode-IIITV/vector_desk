"""
VectorDesk: Pydantic state models for the OpenEnv-compliant environment.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    EMAIL = "email"
    SUPPORT = "support"
    CALENDAR = "calendar"

class EmailPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ActionType(str, Enum):
    CLASSIFY_EMAIL = "classify_email"
    REPLY_EMAIL = "reply_email"
    RESOLVE_TICKET = "resolve_ticket"
    ESCALATE_TICKET = "escalate_ticket"
    ASK_CLARIFICATION = "ask_clarification"
    SCHEDULE_MEETING = "schedule_meeting"
    RESCHEDULE_MEETING = "reschedule_meeting"
    DECLINE_MEETING = "decline_meeting"
    RETRIEVE_CONTEXT = "retrieve_context"
    COMPLETE_TASK = "complete_task"


class Action(BaseModel):
    action_type: ActionType
    task_type: TaskType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    retrieved_context_used: bool = False

    class Config:
        use_enum_values = True


class RetrievedDocument(BaseModel):
    content: str
    source: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    task_type: TaskType
    task_id: str
    step: int
    input_data: Dict[str, Any]
    retrieved_context: List[RetrievedDocument] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    task_complete: bool = False
    feedback: Optional[str] = None

    class Config:
        use_enum_values = True


class RewardBreakdown(BaseModel):
    task_completion: float = 0.0
    context_utilization: float = 0.0
    response_quality: float = 0.0
    policy_compliance: float = 0.0
    efficiency: float = 0.0
    hallucination_penalty: float = 0.0


class Reward(BaseModel):
    total: float = Field(ge=-1.0, le=1.0)
    breakdown: RewardBreakdown
    explanation: str
    is_terminal: bool = False

    @classmethod
    def zero(cls) -> "Reward":
        return cls(total=0.0, breakdown=RewardBreakdown(), explanation="No reward yet.")

    @classmethod
    def penalty(cls, reason: str, amount: float = -0.2) -> "Reward":
        return cls(total=amount, breakdown=RewardBreakdown(hallucination_penalty=amount), explanation=f"Penalty: {reason}")


class EnvState(BaseModel):
    current_task_type: Optional[TaskType] = None
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 10
    task_data: Dict[str, Any] = Field(default_factory=dict)
    agent_memory: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
