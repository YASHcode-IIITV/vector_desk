"""
VectorDesk: Observation space returned after each env.step().
Agents receive structured observations so they always know what happened.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Returned to the agent after every step."""
    task_type: str
    step_number: int
    input_data: Dict[str, Any]              # the raw task data (email, ticket, etc.)
    retrieved_context: List[str]            # RAG-retrieved passages
    feedback: Optional[str] = None         # grader feedback on the last action
    partial_score: float = 0.0             # score earned this step
    cumulative_score: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)

    def to_prompt_str(self) -> str:
        """Serialise the observation into a plain-English prompt for an LLM agent."""
        ctx = "\n".join(f"  - {c}" for c in self.retrieved_context) or "  (none)"
        return (
            f"[Step {self.step_number}] Task: {self.task_type}\n"
            f"Input:\n  {self.input_data}\n"
            f"Retrieved Context:\n{ctx}\n"
            f"Last Feedback: {self.feedback or 'N/A'}\n"
            f"Cumulative Score: {self.cumulative_score:.2f}"
        )
