"""
VectorDesk: Main OpenEnv-compliant environment.
Orchestrates tasks, RAG retrieval, and reward computation.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, Optional, Tuple

from .state import Action, ActionType, EnvState, Observation, Reward, TaskType
from .reward import compute_reward
from tasks.email_task import EmailTask
from tasks.support_task import SupportTask
from tasks.calendar_task import CalendarTask
from rag.retriever import Retriever
from graders.email_grader import EmailGrader
from graders.support_grader import SupportGrader
from graders.calendar_grader import CalendarGrader


class VectorDeskEnv:
    """
    OpenEnv-compliant benchmark environment for memory-augmented AI agents.

    Usage:
        env = VectorDeskEnv()
        obs = env.reset(task_type="email")
        obs, reward, done, info = env.step(action)
    """

    TASK_MAP = {
        TaskType.EMAIL: EmailTask,
        TaskType.SUPPORT: SupportTask,
        TaskType.CALENDAR: CalendarTask,
    }
    GRADER_MAP = {
        TaskType.EMAIL: EmailGrader,
        TaskType.SUPPORT: SupportGrader,
        TaskType.CALENDAR: CalendarGrader,
    }

    def __init__(self, retriever: Optional[Retriever] = None):
        self._state = EnvState()
        self._task = None
        self._grader = None
        self._retriever = retriever or Retriever()

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self, task_type: Optional[str] = None) -> Observation:
        """Reset environment and return initial observation."""
        task_type_enum = TaskType(task_type) if task_type else TaskType.EMAIL
        task_cls = self.TASK_MAP[task_type_enum]
        grader_cls = self.GRADER_MAP[task_type_enum]

        self._task = task_cls()
        self._grader = grader_cls()
        task_data = self._task.generate()

        self._state = EnvState(
            current_task_type=task_type_enum,
            task_id=str(uuid.uuid4())[:8],
            task_data=task_data,
            max_steps=10,
        )

        # Auto-retrieve context on reset so agent always has RAG context
        retrieved = self._retriever.retrieve(
            query=self._task.build_query(task_data),
            task_type=task_type_enum.value,
            top_k=3,
        )

        return Observation(
            task_type=task_type_enum,
            task_id=self._state.task_id,
            step=0,
            input_data=task_data,
            retrieved_context=retrieved,
            available_actions=self._task.available_actions(),
            feedback="Task loaded. Retrieved context available.",
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute one agent action; return (observation, reward, done, info)."""
        if self._state.done:
            raise RuntimeError("Environment is done. Call reset() first.")

        self._state.step_count += 1

        # Validate action belongs to current task
        if action.task_type != self._state.current_task_type:
            reward = Reward.penalty("Action task_type mismatch.")
            return self._make_obs(feedback="Wrong task type for current episode."), reward, False, {}

        # Grade the action
        grader_score, policy_compliant, feedback = self._grader.grade(
            action=action,
            task_data=self._state.task_data,
            step=self._state.step_count,
        )

        # Retrieve fresh context if agent explicitly requested it
        retrieved = []
        if action.action_type == ActionType.RETRIEVE_CONTEXT:
            query = action.parameters.get("query", self._task.build_query(self._state.task_data))
            retrieved = self._retriever.retrieve(query=query, task_type=self._state.current_task_type.value, top_k=3)
            action.retrieved_context_used = True

        # Compute reward
        reward = compute_reward(
            action=action,
            grader_score=grader_score,
            context_was_retrieved=bool(retrieved) or action.retrieved_context_used,
            step=self._state.step_count,
            max_steps=self._state.max_steps,
            policy_compliant=policy_compliant,
        )

        # Update state
        self._state.cumulative_reward += reward.total
        self._state.agent_memory.append({
            "step": self._state.step_count,
            "action": action.action_type,
            "score": grader_score,
        })

        # Determine termination
        done = (
            action.action_type == ActionType.COMPLETE_TASK
            or grader_score >= 0.95
            or self._state.step_count >= self._state.max_steps
        )
        self._state.done = done
        reward.is_terminal = done

        obs = self._make_obs(retrieved=retrieved, feedback=feedback)
        info = {
            "grader_score": grader_score,
            "cumulative_reward": self._state.cumulative_reward,
            "step": self._state.step_count,
        }
        return obs, reward, done, info

    def state(self) -> EnvState:
        """Return current environment state snapshot."""
        return self._state.copy()

    # ── Internals ────────────────────────────────────────────────────────────

    def _make_obs(self, retrieved=None, feedback: str = "") -> Observation:
        return Observation(
            task_type=self._state.current_task_type,
            task_id=self._state.task_id,
            step=self._state.step_count,
            input_data=self._state.task_data,
            retrieved_context=retrieved or [],
            available_actions=self._task.available_actions() if self._task else [],
            task_complete=self._state.done,
            feedback=feedback,
        )
