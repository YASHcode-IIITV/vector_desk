"""
VectorDesk: Reward computation logic.
Step-wise rewards with partial credit and hallucination penalties.
"""
from typing import Any, Dict
from .state import Action, Reward, RewardBreakdown, ActionType


def compute_reward(
    action: Action,
    grader_score: float,
    context_was_retrieved: bool,
    step: int,
    max_steps: int,
    policy_compliant: bool = True,
) -> Reward:
    """
    Compute a step-level reward combining grader output with
    context-utilization bonuses and efficiency penalties.
    """
    # Base task completion reward
    task_completion = grader_score * 0.5

    # Reward for using retrieved context (encourages RAG usage)
    context_utilization = 0.15 if (action.retrieved_context_used and context_was_retrieved) else 0.0

    # Response quality: reward for providing reasoning
    response_quality = 0.10 if action.reasoning and len(action.reasoning) > 20 else 0.0

    # Policy compliance
    policy_compliance = 0.10 if policy_compliant else 0.0

    # Efficiency: small bonus for completing early
    steps_remaining = max_steps - step
    efficiency = min(0.10, steps_remaining * 0.01) if grader_score >= 0.8 else 0.0

    # Hallucination penalty: fired by grader externally, default 0
    hallucination_penalty = 0.0

    total = min(1.0, task_completion + context_utilization + response_quality + policy_compliance + efficiency)
    total = max(-1.0, total + hallucination_penalty)

    return Reward(
        total=round(total, 4),
        breakdown=RewardBreakdown(
            task_completion=task_completion,
            context_utilization=context_utilization,
            response_quality=response_quality,
            policy_compliance=policy_compliance,
            efficiency=efficiency,
            hallucination_penalty=hallucination_penalty,
        ),
        explanation=(
            f"Grader={grader_score:.2f}, CTX={'yes' if context_was_retrieved else 'no'}, "
            f"Policy={'OK' if policy_compliant else 'FAIL'}, Step={step}/{max_steps}"
        ),
        is_terminal=(grader_score >= 0.9 or step >= max_steps),
    )
