"""Customer Support Grader: checks resolution correctness and policy compliance."""
from typing import Any, Dict, Tuple
from environment.state import Action, ActionType

RESOLUTION_SCORES = {
    ("resolve_ticket", "issue_refund"): 1.0,
    ("resolve_ticket", "reset_account"): 1.0,
    ("resolve_ticket", "provide_documentation"): 1.0,
    ("escalate_ticket", "escalate_to_engineering"): 1.0,
    ("resolve_ticket", "escalate_to_engineering"): 0.3,  # Should have escalated
    ("escalate_ticket", "issue_refund"): 0.4,  # Should have resolved
}

class SupportGrader:
    def grade(self, action: Action, task_data: Dict[str, Any], step: int) -> Tuple[float, bool, str]:
        if action.action_type == ActionType.RETRIEVE_CONTEXT:
            return 0.3, True, "Context retrieved. Resolve or escalate the ticket."
        if action.action_type in (ActionType.RESOLVE_TICKET, ActionType.ESCALATE_TICKET):
            return self._grade_resolution(action, task_data)
        if action.action_type == ActionType.ASK_CLARIFICATION:
            return 0.2, True, "Clarification asked. Proceed to resolution."
        if action.action_type == ActionType.COMPLETE_TASK:
            return 0.5, True, "Task completed."
        return 0.1, True, "Action acknowledged."

    def _grade_resolution(self, action: Action, task_data: Dict) -> Tuple[float, bool, str]:
        action_str = action.action_type.value if hasattr(action.action_type, 'value') else str(action.action_type)
        resolution = action.parameters.get("resolution", "")
        expected_action = task_data.get("expected_action", "")
        expected_resolution = task_data.get("expected_resolution", "")

        key = (action_str.replace("ActionType.", ""), resolution)
        score = RESOLUTION_SCORES.get(key, 0.2)

        # Context bonus
        if action.retrieved_context_used:
            score = min(1.0, score + 0.1)

        # Policy compliance: check if response text mentions policy
        response = action.parameters.get("response", "")
        policy_compliant = len(response) > 20
        if not policy_compliant:
            score *= 0.5

        fb = f"Action={action_str}, Resolution={resolution}, Expected={expected_resolution}"
        return round(score, 3), policy_compliant, fb
