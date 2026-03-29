"""Email Triage Grader: deterministic scoring for email classification and reply."""
from typing import Any, Dict, Tuple
from environment.state import Action, ActionType

PRIORITY_MAP = {"urgent": 4, "high": 3, "medium": 2, "low": 1}

class EmailGrader:
    def grade(self, action: Action, task_data: Dict[str, Any], step: int) -> Tuple[float, bool, str]:
        if action.action_type == ActionType.RETRIEVE_CONTEXT:
            return 0.3, True, "Context retrieved. Now classify and reply."
        if action.action_type == ActionType.CLASSIFY_EMAIL:
            return self._grade_classify(action, task_data)
        if action.action_type == ActionType.REPLY_EMAIL:
            return self._grade_reply(action, task_data)
        if action.action_type == ActionType.COMPLETE_TASK:
            # Check if both classification and reply were attempted
            score = 0.5 if action.parameters.get("classification") and action.parameters.get("reply") else 0.2
            return score, True, "Task completed."
        return 0.1, True, "Action acknowledged."

    def _grade_classify(self, action: Action, task_data: Dict) -> Tuple[float, bool, str]:
        predicted_priority = action.parameters.get("priority", "").lower()
        predicted_category = action.parameters.get("category", "").lower()
        expected_priority = task_data.get("expected_priority", "medium")
        expected_category = task_data.get("expected_category", "")

        priority_score = 0.0
        if predicted_priority == expected_priority:
            priority_score = 1.0
        elif abs(PRIORITY_MAP.get(predicted_priority, 0) - PRIORITY_MAP.get(expected_priority, 0)) == 1:
            priority_score = 0.5  # Partial credit for adjacent priority

        category_score = 1.0 if predicted_category == expected_category else 0.0
        score = (priority_score * 0.6 + category_score * 0.4)
        policy = predicted_priority in PRIORITY_MAP
        fb = f"Priority: {predicted_priority} (expected {expected_priority}), Category: {predicted_category} (expected {expected_category})"
        return round(score, 3), policy, fb

    def _grade_reply(self, action: Action, task_data: Dict) -> Tuple[float, bool, str]:
        reply = action.parameters.get("reply", "")
        if not reply:
            return 0.0, False, "No reply generated."
        score = 0.4  # Base for any reply
        if len(reply) > 50:
            score += 0.2
        # Check if reply acknowledges key content
        body_words = set(task_data.get("body", "").lower().split())
        reply_words = set(reply.lower().split())
        overlap = len(body_words & reply_words) / (len(body_words) + 1)
        score += min(0.3, overlap * 3)
        if action.retrieved_context_used:
            score += 0.1
        return round(min(1.0, score), 3), True, f"Reply length: {len(reply)} chars."
