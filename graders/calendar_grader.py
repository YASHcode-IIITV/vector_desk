"""Calendar Scheduling Grader: evaluates slot correctness and conflict resolution."""
from typing import Any, Dict, Tuple
from environment.state import Action, ActionType
from datetime import datetime

class CalendarGrader:
    def grade(self, action: Action, task_data: Dict[str, Any], step: int) -> Tuple[float, bool, str]:
        if action.action_type == ActionType.RETRIEVE_CONTEXT:
            return 0.3, True, "Context retrieved. Schedule the meeting."
        if action.action_type == ActionType.SCHEDULE_MEETING:
            return self._grade_schedule(action, task_data)
        if action.action_type == ActionType.RESCHEDULE_MEETING:
            return 0.6, True, "Rescheduled. Suboptimal but acceptable."
        if action.action_type == ActionType.DECLINE_MEETING:
            return 0.1, True, "Declined — low score unless unavoidable conflict."
        if action.action_type == ActionType.COMPLETE_TASK:
            return 0.5, True, "Task completed."
        return 0.1, True, "Action acknowledged."

    def _grade_schedule(self, action: Action, task_data: Dict) -> Tuple[float, bool, str]:
        proposed_slot = action.parameters.get("time_slot", "")
        attendees = action.parameters.get("attendees", [])
        expected_slot = task_data.get("expected_slot", "")
        existing_events = task_data.get("existing_events", [])

        score = 0.4  # Base for attempting scheduling

        # Check for conflicts
        conflict = self._check_conflict(proposed_slot, existing_events)
        if conflict:
            score = 0.1
            fb = f"CONFLICT detected with {conflict}."
            return score, False, fb

        # Check slot accuracy
        if proposed_slot and expected_slot:
            try:
                proposed_dt = datetime.strptime(proposed_slot[:16], "%Y-%m-%d %H:%M")
                expected_dt = datetime.strptime(expected_slot[:16], "%Y-%m-%d %H:%M")
                diff_hours = abs((proposed_dt - expected_dt).total_seconds()) / 3600
                if diff_hours == 0:
                    score = 1.0
                elif diff_hours <= 1:
                    score = 0.8
                elif diff_hours <= 3:
                    score = 0.6
                else:
                    score = 0.4
            except ValueError:
                score = 0.3

        if action.retrieved_context_used:
            score = min(1.0, score + 0.1)

        # Check attendees coverage
        expected_attendees = set(task_data.get("attendees", []))
        proposed_attendees = set(attendees)
        if expected_attendees and proposed_attendees:
            coverage = len(expected_attendees & proposed_attendees) / len(expected_attendees)
            score = score * 0.7 + coverage * 0.3

        return round(score, 3), True, f"Slot={proposed_slot}, Expected={expected_slot}"

    def _check_conflict(self, proposed_slot: str, existing_events: list) -> str:
        if not proposed_slot:
            return ""
        try:
            proposed_dt = datetime.strptime(proposed_slot[:16], "%Y-%m-%d %H:%M")
        except ValueError:
            return ""
        for event in existing_events:
            try:
                start = datetime.strptime(event["start"][:16], "%Y-%m-%d %H:%M")
                end = datetime.strptime(event["end"][:16], "%Y-%m-%d %H:%M")
                if start <= proposed_dt < end:
                    return event["title"]
            except (ValueError, KeyError):
                continue
        return ""
