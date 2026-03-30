"""Calendar Scheduling Task: schedule meetings, resolve conflicts."""
import random
from typing import Any, Dict, List

CALENDAR_TASKS = [
    {"request": "Schedule a 1-hour strategy meeting with the exec team",
     "attendees": ["ceo@company.com", "cto@company.com", "cfo@company.com"],
     "existing_events": [
         {"title": "Board Call", "start": "2024-02-05 09:00", "end": "2024-02-05 10:00"},
         {"title": "Investor Sync", "start": "2024-02-05 14:00", "end": "2024-02-05 15:00"},
     ],
     "preferred_window": "morning", "expected_slot": "2024-02-05 10:00"},
    {"request": "Book a 30-min daily standup for the engineering team",
     "attendees": ["eng1@company.com", "eng2@company.com", "eng3@company.com"],
     "existing_events": [
         {"title": "Existing Standup", "start": "2024-02-06 09:30", "end": "2024-02-06 10:00"},
     ],
     "preferred_window": "morning", "expected_slot": "2024-02-06 10:00"},
    {"request": "Schedule an urgent client call - they are threatening to churn",
     "attendees": ["client@bigcorp.com", "sales@company.com"],
     "existing_events": [
         {"title": "Team lunch", "start": "2024-02-07 12:00", "end": "2024-02-07 13:00"},
     ],
     "preferred_window": "asap", "expected_slot": "2024-02-07 09:00"},
]

class CalendarTask:
    def generate(self) -> Dict[str, Any]:
        task = random.choice(CALENDAR_TASKS).copy()
        task["task_type"] = "calendar"
        return task

    def build_query(self, task_data: Dict[str, Any]) -> str:
        return f"{task_data['request']} attendees {' '.join(task_data['attendees'])}"

    def available_actions(self) -> List[str]:
        return ["schedule_meeting", "reschedule_meeting", "decline_meeting", "retrieve_context", "complete_task"]
