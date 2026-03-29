"""Email Triage Task: classify, prioritize, and reply to office emails."""
import random
from typing import Any, Dict, List

EMAILS = [
    {"subject": "Server down in production!", "sender": "ops@company.com",
     "body": "The main API server has been throwing 500 errors for 20 minutes. Customers are complaining.",
     "expected_priority": "urgent", "expected_category": "incident"},
    {"subject": "Team lunch next Friday", "sender": "hr@company.com",
     "body": "We're organizing a team lunch at The Grand Bistro. Please RSVP by Wednesday.",
     "expected_priority": "low", "expected_category": "social"},
    {"subject": "Invoice overdue - ACTION REQUIRED", "sender": "billing@vendor.com",
     "body": "Your invoice #INV-2045 for $12,400 is 30 days overdue. Please remit payment immediately.",
     "expected_priority": "high", "expected_category": "finance"},
    {"subject": "Security vulnerability report", "sender": "security@certorg.com",
     "body": "A CVE-2024-5523 critical SQL injection vulnerability has been found in your login endpoint.",
     "expected_priority": "urgent", "expected_category": "security"},
    {"subject": "New feature request from client", "sender": "clientX@bigcorp.com",
     "body": "We'd love to see dark mode and CSV export added to the dashboard by Q3.",
     "expected_priority": "medium", "expected_category": "feature_request"},
]

class EmailTask:
    def generate(self) -> Dict[str, Any]:
        email = random.choice(EMAILS).copy()
        email["task_type"] = "email"
        return email

    def build_query(self, task_data: Dict[str, Any]) -> str:
        return f"{task_data['subject']} {task_data['body'][:100]}"

    def available_actions(self) -> List[str]:
        return ["classify_email", "reply_email", "retrieve_context", "complete_task"]
