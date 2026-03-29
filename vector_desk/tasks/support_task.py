"""Customer Support Task: resolve tickets following company policies."""
import random
from typing import Any, Dict, List

TICKETS = [
    {"ticket_id": "TKT-001", "customer": "Alice Brown", "tier": "premium",
     "issue": "I was charged twice for my subscription this month. I need a refund immediately.",
     "expected_action": "resolve_ticket", "expected_resolution": "issue_refund",
     "policy": "premium_refund_policy"},
    {"ticket_id": "TKT-002", "customer": "Bob Smith", "tier": "basic",
     "issue": "I can't log in to my account. Password reset email never arrives.",
     "expected_action": "resolve_ticket", "expected_resolution": "reset_account",
     "policy": "account_access_policy"},
    {"ticket_id": "TKT-003", "customer": "Carol Jones", "tier": "enterprise",
     "issue": "Our entire team lost access to the API at 3am. This is a P0 for us.",
     "expected_action": "escalate_ticket", "expected_resolution": "escalate_to_engineering",
     "policy": "enterprise_sla_policy"},
    {"ticket_id": "TKT-004", "customer": "Dave Lee", "tier": "basic",
     "issue": "How do I export my data as CSV?",
     "expected_action": "resolve_ticket", "expected_resolution": "provide_documentation",
     "policy": "general_support_policy"},
]

class SupportTask:
    def generate(self) -> Dict[str, Any]:
        ticket = random.choice(TICKETS).copy()
        ticket["task_type"] = "support"
        return ticket

    def build_query(self, task_data: Dict[str, Any]) -> str:
        return f"{task_data['issue']} customer tier {task_data['tier']}"

    def available_actions(self) -> List[str]:
        return ["resolve_ticket", "escalate_ticket", "ask_clarification", "retrieve_context", "complete_task"]
