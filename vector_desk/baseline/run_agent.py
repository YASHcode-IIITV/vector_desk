"""
VectorDesk Baseline Agent: RAG-augmented GPT agent that runs all three tasks.
Usage: python -m baseline.run_agent
"""
from __future__ import annotations
import json
import os
import sys
from typing import Any, Dict, List

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import VectorDeskEnv
from environment.state import Action, ActionType, TaskType


SYSTEM_PROMPT = """You are an intelligent office assistant AI agent.
You will receive task observations containing:
- input_data: the task (email/ticket/calendar request)
- retrieved_context: relevant policies and history from the knowledge base
- available_actions: what you can do

You MUST use the retrieved context to inform your decisions.
Always provide reasoning and structured responses.

Respond ONLY with valid JSON matching this schema:
{
  "action_type": "<one of the available actions>",
  "parameters": { ... action-specific params ... },
  "reasoning": "<your chain of thought>",
  "retrieved_context_used": true
}

Action parameters:
- classify_email: {"priority": "low|medium|high|urgent", "category": "incident|finance|security|feature_request|social|other"}
- reply_email: {"reply": "<email reply text>"}
- resolve_ticket: {"resolution": "issue_refund|reset_account|provide_documentation|other", "response": "<customer response>"}
- escalate_ticket: {"resolution": "escalate_to_engineering|escalate_to_management", "response": "<customer response>"}
- schedule_meeting: {"time_slot": "YYYY-MM-DD HH:MM", "attendees": ["email1", "email2"], "notes": "..."}
- complete_task: {"summary": "..."}
"""


def call_claude_api(messages: List[Dict]) -> str:
    """Call Claude API for agent decisions."""
    import urllib.request
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _fallback_action(messages)
    
    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }).encode()
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def _fallback_action(messages: List[Dict]) -> str:
    """Rule-based fallback when no API key is available."""
    content = str(messages[-1].get("content", ""))
    if "email" in content.lower():
        if "urgent" in content.lower() or "down" in content.lower():
            return json.dumps({"action_type": "classify_email", "parameters": {"priority": "urgent", "category": "incident"}, "reasoning": "Keywords indicate incident.", "retrieved_context_used": True})
        return json.dumps({"action_type": "classify_email", "parameters": {"priority": "medium", "category": "other"}, "reasoning": "Default classification.", "retrieved_context_used": False})
    if "ticket" in content.lower() or "support" in content.lower():
        return json.dumps({"action_type": "resolve_ticket", "parameters": {"resolution": "provide_documentation", "response": "Thank you for contacting support. We are looking into your issue."}, "reasoning": "Default resolve.", "retrieved_context_used": False})
    return json.dumps({"action_type": "complete_task", "parameters": {"summary": "Task completed via fallback."}, "reasoning": "No API key.", "retrieved_context_used": False})


def obs_to_prompt(obs) -> str:
    ctx_text = "\n".join([f"[{d.source}]: {d.content}" for d in obs.retrieved_context[:3]])
    return f"""TASK TYPE: {obs.task_type}
TASK DATA: {json.dumps(obs.input_data, indent=2)}
RETRIEVED CONTEXT:
{ctx_text or '(none)'}
AVAILABLE ACTIONS: {', '.join(obs.available_actions)}
STEP: {obs.step}
FEEDBACK: {obs.feedback or ''}
"""


def run_episode(env: VectorDeskEnv, task_type: str) -> Dict[str, Any]:
    obs = env.reset(task_type=task_type)
    messages = []
    total_reward = 0.0
    steps = []

    print(f"\n{'='*60}")
    print(f"Task: {task_type.upper()} | ID: {obs.task_id}")
    print(f"Input: {json.dumps(obs.input_data, indent=2)[:200]}...")

    for step in range(10):
        prompt = obs_to_prompt(obs)
        messages.append({"role": "user", "content": prompt})

        try:
            response_text = call_claude_api(messages)
            # Strip markdown if present
            cleaned = response_text.strip().lstrip("```json").rstrip("```").strip()
            agent_data = json.loads(cleaned)
        except Exception as e:
            print(f"  [Step {step+1}] Parse error: {e}")
            agent_data = {"action_type": "complete_task", "parameters": {"summary": "Error fallback"}, "reasoning": str(e), "retrieved_context_used": False}

        messages.append({"role": "assistant", "content": json.dumps(agent_data)})

        try:
            action = Action(
                action_type=ActionType(agent_data["action_type"]),
                task_type=TaskType(task_type),
                parameters=agent_data.get("parameters", {}),
                reasoning=agent_data.get("reasoning", ""),
                retrieved_context_used=agent_data.get("retrieved_context_used", False),
            )
        except Exception as e:
            print(f"  [Step {step+1}] Invalid action: {e}")
            break

        obs, reward, done, info = env.step(action)
        total_reward += reward.total

        steps.append({"step": step + 1, "action": agent_data["action_type"], "reward": reward.total, "score": info.get("grader_score", 0)})
        print(f"  [Step {step+1}] {agent_data['action_type']} → reward={reward.total:.3f} | {reward.explanation}")

        if done:
            break

    print(f"  TOTAL REWARD: {total_reward:.3f}")
    return {"task_type": task_type, "task_id": obs.task_id, "total_reward": total_reward, "steps": steps}


def main():
    env = VectorDeskEnv()
    results = []
    for task in ["email", "support", "calendar"]:
        result = run_episode(env, task)
        results.append(result)

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['task_type']:12s} | reward={r['total_reward']:.3f} | steps={len(r['steps'])}")
    avg = sum(r["total_reward"] for r in results) / len(results)
    print(f"\n  AVERAGE REWARD: {avg:.3f}")
    return results


if __name__ == "__main__":
    main()
