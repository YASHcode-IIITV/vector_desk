import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

def get_client():
    return OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
_env = None

def get_env():
    global _env
    if _env is None:
        from environment.env import VectorDeskEnv
        _env = VectorDeskEnv()
    return _env

class ResetRequest(BaseModel):
    task_type: Optional[str] = "email"

class StepRequest(BaseModel):
    action_type: str
    parameters: dict = {}
    reasoning: str = ""

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    print("START", flush=True)
    env = get_env()
    obs = env.reset(task_type=req.task_type)
    print("STEP reset complete", flush=True)
    print("END", flush=True)
    return {
        "task_id": obs.task_id,
        "task_type": obs.task_type.value if hasattr(obs.task_type, 'value') else obs.task_type,
        "input_data": obs.input_data,
        "available_actions": obs.available_actions,
        "feedback": obs.feedback,
    }

@app.post("/step")
def step(req: StepRequest):
    print("START", flush=True)
    from environment.state import Action, ActionType
    env = get_env()
    action = Action(
        action_type=ActionType(req.action_type),
        task_type=env._state.current_task_type,
        parameters=req.parameters,
        reasoning=req.reasoning,
        retrieved_context_used=False,
    )
    obs, reward, done, info = env.step(action)
    print("STEP action complete", flush=True)
    print("END", flush=True)
    return {
        "task_id": obs.task_id,
        "feedback": obs.feedback,
        "reward": reward.total,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    env = get_env()
    s = env.state()
    return {
        "task_id": s.task_id,
        "step_count": s.step_count,
        "cumulative_reward": s.cumulative_reward,
        "done": s.done,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

def _fallback_action(task_type: str, obs) -> dict:
    if task_type == "email":
        return {"action_type": "classify_email", "parameters": {"priority": "medium", "category": "other"}, "reasoning": "Fallback", "retrieved_context_used": False}
    if task_type == "support":
        return {"action_type": "resolve_ticket", "parameters": {"resolution": "provide_documentation", "response": "We are looking into your issue."}, "reasoning": "Fallback", "retrieved_context_used": False}
    return {"action_type": "schedule_meeting", "parameters": {"time_slot": "2024-02-05 10:00", "attendees": [], "notes": "Scheduled."}, "reasoning": "Fallback", "retrieved_context_used": False}

def run_episode(env, task_type: str) -> dict:
    from environment.state import Action, ActionType, TaskType
    obs = env.reset(task_type=task_type)
    total_reward = 0.0
    steps = []

    print(f"START", flush=True)
    print(f"STEP task={task_type} id={obs.task_id}", flush=True)

    for step in range(5):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Task: {task_type}, Data: {json.dumps(obs.input_data)[:200]}, Actions: {obs.available_actions}. Respond with JSON: {{action_type, parameters, reasoning}}"}],
                timeout=30,
            )
            agent_data = json.loads(response.choices[0].message.content)
        except Exception:
            agent_data = _fallback_action(task_type, obs)

        try:
            action = Action(
                action_type=ActionType(agent_data["action_type"]),
                task_type=TaskType(task_type),
                parameters=agent_data.get("parameters", {}),
                reasoning=agent_data.get("reasoning", ""),
                retrieved_context_used=agent_data.get("retrieved_context_used", False),
            )
            obs, reward, done, info = env.step(action)
            total_reward += reward.total
            steps.append({"step": step+1, "action": agent_data["action_type"], "reward": reward.total})
            print(f"STEP {agent_data['action_type']} reward={reward.total:.3f}", flush=True)
            if done:
                break
        except Exception as e:
            print(f"STEP error={e}", flush=True)
            break

    print(f"END reward={total_reward:.3f}", flush=True)
    return {"task_type": task_type, "total_reward": total_reward, "steps": steps}

def main():
    from environment.env import VectorDeskEnv
    env = VectorDeskEnv()
    results = []
    for task in ["email", "support", "calendar"]:
        result = run_episode(env, task)
        results.append(result)

    print("\nBENCHMARK SUMMARY", flush=True)
    for r in results:
        print(f"  {r['task_type']:12s} | reward={r['total_reward']:.3f} | steps={len(r['steps'])}", flush=True)
    avg = sum(r["total_reward"] for r in results) / len(results)
    print(f"  AVERAGE REWARD: {avg:.3f}", flush=True)
    return results

if __name__ == "__main__":
    main()