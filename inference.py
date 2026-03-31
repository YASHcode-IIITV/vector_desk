import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from environment.env import VectorDeskEnv
from environment.state import Action, ActionType, TaskType

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
app = FastAPI()
env = VectorDeskEnv()

class ResetRequest(BaseModel):
    task_type: Optional[str] = "email"

class StepRequest(BaseModel):
    action_type: str
    parameters: dict = {}
    reasoning: str = ""

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = env.reset(task_type=req.task_type)
    return {
        "task_id": obs.task_id,
        "task_type": obs.task_type.value,
        "input_data": obs.input_data,
        "available_actions": obs.available_actions,
        "feedback": obs.feedback,
    }

@app.post("/step")
def step(req: StepRequest):
    action = Action(
        action_type=ActionType(req.action_type),
        task_type=env._state.current_task_type,
        parameters=req.parameters,
        reasoning=req.reasoning,
        retrieved_context_used=False,
    )
    obs, reward, done, info = env.step(action)
    return {
        "task_id": obs.task_id,
        "feedback": obs.feedback,
        "reward": reward.total,
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)