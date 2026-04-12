import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

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
    env = get_env()
    obs = env.reset(task_type=req.task_type)
    return {
        "task_id": obs.task_id,
        "task_type": obs.task_type.value if hasattr(obs.task_type, 'value') else obs.task_type,
        "input_data": obs.input_data,
        "available_actions": obs.available_actions,
        "feedback": obs.feedback,
    }

@app.post("/step")
def step(req: StepRequest):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

def main():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    main()
