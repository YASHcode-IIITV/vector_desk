import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment.env import VectorDeskEnv

app = FastAPI()
env = VectorDeskEnv()

class ResetRequest(BaseModel):
    task_type: Optional[str] = "email"

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

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)