# 🧠 VectorDesk: Memory-Augmented AI Office Benchmark

A production-grade OpenEnv-compliant environment for evaluating AI agents on real-world office workflows using RAG, Vector Databases, and Context-Aware Generation.

## 🎯 Why It Matters

Modern AI assistants fail at office work not because they lack reasoning, but because they lack **memory** and **context**. VectorDesk bridges this gap by evaluating agents on:
- Retrieval-Augmented Generation (RAG) from a living knowledge base
- Multi-turn context-aware decision making
- Adherence to company policies retrieved dynamically

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     VectorDesk                          │
│                                                         │
│  ┌─────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │  Task   │───▶│  VectorDesk  │───▶│    Grader     │  │
│  │Generator│    │     Env      │    │  (0.0 - 1.0)  │  │
│  └─────────┘    └──────┬───────┘    └───────────────┘  │
│                        │                                 │
│               ┌────────▼────────┐                       │
│               │   RAG Retriever │                       │
│               │  ┌───────────┐  │                       │
│               │  │  FAISS /  │  │                       │
│               │  │  Cosine   │  │                       │
│               │  │  Vector   │  │                       │
│               │  │  Store    │  │                       │
│               │  └───────────┘  │                       │
│               │  ┌───────────┐  │                       │
│               │  │Embeddings │  │                       │
│               │  │(MiniLM /  │  │                       │
│               │  │OpenAI /   │  │                       │
│               │  │TF-IDF)    │  │                       │
│               │  └───────────┘  │                       │
│               └─────────────────┘                       │
└─────────────────────────────────────────────────────────┘
         │                    │                │
    Email Triage       Customer Support   Calendar Scheduling
```

## 📋 Tasks

| Task | Description | Key Actions |
|------|-------------|-------------|
| **Email Triage** | Classify priority & category, generate reply | classify_email, reply_email |
| **Customer Support** | Resolve tickets per company policy | resolve_ticket, escalate_ticket |
| **Calendar Scheduling** | Schedule meetings, avoid conflicts | schedule_meeting, reschedule_meeting |

## 🔑 Action & Observation Space

### Actions
```python
Action(
    action_type: ActionType,        # One of the task-specific actions
    task_type: TaskType,            # email | support | calendar
    parameters: Dict[str, Any],     # Action-specific params
    reasoning: str,                 # Agent chain-of-thought
    retrieved_context_used: bool    # Whether RAG context was used
)
```

### Observations
```python
Observation(
    task_type: TaskType,
    task_id: str,
    step: int,
    input_data: Dict,               # Raw task input
    retrieved_context: List[RetrievedDocument],  # RAG results
    available_actions: List[str],
    task_complete: bool,
    feedback: str
)
```

## 🏆 Reward Design

| Component | Weight | Description |
|-----------|--------|-------------|
| Task Completion | 0.50 | Grader score × 0.5 |
| Context Utilization | 0.15 | Bonus for using RAG |
| Response Quality | 0.10 | Non-trivial reasoning |
| Policy Compliance | 0.10 | Follows company policy |
| Efficiency | 0.10 | Early completion bonus |
| Hallucination Penalty | −0.20 | Wrong task type etc. |

## 🚀 Running Locally

```bash
# Install deps
pip install -r requirements.txt

# Run benchmark agent
python -m baseline.run_agent

# Launch Gradio UI
python app/app.py
```

## 🐳 Docker

```bash
docker build -t vectordesk .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... vectordesk
```

## 🤗 Hugging Face Deployment

```bash
# Create a new HF Space (SDK: gradio)
# Upload all files from this repo
# Set OPENAI_API_KEY in Space secrets
```

## 📊 Example Output

```
============================================================
Task: EMAIL | ID: a3f2c1b0
Input: {"subject": "Server down in production!", ...}
  [Step 1] retrieve_context → reward=0.300
  [Step 2] classify_email  → reward=0.820 | Priority: urgent ✓
  [Step 3] complete_task   → reward=0.650
  TOTAL REWARD: 1.770

BENCHMARK SUMMARY
  email        | reward=1.770 | steps=3
  support      | reward=1.420 | steps=3
  calendar     | reward=1.680 | steps=3
  AVERAGE REWARD: 1.623
```

## 📁 Project Structure

```
vector_desk/
├── environment/      # OpenEnv core (env.py, state.py, reward.py)
├── tasks/            # Task generators (email, support, calendar)
├── rag/              # Vector store, retriever, embeddings
├── graders/          # Deterministic scoring per task
├── baseline/         # Claude-based benchmark agent
├── app/              # Gradio UI
├── openenv.yaml      # Environment specification
├── Dockerfile
└── requirements.txt
```
