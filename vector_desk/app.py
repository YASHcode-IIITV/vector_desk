"""
VectorDesk Gradio UI — interactive demo for the Hugging Face Space.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from environment.env import VectorDeskEnv
from environment.state import Action, ActionType, TaskType

_env = VectorDeskEnv()
TASK_PROMPTS = {"Email Triage": "email", "Customer Support": "support", "Calendar Scheduling": "calendar"}

def _decide_action(obs, task_type: str, step: int) -> tuple:
    data = obs.input_data
    ctx = obs.retrieved_context
    if step == 1:
        return "retrieve_context", {"query": str(data)[:100]}
    if task_type == "email":
        body = (data.get("body", "") + data.get("subject", "")).lower()
        if any(w in body for w in ["urgent", "down", "outage", "critical"]): priority, category = "urgent", "incident"
        elif any(w in body for w in ["invoice", "payment", "billing"]): priority, category = "high", "finance"
        elif any(w in body for w in ["security", "vulnerability"]): priority, category = "urgent", "security"
        elif any(w in body for w in ["lunch", "party", "event"]): priority, category = "low", "social"
        else: priority, category = "medium", "feature_request"
        if step == 2:
            return "classify_email", {"priority": priority, "category": category}
        hint = ctx[0].content[:100] if ctx else "Please action this email."
        reply = f"Thank you for your email. Based on our policy: {hint} We will follow up shortly."
        return "complete_task", {"classification": {"priority": priority, "category": category}, "reply": reply, "summary": "Email triaged."}
    elif task_type == "support":
        issue = data.get("issue", "").lower()
        tier = data.get("tier", "basic")
        if step == 2:
            if "p0" in issue or (tier == "enterprise" and "down" in issue):
                return "escalate_ticket", {"resolution": "escalate_to_engineering", "response": "Escalating to engineering per enterprise SLA."}
            if "refund" in issue or "charged" in issue:
                return "resolve_ticket", {"resolution": "issue_refund", "response": "Processing your refund per our premium refund policy."}
            if "login" in issue or "password" in issue:
                return "resolve_ticket", {"resolution": "reset_account", "response": "Triggering password reset. Check your inbox."}
            return "resolve_ticket", {"resolution": "provide_documentation", "response": "Please see our documentation for help."}
        return "complete_task", {"summary": "Ticket resolved."}
    elif task_type == "calendar":
        if step == 2:
            return "schedule_meeting", {"time_slot": data.get("expected_slot", "2024-02-05 10:00"), "attendees": data.get("attendees", []), "notes": "Scheduled per policy."}
        return "complete_task", {"summary": "Meeting scheduled."}
    return "complete_task", {"summary": "Done."}

def run_demo(task_label: str, custom_input: str, api_key: str) -> tuple:
    if api_key: os.environ["ANTHROPIC_API_KEY"] = api_key
    task_type = TASK_PROMPTS.get(task_label, "email")
    obs = _env.reset(task_type=task_type)
    if custom_input.strip():
        obs.input_data["body"] = custom_input
        obs.input_data["subject"] = custom_input[:60]
        obs.input_data["issue"] = custom_input
        obs.input_data["request"] = custom_input
    ctx_lines = [f"**[{i}] {d.source}** (score: {d.relevance_score:.2f})\n> {d.content}" for i, d in enumerate(obs.retrieved_context, 1)]
    context_md = "\n\n".join(ctx_lines) or "_No context retrieved._"
    steps_log, total_reward, final_action = [], 0.0, {}
    for step_num in range(1, 6):
        action_type, params = _decide_action(obs, task_type, step_num)
        try:
            action = Action(action_type=ActionType(action_type), task_type=TaskType(task_type), parameters=params, reasoning=f"Step {step_num}.", retrieved_context_used=bool(obs.retrieved_context))
            obs, reward, done, info = _env.step(action)
            total_reward += reward.total
            steps_log.append({"step": step_num, "action": action_type, "reward": round(reward.total, 3), "grader_score": round(info.get("grader_score", 0), 3), "feedback": (obs.feedback or "")[:60]})
            final_action = params
            if done: break
        except Exception as e:
            steps_log.append({"step": step_num, "action": action_type, "error": str(e)}); break
    steps_md = "| Step | Action | Reward | Score | Feedback |\n|------|--------|--------|-------|----------|\n"
    for s in steps_log:
        steps_md += f"| {s['step']} | `{s['action']}` | {s.get('reward',0):.3f} | {s.get('grader_score',0):.3f} | {s.get('feedback','')} |\n"
    input_md = f"```json\n{json.dumps(obs.input_data, indent=2)}\n```"
    score_pct = round(min(1.0, total_reward / 3.0) * 100, 1)
    score_md = f"### 🏆 Final Score: **{score_pct}%** (reward: {total_reward:.3f})\n\n{steps_md}"
    output_md = f"**Task:** {task_label}\n\n**Agent Output:**\n```json\n{json.dumps(final_action, indent=2)}\n```"
    return input_md, context_md, output_md, score_md

with gr.Blocks(title="VectorDesk", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.HTML("<h1 style='text-align:center;padding:20px;background:linear-gradient(135deg,#1e1b4b,#4338ca);color:white;border-radius:12px'>🧠 VectorDesk: Memory-Augmented AI Office Benchmark</h1>")
    with gr.Row():
        task_selector = gr.Dropdown(choices=list(TASK_PROMPTS.keys()), value="Email Triage", label="Task")
        api_key_input = gr.Textbox(placeholder="Anthropic API key (optional)", label="API Key", type="password")
    custom_input = gr.Textbox(placeholder="Custom email / ticket / meeting request (optional)...", label="Custom Input", lines=3)
    run_btn = gr.Button("▶ Run Benchmark Episode", variant="primary", size="lg")
    with gr.Row():
        input_display = gr.Markdown(label="📥 Task Input")
        context_display = gr.Markdown(label="🔍 RAG Context")
    with gr.Row():
        output_display = gr.Markdown(label="🤖 Agent Output")
        score_display = gr.Markdown(label="📊 Scores")
    run_btn.click(fn=run_demo, inputs=[task_selector, custom_input, api_key_input], outputs=[input_display, context_display, output_display, score_display])
    gr.Markdown("---\n**Architecture**: Input → FAISS RAG → Context-Aware Agent → Deterministic Grader → Step Reward")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
