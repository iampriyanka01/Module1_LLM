# 🧠 GenAI Learning Hub

An interactive Streamlit companion for the Generative AI course. It covers all nine modules with explanations,
real-life analogies, simulators, runnable code, live Gemini demos and a quiz for each module.

| Module | What you can do in the app |
|---|---|
| 1 · Foundations of GenAI | AI/ML/DL diagram, ML type quiz, a one-neuron gradient-descent demo, a GenAI timeline, a **temperature / top-k / top-p simulator** |
| 2 · GenAI Platforms | Step through pre-training → SFT → RLHF, a tokenizer, an **RLHF ranking game**, open vs closed models, SLMs, ethics cases |
| 3 · Prompt Basics | A **prompt builder** (role/instruction/input/format), a loss-in-the-middle test, crafting before/after, a zero/one/few-shot/role lab, dynamic templates |
| 4 · Advanced Prompting | CoT vs direct answers, a few-shot bias experiment, two-phase Auto-CoT, **self-consistency voting**, Tree-of-Thought |
| 5.1 · APIs | Request journey (DNS → TCP → packets → FIN), packet calculator, JSON validator, a **live HTTP playground**, status codes |
| 5.2 · LLM API Integration | API-key safety, Hello Gemini with token usage, cost / **smart-routing** / caching calculators, a retry simulator, input sanitising, a **request log dashboard** |
| 6 · Response Handling | Streaming vs blocking, parallel batch, a **Pydantic playground**, a failure-handling pipeline, a **guardrails Checker → Corrector → Rail → Guard** demo |
| 7 · Conversational Interfaces | Stateless vs chat session, a message builder, a system-prompt playground, decoding strategies, a context-window simulator, context strategies, and **your auto-summary chatbot** |
| 8 · Function Calling | Tool card builder, the lifecycle, a **live multi-tool loop**, a review of your weather script, an **async fan-out/fan-in simulator** |
| 9 · Evaluation | Hallucination test, a quirk-surface test suite, metric calculators, a 3-tier metric layering demo, a rubric builder, **your Study Buddy + LLM-as-judge**, an A/B refinement test |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your Gemini key (optional)
streamlit run app.py
```

You don't need an API key for the explanations and simulators. Only the 🟢 live demos need a Gemini key, which you can
paste in the sidebar or put in `.env` (`GEMINI_API_KEY` or `GOOGLE_API_KEY`). If the sidebar model gives "not found",
pick another model or type a custom ID.

## Project layout

```
app.py                  navigation + sidebar (key, model)
core/llm.py             Gemini client, retry with backoff, friendly errors, request log
core/ui.py              shared UI: cards, analogies, stepper, quizzes, progress
core/sim.py             offline simulators (sampling, toy tokenizer, token estimate)
core/course_prompts.py  Study Buddy + Evaluator prompts and safe_parse_json (from the course project)
lessons/*.py            one page per module
```
