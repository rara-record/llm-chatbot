# Repository Guidelines

## Project Structure & Module Organization
- `chat.py` hosts the Streamlit entry point, UI scaffolding, and session-state chat loop.
- `llm.py` encapsulates the LangChain RAG pipeline, including retriever wiring, prompt templates, and streaming response handling.
- `.env` stores local secrets such as `OPENAI_API_KEY`, `PINECONE_API_KEY`, and project-specific configuration; keep it out of version control.
- Place new data loaders or prompt helpers beside `llm.py`. If you add substantial packages, create a `requirements.txt` or `pyproject.toml` at the repository root for reproducible installs.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates and activates an isolated environment aligned with `.python-version`.
- `pip install streamlit langchain python-dotenv pinecone-client` pulls the minimum runtime stack; freeze updates into `requirements.txt` after dependency changes.
- `streamlit run chat.py` launches the local chatbot, enabling live UI iteration.
- `python -m pip check` verifies dependency consistency; run it before provisioning deployments.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; prefer descriptive snake_case names for variables, functions, and session keys.
- Keep Streamlit widgets and prompts narrated in Korean to match the existing UX, but document helpers and comments in clear English.
- Store constants (model IDs, index names) in uppercase at the module top; consider centralizing them if variants proliferate.
- Favor small, pure functions inside `llm.py` so retriever components remain composable and testable.

## Testing Guidelines
- Adopt `pytest` with tests grouped under `tests/` mirroring module names (e.g., `tests/test_llm.py`).
- Mock external services (OpenAI, Pinecone) via fixtures; validate prompt flows by asserting on rendered templates and retriever parameters.
- Before PR review, run `pytest -q` and attach coverage output (`coverage run -m pytest && coverage report`) once instrumentation is added.

## Commit & Pull Request Guidelines
- Follow the conventional commit pattern seen in history (e.g., `feat(chat): ...`, `refactor(chat): ...`). Scope tags should identify the touched module.
- Limit commits to focused changes and include Korean user-facing copy updates in the description when relevant.
- For PRs, provide a concise summary, testing evidence, affected environment notes, and screenshots of key UI changes.
- Link issues or tasks, flag required secrets, and request reviewers familiar with LangChain integration when altering `llm.py`.

## Environment & Secrets
- Use `.env.example` to document required keys; regenerate embeddings before deployment if Pinecone indexes evolve.
- Rotate API credentials regularly and avoid logging raw prompts or secrets in Streamlit outputs.
