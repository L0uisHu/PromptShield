# PromptShield

A proxy server that sits between any application and the Anthropic API, scanning messages for prompt injection attacks and logging everything to a dashboard.

## Architecture

```
Your App → PromptShield (/v1/messages) → Anthropic API
                  ↓
            Injection Detector
                  ↓
            Dashboard / Logs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```
