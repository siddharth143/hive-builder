## Gmail Digest Action

A lightweight GitHub Actions workflow that **fetches recent Gmail messages from selected labels**, filters for signal, generates **PM‑oriented summaries with Gemini**, and posts a **structured daily digest to Slack**.

This repo is designed to be fork‑and‑go: you configure a handful of GitHub Secrets, and the scheduled workflow runs automatically.

---

## What it does

- **Gmail**: Reads messages from the last \(N\) days for each label in `GMAIL_LABELS`.
  - Supports **hierarchical labels**: if you specify `AI Product Management`, it also includes `AI Product Management/...`.
- **Filter**: Removes obvious noise using keyword matching on subject/snippet.
- **Summarize + score**: Uses Gemini to produce a concise summary plus:
  - **Relevance** \(1–5\)
  - **Urgency** \(High / Medium / Low\)
- **Slack**: Posts a scannable digest using Slack Block Kit.
  - Automatically **splits into multiple Slack messages** to avoid Slack’s 50‑block limit.

---

## How the “agentic” workflow works (high level)

The workflow behaves like a small agentic pipeline:

- **Observe**: pull candidate messages from Gmail by label and time window
- **Decide**: drop low‑signal items via deterministic filters and model judgement (`SKIP`)
- **Synthesize**: summarize remaining emails in a consistent, PM‑friendly format
- **Deliver**: publish the digest to Slack in a structured, easy‑to‑scan layout

It’s intentionally simple: the “agent” is the script + prompt, running on a schedule, producing repeatable output.

---

## Repo layout

```text
.github/workflows/daily-digest.yml   # scheduled + manual workflow
src/digest.py                        # fetch → filter → summarize → Slack
requirements.txt                     # Python dependencies
.env.example                         # config template (no secrets)
```

---

## Quickstart (fork + run)

### 1) Fork this repository

Fork to your GitHub account (or org) so the workflow can run under your ownership.

### 2) Add GitHub Actions secrets

In your forked repo:

- **Settings → Secrets and variables → Actions → Repository secrets**

Create secrets for each key in `.env.example` \(names must match exactly\). You’ll typically need:

- **Gmail OAuth**: `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`
- **Gemini**: `GEMINI_API_KEY`, `GEMINI_MODEL`
- **Slack**: `SLACK_WEBHOOK_URL`
- **Config**: `GMAIL_LABELS`, `DATE_WINDOW_DAYS`, `FILTER_KEYWORDS`, `MAX_EMAILS_PER_LABEL`, `SUMMARISATION_PERSONA`

Notes:

- **Never commit real secrets**. Use `.env` locally and GitHub Secrets in CI.
- If a secret is missing, the script exits with a non‑zero code and the run fails.

### 3) Trigger a manual run (recommended)

Go to:

- **Actions → Daily Gmail digest → Run workflow**

This validates your secrets/config before waiting for the cron schedule.

### 4) Let the schedule run

The workflow is configured to run daily at **9:00 AM IST** and also supports manual triggers:

- File: `.github/workflows/daily-digest.yml`

---

## Local development (optional)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill values locally:

```bash
cp .env.example .env
```

3. Run:

```bash
python src/digest.py
```

---

## Configuration reference

All configuration is read from environment variables (loaded via `python-dotenv` locally, GitHub Secrets in Actions).

- **`GMAIL_LABELS`**: Comma‑separated label names (parent labels supported).
  - Example: `Education,AI Product Management`
- **`DATE_WINDOW_DAYS`**: Lookback window used by Gmail query (`newer_than:Xd`).
- **`FILTER_KEYWORDS`**: Comma‑separated keywords/phrases; if a keyword appears in subject/snippet, the email is dropped.
- **`MAX_EMAILS_PER_LABEL`**: Cap per parent label for the time window.
- **`SUMMARISATION_PERSONA`**: Persona string to steer summaries.
- **`GEMINI_MODEL`**: Model name (example: `gemini-2.0-flash`).

---

## Troubleshooting

- **Digest is empty**: Check the Actions logs for `[label] ... fetched=...` counts.
  - If `fetched=0`, increase `DATE_WINDOW_DAYS` temporarily or verify labels have recent mail.
- **Some emails missing in Slack**: The script now splits posts across multiple Slack messages.
- **Gmail label not found**: Ensure label names match Gmail exactly (case-insensitive, but spacing matters).
- **`unauthorized_client`**: The **refresh token and OAuth client do not match**. `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, and `GMAIL_REFRESH_TOKEN` must all come from the **same** OAuth 2.0 Client ID in Google Cloud. Re-issue the refresh token after fixing the client ID/secret (e.g. do not pair a token from OAuth Playground with a different client’s secrets).
- **`invalid_grant` / “Token has been expired or revoked”**: Your **Gmail OAuth refresh token** is no longer valid for Google’s token endpoint.
  - **Fix**: Complete the OAuth consent flow again locally, copy the new **refresh token**, and update the `GMAIL_REFRESH_TOKEN` repository secret. Confirm `GMAIL_CLIENT_ID` and `GMAIL_CLIENT_SECRET` still match the same OAuth client in Google Cloud.
  - **Why it happens** (you cannot fully “prevent” Google from invalidating tokens, but you can reduce surprises):
    - You **revoked** the app under [Google Account → Third-party access](https://myaccount.google.com/permissions).
    - The OAuth client **secret was regenerated** in Google Cloud Console.
    - The OAuth consent screen is in **Testing** mode: refresh tokens for users outside your workspace can **expire after 7 days** unless you publish the app or use a supported long-lived setup for your use case.
    - Google may invalidate older refresh tokens when **too many** are issued for the same user + OAuth client (typical limit is on the order of **50** per user per client).
  - **FutureWarning about `google.generativeai`**: Harmless for now; the package is deprecated in favor of `google.genai`. Plan to migrate when convenient.

---

## Security

- Treat OAuth refresh tokens, API keys, and webhook URLs as **password‑equivalent**.
- Rotate credentials immediately if they are ever exposed.

---

## License

Add a license if you plan to distribute this broadly.