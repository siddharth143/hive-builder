"""Gmail label digest: fetch, filter, summarise with Gemini, post to Slack."""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.genai import errors as genai_errors
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Constants ──────────────────────────────────────────────────────────────────
GMAIL_SCOPES = ("https://www.googleapis.com/auth/gmail.readonly",)
GMAIL_TOKEN_URI = "https://oauth2.googleapis.com/token"
GMAIL_EXECUTE_RETRIES = 3       # built-in retry in googleapiclient .execute()

GEMINI_MAX_OUTPUT_TOKENS = 2048
GEMINI_TLDR_MAX_OUTPUT_TOKENS = 256
GEMINI_MAX_WORKERS = 5          # concurrent summarisation threads
GEMINI_RETRY_ATTEMPTS = 3       # per-call retry on transient errors
GEMINI_RETRY_BACKOFF = 2.0      # exponential backoff base (seconds)

SLACK_MAX_BLOCKS_PER_MSG = 49   # Slack hard-limits messages to 50 blocks
SLACK_SUMMARY_MAX_CHARS = 700
SLACK_TLDR_MAX_BULLETS = 3
SLACK_RETRY_ATTEMPTS = 3
SLACK_RETRY_BACKOFF = 2.0

VALID_URGENCY: frozenset[str] = frozenset({"High", "Medium", "Low"})

DEBUG = (os.environ.get("DEBUG") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

# Exceptions that warrant a retry on Gemini calls (server-side / network).
# ClientError (4xx, e.g. invalid API key) is intentionally excluded — retrying
# won't help and would just delay the fatal exit.
_RETRYABLE_GEMINI: tuple[type[BaseException], ...] = (
    genai_errors.ServerError,
    ConnectionError,
    TimeoutError,
    requests.exceptions.RequestException,
)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class EmailItem:
    msg_id: str
    thread_id: str
    subject: str
    from_addr: str
    date: str
    snippet: str


@dataclass
class ScoredSummary:
    summary: str
    relevance: int
    urgency: str  # "High" | "Medium" | "Low"


@dataclass(frozen=True)
class GmailLabel:
    id: str
    name: str


# ── Utilities ──────────────────────────────────────────────────────────────────

def _slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 1)].rstrip() + "…"


def _parse_csv_env(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _debug(msg: str) -> None:
    if DEBUG:
        print(msg, file=sys.stderr)


def _with_retry(
    fn: Any,
    *,
    max_attempts: int,
    backoff_base: float,
    retryable: tuple[type[BaseException], ...],
    label: str = "operation",
) -> Any:
    """Execute *fn()* up to *max_attempts* times, backing off on *retryable* exceptions."""
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt < max_attempts:
                wait = backoff_base ** attempt
                _debug(f"[retry] {label} attempt {attempt} failed ({exc!r}), retrying in {wait:.1f}s")
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ── Config ─────────────────────────────────────────────────────────────────────

def _load_config() -> dict[str, Any]:
    load_dotenv()

    missing = [
        k
        for k in (
            "GMAIL_CLIENT_ID",
            "GMAIL_CLIENT_SECRET",
            "GMAIL_REFRESH_TOKEN",
            "GMAIL_LABELS",
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "SUMMARISATION_PERSONA",
            "SLACK_WEBHOOK_URL",
        )
        if not (os.environ.get(k) or "").strip()
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)

    try:
        date_window = int((os.environ.get("DATE_WINDOW_DAYS") or "1").strip())
    except ValueError:
        print("DATE_WINDOW_DAYS must be an integer.", file=sys.stderr)
        raise SystemExit(1)
    if date_window < 1:
        print("DATE_WINDOW_DAYS must be at least 1.", file=sys.stderr)
        raise SystemExit(1)

    try:
        max_emails = int((os.environ.get("MAX_EMAILS_PER_LABEL") or "50").strip())
    except ValueError:
        print("MAX_EMAILS_PER_LABEL must be an integer.", file=sys.stderr)
        raise SystemExit(1)
    max_emails = max(1, min(max_emails, 500))

    return {
        "gmail_client_id": os.environ["GMAIL_CLIENT_ID"].strip(),
        "gmail_client_secret": os.environ["GMAIL_CLIENT_SECRET"].strip(),
        "gmail_refresh_token": os.environ["GMAIL_REFRESH_TOKEN"].strip(),
        "labels": _parse_csv_env(os.environ["GMAIL_LABELS"]),
        "date_window_days": date_window,
        "filter_keywords": _parse_csv_env(os.environ.get("FILTER_KEYWORDS") or ""),
        "gemini_api_key": os.environ["GEMINI_API_KEY"].strip(),
        "gemini_model": os.environ["GEMINI_MODEL"].strip(),
        "persona": os.environ["SUMMARISATION_PERSONA"].strip(),
        "slack_webhook": os.environ["SLACK_WEBHOOK_URL"].strip(),
        "max_emails_per_label": max_emails,
    }


# ── Gmail ──────────────────────────────────────────────────────────────────────

def _gmail_credentials(cfg: dict[str, Any]) -> Credentials:
    creds = Credentials(
        token=None,
        refresh_token=cfg["gmail_refresh_token"],
        token_uri=GMAIL_TOKEN_URI,
        client_id=cfg["gmail_client_id"],
        client_secret=cfg["gmail_client_secret"],
        scopes=GMAIL_SCOPES,
    )
    try:
        creds.refresh(Request())
    except RefreshError as e:
        err_txt = str(e).lower()
        if "unauthorized_client" in err_txt:
            print(
                "Gmail OAuth refresh failed (unauthorized_client). Google rejected the client credentials.\n"
                "This almost always means GMAIL_CLIENT_ID + GMAIL_CLIENT_SECRET + GMAIL_REFRESH_TOKEN "
                "do not belong together.\n"
                "What to do:\n"
                "  1. In Google Cloud Console → APIs & Services → Credentials, open ONE OAuth 2.0 "
                "Client ID (Desktop or Web).\n"
                "  2. Copy that client's ID and secret into GitHub Secrets GMAIL_CLIENT_ID and "
                "GMAIL_CLIENT_SECRET.\n"
                "  3. Obtain a NEW refresh token using that same client (OAuth Playground or your "
                "local script) and update GMAIL_REFRESH_TOKEN.\n"
                "Do not mix: a refresh token from client A with secrets from client B. If you use "
                "OAuth Playground, enable 'Use your own OAuth credentials' with the same client.\n"
                f"Underlying error: {e}",
                file=sys.stderr,
            )
        else:
            print(
                "Gmail OAuth refresh failed (invalid_grant). The refresh token is no longer accepted.\n"
                "What to do:\n"
                "  1. Run your local OAuth flow again to obtain a new refresh token.\n"
                "  2. Update GitHub Secrets: GMAIL_REFRESH_TOKEN (and verify GMAIL_CLIENT_ID / "
                "GMAIL_CLIENT_SECRET match that OAuth client).\n"
                "Common causes: you revoked app access in Google Account, the OAuth client secret was "
                "rotated, the app is in OAuth 'Testing' mode (refresh tokens can expire for external "
                "test users), or Google invalidated old tokens when issuing new ones.\n"
                f"Underlying error: {e}",
                file=sys.stderr,
            )
        raise SystemExit(1) from e
    return creds


def _list_gmail_labels(service: Any) -> list[GmailLabel]:
    resp = service.users().labels().list(userId="me").execute(num_retries=GMAIL_EXECUTE_RETRIES)
    out: list[GmailLabel] = []
    for lab in resp.get("labels", []):
        lid = (lab.get("id") or "").strip()
        name = (lab.get("name") or "").strip()
        if lid and name:
            out.append(GmailLabel(id=lid, name=name))
    return out


def _expand_label_family(labels: list[GmailLabel], parent_name: str) -> list[GmailLabel]:
    """Return the label matching *parent_name* and all its '/' sub-labels, sorted."""
    want = parent_name.strip()
    if not want:
        return []
    want_l = want.lower()
    prefix_l = want_l + "/"
    family = [l for l in labels if l.name.lower() == want_l or l.name.lower().startswith(prefix_l)]
    family.sort(key=lambda x: x.name.lower())
    return family


def _header(headers: list[dict[str, str]], want: str) -> str:
    wl = want.lower()
    for h in headers:
        if (h.get("name") or "").lower() == wl:
            return (h.get("value") or "").strip()
    return ""


def _matches_filter(subject: str, snippet: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    hay = f"{subject}\n{snippet}".lower()
    return any(k.lower() in hay for k in keywords)


def _list_message_refs(
    service: Any,
    label_id: str,
    *,
    q: str | None,
    max_count: int,
    include_spam_trash: bool = False,
) -> list[dict[str, str]]:
    _debug(f"[gmail] list: label_id={label_id} q={(q or '')!r} max={max_count}")
    refs: list[dict[str, str]] = []
    page_token: str | None = None
    while len(refs) < max_count:
        batch_size = min(500, max_count - len(refs))
        resp = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=[label_id],
                q=(q or None),
                maxResults=batch_size,
                pageToken=page_token,
                includeSpamTrash=include_spam_trash,
            )
            .execute(num_retries=GMAIL_EXECUTE_RETRIES)
        )
        batch = resp.get("messages") or []
        if not batch:
            break
        refs.extend(batch)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return refs[:max_count]


def _fetch_email_item(service: Any, msg_id: str) -> EmailItem:
    msg = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=msg_id,
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        )
        .execute(num_retries=GMAIL_EXECUTE_RETRIES)
    )
    headers = (msg.get("payload") or {}).get("headers") or []
    subject = _header(headers, "Subject") or "(no subject)"
    from_addr = _header(headers, "From") or "(unknown)"
    date = _header(headers, "Date") or ""
    if not date and msg.get("internalDate"):
        try:
            ms = int(msg["internalDate"])
            date = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (OSError, ValueError, TypeError):
            date = ""
    snippet = (msg.get("snippet") or "").strip()
    return EmailItem(
        msg_id=msg_id,
        thread_id=msg.get("threadId") or msg_id,
        subject=subject,
        from_addr=from_addr,
        date=date,
        snippet=snippet,
    )


# ── Gemini ─────────────────────────────────────────────────────────────────────

def _parse_gemini_response(text: str) -> ScoredSummary | None:
    """
    Parse a Gemini summarisation response.

    Returns ``None``  → Gemini decided to SKIP this email.
    Returns ``ScoredSummary`` → parsed (or fallback) result.

    Handles:
    - Optional whitespace around the colon (``Relevance:3`` and ``Relevance: 3``)
    - Case-insensitive urgency (``high`` → ``High``)
    - Multi-line summaries
    - Fallback to full-response-as-summary when structured fields are absent
    """
    stripped = text.strip()

    if re.match(r"^SKIP\s*$", stripped, re.IGNORECASE):
        return None

    rel_m = re.search(r"(?im)^\s*Relevance\s*:\s*([1-5])\s*$", stripped)
    urg_m = re.search(r"(?im)^\s*Urgency\s*:\s*(High|Medium|Low)\s*$", stripped, re.IGNORECASE)
    # Capture first word(s) on the Summary line; everything after that line is
    # also part of the summary (multi-line responses).
    sum_m = re.search(r"(?im)^\s*Summary\s*:\s*(.+)", stripped)

    if rel_m and urg_m and sum_m:
        relevance = int(rel_m.group(1))  # regex already constrains to [1-5]
        raw_urgency = urg_m.group(1).strip()
        urgency = next(u for u in VALID_URGENCY if u.lower() == raw_urgency.lower())
        # Take everything from after "Summary: " to end of string so multi-line
        # summaries aren't truncated to just the first line.
        trailing = stripped[sum_m.end():].strip()
        summary = (sum_m.group(1).strip() + (" " + trailing if trailing else "")).strip()
        return ScoredSummary(summary=summary, relevance=relevance, urgency=urgency)

    # Fallback: treat full response as summary with neutral scores and log it so
    # operators can tune the prompt if this happens frequently.
    _debug(f"[gemini] structured parse failed — using full response as summary:\n{stripped[:200]}")
    return ScoredSummary(summary=stripped, relevance=3, urgency="Low")


def _build_summarise_prompt(cfg: dict[str, Any], item: EmailItem) -> str:
    system_instructions = f"""You are an intelligent email digest assistant for a product manager. Your job is to monitor incoming emails, filter for signal over noise, summarise each one clearly, and deliver a structured daily digest that is fast to scan and easy to act on.

YOUR PERSONA & GOAL
You think like a senior product manager. You prioritise:
- Strategic insights, market trends, and competitive intelligence
- Frameworks, case studies, and research findings
- Actionable recommendations with clear next steps
- Data points that inform product decisions

You ruthlessly filter out noise. You never waste the PM's time.

FILTER: KEEP SIGNAL, DROP NOISE
Discard an email (respond SKIP) if it is promotional, course/event sales, onboarding drip/cart abandonment, padding-heavy with no substance, or purely transactional.
Keep emails that look like real editorial content, analysis, research, case studies, frameworks, insights, or prose with genuine ideas. When in doubt, KEEP.

SUMMARISE EACH EMAIL
For each kept email, produce a summary of 120–200 words:
- Start with the core insight (no vague openers like "this email discusses")
- Name specifics (frameworks, stats, companies, products, findings)
- Extract PM-relevant value: what decision/action could this inform?
- Flag if actionable
- Plain, confident prose. No bullet points inside the summary.

SCORE EACH EMAIL
Also rate:
- Relevance (1–5)
- Urgency (High / Medium / Low)

OUTPUT FORMAT (strict)
- If you decide to discard: output exactly: SKIP
- Otherwise output exactly 3 lines:
Relevance: <1-5>
Urgency: <High|Medium|Low>
Summary: <single-paragraph 120-200 word summary>

Reader persona: {cfg["persona"]}
"""
    return f"""{system_instructions}

Email details:
Subject: {item.subject}
Sender: {item.from_addr}
Date received: {item.date}
Snippet / body preview:
{item.snippet}
"""


def _summarise(
    cfg: dict[str, Any],
    client: genai.Client,
    item: EmailItem,
) -> ScoredSummary | None:
    """
    Call Gemini to score and summarise *item*.

    Returns ``None`` if Gemini SKIPped the email.
    Raises on API error after retries are exhausted (caller handles per-email).
    """
    prompt = _build_summarise_prompt(cfg, item)

    def _call() -> Any:
        return client.models.generate_content(
            model=cfg["gemini_model"],
            contents=prompt,
            config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
        )

    resp = _with_retry(
        _call,
        max_attempts=GEMINI_RETRY_ATTEMPTS,
        backoff_base=GEMINI_RETRY_BACKOFF,
        retryable=_RETRYABLE_GEMINI,
        label=f"gemini/summarise {item.msg_id}",
    )
    text = (getattr(resp, "text", "") or "").strip()
    return _parse_gemini_response(text)


def _generate_tldr(
    cfg: dict[str, Any],
    client: genai.Client,
    scored: list[tuple[str, EmailItem, ScoredSummary]],
) -> list[str]:
    if not scored:
        return []
    items = scored[:20]
    joined = "\n\n".join(
        f"Label: {label}\nSubject: {it.subject}\nFrom: {it.from_addr}\n"
        f"Summary: {sc.summary}\nRelevance: {sc.relevance}/5 Urgency: {sc.urgency}"
        for label, it, sc in items
    )
    prompt = (
        'You are a PM chief-of-staff. Produce exactly 3 punchy takeaways (one sentence each) '
        'from this daily email digest.\n'
        'No preamble, no numbering. Output exactly 3 lines, each starting with "- ".\n\n'
        f'Digest items:\n{joined}\n'
    )

    def _call() -> Any:
        return client.models.generate_content(
            model=cfg["gemini_model"],
            contents=prompt,
            config={"max_output_tokens": GEMINI_TLDR_MAX_OUTPUT_TOKENS},
        )

    resp = _with_retry(
        _call,
        max_attempts=GEMINI_RETRY_ATTEMPTS,
        backoff_base=GEMINI_RETRY_BACKOFF,
        retryable=_RETRYABLE_GEMINI,
        label="gemini/tldr",
    )
    text = (getattr(resp, "text", "") or "").strip()
    lines = [re.sub(r"^\s*[-•]\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln][:SLACK_TLDR_MAX_BULLETS]


# ── Slack ──────────────────────────────────────────────────────────────────────

def _gmail_open_url(thread_id: str) -> str:
    return f"https://mail.google.com/mail/u/0/#all/{thread_id}"


def _digest_header_blocks(*, title_suffix: str | None = None) -> list[dict[str, Any]]:
    title = "Gmail digest" if not title_suffix else f"Gmail digest ({title_suffix})"
    return [
        {"type": "header", "text": {"type": "plain_text", "text": title, "emoji": True}},
        {"type": "divider"},
    ]


def _label_header_block(label_name: str, *, continued: bool) -> dict[str, Any]:
    suffix = " (cont.)" if continued else ""
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": f"Label: {label_name}{suffix}", "emoji": False},
    }


def _email_card_blocks(item: EmailItem, scored: ScoredSummary, source_label: str) -> list[dict[str, Any]]:
    url = _gmail_open_url(item.thread_id)
    urgency_dot = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}.get(scored.urgency, "⚪")
    return [
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{_slack_escape(item.subject)}*"},
            "fields": [
                {"type": "mrkdwn", "text": f"*From*\n{_slack_escape(item.from_addr)}"},
                {"type": "mrkdwn", "text": f"*Date*\n{_slack_escape(item.date)}"},
                {"type": "mrkdwn", "text": f"*Sublabel*\n{_slack_escape(source_label)}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Score*\nRelevance {scored.relevance}/5 · {urgency_dot} {scored.urgency}",
                },
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": _slack_escape(_truncate(scored.summary, SLACK_SUMMARY_MAX_CHARS)),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Open in Gmail"},
                    "url": url,
                }
            ],
        },
    ]


def _build_slack_messages(
    label_sections: list[tuple[str, list[tuple[EmailItem, ScoredSummary, str]]]],
    *,
    tldr_lines: list[str] | None = None,
    max_blocks_per_message: int = SLACK_MAX_BLOCKS_PER_MSG,
) -> list[list[dict[str, Any]]]:
    """Split the digest into Slack messages that each fit within the 50-block limit."""
    messages: list[list[dict[str, Any]]] = []
    blocks: list[dict[str, Any]] = _digest_header_blocks()

    def flush() -> None:
        nonlocal blocks
        if blocks:
            messages.append(blocks)
        blocks = _digest_header_blocks(title_suffix=f"{len(messages) + 1}")

    for label_name, rows in label_sections:
        label_started = False
        continued = False

        if not rows:
            if len(blocks) + 2 > max_blocks_per_message:
                flush()
            blocks.append(_label_header_block(label_name, continued=False))
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "_No qualifying emails for this period._"},
                }
            )
            continue

        for item, scored, source_label in rows:
            card = _email_card_blocks(item, scored, source_label)
            needed = (0 if label_started else 1) + len(card)
            if len(blocks) + needed > max_blocks_per_message:
                flush()
                continued = True
                label_started = False

            if not label_started:
                blocks.append(_label_header_block(label_name, continued=continued))
                label_started = True
            blocks.extend(card)

    if tldr_lines:
        tldr_blocks: list[dict[str, Any]] = [
            {"type": "divider"},
            {"type": "header", "text": {"type": "plain_text", "text": "TL;DR", "emoji": False}},
        ]
        bullets = "\n".join(
            f"- {_slack_escape(x)}" for x in tldr_lines[:SLACK_TLDR_MAX_BULLETS] if x.strip()
        )
        if bullets:
            tldr_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": bullets}})
        if len(blocks) + len(tldr_blocks) > max_blocks_per_message:
            flush()
        blocks.extend(tldr_blocks)

    if blocks:
        messages.append(blocks)
    return messages


def _post_slack(webhook: str, blocks: list[dict[str, Any]]) -> bool:
    """
    Post *blocks* to Slack.

    Retries on 5xx errors and network failures with exponential backoff.
    Returns ``False`` (without retrying) on 4xx errors — those indicate a
    misconfigured webhook and won't improve on retry.
    """

    def _do_post() -> requests.Response:
        r = requests.post(
            webhook,
            json={"blocks": blocks},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        if r.status_code >= 500:
            # Raise so _with_retry treats this as a retryable failure.
            raise requests.exceptions.RequestException(
                f"Slack server error {r.status_code}: {r.text}"
            )
        return r

    try:
        r = _with_retry(
            _do_post,
            max_attempts=SLACK_RETRY_ATTEMPTS,
            backoff_base=SLACK_RETRY_BACKOFF,
            retryable=(requests.exceptions.RequestException,),
            label="slack/post",
        )
    except requests.exceptions.RequestException as e:
        print(f"Slack request failed after {SLACK_RETRY_ATTEMPTS} attempts: {e}", file=sys.stderr)
        return False

    if r.status_code >= 400:
        print(f"Slack webhook error {r.status_code}: {r.text}", file=sys.stderr)
        return False
    return True


# ── Orchestration ──────────────────────────────────────────────────────────────

def main() -> int:
    cfg = _load_config()
    print(
        "[config] "
        + f"labels={cfg['labels']!r} date_window_days={cfg['date_window_days']} "
        + f"max_emails_per_label={cfg['max_emails_per_label']} "
        + f"filter_keywords={len(cfg['filter_keywords'])}",
        file=sys.stderr,
    )

    try:
        creds = _gmail_credentials(cfg)
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except HttpError as e:
        print(f"Gmail API error: {e}", file=sys.stderr)
        return 1

    gemini_client = genai.Client(api_key=cfg["gemini_api_key"])

    try:
        all_labels = _list_gmail_labels(service)
    except HttpError as e:
        print(f"Gmail API error listing labels: {e}", file=sys.stderr)
        return 1

    label_sections: list[tuple[str, list[tuple[EmailItem, ScoredSummary, str]]]] = []
    all_scored: list[tuple[str, EmailItem, ScoredSummary]] = []

    for parent_label in cfg["labels"]:
        family = _expand_label_family(all_labels, parent_label)
        if not family:
            print(f"Gmail label not found: {parent_label!r}", file=sys.stderr)
            return 1

        remaining = cfg["max_emails_per_label"]
        q = f"newer_than:{cfg['date_window_days']}d"
        refs_with_source: list[tuple[dict[str, str], str]] = []

        for lab in family:
            if remaining <= 0:
                break
            _debug(f"[gmail] resolved label {lab.name!r} -> {lab.id!r} (parent {parent_label!r})")
            try:
                refs = _list_message_refs(service, lab.id, q=q, max_count=remaining)
            except HttpError as e:
                print(f"Gmail API error for label {lab.name!r}: {e}", file=sys.stderr)
                return 1
            refs_with_source.extend((r, lab.name) for r in refs)
            remaining -= len(refs)

        fetched = len(refs_with_source)

        if fetched == 0:
            try:
                any_in_family = False
                for lab in family:
                    any_refs = _list_message_refs(service, lab.id, q=None, max_count=1)
                    if any_refs:
                        any_in_family = True
                        break
                print(
                    "[label-check] "
                    + f"{parent_label!r}: any_in_label_family={any_in_family} "
                    + "(no date filter, includes sublabels)",
                    file=sys.stderr,
                )
            except HttpError as e:
                print(f"Gmail API error during label-check for {parent_label!r}: {e}", file=sys.stderr)
                return 1

        # Deduplicate before issuing any email-detail API calls so we don't
        # fetch the same message twice when it appears in both a parent label
        # and a child label.
        seen_msg_ids: set[str] = set()
        deduped: list[tuple[str, str]] = []  # (msg_id, source_label_name)
        for ref, source_label_name in refs_with_source:
            mid = ref.get("id")
            if mid and mid not in seen_msg_ids:
                seen_msg_ids.add(mid)
                deduped.append((mid, source_label_name))

        # Fetch email metadata (fast, serial — each call is a tiny metadata GET).
        keyword_filtered = 0
        items_to_summarise: list[tuple[EmailItem, str]] = []
        for mid, source_label_name in deduped:
            try:
                item = _fetch_email_item(service, mid)
            except HttpError as e:
                print(f"Gmail API error fetching message {mid}: {e}", file=sys.stderr)
                return 1
            if _matches_filter(item.subject, item.snippet, cfg["filter_keywords"]):
                keyword_filtered += 1
                _debug(f"[filter] keyword drop: {item.subject!r}")
                continue
            items_to_summarise.append((item, source_label_name))

        # Parallel Gemini summarisation — each email is an independent API call
        # so we saturate the quota allowance instead of waiting serially.
        gemini_skipped = 0
        gemini_errors = 0
        rows: list[tuple[EmailItem, ScoredSummary, str]] = []

        with ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS) as executor:
            futures = [
                executor.submit(_summarise, cfg, gemini_client, item)
                for item, _ in items_to_summarise
            ]
            for future, (item, source_label_name) in zip(futures, items_to_summarise):
                try:
                    scored = future.result()
                except Exception as e:
                    print(f"Gemini error for {item.subject!r}: {e}", file=sys.stderr)
                    gemini_errors += 1
                    continue
                if scored is None:
                    gemini_skipped += 1
                    _debug(f"[gemini] SKIP: {item.subject!r}")
                    continue
                rows.append((item, scored, source_label_name))
                all_scored.append((parent_label, item, scored))

        kept = len(rows)
        print(
            "[label] "
            + f"{parent_label!r}: fetched={fetched} deduped={len(deduped)} "
            + f"keyword_filtered={keyword_filtered} gemini_skipped={gemini_skipped} "
            + f"gemini_errors={gemini_errors} kept={kept} labels_included={len(family)}",
            file=sys.stderr,
        )
        label_sections.append((parent_label, rows))

    # Only attempt TL;DR when there are scored emails — avoids a wasted API
    # call on days with no qualifying content.
    tldr: list[str] = []
    if all_scored:
        try:
            tldr = _generate_tldr(cfg, gemini_client, all_scored)
        except Exception as e:
            _debug(f"[tldr] failed: {e}")

    messages = _build_slack_messages(label_sections, tldr_lines=tldr)
    for i, blocks in enumerate(messages, start=1):
        _debug(f"[slack] posting message {i}/{len(messages)} blocks={len(blocks)}")
        if not _post_slack(cfg["slack_webhook"], blocks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
