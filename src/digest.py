"""Gmail label digest: fetch, filter, summarise with Gemini, upload Markdown brief to Dropbox."""

from __future__ import annotations

import base64
import html as html_module
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import dropbox as dropbox_sdk
import requests
from dotenv import load_dotenv
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import WriteMode
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
GEMINI_MAX_WORKERS = 5          # concurrent summarisation threads
GEMINI_RETRY_ATTEMPTS = 3       # per-call retry on transient errors
GEMINI_RETRY_BACKOFF = 2.0      # exponential backoff base (seconds)

DROPBOX_RETRY_ATTEMPTS = 3
DROPBOX_RETRY_BACKOFF = 2.0

MAX_BODY_CHARS = 6000           # max email body chars forwarded to Gemini

VALID_URGENCY: frozenset[str] = frozenset({"High", "Medium", "Low"})

DEBUG = (os.environ.get("DEBUG") or "").strip().lower() in {"1", "true", "yes", "y", "on"}

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
    snippet: str        # short preview — used for keyword filtering only
    body: str           # full body text — used for Gemini summarisation


@dataclass
class ScoredSummary:
    headline: str       # AI one-liner title
    tldr: str           # 1-2 sentence high-level summary
    main_points: list   # key bullet points
    topics: list        # topic hashtags e.g. ['#prompt-engineering']
    companies: list     # company hashtags e.g. ['#openai']
    urgency: str        # "High" | "Medium" | "Low"


@dataclass(frozen=True)
class GmailLabel:
    id: str
    name: str


# ── Utilities ──────────────────────────────────────────────────────────────────

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


def _extract_display_name(from_addr: str) -> str:
    """Extract display name from 'Name <email@domain>' format, else return raw."""
    m = re.match(r'^"?([^"<]+?)"?\s*<[^>]+>$', from_addr.strip())
    if m:
        return m.group(1).strip()
    return from_addr.strip()


def _normalize_date(raw: str) -> str:
    """Normalize RFC 2822 date header to 'D Mon YYYY' (e.g. '11 Apr 2026')."""
    try:
        dt = parsedate_to_datetime(raw.strip())
        return f"{dt.day} {dt.strftime('%b %Y')}"
    except Exception:
        return raw


def _parse_hashtags(raw: str) -> list[str]:
    """Extract and normalize hashtag tokens from a Gemini output field value."""
    raw = raw.strip()
    if raw.lower() in ("none", "n/a", "-", ""):
        return []
    tokens = re.split(r"[,;\s]+", raw)
    result: list[str] = []
    for token in tokens:
        token = token.strip().lstrip("#").strip(".,;:")
        if not token or token.lower() in ("none", "n/a"):
            continue
        # Normalize to lowercase-hyphenated, strip non-alphanumeric except hyphens
        normalized = re.sub(r"[\s_]+", "-", token.lower())
        normalized = re.sub(r"[^a-z0-9\-]", "", normalized)
        if normalized:
            result.append(f"#{normalized}")
    return result


# ── Email body extraction ──────────────────────────────────────────────────────

def _strip_html(raw: str) -> str:
    """Strip HTML tags and decode entities to readable plain text."""
    raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # Convert block elements to newlines for readability
    raw = re.sub(r"<(?:br|p|div|h[1-6]|li|tr)[^>]*>", "\n", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html_module.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _extract_body_text(payload: dict, depth: int = 0) -> str:
    """Recursively extract plain text from a Gmail message payload (MIME tree)."""
    if depth > 5:
        return ""
    mime_type = (payload.get("mimeType") or "").lower()
    body_data = (payload.get("body") or {}).get("data", "")

    # Leaf node with content
    if body_data and "parts" not in payload:
        try:
            text = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        except Exception:
            return ""
        return _strip_html(text) if "html" in mime_type else text.strip()

    parts = payload.get("parts") or []
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in parts:
        part_mime = (part.get("mimeType") or "").lower()
        if any(x in part_mime for x in ("alternative", "mixed", "related")):
            nested = _extract_body_text(part, depth + 1)
            if nested:
                plain_parts.append(nested)
        elif "plain" in part_mime:
            data = (part.get("body") or {}).get("data", "")
            if data:
                try:
                    t = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace").strip()
                    if t:
                        plain_parts.append(t)
                except Exception:
                    pass
        elif "html" in part_mime:
            data = (part.get("body") or {}).get("data", "")
            if data:
                try:
                    t = _strip_html(
                        base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                    )
                    if t:
                        html_parts.append(t)
                except Exception:
                    pass
        elif "parts" in part:
            nested = _extract_body_text(part, depth + 1)
            if nested:
                plain_parts.append(nested)

    return "\n\n".join(plain_parts) or "\n\n".join(html_parts)


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
            "DROPBOX_ACCESS_TOKEN",
            "DROPBOX_FOLDER_PATH",
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
        "dropbox_access_token": os.environ["DROPBOX_ACCESS_TOKEN"].strip(),
        "dropbox_folder_path": os.environ["DROPBOX_FOLDER_PATH"].strip().rstrip("/"),
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
                "Do not mix: a refresh token from client A with secrets from client B.\n"
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


def _matches_filter(subject: str, body: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    hay = f"{subject}\n{body}".lower()
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
        .get(userId="me", id=msg_id, format="full")
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

    body = _extract_body_text(msg.get("payload") or {})
    if not body:
        body = snippet
    else:
        body = body[:MAX_BODY_CHARS]

    return EmailItem(
        msg_id=msg_id,
        thread_id=msg.get("threadId") or msg_id,
        subject=subject,
        from_addr=from_addr,
        date=date,
        snippet=snippet,
        body=body,
    )


# ── Gemini ─────────────────────────────────────────────────────────────────────

def _parse_gemini_response(text: str) -> ScoredSummary | None:
    """
    Parse a Gemini summarisation response.

    Returns ``None``          → Gemini decided to SKIP this email.
    Returns ``ScoredSummary`` → parsed (or fallback) result.

    Expected format:
        Headline: <text>
        TL;DR: <text>
        Main Points:
        - point 1
        - point 2
        Topic: #tag1 #tag2 or None
        Company: #tag1 or None
        Urgency: High|Medium|Low
    """
    stripped = text.strip()

    if re.match(r"^SKIP\s*$", stripped, re.IGNORECASE):
        return None

    head_m = re.search(r"(?im)^\s*Headline\s*:\s*(.+)", stripped)
    # TL;DR: from field value up to "Main Points:" or end
    tldr_m = re.search(
        r"(?im)^\s*TL;?DR\s*:\s*(.+?)(?=^\s*Main\s+Points\s*:|\Z)",
        stripped,
        re.DOTALL | re.MULTILINE,
    )
    # Main Points: from section header up to "Topic:" or end
    points_m = re.search(
        r"(?im)^\s*Main\s+Points\s*:\s*\n(.*?)(?=^\s*Topic\s*:|\Z)",
        stripped,
        re.DOTALL | re.MULTILINE,
    )
    topic_m = re.search(r"(?im)^\s*Topic\s*:\s*(.+)", stripped)
    company_m = re.search(r"(?im)^\s*Company\s*:\s*(.+)", stripped)
    urg_m = re.search(r"(?im)^\s*Urgency\s*:\s*(High|Medium|Low)\s*$", stripped, re.IGNORECASE)

    if head_m and tldr_m and urg_m:
        raw_urgency = urg_m.group(1).strip()
        urgency = next(u for u in VALID_URGENCY if u.lower() == raw_urgency.lower())
        headline = head_m.group(1).strip()
        tldr = tldr_m.group(1).strip()

        main_points: list[str] = []
        if points_m:
            main_points = [
                re.sub(r"^\s*[-•*]\s*", "", line).strip()
                for line in points_m.group(1).splitlines()
                if re.match(r"\s*[-•*]", line) and line.strip()
            ]

        topics = _parse_hashtags(topic_m.group(1)) if topic_m else []
        companies = _parse_hashtags(company_m.group(1)) if company_m else []

        return ScoredSummary(
            headline=headline,
            tldr=tldr,
            main_points=main_points,
            topics=topics,
            companies=companies,
            urgency=urgency,
        )

    # Fallback: first non-empty line as headline, full text as TL;DR
    _debug(f"[gemini] structured parse failed — using fallback:\n{stripped[:200]}")
    first_line = next((ln.strip() for ln in stripped.splitlines() if ln.strip()), stripped)
    return ScoredSummary(
        headline=first_line[:100],
        tldr=stripped,
        main_points=[],
        topics=[],
        companies=[],
        urgency="Low",
    )


def _build_summarise_prompt(cfg: dict[str, Any], item: EmailItem) -> str:
    return f"""You are a senior product manager's daily briefing assistant. Read the email below and produce a structured summary or discard it.

PERSONA
Think like a senior PM. Prioritise: strategic insights, market trends, competitive intelligence, frameworks with real case studies, actionable data points. Ruthlessly filter noise.

DISCARD RULE
Output the single word SKIP — nothing else — if the email is:
- Purely promotional, course/event sales, or cart abandonment
- Onboarding drip with no substantive editorial content
- Transactional (receipts, notifications, confirmations)
When in doubt, keep it.

SUMMARISE RULE
For emails you keep, produce all six fields below.

Headline — One punchy, specific title (max 10 words). Name the actual subject: a company, stat, framework, or finding. Never start with "This", "How", or "Why".

TL;DR — 2 complete sentences maximum. Lead with the core insight. No vague openers like "This article covers…".

Main Points — 3 to 5 bullets. Each bullet must be one complete sentence with a concrete finding, stat, or action. Omit a bullet entirely rather than writing a vague or incomplete one.

Topic — Up to 3 topic hashtags in kebab-case (e.g. #prompt-engineering, #ai-agents, #product-strategy). Use None if no clear topic applies.

Company — Up to 3 company hashtags in kebab-case (e.g. #openai, #anthropic, #google). Use None if no specific company is a key subject of the article.

Urgency — Exactly one of: High, Medium, Low.

OUTPUT RULES — READ CAREFULLY
- Output ONLY the six fields in the exact format below
- Do not write any reasoning, thinking steps, notes to yourself, or commentary outside the fields
- Do not truncate any field mid-sentence — omit the bullet rather than leaving it incomplete
- Each Main Points bullet must be a complete sentence

OUTPUT FORMAT:
Headline: <max 10 words>
TL;DR: <2 sentences max>
Main Points:
- <complete sentence>
- <complete sentence>
- <complete sentence>
Topic: <#hashtag #hashtag or None>
Company: <#hashtag #hashtag or None>
Urgency: <High|Medium|Low>

Reader persona: {cfg["persona"]}

---
Subject: {item.subject}
From: {item.from_addr}
Date: {item.date}
Body:
{item.body}
"""


def _summarise(
    cfg: dict[str, Any],
    client: genai.Client,
    item: EmailItem,
) -> ScoredSummary | None:
    """Call Gemini to summarise *item*. Returns None if Gemini SKIPped the email."""
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


# ── Markdown ───────────────────────────────────────────────────────────────────

def _gmail_open_url(thread_id: str) -> str:
    return f"https://mail.google.com/mail/u/0/#all/{thread_id}"


_URGENCY_ICON = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}


def _email_to_markdown_section(item: EmailItem, scored: ScoredSummary, source_label: str) -> str:
    """Render a single email as a structured Markdown section."""
    icon = _URGENCY_ICON.get(scored.urgency, "⚪")
    url = _gmail_open_url(item.thread_id)
    display_name = _extract_display_name(item.from_addr)
    display_date = _normalize_date(item.date)

    lines = [
        f"### {scored.headline}",
        f"> **From:** {display_name} · **Date:** {display_date} · **Label:** {source_label} · {icon} {scored.urgency}",
        "",
    ]

    # Hashtag line — only emit when at least one tag exists
    tag_parts: list[str] = []
    if scored.topics:
        tag_parts.append("topic: " + " ".join(scored.topics))
    if scored.companies:
        tag_parts.append("company: " + " ".join(scored.companies))
    if tag_parts:
        lines.append(" · ".join(tag_parts))
        lines.append("")

    lines += [
        "**TL;DR**",
        scored.tldr,
        "",
    ]

    if scored.main_points:
        lines.append("**Main Points**")
        for point in scored.main_points:
            lines.append(f"- {point}")
        lines.append("")

    lines += [
        f"[Open in Gmail →]({url})",
        "",
        "---",
        "",
    ]

    return "\n".join(lines)


def _build_daily_brief_markdown(
    label_sections: list[tuple[str, list[tuple[EmailItem, ScoredSummary, str]]]],
    *,
    date_str: str,
) -> str:
    """Build the full Obsidian-compatible Markdown document for the daily brief."""
    parts: list[str] = []

    # YAML frontmatter
    parts.append(f"---\ndate: {date_str}\ntags: [daily-brief, email-digest]\n---\n")

    # Title
    parts.append(f"# Daily Brief — {date_str}\n")

    # Index: one line per article across all labels
    all_rows = [
        (item, scored)
        for _, rows in label_sections
        for item, scored, _ in rows
    ]
    if all_rows:
        parts.append("## Articles Processed\n")
        for i, (item, scored) in enumerate(all_rows, start=1):
            display_name = _extract_display_name(item.from_addr)
            display_date = _normalize_date(item.date)
            parts.append(f"{i}. **{scored.headline}** — {display_name} · {display_date}")
        parts.append("\n---\n")

    # Per-label detailed sections
    for label_name, rows in label_sections:
        parts.append(f"## {label_name}\n")
        if not rows:
            parts.append("_No qualifying emails for this period._\n")
            parts.append("---\n")
            continue
        for item, scored, source_label in rows:
            parts.append(_email_to_markdown_section(item, scored, source_label))

    return "\n".join(parts)


# ── Dropbox ────────────────────────────────────────────────────────────────────

def _upload_to_dropbox(cfg: dict[str, Any], content: str, filename: str) -> bool:
    """
    Upload *content* as *filename* into the configured Dropbox folder.

    Always overwrites an existing file with the same name so re-runs on the
    same day produce a fresh brief rather than an error.
    Returns ``False`` on auth or API errors.
    """
    folder = cfg["dropbox_folder_path"]
    path = f"{folder}/{filename}"
    data = content.encode("utf-8")

    try:
        dbx = dropbox_sdk.Dropbox(oauth2_access_token=cfg["dropbox_access_token"])

        def _do_upload() -> None:
            dbx.files_upload(data, path, mode=WriteMode("overwrite"))

        _with_retry(
            _do_upload,
            max_attempts=DROPBOX_RETRY_ATTEMPTS,
            backoff_base=DROPBOX_RETRY_BACKOFF,
            retryable=(requests.exceptions.RequestException, ConnectionError, TimeoutError),
            label=f"dropbox/upload {filename}",
        )
        _debug(f"[dropbox] uploaded {path}")
        return True
    except AuthError as e:
        print(
            f"Dropbox authentication failed. Check DROPBOX_ACCESS_TOKEN.\n"
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        return False
    except ApiError as e:
        print(f"Dropbox API error uploading {path}: {e}", file=sys.stderr)
        return False


# ── Orchestration ──────────────────────────────────────────────────────────────

def main() -> int:
    cfg = _load_config()
    print(
        "[config] "
        + f"labels={cfg['labels']!r} date_window_days={cfg['date_window_days']} "
        + f"max_emails_per_label={cfg['max_emails_per_label']} "
        + f"filter_keywords={len(cfg['filter_keywords'])} "
        + f"dropbox_folder={cfg['dropbox_folder_path']!r}",
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

        seen_msg_ids: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for ref, source_label_name in refs_with_source:
            mid = ref.get("id")
            if mid and mid not in seen_msg_ids:
                seen_msg_ids.add(mid)
                deduped.append((mid, source_label_name))

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

        kept = len(rows)
        print(
            "[label] "
            + f"{parent_label!r}: fetched={fetched} deduped={len(deduped)} "
            + f"keyword_filtered={keyword_filtered} gemini_skipped={gemini_skipped} "
            + f"gemini_errors={gemini_errors} kept={kept} labels_included={len(family)}",
            file=sys.stderr,
        )
        label_sections.append((parent_label, rows))

    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    filename = f"{date_str}-daily-brief.md"
    markdown = _build_daily_brief_markdown(label_sections, date_str=date_str)

    _debug(f"[markdown] built brief: {len(markdown)} chars")

    if not _upload_to_dropbox(cfg, markdown, filename):
        return 1

    print(
        f"[done] uploaded {filename} to {cfg['dropbox_folder_path']}/{filename}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
