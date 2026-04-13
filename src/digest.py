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

GEMINI_MAX_OUTPUT_TOKENS = 3072
GEMINI_KEY_NUMBERS_MAX_TOKENS = 768
GEMINI_MAX_WORKERS = 5          # concurrent summarisation threads
GEMINI_RETRY_ATTEMPTS = 3       # per-call retry on transient errors
GEMINI_RETRY_BACKOFF = 2.0      # exponential backoff base (seconds)

DROPBOX_RETRY_ATTEMPTS = 3
DROPBOX_RETRY_BACKOFF = 2.0

MAX_BODY_CHARS = 6000           # max email body chars forwarded to Gemini

# Canonical theme buckets for index + grouping (must match the summarisation prompt).
THEME_SECTION_ORDER: tuple[str, ...] = (
    "AI infrastructure & compute",
    "System design & engineering",
    "AI product strategy & agents",
    "Business & market signals",
    "Productivity & workflows",
    "Vibe coding",
    "AI Product Management",
    "AI Models",
)
_DEFAULT_THEME = "AI Product Management"

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
    headline: str       # sentence-case article title for the brief
    theme: str          # one of THEME_SECTION_ORDER (normalized)
    read_minutes: int   # 2, 3, or 4 — estimated read time for this piece
    tldr: str           # 2–3 sentences
    key_insight: str    # single most useful line
    main_points: list   # bullet strings (concrete points)
    topics: list        # topic hashtags e.g. ['#prompt-engineering']
    companies: list     # company hashtags e.g. ['#openai']


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


def _normalize_theme(raw: str) -> str:
    """Map model output to a canonical theme from THEME_SECTION_ORDER."""
    s = (raw or "").strip()
    if not s:
        return _DEFAULT_THEME
    sl = s.lower()
    for canonical in THEME_SECTION_ORDER:
        if sl == canonical.lower():
            return canonical
    for canonical in THEME_SECTION_ORDER:
        if sl in canonical.lower() or canonical.lower() in sl:
            return canonical
    return _DEFAULT_THEME


def _email_sort_timestamp(item: EmailItem) -> float:
    """Parse email date for descending sort (newest first); unparseable → 0."""
    try:
        dt = parsedate_to_datetime(item.date.strip())
        return dt.timestamp()
    except Exception:
        return 0.0


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

    dropbox_access_token = (os.environ.get("DROPBOX_ACCESS_TOKEN") or "").strip()
    dropbox_refresh_token = (os.environ.get("DROPBOX_REFRESH_TOKEN") or "").strip()
    dropbox_app_key = (os.environ.get("DROPBOX_APP_KEY") or "").strip()
    dropbox_app_secret = (os.environ.get("DROPBOX_APP_SECRET") or "").strip()

    has_refresh_set = all([dropbox_refresh_token, dropbox_app_key, dropbox_app_secret])
    has_any_refresh_field = any([dropbox_refresh_token, dropbox_app_key, dropbox_app_secret])

    if not dropbox_access_token and not has_refresh_set:
        if has_any_refresh_field:
            missing_refresh = [
                k
                for k, v in (
                    ("DROPBOX_REFRESH_TOKEN", dropbox_refresh_token),
                    ("DROPBOX_APP_KEY", dropbox_app_key),
                    ("DROPBOX_APP_SECRET", dropbox_app_secret),
                )
                if not v
            ]
            print(
                "Dropbox auth is partially configured. Provide either:\n"
                "  - DROPBOX_ACCESS_TOKEN\n"
                "  - OR all of DROPBOX_REFRESH_TOKEN, DROPBOX_APP_KEY, DROPBOX_APP_SECRET\n"
                f"Missing from refresh-token set: {', '.join(missing_refresh)}",
                file=sys.stderr,
            )
        else:
            print(
                "Missing Dropbox auth configuration. Provide either:\n"
                "  - DROPBOX_ACCESS_TOKEN\n"
                "  - OR all of DROPBOX_REFRESH_TOKEN, DROPBOX_APP_KEY, DROPBOX_APP_SECRET",
                file=sys.stderr,
            )
        raise SystemExit(1)

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
        "dropbox_access_token": dropbox_access_token,
        "dropbox_refresh_token": dropbox_refresh_token,
        "dropbox_app_key": dropbox_app_key,
        "dropbox_app_secret": dropbox_app_secret,
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

    Expected machine-readable block (see ``_build_summarise_prompt``).
    """
    stripped = text.strip()

    if re.match(r"^SKIP\s*$", stripped, re.IGNORECASE):
        return None

    theme_m = re.search(r"(?im)^\s*Theme\s*:\s*(.+)$", stripped, re.MULTILINE)
    read_m = re.search(r"(?im)^\s*Read\s+minutes\s*:\s*([234])\s*$", stripped, re.MULTILINE)
    title_m = re.search(
        r"(?im)^\s*(?:Article\s+title|Headline)\s*:\s*(.+)$",
        stripped,
        re.MULTILINE,
    )
    tldr_m = re.search(
        r"(?im)^\s*TL;?DR\s*:\s*(.+?)(?=^\s*Key\s+insight\s*:|\Z)",
        stripped,
        re.DOTALL | re.MULTILINE,
    )
    insight_m = re.search(
        r"(?im)^\s*Key\s+insight\s*:\s*(.+?)(?=^\s*Main\s+points\s*:|\Z)",
        stripped,
        re.DOTALL | re.MULTILINE,
    )
    points_m = re.search(
        r"(?im)^\s*Main\s+points\s*:\s*\n(.*?)(?=^\s*Topic\s*:|\Z)",
        stripped,
        re.DOTALL | re.MULTILINE,
    )
    topic_m = re.search(r"(?im)^\s*Topic\s*:\s*(.+)$", stripped, re.MULTILINE)
    company_m = re.search(r"(?im)^\s*Company\s*:\s*(.+)$", stripped, re.MULTILINE)

    if theme_m and read_m and title_m and tldr_m and insight_m:
        headline = title_m.group(1).strip()
        tldr = tldr_m.group(1).strip()
        key_insight = insight_m.group(1).strip()
        theme = _normalize_theme(theme_m.group(1).strip())
        try:
            read_minutes = int(read_m.group(1).strip())
        except ValueError:
            read_minutes = 3
        read_minutes = max(2, min(4, read_minutes))

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
            theme=theme,
            read_minutes=read_minutes,
            tldr=tldr,
            key_insight=key_insight,
            main_points=main_points,
            topics=topics,
            companies=companies,
        )

    _debug(f"[gemini] structured parse failed — using fallback:\n{stripped[:200]}")
    first_line = next((ln.strip() for ln in stripped.splitlines() if ln.strip()), stripped)
    return ScoredSummary(
        headline=first_line[:100],
        theme=_DEFAULT_THEME,
        read_minutes=3,
        tldr=stripped,
        key_insight="",
        main_points=[],
        topics=[],
        companies=[],
    )


def _build_summarise_prompt(cfg: dict[str, Any], item: EmailItem, brief_date_str: str) -> str:
    theme_list = "\n".join(f"- {t}" for t in THEME_SECTION_ORDER)
    return f"""#role:
You are a senior product manager's daily briefing assistant. Your job is to convert newsletter emails into structured fields for a daily Obsidian brief. Follow every formatting rule below exactly — no deviations, no extra or placeholder fields.

The full brief for **{brief_date_str}** is assembled by software from many emails. **You only see one email.** Your output must be the plain **OUTPUT FIELD BLOCK** at the bottom — not the full Markdown document.

#PERSONA
Think like a senior PM. Prioritise: strategic insights, market trends, competitive intelligence, frameworks with real case studies, actionable data points. Ruthlessly filter noise.

#DOCUMENT STRUCTURE (assembled elsewhere — context only)
The final document includes, in order:
1. TITLE — `# Daily Brief — {brief_date_str}`
2. KEY NUMBERS — blockquotes with 2–3 striking numeric facts across today's articles (another step)
3. THEME INDEX — markdown table by theme
4. ARTICLES — grouped under `## {{theme}}` headers

You supply **classification + per-article content** for step 4 via the field block.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PER-ARTICLE RULES (this email)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- **Article title** — sentence case, specific (this becomes the `###` heading in the brief).
- **Read time minutes** — integer **2**, **3**, or **4** only:
  - Short (1–2 thin points) = 2
  - Standard (3–5 solid points) = 3
  - Dense (long / technical) = 4
- **TL;DR** — 2–3 sentences. Concrete numbers, names, or outcomes where they exist. Never vague.
- **Key insight** — **exactly one sentence**: the single most useful or surprising thing; what to remember if nothing else is read.
- **Main points** — 3–5 bullets (`- ` lines). Each a complete, concrete sentence — **not** a restatement of the TL;DR. Add a 4th or 5th bullet only if the article is genuinely dense.
- **Topic** / **Company** — hashtags in kebab-case (e.g. `#openai`). Use `None` if none apply.
- **Theme** — exactly **one** line from this list (copy spelling and casing exactly):
{theme_list}

#RULES

NEVER include:
- Urgency ratings, urgency emojis (🔴🟡⚪), or any urgency field
- Markdown `###` headings, YAML front matter, or table syntax in your output
- Placeholder lines like "TBD" or "N/A" except `None` for Topic/Company as specified

ALWAYS:
- Put **only** the OUTPUT FIELD BLOCK in your response after you decide to keep the email (no preamble)
- Use sentence case for Article title

DISCARD RULE
Output the single word SKIP — nothing else — if the email is:
- Purely promotional, course/event sales, or cart abandonment
- Onboarding drip with no substantive editorial content
- Transactional (receipts, notifications, confirmations)
When in doubt, keep it.
- Do not include placeholder entries.

---

OUTPUT FIELD BLOCK (required when not SKIP — plain lines only):
Theme: <one theme from the list above>
Read minutes: <2|3|4>
Article title: <sentence case title>
TL;DR: <2–3 sentences>
Key insight: <one sentence only>
Main points:
- <sentence>
- <sentence>
- <sentence>
Topic: <#tags or None>
Company: <#tags or None>

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
    *,
    brief_date_str: str,
) -> ScoredSummary | None:
    """Call Gemini to summarise *item*. Returns None if Gemini SKIPped the email."""
    prompt = _build_summarise_prompt(cfg, item, brief_date_str)

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


def _build_key_numbers_prompt(date_str: str, digest_excerpts: str) -> str:
    return f"""You are a senior product manager's briefing editor.

Below are excerpts from newsletter articles that will appear in the **Daily Brief — {date_str}**.

Write **only** the "Key numbers" body: **2–3** markdown blockquote lines. Each line must follow this pattern exactly:
> **{{number or stat}}** — {{one sentence of context}}

Use genuinely striking or significant numbers mentioned in the excerpts. If there are **zero** strong numeric facts, output exactly this single line (italic):
_No striking numeric facts in today's articles._

Do not add a heading, preamble, or bullet list outside those rules.

---

{digest_excerpts}
"""


def _generate_key_numbers_markdown(
    cfg: dict[str, Any],
    client: genai.Client,
    rows: list[tuple[EmailItem, ScoredSummary, str]],
    date_str: str,
) -> str:
    """Second-pass Gemini call: pull 2–3 numeric highlights across all articles."""
    if not rows:
        return ""
    excerpts: list[str] = []
    for item, scored, _ in rows:
        points_join = " ".join(scored.main_points[:5])
        excerpts.append(
            f"Article title: {scored.headline}\n"
            f"TL;DR: {scored.tldr}\n"
            f"Key insight: {scored.key_insight}\n"
            f"Main points: {points_join}"
        )
    prompt = _build_key_numbers_prompt(date_str, "\n\n---\n\n".join(excerpts))

    def _call() -> Any:
        return client.models.generate_content(
            model=cfg["gemini_model"],
            contents=prompt,
            config={"max_output_tokens": GEMINI_KEY_NUMBERS_MAX_TOKENS},
        )

    resp = _with_retry(
        _call,
        max_attempts=GEMINI_RETRY_ATTEMPTS,
        backoff_base=GEMINI_RETRY_BACKOFF,
        retryable=_RETRYABLE_GEMINI,
        label="gemini/key_numbers",
    )
    return (getattr(resp, "text", "") or "").strip()


# ── Markdown ───────────────────────────────────────────────────────────────────

def _gmail_open_url(thread_id: str) -> str:
    return f"https://mail.google.com/mail/u/0/#all/{thread_id}"


def _email_to_markdown_section(item: EmailItem, scored: ScoredSummary, source_label: str) -> str:
    """Render a single email as a structured Markdown section (Obsidian brief style)."""
    url = _gmail_open_url(item.thread_id)
    display_name = _extract_display_name(item.from_addr)
    display_date = _normalize_date(item.date)
    rm = max(2, min(4, scored.read_minutes))

    lines = [
        f"### {scored.headline}",
        (
            f"> **From:** {display_name} · **Date:** {display_date} · **Label:** {source_label} "
            f"· ⏱ {rm} min read"
        ),
        "",
    ]

    tag_line_parts: list[str] = []
    if scored.topics:
        tag_line_parts.append("topic: " + " ".join(scored.topics))
    if scored.companies:
        tag_line_parts.append("company: " + " ".join(scored.companies))
    if tag_line_parts:
        lines.append(" · ".join(tag_line_parts))
        lines.append("")

    lines += [
        "**TL;DR**",
        scored.tldr,
        "",
        "**Key insight**",
        scored.key_insight or "_No single insight extracted._",
        "",
    ]

    if scored.main_points:
        lines.append("**Main points**")
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
    key_numbers_markdown: str,
) -> str:
    """Build the full Obsidian-compatible Markdown document for the daily brief."""
    parts: list[str] = []

    parts.append(f"---\ndate: {date_str}\ntags: [daily-brief, email-digest]\n---\n")

    parts.append(f"# Daily Brief — {date_str}\n")

    parts.append("## Key numbers\n\n")
    if key_numbers_markdown.strip():
        parts.append(key_numbers_markdown.strip() + "\n\n")
    else:
        parts.append("_No striking numeric facts in today's articles._\n\n")

    all_rows: list[tuple[EmailItem, ScoredSummary, str]] = [
        (item, scored, sl)
        for _, rows in label_sections
        for item, scored, sl in rows
    ]

    parts.append("## Today's reading\n\n")
    if not all_rows:
        parts.append("_No qualifying emails for this period._\n")
        return "".join(parts)

    parts.append("| Theme | Article | Source | Read time |\n")
    parts.append("|-------|---------|--------|----------|\n")
    for item, scored, _sl in sorted(all_rows, key=lambda r: _email_sort_timestamp(r[0]), reverse=True):
        theme = scored.theme
        title = scored.headline.replace("|", "\\|")
        src = _extract_display_name(item.from_addr).replace("|", "\\|")
        parts.append(f"| {theme} | {title} | {src} | {scored.read_minutes} min |\n")
    parts.append("\n")

    by_theme: dict[str, list[tuple[EmailItem, ScoredSummary, str]]] = {t: [] for t in THEME_SECTION_ORDER}
    for item, scored, sl in all_rows:
        th = scored.theme if scored.theme in by_theme else _normalize_theme(scored.theme)
        if th not in by_theme:
            th = _DEFAULT_THEME
        by_theme[th].append((item, scored, sl))

    for theme in THEME_SECTION_ORDER:
        bucket = by_theme.get(theme, [])
        if not bucket:
            continue
        parts.append(f"## {theme}\n\n")
        bucket.sort(key=lambda row: _email_sort_timestamp(row[0]), reverse=True)
        for item, scored, source_label in bucket:
            parts.append(_email_to_markdown_section(item, scored, source_label))

    return "".join(parts)


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
        # Prefer refresh-token auth for long-running automation in CI.
        if (
            cfg.get("dropbox_refresh_token")
            and cfg.get("dropbox_app_key")
            and cfg.get("dropbox_app_secret")
        ):
            dbx = dropbox_sdk.Dropbox(
                oauth2_refresh_token=cfg["dropbox_refresh_token"],
                app_key=cfg["dropbox_app_key"],
                app_secret=cfg["dropbox_app_secret"],
            )
        elif cfg.get("dropbox_access_token"):
            dbx = dropbox_sdk.Dropbox(oauth2_access_token=cfg["dropbox_access_token"])
        else:
            print(
                "Dropbox auth configuration missing at upload time. "
                "Set DROPBOX_ACCESS_TOKEN or "
                "DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET.",
                file=sys.stderr,
            )
            return False

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
            "Dropbox authentication failed. Check DROPBOX credentials "
            "(preferred: DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET; "
            "fallback: DROPBOX_ACCESS_TOKEN).\n"
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
    brief_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
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
                executor.submit(_summarise, cfg, gemini_client, item, brief_date_str=brief_date)
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

    all_rows_flat: list[tuple[EmailItem, ScoredSummary, str]] = [
        (item, scored, sl)
        for _, rows in label_sections
        for item, scored, sl in rows
    ]
    key_numbers_md = ""
    if all_rows_flat:
        key_numbers_md = _generate_key_numbers_markdown(
            cfg, gemini_client, all_rows_flat, brief_date
        )

    filename = f"{brief_date}-daily-brief.md"
    markdown = _build_daily_brief_markdown(
        label_sections,
        date_str=brief_date,
        key_numbers_markdown=key_numbers_md,
    )

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
