"""Gmail label digest: fetch, filter, summarise with Gemini, post to Slack."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

GMAIL_SCOPES = ("https://www.googleapis.com/auth/gmail.readonly",)
GMAIL_TOKEN_URI = "https://oauth2.googleapis.com/token"
DEBUG = (os.environ.get("DEBUG") or "").strip().lower() in {"1", "true", "yes", "y", "on"}


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
    urgency: str  # High | Medium | Low


def _slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _parse_csv_env(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


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


def _gmail_credentials(cfg: dict[str, Any]) -> Credentials:
    creds = Credentials(
        token=None,
        refresh_token=cfg["gmail_refresh_token"],
        token_uri=GMAIL_TOKEN_URI,
        client_id=cfg["gmail_client_id"],
        client_secret=cfg["gmail_client_secret"],
        scopes=GMAIL_SCOPES,
    )
    creds.refresh(Request())
    return creds


def _label_id_by_name(service: Any, name: str) -> str:
    resp = service.users().labels().list(userId="me").execute()
    want = name.strip().lower()
    for lab in resp.get("labels", []):
        if (lab.get("name") or "").lower() == want:
            return lab["id"]
    print(f"Gmail label not found: {name!r}", file=sys.stderr)
    raise SystemExit(1)


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


def _debug(msg: str) -> None:
    if DEBUG:
        print(msg, file=sys.stderr)


def _list_message_refs(
    service: Any,
    label_id: str,
    *,
    q: str | None,
    max_count: int,
    include_spam_trash: bool = False,
) -> list[dict[str, str]]:
    _debug(f"[gmail] list: label_id={label_id} q={(q or '')!r} max={max_count} include_spam_trash={include_spam_trash}")
    refs: list[dict[str, str]] = []
    page_token: str | None = None
    while len(refs) < max_count:
        batch_size = min(500, max_count - len(refs))
        req = (
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
        )
        resp = req.execute()
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
        .execute()
    )
    headers = (msg.get("payload") or {}).get("headers") or []
    subject = _header(headers, "Subject") or "(no subject)"
    from_addr = _header(headers, "From") or "(unknown)"
    date = _header(headers, "Date") or ""
    if not date and msg.get("internalDate"):
        try:
            from datetime import datetime, timezone

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


def _summarise(
    cfg: dict[str, Any],
    model: Any,
    item: EmailItem,
) -> ScoredSummary | None:
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
- Start with the core insight (no vague openers like \"this email discusses\")
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

    prompt = f"""{system_instructions}

Email details:
Subject: {item.subject}
Sender: {item.from_addr}
Date received: {item.date}
Snippet / body preview:
{item.snippet}
"""
    try:
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 2048},
        )
    except Exception as e:
        print(f"Gemini API error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    text = ""
    try:
        text = (resp.text or "").strip()
    except Exception:
        text = ""
    if not text and resp.candidates:
        parts: list[str] = []
        for c in resp.candidates:
            content = getattr(c, "content", None)
            if not content:
                continue
            for p in getattr(content, "parts", None) or []:
                t = getattr(p, "text", None)
                if t:
                    parts.append(t)
        text = "".join(parts).strip()

    if re.match(r"^SKIP\s*$", text, re.IGNORECASE):
        return None

    rel_m = re.search(r"(?im)^\s*Relevance:\s*([1-5])\s*$", text)
    urg_m = re.search(r"(?im)^\s*Urgency:\s*(High|Medium|Low)\s*$", text)
    sum_m = re.search(r"(?im)^\s*Summary:\s*(.+)\s*$", text)
    if not (rel_m and urg_m and sum_m):
        # Fallback: treat full response as summary with default scores.
        return ScoredSummary(summary=text.strip(), relevance=3, urgency="Low")

    summary = sum_m.group(1).strip()
    relevance = int(rel_m.group(1))
    urgency = urg_m.group(1)
    return ScoredSummary(summary=summary, relevance=relevance, urgency=urgency)


def _gmail_open_url(thread_id: str) -> str:
    return f"https://mail.google.com/mail/u/0/#all/{thread_id}"


def _build_blocks(
    label_sections: list[tuple[str, list[tuple[EmailItem, ScoredSummary]]]],
    *,
    tldr_lines: list[str] | None = None,
    block_budget: int = 45,
) -> list[dict[str, Any]]:
    """Build Slack Block Kit payload (max 50 blocks per message)."""
    blocks: list[dict[str, Any]] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Gmail digest", "emoji": True}},
        {"type": "divider"},
    ]
    used = len(blocks)

    for label_name, rows in label_sections:
        if used + 1 > block_budget:
            break
        blocks.append(
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"Label: {label_name}", "emoji": False},
            }
        )
        used += 1

        if not rows:
            if used + 1 > block_budget:
                break
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "_No qualifying emails for this period._",
                    },
                }
            )
            used += 1
            continue

        omitted = 0
        for idx, (item, scored) in enumerate(rows):
            if used + 1 > block_budget:
                omitted = len(rows) - idx
                break
            url = _gmail_open_url(item.thread_id)
            urgency_dot = {"High": "🔴", "Medium": "🟡", "Low": "⚪"}.get(scored.urgency, "⚪")
            body = (
                f"*Subject:* {_slack_escape(item.subject)}\n"
                f"*From:* {_slack_escape(item.from_addr)}\n"
                f"*Date:* {_slack_escape(item.date)}\n\n"
                f"*Relevance:* {scored.relevance}/5   *Urgency:* {urgency_dot} {scored.urgency}\n\n"
                f"{_slack_escape(scored.summary)}\n\n"
                f"<{url}|Open in Gmail>"
            )
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": body}})
            used += 1

        if omitted > 0 and used + 1 <= block_budget:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_{omitted} more email(s) in this label omitted (Slack block limit)._",
                        }
                    ],
                }
            )
            used += 1

    if tldr_lines and used + 2 <= block_budget:
        blocks.append({"type": "divider"})
        used += 1
        blocks.append(
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "TL;DR", "emoji": False},
            }
        )
        used += 1
        bullets = "\n".join(f"- {_slack_escape(x)}" for x in tldr_lines[:3] if x.strip())
        if bullets and used + 1 <= block_budget:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": bullets}})
            used += 1

    return blocks


def _generate_tldr(model: Any, scored: list[tuple[str, EmailItem, ScoredSummary]]) -> list[str]:
    if not scored:
        return []
    # Keep the prompt bounded to avoid huge token usage.
    items = scored[:20]
    joined = "\n\n".join(
        f"Label: {label}\nSubject: {it.subject}\nFrom: {it.from_addr}\nSummary: {sc.summary}\nRelevance: {sc.relevance}/5 Urgency: {sc.urgency}"
        for label, it, sc in items
    )
    prompt = f"""You are a PM chief-of-staff. Produce exactly 3 punchy takeaways (one sentence each) from this daily email digest.
No preamble, no numbering. Output exactly 3 lines, each starting with \"- \".

Digest items:
{joined}
"""
    resp = model.generate_content(prompt, generation_config={"max_output_tokens": 256})
    text = (getattr(resp, "text", "") or "").strip()
    lines = [re.sub(r"^\s*[-•]\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln][:3]


def _post_slack(webhook: str, blocks: list[dict[str, Any]]) -> bool:
    r = requests.post(
        webhook,
        json={"blocks": blocks},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if r.status_code >= 400:
        print(f"Slack webhook error {r.status_code}: {r.text}", file=sys.stderr)
        return False
    return True


def main() -> int:
    cfg = _load_config()
    print(
        "[config] "
        + f"labels={cfg['labels']!r} date_window_days={cfg['date_window_days']} "
        + f"max_emails_per_label={cfg['max_emails_per_label']} filter_keywords={len(cfg['filter_keywords'])}",
        file=sys.stderr,
    )
    try:
        creds = _gmail_credentials(cfg)
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except HttpError as e:
        print(f"Gmail API error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Gmail setup failed: {e}", file=sys.stderr)
        return 1

    genai.configure(api_key=cfg["gemini_api_key"])
    try:
        model = genai.GenerativeModel(cfg["gemini_model"])
    except Exception as e:
        print(f"Gemini model init failed: {e}", file=sys.stderr)
        return 1

    label_sections: list[tuple[str, list[tuple[EmailItem, ScoredSummary]]]] = []
    all_scored: list[tuple[str, EmailItem, ScoredSummary]] = []

    for label_name in cfg["labels"]:
        try:
            label_id = _label_id_by_name(service, label_name)
            _debug(f"[gmail] resolved label {label_name!r} -> {label_id!r}")
            q = f"newer_than:{cfg['date_window_days']}d"
            refs = _list_message_refs(
                service,
                label_id,
                q=q,
                max_count=cfg["max_emails_per_label"],
            )
        except HttpError as e:
            print(f"Gmail API error for label {label_name!r}: {e}", file=sys.stderr)
            return 1

        rows: list[tuple[EmailItem, ScoredSummary]] = []
        fetched = len(refs)
        keyword_filtered = 0
        gemini_skipped = 0
        kept = 0
        # If we fetched nothing, do a fast sanity check: does the label contain ANY mail?
        # This helps distinguish "date window too small" vs "label has no messages".
        if fetched == 0:
            try:
                any_refs = _list_message_refs(
                    service,
                    label_id,
                    q=None,
                    max_count=1,
                )
                print(
                    "[label-check] "
                    + f"{label_name!r}: any_in_label={bool(any_refs)} (no date filter)",
                    file=sys.stderr,
                )
            except HttpError as e:
                print(f"Gmail API error during label-check for {label_name!r}: {e}", file=sys.stderr)
                return 1

        for ref in refs:
            mid = ref.get("id")
            if not mid:
                continue
            try:
                item = _fetch_email_item(service, mid)
            except HttpError as e:
                print(f"Gmail API error fetching message {mid}: {e}", file=sys.stderr)
                return 1
            if _matches_filter(item.subject, item.snippet, cfg["filter_keywords"]):
                keyword_filtered += 1
                _debug(f"[filter] keyword drop: {item.subject!r}")
                continue
            scored = _summarise(cfg, model, item)
            if scored is None:
                gemini_skipped += 1
                _debug(f"[gemini] SKIP: {item.subject!r}")
                continue
            rows.append((item, scored))
            all_scored.append((label_name, item, scored))
            kept += 1

        # Always log aggregate counts (safe, no secrets) so Actions logs explain empty digests.
        print(
            "[label] "
            + f"{label_name!r}: fetched={fetched} keyword_filtered={keyword_filtered} gemini_skipped={gemini_skipped} kept={kept}",
            file=sys.stderr,
        )
        label_sections.append((label_name, rows))

    tldr = []
    try:
        tldr = _generate_tldr(model, all_scored)
    except Exception as e:
        _debug(f"[tldr] failed: {e}")

    blocks = _build_blocks(label_sections, tldr_lines=tldr)
    try:
        if not _post_slack(cfg["slack_webhook"], blocks):
            return 1
    except requests.RequestException as e:
        print(f"Slack request failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
