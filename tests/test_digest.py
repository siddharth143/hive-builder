"""Unit tests for digest.py — covers all pure utility and parsing functions."""
from __future__ import annotations

import pytest

from digest import (
    EmailItem,
    GmailLabel,
    ScoredSummary,
    _build_slack_messages,
    _email_card_blocks,
    _expand_label_family,
    _gmail_open_url,
    _matches_filter,
    _parse_csv_env,
    _parse_gemini_response,
    _slack_escape,
    _truncate,
)


# ── _slack_escape ──────────────────────────────────────────────────────────────

def test_slack_escape_ampersand():
    assert _slack_escape("a & b") == "a &amp; b"


def test_slack_escape_lt_gt():
    assert _slack_escape("<tag>") == "&lt;tag&gt;"


def test_slack_escape_combined():
    assert _slack_escape("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_slack_escape_no_special_chars():
    assert _slack_escape("hello world") == "hello world"


def test_slack_escape_empty_string():
    assert _slack_escape("") == ""


def test_slack_escape_idempotent_entities():
    # Already-escaped entities should be double-escaped (we escape, not unescape).
    assert _slack_escape("&amp;") == "&amp;amp;"


# ── _truncate ──────────────────────────────────────────────────────────────────

def test_truncate_short_string_unchanged():
    assert _truncate("hi", 10) == "hi"


def test_truncate_exact_length_unchanged():
    assert _truncate("hello", 5) == "hello"


def test_truncate_long_string_ends_with_ellipsis():
    result = _truncate("hello world", 7)
    assert result.endswith("…")
    assert len(result) <= 7


def test_truncate_empty_string():
    assert _truncate("", 10) == ""


def test_truncate_none_treated_as_empty():
    assert _truncate(None, 10) == ""  # type: ignore[arg-type]


def test_truncate_strips_whitespace():
    assert _truncate("  hi  ", 10) == "hi"


# ── _parse_csv_env ─────────────────────────────────────────────────────────────

def test_parse_csv_env_basic():
    assert _parse_csv_env("a, b, c") == ["a", "b", "c"]


def test_parse_csv_env_skips_empty_entries():
    assert _parse_csv_env("a,,b") == ["a", "b"]


def test_parse_csv_env_empty_string():
    assert _parse_csv_env("") == []


def test_parse_csv_env_single_value():
    assert _parse_csv_env("only") == ["only"]


def test_parse_csv_env_trims_whitespace():
    assert _parse_csv_env("  x , y  ") == ["x", "y"]


# ── _matches_filter ────────────────────────────────────────────────────────────

def test_matches_filter_no_keywords_always_false():
    assert _matches_filter("subject", "snippet", []) is False


def test_matches_filter_match_in_subject():
    assert _matches_filter("Buy Now!", "details", ["buy now"]) is True


def test_matches_filter_match_in_snippet():
    assert _matches_filter("Hello", "Click here to unsubscribe", ["unsubscribe"]) is True


def test_matches_filter_case_insensitive():
    assert _matches_filter("NEWSLETTER", "content", ["newsletter"]) is True


def test_matches_filter_no_match():
    assert _matches_filter("Interesting research", "good content", ["buy", "sale"]) is False


def test_matches_filter_partial_word_matches():
    # "sale" appears inside "wholesale" — should still match.
    assert _matches_filter("Wholesale pricing", "n/a", ["sale"]) is True


# ── _expand_label_family ───────────────────────────────────────────────────────

def _labels(*names: str) -> list[GmailLabel]:
    return [GmailLabel(id=f"id_{i}", name=n) for i, n in enumerate(names)]


def test_expand_label_family_includes_parent_and_children():
    labs = _labels("Inbox", "Work", "Work/Projects", "Work/Projects/2024")
    result = _expand_label_family(labs, "Work")
    assert [l.name for l in result] == ["Work", "Work/Projects", "Work/Projects/2024"]


def test_expand_label_family_no_match_returns_empty():
    labs = _labels("Inbox", "Personal")
    assert _expand_label_family(labs, "Work") == []


def test_expand_label_family_case_insensitive_lookup():
    labs = _labels("AI Product Management", "AI Product Management/News")
    result = _expand_label_family(labs, "ai product management")
    assert len(result) == 2


def test_expand_label_family_empty_name_returns_empty():
    labs = _labels("Inbox")
    assert _expand_label_family(labs, "") == []


def test_expand_label_family_no_partial_word_match():
    """'Work' must not match 'Workshop'."""
    labs = _labels("Work", "Workshop", "Workshop/Advanced")
    result = _expand_label_family(labs, "Work")
    assert [l.name for l in result] == ["Work"]


def test_expand_label_family_sorted_alphabetically():
    labs = _labels("Z/Beta", "Z/Alpha", "Z")
    result = _expand_label_family(labs, "Z")
    assert [l.name for l in result] == ["Z", "Z/Alpha", "Z/Beta"]


# ── _gmail_open_url ────────────────────────────────────────────────────────────

def test_gmail_open_url_format():
    assert _gmail_open_url("abc123") == "https://mail.google.com/mail/u/0/#all/abc123"


def test_gmail_open_url_special_chars_in_thread_id():
    # Thread IDs are hex strings; just verify it's embedded verbatim.
    tid = "1234abcd"
    assert tid in _gmail_open_url(tid)


# ── _parse_gemini_response ─────────────────────────────────────────────────────

def test_parse_gemini_response_skip_uppercase():
    assert _parse_gemini_response("SKIP") is None


def test_parse_gemini_response_skip_lowercase():
    assert _parse_gemini_response("skip") is None


def test_parse_gemini_response_skip_with_trailing_newline():
    assert _parse_gemini_response("SKIP\n") is None


def test_parse_gemini_response_valid_structured():
    text = "Relevance: 4\nUrgency: High\nSummary: This is a great insight."
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.relevance == 4
    assert result.urgency == "High"
    assert "great insight" in result.summary


def test_parse_gemini_response_no_space_after_colon():
    """Regression: Gemini sometimes omits the space after the colon."""
    text = "Relevance:3\nUrgency:Medium\nSummary:Short summary here."
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.relevance == 3
    assert result.urgency == "Medium"


def test_parse_gemini_response_case_insensitive_urgency():
    text = "Relevance: 2\nUrgency: high\nSummary: A summary."
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.urgency == "High"  # normalised to title-case


def test_parse_gemini_response_multiline_summary():
    text = "Relevance: 5\nUrgency: Low\nSummary: Line one.\nLine two continues the thought."
    result = _parse_gemini_response(text)
    assert result is not None
    assert "Line one" in result.summary
    assert "Line two" in result.summary


def test_parse_gemini_response_all_relevance_values():
    for score in range(1, 6):
        text = f"Relevance: {score}\nUrgency: Low\nSummary: Test."
        result = _parse_gemini_response(text)
        assert result is not None
        assert result.relevance == score


def test_parse_gemini_response_all_urgency_values():
    for urg in ("High", "Medium", "Low"):
        text = f"Relevance: 3\nUrgency: {urg}\nSummary: Test."
        result = _parse_gemini_response(text)
        assert result is not None
        assert result.urgency == urg


def test_parse_gemini_response_fallback_on_malformed():
    """Unstructured response gets fallback scores."""
    text = "This is just some random text without structure."
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.relevance == 3
    assert result.urgency == "Low"
    assert result.summary == text


def test_parse_gemini_response_fallback_preserves_full_text():
    text = "A paragraph with no scoring fields at all."
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.summary == text


# ── _email_card_blocks ─────────────────────────────────────────────────────────

def _make_item(subject: str = "Test Subject") -> EmailItem:
    return EmailItem(
        msg_id="msg1",
        thread_id="thread1",
        subject=subject,
        from_addr="sender@example.com",
        date="2024-01-01",
        snippet="snippet text",
    )


def _make_scored(relevance: int = 3, urgency: str = "Low") -> ScoredSummary:
    return ScoredSummary(summary="A good summary.", relevance=relevance, urgency=urgency)


def test_email_card_blocks_returns_four_blocks():
    card = _email_card_blocks(_make_item(), _make_scored(), "MyLabel")
    assert len(card) == 4


def test_email_card_blocks_contains_gmail_url():
    card = _email_card_blocks(_make_item(), _make_scored(), "MyLabel")
    actions = next(b for b in card if b["type"] == "actions")
    url = actions["elements"][0]["url"]
    assert "mail.google.com" in url
    assert "thread1" in url


def test_email_card_blocks_subject_escaped():
    item = _make_item(subject="<Alert> & Warning")
    card = _email_card_blocks(item, _make_scored(), "L")
    header_block = next(b for b in card if b["type"] == "section" and "fields" in b)
    assert "&lt;Alert&gt;" in header_block["text"]["text"]


def test_email_card_blocks_high_urgency_uses_red_dot():
    card = _email_card_blocks(_make_item(), _make_scored(urgency="High"), "L")
    fields_block = next(b for b in card if b["type"] == "section" and "fields" in b)
    score_field = fields_block["fields"][3]["text"]
    assert "🔴" in score_field


# ── _build_slack_messages ──────────────────────────────────────────────────────

def test_build_slack_messages_empty_section_shows_placeholder():
    msgs = _build_slack_messages([("MyLabel", [])])
    assert len(msgs) == 1
    texts = [
        b.get("text", {}).get("text", "")
        for b in msgs[0]
        if isinstance(b.get("text"), dict)
    ]
    assert any("No qualifying" in t for t in texts)


def test_build_slack_messages_single_email():
    sections = [("L", [(_make_item(), _make_scored(), "L/Sub")])]
    msgs = _build_slack_messages(sections)
    assert len(msgs) == 1
    block_types = [b["type"] for b in msgs[0]]
    assert "header" in block_types
    assert "actions" in block_types


def test_build_slack_messages_all_blocks_within_limit():
    rows = [(_make_item(), _make_scored(), "Label") for _ in range(5)]
    msgs = _build_slack_messages([("Label", rows)])
    for msg in msgs:
        assert len(msg) <= 49


def test_build_slack_messages_splits_on_block_limit():
    """A tight max forces the digest into multiple messages."""
    # Each email card = 4 blocks + label header = 1 → 5 per email.
    # Header + divider = 2. So 2 emails = 2 + 1 + 4 + 4 = 11 blocks.
    # Setting limit to 7 forces a split after the first email.
    rows = [(_make_item(), _make_scored(), "L"), (_make_item(), _make_scored(), "L")]
    msgs = _build_slack_messages([("L", rows)], max_blocks_per_message=7)
    assert len(msgs) >= 2


def test_build_slack_messages_with_tldr_appears_in_last_message():
    sections = [("L", [])]
    msgs = _build_slack_messages(sections, tldr_lines=["Point one", "Point two", "Point three"])
    last_blocks = msgs[-1]
    texts = [
        b.get("text", {}).get("text", "")
        for b in last_blocks
        if isinstance(b.get("text"), dict)
    ]
    assert any("TL;DR" in t for t in texts)


def test_build_slack_messages_tldr_capped_at_three_bullets():
    sections = [("L", [])]
    lines = ["A", "B", "C", "D", "E"]
    msgs = _build_slack_messages(sections, tldr_lines=lines)
    # Find the TL;DR section block and count bullet lines.
    all_blocks = [b for msg in msgs for b in msg]
    tldr_section = next(
        (b for b in all_blocks if b["type"] == "section" and "- " in b.get("text", {}).get("text", "")),
        None,
    )
    assert tldr_section is not None
    bullet_count = tldr_section["text"]["text"].count("\n- ") + 1  # n items = n-1 newlines + 1
    assert bullet_count <= 3


def test_build_slack_messages_multiple_labels():
    rows1 = [(_make_item("Subj A"), _make_scored(), "L1")]
    rows2 = [(_make_item("Subj B"), _make_scored(), "L2")]
    msgs = _build_slack_messages([("Label1", rows1), ("Label2", rows2)])
    all_text = " ".join(
        b.get("text", {}).get("text", "")
        for msg in msgs
        for b in msg
        if isinstance(b.get("text"), dict)
    )
    assert "Label1" in all_text
    assert "Label2" in all_text
