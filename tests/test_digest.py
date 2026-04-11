"""Unit tests for digest.py — covers all pure utility and parsing functions."""
from __future__ import annotations

import pytest

from digest import (
    EmailItem,
    GmailLabel,
    ScoredSummary,
    _build_daily_brief_markdown,
    _email_to_markdown_section,
    _expand_label_family,
    _extract_display_name,
    _gmail_open_url,
    _matches_filter,
    _normalize_date,
    _parse_csv_env,
    _parse_gemini_response,
    _parse_hashtags,
    _truncate,
)


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
    assert _matches_filter("subject", "body", []) is False


def test_matches_filter_match_in_subject():
    assert _matches_filter("Buy Now!", "details", ["buy now"]) is True


def test_matches_filter_match_in_snippet():
    assert _matches_filter("Hello", "Click here to unsubscribe", ["unsubscribe"]) is True


def test_matches_filter_case_insensitive():
    assert _matches_filter("NEWSLETTER", "content", ["newsletter"]) is True


def test_matches_filter_no_match():
    assert _matches_filter("Interesting research", "good content", ["buy", "sale"]) is False


def test_matches_filter_partial_word_matches():
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


def test_gmail_open_url_contains_thread_id():
    assert "1234abcd" in _gmail_open_url("1234abcd")


# ── _extract_display_name ──────────────────────────────────────────────────────

def test_extract_display_name_standard_format():
    assert _extract_display_name("ByteByteGo <bytebytego@substack.com>") == "ByteByteGo"


def test_extract_display_name_quoted_name():
    assert _extract_display_name('"Paweł from Product Compass" <p@sub.com>') == "Paweł from Product Compass"


def test_extract_display_name_name_with_spaces():
    result = _extract_display_name("Guillermo Flor from Product Market Fit <g@sub.com>")
    assert result == "Guillermo Flor from Product Market Fit"


def test_extract_display_name_no_angle_brackets():
    assert _extract_display_name("just@email.com") == "just@email.com"


def test_extract_display_name_strips_whitespace():
    assert _extract_display_name("  TLDR  <dan@tldrnewsletter.com>") == "TLDR"


# ── _normalize_date ────────────────────────────────────────────────────────────

def test_normalize_date_rfc2822():
    assert _normalize_date("Sat, 11 Apr 2026 15:30:49 +0000") == "11 Apr 2026"


def test_normalize_date_with_offset():
    assert _normalize_date("Mon, 06 Apr 2026 12:40:49 +0530") == "6 Apr 2026"


def test_normalize_date_no_leading_zero_on_day():
    result = _normalize_date("Wed, 08 Apr 2026 10:19:10 +0000")
    assert result == "8 Apr 2026"
    assert not result.startswith("0")


def test_normalize_date_unparseable_returns_raw():
    raw = "not a real date"
    assert _normalize_date(raw) == raw


# ── _parse_hashtags ────────────────────────────────────────────────────────────

def test_parse_hashtags_with_hash_prefix():
    assert _parse_hashtags("#openai #anthropic") == ["#openai", "#anthropic"]


def test_parse_hashtags_without_hash_prefix():
    assert _parse_hashtags("openai anthropic") == ["#openai", "#anthropic"]


def test_parse_hashtags_mixed_case_normalized():
    assert _parse_hashtags("#OpenAI") == ["#openai"]


def test_parse_hashtags_kebab_case_preserved():
    assert _parse_hashtags("#prompt-engineering") == ["#prompt-engineering"]


def test_parse_hashtags_comma_separated():
    result = _parse_hashtags("#ai-agents, #product-strategy")
    assert "#ai-agents" in result
    assert "#product-strategy" in result


def test_parse_hashtags_none_returns_empty():
    assert _parse_hashtags("None") == []


def test_parse_hashtags_na_returns_empty():
    assert _parse_hashtags("N/A") == []


def test_parse_hashtags_empty_string_returns_empty():
    assert _parse_hashtags("") == []


def test_parse_hashtags_single_tag():
    assert _parse_hashtags("#google") == ["#google"]


# ── _parse_gemini_response ─────────────────────────────────────────────────────

_VALID_RESPONSE = (
    "Headline: Anthropic Surpasses OpenAI in Enterprise Revenue\n"
    "TL;DR: Anthropic has reached a $30B run-rate, overtaking OpenAI's $24B. "
    "Enterprise adoption of Claude is accelerating faster than GPT-4 equivalents.\n"
    "Main Points:\n"
    "- Anthropic's revenue run-rate hit $30B, ahead of OpenAI's $24B.\n"
    "- Safety-focused positioning is winning enterprise procurement teams.\n"
    "- OpenAI faces internal friction with CFO excluded from strategic decisions.\n"
    "Topic: #ai-industry #revenue\n"
    "Company: #anthropic #openai\n"
    "Urgency: High\n"
)


def test_parse_gemini_response_skip_uppercase():
    assert _parse_gemini_response("SKIP") is None


def test_parse_gemini_response_skip_lowercase():
    assert _parse_gemini_response("skip") is None


def test_parse_gemini_response_skip_with_trailing_newline():
    assert _parse_gemini_response("SKIP\n") is None


def test_parse_gemini_response_valid_headline():
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert result.headline == "Anthropic Surpasses OpenAI in Enterprise Revenue"


def test_parse_gemini_response_valid_tldr():
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert "$30B" in result.tldr


def test_parse_gemini_response_valid_main_points():
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert len(result.main_points) == 3
    assert any("$30B" in p for p in result.main_points)


def test_parse_gemini_response_valid_topics():
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert "#ai-industry" in result.topics
    assert "#revenue" in result.topics


def test_parse_gemini_response_valid_companies():
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert "#anthropic" in result.companies
    assert "#openai" in result.companies


def test_parse_gemini_response_valid_urgency():
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert result.urgency == "High"


def test_parse_gemini_response_no_relevance_field():
    """Relevance has been removed — responses without it should parse fine."""
    result = _parse_gemini_response(_VALID_RESPONSE)
    assert result is not None
    assert not hasattr(result, "relevance")


def test_parse_gemini_response_topic_none_returns_empty_list():
    text = (
        "Headline: A headline\n"
        "TL;DR: A summary.\n"
        "Main Points:\n- Point one.\n"
        "Topic: None\n"
        "Company: #openai\n"
        "Urgency: Low\n"
    )
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.topics == []


def test_parse_gemini_response_company_none_returns_empty_list():
    text = (
        "Headline: A headline\n"
        "TL;DR: A summary.\n"
        "Main Points:\n- Point one.\n"
        "Topic: #ai-agents\n"
        "Company: None\n"
        "Urgency: Medium\n"
    )
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.companies == []


def test_parse_gemini_response_case_insensitive_urgency():
    text = (
        "Headline: H\nTL;DR: S.\nMain Points:\n- P.\n"
        "Topic: None\nCompany: None\nUrgency: high\n"
    )
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.urgency == "High"


def test_parse_gemini_response_all_urgency_values():
    for urg in ("High", "Medium", "Low"):
        text = (
            f"Headline: H\nTL;DR: S.\nMain Points:\n- P.\n"
            f"Topic: None\nCompany: None\nUrgency: {urg}\n"
        )
        result = _parse_gemini_response(text)
        assert result is not None
        assert result.urgency == urg


def test_parse_gemini_response_fallback_on_malformed():
    text = "This is just some random text without structure."
    result = _parse_gemini_response(text)
    assert result is not None
    assert result.urgency == "Low"
    assert text in result.tldr
    assert result.topics == []
    assert result.companies == []


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_item(subject: str = "Test Subject") -> EmailItem:
    return EmailItem(
        msg_id="msg1",
        thread_id="thread1",
        subject=subject,
        from_addr="ByteByteGo <bytebytego@substack.com>",
        date="Sat, 11 Apr 2026 15:30:49 +0000",
        snippet="Short preview of the email.",
        body="Full body text of the email goes here.",
    )


def _make_scored(
    headline: str = "A great insight on product thinking",
    tldr: str = "A concise high-level summary.",
    main_points: list | None = None,
    topics: list | None = None,
    companies: list | None = None,
    urgency: str = "Low",
) -> ScoredSummary:
    return ScoredSummary(
        headline=headline,
        tldr=tldr,
        main_points=main_points if main_points is not None else ["Point one.", "Point two."],
        topics=topics if topics is not None else ["#ai-agents"],
        companies=companies if companies is not None else ["#openai"],
        urgency=urgency,
    )


# ── _email_to_markdown_section ─────────────────────────────────────────────────

def test_email_to_markdown_section_headline_as_h3():
    md = _email_to_markdown_section(_make_item(), _make_scored(headline="My Headline"), "L")
    assert "### My Headline" in md


def test_email_to_markdown_section_shows_display_name_not_raw_address():
    md = _email_to_markdown_section(_make_item(), _make_scored(), "L")
    assert "ByteByteGo" in md
    assert "bytebytego@substack.com" not in md


def test_email_to_markdown_section_shows_normalized_date():
    md = _email_to_markdown_section(_make_item(), _make_scored(), "L")
    assert "11 Apr 2026" in md
    assert "+0000" not in md


def test_email_to_markdown_section_shows_label():
    md = _email_to_markdown_section(_make_item(), _make_scored(), "Education/Newsletter")
    assert "Education/Newsletter" in md


def test_email_to_markdown_section_topic_hashtags():
    scored = _make_scored(topics=["#prompt-engineering", "#ai-agents"])
    md = _email_to_markdown_section(_make_item(), scored, "L")
    assert "topic: #prompt-engineering #ai-agents" in md


def test_email_to_markdown_section_company_hashtags():
    scored = _make_scored(companies=["#openai", "#anthropic"])
    md = _email_to_markdown_section(_make_item(), scored, "L")
    assert "company: #openai #anthropic" in md


def test_email_to_markdown_section_no_hashtag_line_when_empty():
    scored = _make_scored(topics=[], companies=[])
    md = _email_to_markdown_section(_make_item(), scored, "L")
    assert "topic:" not in md
    assert "company:" not in md


def test_email_to_markdown_section_tldr_section():
    scored = _make_scored(tldr="Core insight sentence here.")
    md = _email_to_markdown_section(_make_item(), scored, "L")
    assert "**TL;DR**" in md
    assert "Core insight sentence here." in md


def test_email_to_markdown_section_main_points():
    scored = _make_scored(main_points=["Alpha is first.", "Beta is second."])
    md = _email_to_markdown_section(_make_item(), scored, "L")
    assert "**Main Points**" in md
    assert "- Alpha is first." in md
    assert "- Beta is second." in md


def test_email_to_markdown_section_no_main_points_header_when_empty():
    scored = _make_scored(main_points=[])
    md = _email_to_markdown_section(_make_item(), scored, "L")
    assert "**Main Points**" not in md


def test_email_to_markdown_section_gmail_link_last():
    md = _email_to_markdown_section(_make_item(), _make_scored(), "L")
    assert "[Open in Gmail →]" in md
    assert "thread1" in md
    assert md.index("Open in Gmail") > md.index("TL;DR")


def test_email_to_markdown_section_high_urgency_red_dot():
    md = _email_to_markdown_section(_make_item(), _make_scored(urgency="High"), "L")
    assert "🔴" in md


def test_email_to_markdown_section_no_relevance_score():
    """Score field (e.g. '4/5') must no longer appear in sections."""
    md = _email_to_markdown_section(_make_item(), _make_scored(), "L")
    assert "/5" not in md


# ── _build_daily_brief_markdown ────────────────────────────────────────────────

def test_build_daily_brief_markdown_has_frontmatter():
    md = _build_daily_brief_markdown([], date_str="2024-01-01")
    assert md.startswith("---\n")
    assert "date: 2024-01-01" in md


def test_build_daily_brief_markdown_has_title():
    md = _build_daily_brief_markdown([], date_str="2024-01-15")
    assert "# Daily Brief — 2024-01-15" in md


def test_build_daily_brief_markdown_index_section():
    rows = [(_make_item(), _make_scored(headline="AI Strategy Insight"), "L")]
    md = _build_daily_brief_markdown([("Education", rows)], date_str="2024-01-01")
    assert "## Articles Processed" in md
    assert "AI Strategy Insight" in md


def test_build_daily_brief_markdown_index_uses_display_name():
    rows = [(_make_item(), _make_scored(), "L")]
    md = _build_daily_brief_markdown([("Education", rows)], date_str="2024-01-01")
    index_section = md.split("## Articles Processed")[1].split("## Education")[0]
    assert "ByteByteGo" in index_section
    assert "bytebytego@substack.com" not in index_section


def test_build_daily_brief_markdown_index_uses_normalized_date():
    rows = [(_make_item(), _make_scored(), "L")]
    md = _build_daily_brief_markdown([("Education", rows)], date_str="2024-01-01")
    index_section = md.split("## Articles Processed")[1].split("## Education")[0]
    assert "11 Apr 2026" in index_section
    assert "+0000" not in index_section


def test_build_daily_brief_markdown_index_numbered():
    rows = [
        (_make_item(), _make_scored(headline="First"), "L"),
        (_make_item(), _make_scored(headline="Second"), "L"),
    ]
    md = _build_daily_brief_markdown([("L", rows)], date_str="2024-01-01")
    assert "1. **First**" in md
    assert "2. **Second**" in md


def test_build_daily_brief_markdown_no_index_when_no_emails():
    md = _build_daily_brief_markdown([("Education", [])], date_str="2024-01-01")
    assert "## Articles Processed" not in md


def test_build_daily_brief_markdown_empty_label_placeholder():
    md = _build_daily_brief_markdown([("Education", [])], date_str="2024-01-01")
    assert "No qualifying emails" in md


def test_build_daily_brief_markdown_multiple_labels():
    rows1 = [(_make_item(), _make_scored(headline="Headline A"), "L1")]
    rows2 = [(_make_item(), _make_scored(headline="Headline B"), "L2")]
    md = _build_daily_brief_markdown(
        [("Label1", rows1), ("Label2", rows2)], date_str="2024-01-01"
    )
    assert "## Label1" in md
    assert "## Label2" in md
    assert "Headline A" in md
    assert "Headline B" in md


def test_build_daily_brief_markdown_index_before_sections():
    rows = [(_make_item(), _make_scored(headline="Test"), "L")]
    md = _build_daily_brief_markdown([("Education", rows)], date_str="2024-01-01")
    assert md.index("Articles Processed") < md.index("## Education")


def test_build_daily_brief_markdown_no_relevance_anywhere():
    """Relevance scores must not appear anywhere in the document."""
    rows = [(_make_item(), _make_scored(), "L")]
    md = _build_daily_brief_markdown([("Education", rows)], date_str="2024-01-01")
    assert "/5" not in md
