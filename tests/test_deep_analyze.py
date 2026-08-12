from types import SimpleNamespace

import pytest

from zotero_arxiv_daily import deep_analyze


def test_full_text_analysis_rejects_missing_document():
    with pytest.raises(ValueError, match="Full text is required"):
        deep_analyze.call_llm(
            "2607.00001",
            {"title": "Test", "abstract": "Abstract", "authors": []},
            full_text="",
            text_source="none",
        )


def test_fetch_paper_info_from_abs_page_reads_citation_metadata(monkeypatch):
    html = """
    <html><head>
      <meta name="citation_title" content="A &amp; B">
      <meta name="citation_author" content="First Author">
      <meta name="citation_author" content="Second Author">
      <meta name="citation_abstract" content="A full abstract.">
    </head></html>
    """
    calls = []

    def get(url, *, headers, timeout):
        calls.append((url, headers, timeout))
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    monkeypatch.setattr(deep_analyze, "requests", SimpleNamespace(get=get), raising=False)

    paper_info = deep_analyze.fetch_paper_info_from_abs_page("2608.09929")

    assert paper_info == {
        "title": "A & B",
        "abstract": "A full abstract.",
        "authors": ["First Author", "Second Author"],
    }
    assert calls[0][0] == "https://arxiv.org/abs/2608.09929"


def test_chunk_text_covers_the_complete_document():
    chunks = deep_analyze.chunk_full_text("a " * 20, chunk_tokens=5, overlap_tokens=1)

    assert len(chunks) > 1
    assert chunks[-1]


def test_full_text_analysis_uses_evidence_from_every_chunk(monkeypatch):
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            if len(requests) < 3:
                content = f"evidence {len(requests)}"
            else:
                content = '{"domain":"宇宙学","overall_score":8,"scores":{},"highlights":[],"sections":{}}'
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]})()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(deep_analyze, "OpenAI", lambda **_: FakeClient())
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(deep_analyze, "chunk_full_text", lambda *_: ["first section", "second section"])

    analysis = deep_analyze.call_llm(
        "2607.00001",
        {"title": "Test", "abstract": "Abstract", "authors": ["Author"]},
        full_text="complete document",
        text_source="TeX",
    )

    assert analysis["domain"] == "宇宙学"
    assert len(requests) == 3
    assert "first section" in requests[0]["messages"][1]["content"]
    assert "second section" in requests[1]["messages"][1]["content"]
    assert "evidence 1" in requests[2]["messages"][1]["content"]
    assert "evidence 2" in requests[2]["messages"][1]["content"]


def test_deepseek_evidence_extraction_disables_thinking(monkeypatch):
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            content = "evidence" if len(requests) == 1 else '{"domain":"宇宙学","overall_score":8,"scores":{},"highlights":[],"sections":{}}'
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]})()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(deep_analyze, "OpenAI", lambda **_: FakeClient())
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(deep_analyze, "chunk_full_text", lambda *_: ["one section"])

    deep_analyze.call_llm(
        "2607.00001",
        {"title": "Test", "abstract": "Abstract", "authors": ["Author"]},
        full_text="complete document",
        text_source="PDF",
    )

    assert requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in requests[1]


def test_full_text_prefers_source_and_never_uses_abstract(monkeypatch):
    calls = []

    def source(_paper):
        calls.append("TeX")
        return "source text"

    def unavailable(_paper):
        calls.append("unexpected")
        return None

    monkeypatch.setattr(deep_analyze, "extract_text_from_tar", source)
    monkeypatch.setattr(deep_analyze, "extract_text_from_html", unavailable)
    monkeypatch.setattr(deep_analyze, "extract_text_from_pdf", unavailable)

    text, source_name = deep_analyze.fetch_full_text("2607.00001", "Test")

    assert (text, source_name) == ("source text", "TeX")
    assert calls == ["TeX"]
