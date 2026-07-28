"""Tests for zotero_arxiv_daily.construct_email: render_email."""

from zotero_arxiv_daily.construct_email import render_email
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "★ 7.5" in html
    assert "共推荐" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "今天没有新论文" in html


def test_render_email_shows_top_5():
    papers = [make_sample_paper(title=f"Paper {i}", score=float(10 - i), tldr=f"TLDR {i}") for i in range(10)]
    html = render_email(papers)
    assert "Paper 0" in html
    assert "Paper 4" in html
    assert "共推荐" in html
    assert "<strong>10</strong>" in html
    # Papers beyond top 5 should not appear
    assert "Paper 5" not in html
    assert "Paper 9" not in html


def test_render_email_has_site_link():
    html = render_email([make_sample_paper()])
    assert "helemnmmm.github.io/zotero-arxiv-daily" in html
    assert "浏览全部论文" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "et al." in html
