import importlib.util
import json
from pathlib import Path


def load_site_build():
    path = Path(__file__).resolve().parent.parent / "site" / "build.py"
    spec = importlib.util.spec_from_file_location("site_build_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_render_markdown_renders_tables():
    site_build = load_site_build()

    html = site_build.render_markdown("| A | B |\n| --- | --- |\n| 1 | 2 |")

    assert "<table>" in html


def test_search_index_marks_analyzed_papers(tmp_path, monkeypatch):
    site_build = load_site_build()
    data_dir = tmp_path / "data"
    daily_dir = data_dir / "daily"
    daily_dir.mkdir(parents=True)
    (daily_dir / "2026-07-30.json").write_text(json.dumps({"papers": [
        {"arxiv_id": "2607.00001", "title": "Read"},
        {"arxiv_id": "2607.00002", "title": "Unread"},
    ]}), encoding="utf-8")
    monkeypatch.setattr(site_build, "DATA_DIR", data_dir)
    monkeypatch.setattr(site_build, "DOCS_DIR", tmp_path / "docs")
    site_build.DOCS_DIR.mkdir()

    site_build.build_search_index({"2607.00001"})

    papers = json.loads((site_build.DOCS_DIR / "search-index.json").read_text())
    assert [paper["has_analysis"] for paper in papers] == [True, False]


def test_deep_read_list_marks_old_and_full_text_reports(tmp_path, monkeypatch):
    site_build = load_site_build()
    data_dir = tmp_path / "data"
    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "index.json").write_text(json.dumps([
        {
            "arxiv_id": "2607.00001",
            "title": "Full text",
            "authors": [],
            "analysis_date": "2026-07-31",
            "overall_score": 8,
            "domain": "宇宙学",
            "analysis_type": "full_text_companion",
            "text_source": "TeX",
        },
        {
            "arxiv_id": "2607.00002",
            "title": "Old report",
            "authors": [],
            "analysis_date": "2026-07-30",
            "overall_score": 7,
            "domain": "宇宙学",
        },
    ]), encoding="utf-8")
    monkeypatch.setattr(site_build, "DATA_DIR", data_dir)
    monkeypatch.setattr(site_build, "DOCS_DIR", tmp_path / "docs")

    site_build.build_deep_read_list("/", "/daily/2026-07-31/")

    html = (site_build.DOCS_DIR / "deep-read" / "index.html").read_text()
    assert "全文伴读报告（TeX）" in html
    assert "旧版摘要分析" in html
