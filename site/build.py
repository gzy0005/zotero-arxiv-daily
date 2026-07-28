#!/usr/bin/env python3
"""Static site generator for arXiv Daily website."""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import mistune
from jinja2 import Environment, FileSystemLoader, select_autoescape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)
markdown = mistune.create_markdown(escape=False)


def render_markdown(text: str) -> str:
    return markdown(text)


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_analyzed_ids() -> set[str]:
    index_path = DATA_DIR / "analysis" / "index.json"
    index = load_json(index_path)
    if not index:
        return set()
    return {entry["arxiv_id"] for entry in index}


def get_available_dates() -> list[str]:
    daily_dir = DATA_DIR / "daily"
    if not daily_dir.exists():
        return []
    dates = []
    for f in sorted(daily_dir.glob("*.json"), reverse=True):
        dates.append(f.stem)
    return dates


def build_daily_pages(analyzed_ids: set[str], site_base: str) -> list[str]:
    dates = get_available_dates()
    today = datetime.now().strftime("%Y-%m-%d")
    template = jinja_env.get_template("daily.html")

    for i, date_str in enumerate(dates):
        daily_data = load_json(DATA_DIR / "daily" / f"{date_str}.json")
        if not daily_data:
            continue

        prev_date = dates[i + 1] if i + 1 < len(dates) else None
        next_date = dates[i - 1] if i > 0 else None

        html = template.render(
            site_base=site_base,
            today=today,
            active_tab="daily",
            date=date_str,
            prev_date=prev_date,
            next_date=next_date,
            papers=daily_data.get("papers", []),
            analyzed_ids=analyzed_ids,
        )

        out_dir = DOCS_DIR / "daily" / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(html, encoding="utf-8")

    return dates


def build_analysis_pages(site_base: str):
    analysis_dir = DATA_DIR / "analysis"
    if not analysis_dir.exists():
        return []

    template = jinja_env.get_template("analysis.html")
    today = datetime.now().strftime("%Y-%m-%d")

    for f in sorted(analysis_dir.glob("*.json")):
        if f.stem == "index":
            continue
        paper = load_json(f)
        if not paper:
            continue

        if "sections" in paper:
            for key, content in paper["sections"].items():
                if content:
                    paper["sections"][key] = render_markdown(content)

        html = template.render(site_base=site_base, today=today, active_tab="deepread", paper=paper)

        out_dir = DOCS_DIR / "papers" / paper["arxiv_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "index.html").write_text(html, encoding="utf-8")


def build_deep_read_list(site_base: str):
    index = load_json(DATA_DIR / "analysis" / "index.json") or []
    template = jinja_env.get_template("deep_read_list.html")
    today = datetime.now().strftime("%Y-%m-%d")

    html = template.render(site_base=site_base, today=today, active_tab="deepread", papers=index)
    out_dir = DOCS_DIR / "deep-read"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def build_index(site_base: str):
    dates = get_available_dates()
    if dates:
        latest = dates[0]
        redirect_html = f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0;url={site_base}daily/{latest}/"></head><body></body></html>'
    else:
        redirect_html = '<!DOCTYPE html><html><head><meta charset="utf-8"><title>arXiv Daily</title></head><body><p>No papers yet.</p></body></html>'
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(redirect_html, encoding="utf-8")


def build_search_index():
    daily_dir = DATA_DIR / "daily"
    papers = []
    if daily_dir.exists():
        for f in sorted(daily_dir.glob("*.json"), reverse=True):
            daily_data = load_json(f)
            if not daily_data:
                continue
            date_str = f.stem
            for p in daily_data.get("papers", []):
                papers.append({
                    "title": p.get("title", ""),
                    "authors": p.get("authors", []),
                    "abstract": p.get("abstract", ""),
                    "tldr": p.get("tldr", ""),
                    "categories": p.get("categories", []),
                    "arxiv_id": p.get("arxiv_id"),
                    "date": date_str,
                    "score": p.get("score"),
                    "url": p.get("url", ""),
                })
    out_path = DOCS_DIR / "search-index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False)
    print(f"Search index with {len(papers)} papers written to {out_path}")


def build_search_page(site_base: str):
    template = jinja_env.get_template("search.html")
    today = datetime.now().strftime("%Y-%m-%d")
    html = template.render(site_base=site_base, today=today, active_tab="search")
    out_dir = DOCS_DIR / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def copy_static_assets():
    css_src = TEMPLATES_DIR.parent / "style.css"
    if css_src.exists():
        shutil.copy2(css_src, DOCS_DIR / "style.css")


def main():
    local = "--local" in sys.argv
    site_base = os.environ.get("SITE_BASE", "/" if local else "/zotero-arxiv-daily/")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    analyzed_ids = load_analyzed_ids()

    build_daily_pages(analyzed_ids, site_base)
    build_analysis_pages(site_base)
    build_deep_read_list(site_base)
    build_search_index()
    build_search_page(site_base)
    build_index(site_base)
    copy_static_assets()

    print(f"Site built to {DOCS_DIR} (site_base={site_base!r})")


if __name__ == "__main__":
    main()
