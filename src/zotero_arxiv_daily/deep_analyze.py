#!/usr/bin/env python3
"""Deep paper analysis via LLM — called by analyze.yml workflow."""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = """You are an astrophysics professor writing detailed paper analysis notes for a PhD student. Analyze the given paper thoroughly and output a JSON object with the following structure. ALL content MUST be in Chinese.

Scoring criteria (0-10):
- Innovation (创新性): 9-10=breakthrough, 7-8=significant, 5-6=minor, 3-4=incremental, 1-2=known
- Technical Quality (技术质量): 9-10=rigorous, 7-8=good, 5-6=acceptable, 3-4=problematic, 1-2=poor
- Experiment (实验充分性): 9-10=comprehensive, 7-8=good, 5-6=acceptable, 3-4=limited, 1-2=poor
- Writing (写作质量): 9-10=clear, 7-8=mostly clear, 5-6=understandable, 3-4=confusing, 1-2=poor
- Practicality (实用性): 9-10=high impact, 7-8=good potential, 5-6=moderate, 3-4=limited, 1-2=theoretical only

Output strictly this JSON structure (no markdown fences, no extra text):

{
  "domain": "领域名称 (e.g., 引力波与相对论, 宇宙学, 星系天体物理, 高能天体物理)",
  "overall_score": 8.0,
  "scores": {"innovation": 8, "technical_quality": 9, "experiment": 7, "writing": 8, "practicality": 7},
  "highlights": ["亮点1", "亮点2", "亮点3"],
  "sections": {
    "core_info": "## 核心信息\\n\\n- **论文ID**: arXiv:XXXX.XXXXX\\n- **作者**: ...\\n- **机构**: ...\\n- **发布时间**: ...\\n- **链接**: ...",
    "abstract_translation": "## 摘要翻译\\n\\n### 英文摘要\\n\\n...\\n\\n### 中文翻译\\n\\n...\\n\\n### 核心要点提炼\\n\\n...",
    "background": "## 研究背景与动机\\n\\n### 领域现状\\n\\n...\\n\\n### 现有方法的局限性\\n\\n...\\n\\n### 研究动机\\n\\n...",
    "method": "## 方法概述\\n\\n### 核心思想\\n\\n...\\n\\n### 方法框架\\n\\n...\\n\\n### 关键创新\\n\\n...",
    "equations": "## 关键公式 / 理论基础\\n\\n(如有重要公式，用 $...$ 行内和 $$...$$ 块级 LaTeX 写出；如无公式，此节留空字符串)",
    "results": "## 实验结果\\n\\n### 实验设置\\n\\n...\\n\\n### 主要结果\\n\\n...",
    "deep_analysis": "## 深度分析\\n\\n### 研究价值评估\\n\\n...\\n\\n### 方法优势详解\\n\\n...\\n\\n### 局限性分析\\n\\n...",
    "roadmap": "## 技术路线定位\\n\\n...",
    "future_work": "## 未来工作建议\\n\\n...",
    "assessment": "## 我的综合评价\\n\\n### 价值评分\\n\\n...\\n\\n### 突出亮点\\n\\n...\\n\\n### 可借鉴点\\n\\n...\\n\\n### 批判性思考\\n\\n..."
  },
}

Rules:
- Each section value must be Markdown-formatted Chinese text (2-5 paragraphs for substantive sections).
- Use $...$ for inline LaTeX and $$...$$ for block LaTeX on separate lines.
- Use Markdown table syntax for comparison tables.
- Output ONLY the JSON object. No markdown code fences, no explanation."""


def fetch_paper_info(arxiv_id: str) -> dict | None:
    """Fetch paper metadata with robust retry loop."""
    import time
    import random
    import arxiv

    max_retries = 8
    for attempt in range(max_retries):
        try:
            client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=0)
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(client.results(search))
            return {
                "title": result.title or "",
                "abstract": result.summary or "",
                "authors": [a.name for a in result.authors],
            }
        except StopIteration:
            print(f"No paper found for {arxiv_id}")
            return None
        except Exception as e:
            msg = str(e)
            if attempt < max_retries - 1:
                wait = min(2 ** attempt + random.uniform(1, 3), 60)
                print(f"arXiv API error (attempt {attempt + 1}/{max_retries}): {msg[:100]}")
                print(f"  Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"Failed after {max_retries} attempts: {msg[:100]}")
                return None


def call_llm(arxiv_id: str, paper_info: dict) -> dict:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    model = os.environ.get("MODEL_NAME", "gpt-4o")

    user_prompt = f"""Analyze the following paper thoroughly:

arXiv ID: {arxiv_id}
Title: {paper_info.get('title', 'Unknown')}
Authors: {', '.join(paper_info.get('authors', []))}
Abstract: {paper_info.get('abstract', 'No abstract available')}

Provide a complete analysis following the system prompt instructions. Output ONLY valid JSON (no markdown code fences, no extra text)."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=16384,
    )
    raw = response.choices[0].message.content

    # Extract JSON — try direct parse first, then regex extraction
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"Failed to parse LLM JSON output. Raw (first 500 chars): {raw[:500]}")
    sys.exit(1)


def main():
    arxiv_id = os.environ.get("ARXIV_ID")
    if not arxiv_id:
        print("ARXIV_ID environment variable is required")
        sys.exit(1)

    print(f"Fetching paper info for {arxiv_id}...")
    paper_info = fetch_paper_info(arxiv_id)
    if not paper_info:
        print(f"Failed to fetch paper info for {arxiv_id}")
        sys.exit(1)
    print(f"Title: {paper_info['title']}")

    print("Calling LLM for deep analysis...")
    analysis = call_llm(arxiv_id, paper_info)

    # Add metadata
    analysis["arxiv_id"] = arxiv_id
    analysis["title"] = paper_info["title"]
    analysis["authors"] = paper_info["authors"]
    analysis["analysis_date"] = datetime.now().strftime("%Y-%m-%d")

    # Save analysis JSON
    analysis_dir = Path("data/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / f"{arxiv_id}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"Analysis saved to {analysis_path}")

    # Update index
    index_path = analysis_dir / "index.json"
    index = []
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    # Remove existing entry for this paper if any
    index = [e for e in index if e.get("arxiv_id") != arxiv_id]
    index.insert(0, {
        "arxiv_id": arxiv_id,
        "title": paper_info["title"],
        "authors": paper_info["authors"],
        "analysis_date": analysis["analysis_date"],
        "overall_score": analysis.get("overall_score", 0),
        "domain": analysis.get("domain", ""),
    })
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Index updated at {index_path}")


if __name__ == "__main__":
    main()
