#!/usr/bin/env python3
"""Deep paper analysis via LLM — called by analyze.yml workflow."""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI
import tiktoken

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    extract_text_from_html,
    extract_text_from_pdf,
    extract_text_from_tar,
)

CHUNK_TOKENS = 6000
CHUNK_OVERLAP_TOKENS = 300
EVIDENCE_MAX_TOKENS = 1000

SYSTEM_PROMPT = """You are an astrophysics professor writing a full-text paper companion report for a PhD student. The final report must be grounded only in the supplied full-text evidence notes. ALL content MUST be in Chinese.

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
    "core_info": "## 核心信息\\n\\n- **论文ID**: arXiv:XXXX.XXXXX\\n- **作者**: ...\\n- **全文来源**: TeX / HTML / PDF",
    "problem_and_context": "## 问题与背景\\n\\n说明作者要解决的具体问题、已有工作的缺口，以及本文的研究位置。每个关键判断附原文定位。",
    "argument_map": "## 论证地图\\n\\n按假设 -> 方法 -> 证据 -> 结论说明推理链，并在每一步标明 §、Fig.、Table、Eq. 或 Appendix 定位。",
    "method_and_equations": "## 方法与关键公式\\n\\n解释支撑结论的方法、重要公式和符号物理含义。仅在全文明确给出公式时使用 $...$ 或 $$...$$。",
    "figure_and_table_guide": "## 图表导读\\n\\n基于正文中对图表的引用和 caption，说明关键 Fig./Table 要回答的问题、比较对象和作者据此得到的结论；不要声称看到了未提供的图像细节。",
    "results_and_limits": "## 结果、适用范围与局限\\n\\n区分论文直接陈述、AI解释和待核验问题，并标明原文定位。",
    "reading_route": "## 建议阅读路线\\n\\n给出阅读 PDF 时的顺序，以及最应亲自核验的 3 到 5 个位置。",
    "assessment": "## 综合评价\\n\\n评价创新性、技术可信度和与相关研究的关系；明确哪些是基于全文的推断，哪些需要读者自行核验。"
  },
}

Rules:
- Each section value must be Markdown-formatted Chinese text (2-5 paragraphs for substantive sections).
- Use $...$ for inline LaTeX and $$...$$ for block LaTeX on separate lines.
- Use Markdown table syntax for comparison tables.
- Every Markdown table must use one header row, one separator row, and one data row per physical line.
- Cite the available full-text location for every material claim using formats such as §3.2, Fig. 2, Table 1, Eq. (5), or Appendix A. If a location is not present in the evidence, say it is unavailable rather than inventing one.
- Clearly label AI interpretation as “AI解释” and unresolved points as “待核验”. Do not claim to have inspected image pixels; use only the text, captions, and cross-references supplied in the evidence.
- Output ONLY the JSON object. No markdown code fences, no explanation."""

EVIDENCE_SYSTEM_PROMPT = """You are extracting evidence from one contiguous chunk of an astrophysics paper's full text for a later companion report. Write concise Chinese Markdown notes. Cover only information present in this chunk: section/topic, assumptions, methods, equations, figure/table captions and references, results, limitations, and exact locations such as §, Fig., Table, Eq., or Appendix when visible. Do not infer missing content, do not summarize the whole paper, and do not claim to inspect figure pixels."""


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


def chunk_full_text(
    full_text: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split complete extracted text into overlapping, model-sized chunks."""
    if not full_text.strip():
        return []
    if chunk_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("Invalid full-text chunk settings")

    encoding = tiktoken.get_encoding("o200k_base")
    tokens = encoding.encode(full_text)
    step = chunk_tokens - overlap_tokens
    return [
        encoding.decode(tokens[start:start + chunk_tokens])
        for start in range(0, len(tokens), step)
        if tokens[start:start + chunk_tokens]
    ]


def fetch_full_text(arxiv_id: str, title: str) -> tuple[str | None, str | None]:
    """Retrieve text from arXiv source, then HTML, then PDF without fallback to abstract."""
    paper = SimpleNamespace(
        title=title,
        entry_id=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        source_url=lambda: f"https://arxiv.org/e-print/{arxiv_id}",
    )
    for source, extractor in (
        ("TeX", extract_text_from_tar),
        ("HTML", extract_text_from_html),
        ("PDF", extract_text_from_pdf),
    ):
        text = extractor(paper)
        if text and text.strip():
            return text, source
    return None, None


def _parse_llm_json(raw: str) -> dict:
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
    raise ValueError(f"Failed to parse LLM JSON output: {raw[:500]}")


def call_llm(
    arxiv_id: str,
    paper_info: dict,
    *,
    full_text: str,
    text_source: str,
) -> dict:
    if not full_text.strip() or text_source == "none":
        raise ValueError("Full text is required for deep analysis")

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    model = os.environ.get("MODEL_NAME", "gpt-4o")
    chunks = chunk_full_text(full_text)
    if not chunks:
        raise ValueError("Full text is required for deep analysis")

    evidence_notes = []
    for index, chunk in enumerate(chunks, start=1):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Paper arXiv ID: {arxiv_id}\nChunk {index}/{len(chunks)}:\n\n{chunk}",
                    },
                ],
                model=model,
                max_tokens=EVIDENCE_MAX_TOKENS,
            )
            note = response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"Full-text evidence extraction failed for chunk {index}") from exc
        if not note or not note.strip():
            raise RuntimeError(f"Full-text evidence extraction returned no content for chunk {index}")
        evidence_notes.append(f"## 全文分段 {index}/{len(chunks)}\n{note}")

    evidence_text = "\n\n".join(evidence_notes)
    user_prompt = f"""Write a full-text companion report from the complete evidence notes below.

arXiv ID: {arxiv_id}
Title: {paper_info.get('title', 'Unknown')}
Authors: {', '.join(paper_info.get('authors', []))}
Abstract: {paper_info.get('abstract', 'No abstract available')}
Full-text source: {text_source}

FULL-TEXT EVIDENCE NOTES:

{evidence_text}

Provide a complete analysis following the system prompt instructions. Output ONLY valid JSON (no markdown code fences, no extra text)."""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=16384,
    )
    return _parse_llm_json(response.choices[0].message.content)


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

    print("Retrieving complete paper text...")
    full_text, text_source = fetch_full_text(arxiv_id, paper_info["title"])
    if not full_text:
        print("Full text could not be extracted. No abstract-only analysis was created.")
        sys.exit(1)
    print(f"Full text retrieved from {text_source}.")

    print("Calling LLM for full-text deep analysis...")
    analysis = call_llm(
        arxiv_id,
        paper_info,
        full_text=full_text,
        text_source=text_source or "none",
    )

    # Add metadata
    analysis["arxiv_id"] = arxiv_id
    analysis["title"] = paper_info["title"]
    analysis["authors"] = paper_info["authors"]
    analysis["analysis_date"] = datetime.now().strftime("%Y-%m-%d")
    analysis["analysis_type"] = "full_text_companion"
    analysis["text_source"] = text_source

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
        "analysis_type": analysis["analysis_type"],
        "text_source": analysis["text_source"],
    })
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Index updated at {index_path}")


if __name__ == "__main__":
    main()
