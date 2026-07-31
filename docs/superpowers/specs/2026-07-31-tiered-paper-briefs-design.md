# Tiered Paper Briefs Design

## Goal

Replace the daily one-sentence TLDR with a 150-250 Chinese-character paper brief that helps the user understand each paper's research content before opening the original paper. A brief must state the research problem, core method or data, and main result or contribution.

The system must use full-text evidence for the top 20 papers ranked by relevance. Remaining ranked papers use title-and-abstract evidence. The website shows two lines by default and expands the complete brief on demand.

## Current Limitation

The current pipeline extracts full text before reranking every candidate. It then combines title, abstract, and full text, but truncates that prompt to its first 4,000 tokens. This is expensive for all candidates and can omit methods, results, and conclusions that appear later in a paper.

## Proposed Flow

1. Retrieve candidates with metadata and abstracts only.
2. Rerank candidates against the filtered Zotero corpus using abstracts, as today.
3. Select the configured maximum number of displayed papers.
4. Select the first 20 displayed papers for full-text enrichment. The other displayed papers remain abstract-only.
5. Generate a brief for every displayed paper:
   - Full-text tier: extract source in TeX, then arXiv HTML, then PDF order; process every usable text chunk into structured evidence; synthesize a final brief from that evidence.
   - Abstract tier: generate the same-length brief from title and abstract only, without asserting unsupported methodological or result details.
6. Persist results, build the static site, and deliver the existing email.

## Full-Text Evidence Pipeline

The full-text tier is a two-stage LLM workflow.

### Stage 1: Evidence extraction

Split the complete extracted text into chunks of at most 6,000 input tokens, with a 300-token overlap between adjacent chunks. Prefer natural section boundaries before applying a hard token boundary. Every chunk is processed; no trailing section is silently discarded. For each chunk, the model returns structured, source-grounded facts under these fields when present:

- research question or objective;
- method, model, experiment, or observational setup;
- data, samples, assumptions, or evaluation setup;
- reported results, quantitative findings, and comparisons;
- conclusions, limitations, and scope.

The prompt instructs the model to omit unavailable facts rather than infer them. This stage covers the entire extracted document rather than just the opening text.

### Stage 2: Brief synthesis

Use the full set of extracted facts to produce one 150-250 Chinese-character paragraph. The synthesis must cover:

1. the problem or research object;
2. the central approach and relevant data or setting;
3. the main result and concrete contribution.

The output is a natural paragraph, not a bullet list. It must avoid generic claims, unsupported details, and evaluation of relevance to the user.

## Data Model

Extend each persisted daily `Paper` record with:

- `tldr`: the final 150-250 character Chinese brief;
- `summary_source`: `full_text`, `abstract`, or `fallback_abstract`;
- `summary_error`: optional diagnostic text for logs and debugging only.

The public daily page continues to render `tldr`. The internal source and error fields are not shown to readers.

## Website Behavior

Each paper card renders a two-line clamped preview of its brief and a compact expand control. Selecting it reveals the complete brief in the same card; selecting it again collapses the text. The card layout must not shift unpredictably or reveal technical implementation labels.

## Site Correctness Fixes

The same implementation also corrects three existing site defects.

### Search analysis state

Build the search index with the set of analyzed arXiv IDs. Each indexed paper includes a derived `has_analysis` boolean. Search results use that value rather than merely testing for an arXiv ID:

- `has_analysis: false`: render an actionable `精读` control that dispatches the existing deep-analysis workflow.
- `has_analysis: true`: render a clickable `已审阅` link to the existing analysis page.

The search-page control follows the same localStorage PAT requirement and workflow-dispatch behavior as the daily recommendation page.

### Stable daily navigation

Derive the navigation target from the newest date with an existing `data/daily/YYYY-MM-DD.json` file, not from the wall-clock date when the static site is built. Pass that target to every page template. The global `每日推荐` link must therefore always open the latest available daily page, including on days with no new arXiv release. If no daily data exists, it links to the site root.

### Deep-analysis tables

Enable Mistune's Markdown table plugin when rendering analysis sections so valid pipe-table syntax produces semantic HTML tables. Configure the analysis table container for horizontal scrolling on narrow screens without clipping its cells. Update the deep-analysis LLM prompt to require standard Markdown table structure: one header row, one separator row, and one data row per physical line. This improves model output consistency; the renderer remains responsible for displaying every valid table.

## Failure Handling

- If TeX extraction fails, try arXiv HTML and then PDF.
- If no full text can be extracted for a top-20 paper, generate its brief from title and abstract and set `summary_source` to `abstract`.
- If a chunk-level LLM request fails, record the error and continue with evidence from the other chunks. If no usable evidence remains, use abstract-tier generation.
- If final synthesis fails, preserve the extracted fallback abstract as `tldr` and set `summary_source` to `fallback_abstract`.
- A failure for one paper must not stop the daily pipeline, data persistence, website build, or email delivery.

## Configuration

Introduce these settings:

- `executor.full_text_brief_paper_num`, default `20`: the number of top-ranked displayed papers receiving full-text processing. It applies after relevance ranking and after the existing `max_paper_num` display limit.
- `llm.full_text_chunk_tokens`, default `6000`: maximum input tokens per evidence-extraction chunk.
- `llm.full_text_chunk_overlap_tokens`, default `300`: overlap between adjacent chunks, preserving evidence that crosses a boundary.

The implementation must reject a non-positive paper count or chunk size, and an overlap equal to or larger than the chunk size.

## Testing

Tests must verify:

- ranking uses abstracts only and occurs before full-text enrichment;
- rank 20 uses the full-text tier and rank 21 uses the abstract tier;
- every full-text chunk participates in the evidence collection request sequence;
- TeX, HTML, and PDF extraction fall through in order;
- full-text and LLM failures downgrade only the affected paper;
- persisted JSON contains the new summary fields;
- static daily pages show a collapsed preview and working expand/collapse behavior;
- existing email rendering remains compatible with the new `tldr` value.
- search-index generation marks analyzed and unanalyzed papers correctly;
- search rendering offers `精读` only for unanalyzed papers and links `已审阅` papers to their analysis page;
- global daily navigation uses the latest available data date rather than the build date;
- a representative Markdown table renders to a `table` element and remains horizontally accessible on a narrow viewport.

## Scope Boundaries

This change does not alter relevance scoring, Zotero filtering, paper sources, or the substantive deep-analysis workflow. It changes only the deep-analysis prompt's table-format instruction and its website rendering. It does not expose generated evidence, source diagnostics, or an evidence-quality badge in the reader-facing interface.
