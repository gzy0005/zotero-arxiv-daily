from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
RawPaperItem = TypeVar('RawPaperItem')

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    categories: Optional[list[str]] = None
    summary_source: Optional[str] = None
    summary_error: Optional[str] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        tldr = response.choices[0].message.content
        return tldr
    
    def _chat(self, openai_client: OpenAI, llm_params: dict, prompt: str, system: str) -> str:
        response = openai_client.chat.completions.create(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **llm_params.get("generation_kwargs", {}),
        )
        return response.choices[0].message.content

    def _full_text_chunks(self, llm_params: dict) -> list[str]:
        chunk_size = int(llm_params.get("full_text_chunk_tokens", 6000))
        overlap = int(llm_params.get("full_text_chunk_overlap_tokens", 300))
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Invalid full-text chunk configuration")
        tokens = tiktoken.encoding_for_model("gpt-4o").encode(self.full_text or "")
        step = chunk_size - overlap
        return [tiktoken.encoding_for_model("gpt-4o").decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), step)]

    def generate_tldr(self, openai_client:OpenAI, llm_params:dict, *, use_full_text: bool = True) -> str:
        try:
            lang = llm_params.get("language", "Chinese")
            if not self.abstract and not self.full_text:
                logger.warning(f"Neither full text nor abstract is provided for {self.url}")
                self.tldr = "Failed to generate TLDR. Neither full text nor abstract is provided"
                self.summary_source = "fallback_abstract"
                return self.tldr
            if not use_full_text or not self.full_text:
                prompt = f"Write one {lang} research brief of 150-250 Chinese characters. Cover the research problem, method or data, and main result. Use only the title and abstract; do not infer unstated details.\n\nTitle: {self.title}\n\nAbstract: {self.abstract}"
                tldr = self._chat(openai_client, llm_params, prompt, "You summarize scientific papers accurately.")
                self.summary_source = "abstract"
            else:
                facts = []
                for index, chunk in enumerate(self._full_text_chunks(llm_params), start=1):
                    prompt = f"Extract only source-grounded facts from part {index} of this paper. Record research objective, method, data or setup, results, conclusions, and limitations when present. Do not infer missing facts.\n\n{chunk}"
                    try:
                        facts.append(self._chat(openai_client, llm_params, prompt, "You extract factual evidence from scientific papers."))
                    except Exception as exc:
                        logger.warning(f"Failed to extract evidence from {self.url} part {index}: {exc}")
                if not facts:
                    raise ValueError("No full-text evidence extracted")
                prompt = f"Using only the evidence below, write one {lang} research brief of 150-250 Chinese characters. Cover the research problem, core method or data, and main result or contribution. Do not add unsupported claims.\n\n" + "\n\n".join(facts)
                tldr = self._chat(openai_client, llm_params, prompt, "You summarize scientific papers accurately.")
                self.summary_source = "full_text"
            self.tldr = tldr
            self.summary_error = None
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            self.summary_source = "fallback_abstract"
            self.summary_error = str(e)
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        text = self.full_text or self.abstract
        if text is None:
            logger.warning(f"No text available for affiliation extraction: {self.url}")
            return None

        if self.full_text is None:
            logger.info(f"Using abstract as fallback for affiliation extraction: {self.url}")

        prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{text}"
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:2000]
        prompt = enc.decode(prompt_tokens)
        affiliations = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        raw = affiliations.choices[0].message.content

        match = re.search(r'\[.*\]', raw, flags=re.DOTALL)
        if not match:
            logger.warning(f"Failed to parse affiliation list from LLM response: {raw[:200]}")
            return None
        affiliations = json.loads(match.group(0))
        affiliations = list(set(affiliations))
        affiliations = [str(a) for a in affiliations]

        return affiliations
    
    def to_dict(self) -> dict:
        import re
        arxiv_id = None
        if self.url:
            cleaned = re.sub(r'(\.pdf|v\d+)$', '', self.url)
            m = re.search(r'/(?:abs|pdf)/([\w./-]+)$', cleaned)
            if m:
                arxiv_id = m.group(1)
        return {
            "source": self.source,
            "arxiv_id": arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "tldr": self.tldr,
            "affiliations": self.affiliations,
            "score": round(self.score, 1) if self.score else None,
            "categories": self.categories or [],
            "summary_source": self.summary_source,
            "summary_error": self.summary_error,
        }

    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
