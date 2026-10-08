"""LLM client for the patch network. Speaks the OpenAI-compatible chat
completions protocol, which Azure AI Foundry, Hugging Face TGI / vLLM and OpenAI
all expose."""

import json
import logging
from importlib import resources
from typing import Dict, List

import httpx

from ..config import Settings
from .candidates import Candidate, TilePair
from .decision import Merge, PairDecision

log = logging.getLogger(__name__)


def load_prompt(version: str) -> str:
    return resources.files("drawing_ai.prompts").joinpath(f"{version}.txt").read_text(encoding="utf-8")


class LLMMergeClient:
    def __init__(self, s: Settings, client: httpx.Client = None):
        if not s.llm_base_url:
            raise ValueError("LLM_BASE_URL is not set")
        self.s = s
        self.system_prompt = load_prompt(s.prompt_version)
        headers = {"Content-Type": "application/json"}
        if s.llm_api_key:
            headers["Authorization"] = f"Bearer {s.llm_api_key}"
            headers["api-key"] = s.llm_api_key  # Azure-style header
        self.http = client or httpx.Client(timeout=s.llm_timeout_s, headers=headers)
        self.url = s.llm_base_url.rstrip("/") + "/chat/completions"
        self.params = {"api-version": s.llm_api_version} if s.llm_api_version else None

    def build_messages(self, pair: TilePair, a: List[Candidate], b: List[Candidate], hints: PairDecision) -> List[dict]:
        payload = {
            "seam": {"orientation": pair.orientation, "overlap_region": [round(v, 1) for v in pair.overlap]},
            "tile_a": {"tile_id": pair.a, "items": [c.to_prompt() for c in a]},
            "tile_b": {"tile_id": pair.b, "items": [c.to_prompt() for c in b]},
            "heuristic_suggestions": [m.ids for m in hints.merges],
        }
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def decide(self, pair: TilePair, a: List[Candidate], b: List[Candidate], hints: PairDecision) -> PairDecision:
        body = {
            "model": self.s.llm_model,
            "messages": self.build_messages(pair, a, b, hints),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        resp = self.http.post(self.url, params=self.params, json=body)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return parse_decision(content, {c.uid: c for c in a + b})


def parse_decision(content: str, known: Dict[str, Candidate]) -> PairDecision:
    """Validate the LLM output; anything referring to unknown ids or mixing
    kinds is dropped rather than trusted."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    data = json.loads(text)
    merges = []
    for m in data.get("merges", []) or []:
        ids = [i for i in dict.fromkeys(m.get("ids", [])) if i in known]
        if len(ids) < 2 or len({known[i].kind for i in ids}) != 1:
            continue
        label = m.get("label")
        merges.append(Merge(ids, str(label) if label else None))
    relabels = {}
    for r in data.get("relabels", []) or []:
        if r.get("id") in known and r.get("label"):
            relabels[r["id"]] = str(r["label"])
    return PairDecision(merges=merges, relabels=relabels, source="llm")
