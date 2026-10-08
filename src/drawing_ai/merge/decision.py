from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Merge:
    ids: List[str]  # candidate uids that are the same physical entity
    label: Optional[str] = None  # optional corrected class / text for the merged entity


@dataclass
class PairDecision:
    merges: List[Merge] = field(default_factory=list)
    relabels: Dict[str, str] = field(default_factory=dict)  # uid -> corrected class / text
    source: str = "heuristic"  # heuristic | llm | heuristic_fallback
