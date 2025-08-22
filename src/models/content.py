from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SectionDraft:
    title: str
    topics: List[str] = field(default_factory=list)
    summary_latex: Optional[str] = None


