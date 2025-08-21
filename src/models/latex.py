from typing import List, Literal
from pydantic import BaseModel, field_validator, model_validator
import os
import re


def _balanced(s: str, open_ch: str = '{', close_ch: str = '}') -> bool:
    depth = 0
    for ch in s:
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


class SectionRef(BaseModel):
    order: int
    type: Literal["section", "subsection"]
    title: str
    path: str

    @field_validator("order")
    def order_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("order must be non-negative")
        return v

    @field_validator("title")
    def title_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("title must not be empty")
        return v

    @field_validator("path")
    def file_must_exist(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"section path does not exist: {v}")
        return v


class LatexMetadata(BaseModel):
    sections: List[SectionRef]

    @model_validator(mode="after")
    def validate_sections(self):
        if not self.sections:
            raise ValueError("sections must not be empty")
        orders = [s.order for s in self.sections]
        if len(orders) != len(set(orders)):
            raise ValueError("section orders must be unique")
        # enforce a section at order 0
        root = next((s for s in self.sections if s.order == 0), None)
        if root is None or root.type != "section":
            raise ValueError("order 0 must be a section")
        return self


class SectionContent(BaseModel):
    ref: SectionRef
    text: str

    @model_validator(mode="after")
    def validate_text(self):
        txt = self.text or ""
        # basic checks
        if "\\end{document}" in txt:
            raise ValueError("section content must not contain \\end{document}")
        if not _balanced(txt, '{', '}'):
            raise ValueError("unbalanced braces in section content")
        # command presence
        if self.ref.type == "section" and "\\section" not in txt:
            raise ValueError("section content must contain \\section")
        if self.ref.type == "subsection" and "\\subsection" not in txt:
            raise ValueError("subsection content must contain \\subsection")
        # simple $ pairing check
        if txt.count("$") % 2 != 0:
            raise ValueError("unpaired $ in math mode")

        # environment matching check: \begin{env} ... \end{env}
        env_stack: List[str] = []
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", txt):
            if kind == "begin":
                env_stack.append(env)
            else:
                if not env_stack or env_stack[-1] != env:
                    raise ValueError(f"mismatched environment: \\end{{{env}}} without matching \\begin")
                env_stack.pop()
        if env_stack:
            raise ValueError(f"unclosed environments: {env_stack}")

        # detect math-only macros used outside math mode (common cause of 'Missing $ inserted')
        math_cmds = [
            r"operatorname",
            r"mathbb",
            r"mathcal",
            r"frac",
            r"sum",
            r"int",
            r"lim",
            r"sqrt",
        ]
        # strip math regions: $, $$, \( \), \[ \]
        def _strip_math_regions(s: str) -> str:
            patterns = [
                (r"\\\(.*?\\\)", re.DOTALL),
                (r"\\\[.*?\\\]", re.DOTALL),
                (r"\$\$.*?\$\$", re.DOTALL),
                (r"\$.*?\$", re.DOTALL),
            ]
            out = s
            for pat, flags in patterns:
                out = re.sub(pat, " ", out, flags=flags)
            return out

        outside_math = _strip_math_regions(txt)
        pattern = re.compile(r"\\(" + r"|".join(math_cmds) + r")\b")
        if pattern.search(outside_math):
            raise ValueError("math macro used outside math mode; wrap with $...$ or \\( ... \\)")
        return self


