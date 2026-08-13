"""Skill loader with progressive disclosure.

Anchors the design from Claude Code ``src/skills/loadSkillsDir.ts``:

* ``list_skills()`` reads **only** the YAML frontmatter of each
  ``SKILL.md`` (Level 1). Cost: ~20-50 tokens per skill.
* ``load(name)`` reads the markdown body on demand (Level 2). Cost:
  up to a few thousand tokens.
* Skills may declare ``allowed-tools`` so the orchestrator can constrain
  which tools the agent invokes *while* following the skill's workflow.

The router is intentionally trivial — keyword matching on
``description`` is enough for the toy-repo scenario. A production version
would use TF-IDF (Claude Code's ``localSearch.ts``) or an LLM-as-judge.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

log = logging.getLogger("skill_loader")

_FRONTMATTER_END = "\n---\n"


class SkillLoader:
    """Catalogue of skills rooted at ``skills_dir`` (each is a sub-folder
    containing ``SKILL.md``).
    """

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        # name → {description, when_to_use, allowed_tools, path}
        self._meta: Dict[str, Dict[str, object]] = {}
        if not self.skills_dir.exists():
            log.warning("skills_dir does not exist: %s", self.skills_dir)
            return
        self._scan()

    # -- public API --------------------------------------------------------

    def list_skills(self) -> List[Dict[str, str]]:
        """Level 1 — cheap enumeration for the system prompt."""
        out: List[Dict[str, str]] = []
        for name, meta in sorted(self._meta.items()):
            out.append(
                {
                    "name": name,
                    "description": str(meta["description"]),
                }
            )
        return out

    def load(self, name: str) -> str:
        """Level 2 — return the full markdown body (frontmatter stripped)."""
        meta = self._meta.get(name)
        if meta is None:
            raise KeyError(f"skill not found: {name}")
        path = meta["path"]
        text = path.read_text(encoding="utf-8")  # type: ignore[arg-type]
        _, body = self._parse_frontmatter(text)
        return body

    def get_meta(self, name: str) -> Optional[Dict[str, object]]:
        """Return the rich metadata (allowed_tools, when_to_use, ...)."""
        return self._meta.get(name)

    def find_relevant(self, issue: str, k: int = 1) -> List[str]:
        """Naive keyword router — return up to ``k`` matching skill names."""
        issue_lc = issue.lower()
        scored: List[tuple[int, str]] = []
        for name, meta in self._meta.items():
            haystack = (
                name.lower()
                + " "
                + str(meta["description"]).lower()
                + " "
                + str(meta.get("when_to_use") or "").lower()
            )
            score = sum(1 for word in issue_lc.split() if word in haystack)
            if score > 0:
                scored.append((score, name))
        scored.sort(reverse=True)
        return [name for _, name in scored[:k]]

    # -- internals ---------------------------------------------------------

    def _scan(self) -> None:
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except OSError as e:
                log.warning("skip unreadable skill %s: %s", md, e)
                continue
            meta, _body = self._parse_frontmatter(text)
            name = meta.get("name") or md.parent.name
            if "name" not in meta or "description" not in meta:
                log.warning(
                    "skill %s missing name/description; skipping", md.parent.name
                )
                continue
            self._meta[name] = {
                "description": str(meta["description"]).strip(),
                "when_to_use": str(meta.get("when_to_use") or "").strip(),
                "allowed_tools": [
                    str(t) for t in (meta.get("allowed-tools") or [])
                ],
                "path": md,
            }

    @staticmethod
    def _parse_frontmatter(text: str) -> tuple[Dict[str, object], str]:
        """Split ``---\\n...\\n---\\nbody`` into (dict, body)."""
        if not text.startswith("---\n"):
            return {}, text
        end = text.find(_FRONTMATTER_END, 4)
        if end == -1:
            return {}, text
        meta = yaml.safe_load(text[4:end]) or {}
        body = text[end + len(_FRONTMATTER_END):]
        if not isinstance(meta, dict):
            return {}, text
        return meta, body


# ---------------------------------------------------------------------------
# Helper used by the agent's system prompt builder
# ---------------------------------------------------------------------------

def format_skill_list(skills: List[Dict[str, str]]) -> str:
    """Render the Level-1 list as a markdown bullet block."""
    if not skills:
        return "(no skills available)"
    lines = ["Available skills (Level 1 — descriptions only):"]
    for s in skills:
        lines.append(f"- **{s['name']}** — {s['description']}")
    return "\n".join(lines)
