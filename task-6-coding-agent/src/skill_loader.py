"""Skill loader with progressive disclosure.

Per blueprint Part II §2.3:

* ``scan()`` reads YAML frontmatter of every ``SKILL.md`` (Level 1 — index).
* ``search(query, k=3)`` returns the top-k matching skill names (token
  overlap scorer — TF-IDF or embedding overkill for our 3 skills).
* ``load(name)`` reads the full markdown body (Level 2).
* ``system_prompt_section(char_budget=8000)`` renders the Level-1 list as
  a markdown bullet block for inclusion in the system prompt.

Aligns with Anthropic's ``packages/builtin-tools/.../SkillTool/prompt.ts``:
the index takes ~1 % of the ctx window and individual descriptions are
capped at ~1024 characters.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

log = logging.getLogger("skill_loader")

DEFAULT_CHAR_BUDGET = 8000  # ≈ 1 % of an 8 K model, or ~1.5 K tokens
MAX_LISTING_DESC_CHARS = 1024

_FRONTMATTER_END = "\n---\n"


class SkillLoader:
    """Catalogue of skills rooted at ``skills_dir``.

    Each skill lives in its own sub-directory containing a single
    ``SKILL.md`` with YAML frontmatter (``name``, ``description``,
    optionally ``when_to_use`` and ``allowed-tools``).
    """

    def __init__(self, skills_dir: str | Path) -> None:
        self.skills_dir = Path(skills_dir)
        self._meta: Dict[str, Dict[str, object]] = {}
        self._bodies: Dict[str, str] = {}
        if not self.skills_dir.exists():
            log.warning("skills_dir does not exist: %s", self.skills_dir)
            return
        self.scan()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_skills(self) -> List[Dict[str, str]]:
        """Level 1 — name + description only (cheap to enumerate)."""
        return [
            {"name": name, "description": str(meta["description"])}
            for name, meta in sorted(self._meta.items())
        ]

    def load(self, name: str) -> str:
        """Level 2 — return the markdown body with frontmatter stripped."""
        if name not in self._bodies:
            raise KeyError(f"skill not found: {name}")
        return self._bodies[name]

    def get_meta(self, name: str) -> Optional[Dict[str, object]]:
        return self._meta.get(name)

    # ------------------------------------------------------------------
    # Lazy resources — the agent can read scripts / references on
    # demand, mirroring Anthropic's Skills design (blueprint Part II §2.3).
    # ------------------------------------------------------------------

    def list_scripts(self, name: str) -> List[Path]:
        """List paths under ``<skills_dir>/<name>/scripts/`` (recursively).

        Returns an empty list if the skill is unknown or has no scripts/.
        Use ``run_bash`` (future tool) or your own subprocess to execute
        them.
        """
        meta = self._meta.get(name)
        if meta is None:
            return []
        skill_dir = meta["path"].parent  # type: ignore[arg-type]
        scripts_dir = skill_dir / "scripts"
        if not scripts_dir.is_dir():
            return []
        return sorted(scripts_dir.rglob("*"))

    def read_script(self, name: str, relative: str) -> Optional[str]:
        """Read a file under ``<skills_dir>/<name>/scripts/<relative>``.

        Returns ``None`` if the file doesn't exist, isn't inside the
        skill's ``scripts/`` tree, or exceeds the 256 KB cap.
        """
        meta = self._meta.get(name)
        if meta is None:
            return None
        skill_dir = meta["path"].parent  # type: ignore[arg-type]
        target = (skill_dir / "scripts" / relative).resolve()
        # Containment check — the path must stay inside the skill's
        # scripts/ directory.
        scripts_root = (skill_dir / "scripts").resolve()
        try:
            target.relative_to(scripts_root)
        except ValueError:
            return None
        if not target.is_file():
            return None
        if target.stat().st_size > 256 * 1024:
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def list_references(self, name: str) -> List[Path]:
        """List paths under ``<skills_dir>/<name>/references/``."""
        meta = self._meta.get(name)
        if meta is None:
            return []
        skill_dir = meta["path"].parent  # type: ignore[arg-type]
        ref_dir = skill_dir / "references"
        if not ref_dir.is_dir():
            return []
        return sorted(ref_dir.rglob("*"))

    def read_reference(self, name: str, relative: str) -> Optional[str]:
        """Read a file under ``<skills_dir>/<name>/references/<relative>``."""
        meta = self._meta.get(name)
        if meta is None:
            return None
        skill_dir = meta["path"].parent  # type: ignore[arg-type]
        target = (skill_dir / "references" / relative).resolve()
        ref_root = (skill_dir / "references").resolve()
        try:
            target.relative_to(ref_root)
        except ValueError:
            return None
        if not target.is_file():
            return None
        if target.stat().st_size > 256 * 1024:
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def search(self, query: str, k: int = 3) -> List[Dict[str, object]]:
        """Return top-k skill hits for ``query`` (token overlap scorer).

        Returns ``[{name, description, score}, ...]`` in descending score
        order. Skills with score 0 are skipped — a ``query`` that matches
        nothing returns an empty list (callers should fall back to the
        full catalogue).
        """
        query_lc = query.lower()
        query_tokens = _tokenise(query_lc)
        if not query_tokens:
            return []
        scored: List[Tuple[float, Dict[str, object]]] = []
        for name, meta in self._meta.items():
            haystack = (
                name.lower()
                + " "
                + str(meta["description"]).lower()
                + " "
                + str(meta.get("when_to_use") or "").lower()
            )
            hay_tokens = set(_tokenise(haystack))
            if not hay_tokens:
                continue
            overlap = sum(1 for t in query_tokens if t in hay_tokens)
            # Normalise by √|haystack| so longer descriptions don't dominate.
            score = overlap / (len(hay_tokens) ** 0.5)
            if score > 0:
                scored.append((score, {
                    "name": name,
                    "description": str(meta["description"]),
                    "score": round(score, 3),
                }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in scored[:k]]

    def system_prompt_section(self, char_budget: int = DEFAULT_CHAR_BUDGET) -> str:
        """Render the Level-1 list as a markdown block for the system prompt."""
        if not self._meta:
            return "(no skills available)"
        lines = ["Available skills (Level 1 — descriptions only):"]
        used = 0
        for name in sorted(self._meta):
            desc = str(self._meta[name]["description"]).strip()
            if len(desc) > MAX_LISTING_DESC_CHARS:
                desc = desc[: MAX_LISTING_DESC_CHARS - 1] + "…"
            entry = f"- `{name}` — {desc}"
            if used + len(entry) > char_budget:
                break
            lines.append(entry)
            used += len(entry)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """Re-scan the skills directory (idempotent)."""
        self._meta.clear()
        self._bodies.clear()
        if not self.skills_dir.exists():
            return
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                text = md.read_text(encoding="utf-8")
            except OSError as e:
                log.warning("skip unreadable skill %s: %s", md, e)
                continue
            meta, body = _parse_frontmatter(text)
            if "name" not in meta or "description" not in meta:
                log.warning(
                    "skill %s missing name/description in frontmatter; skipping",
                    md.parent.name,
                )
                continue
            name = str(meta["name"])
            self._meta[name] = {
                "description": str(meta["description"]).strip(),
                "when_to_use": str(meta.get("when_to_use") or "").strip(),
                "allowed_tools": [str(t) for t in (meta.get("allowed-tools") or [])],
                "path": md,
            }
            self._bodies[name] = body.strip() + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    return [tok for tok in text.replace("\n", " ").split() if tok]


def _parse_frontmatter(text: str) -> Tuple[Dict[str, object], str]:
    """Split ``---\\n...\\n---\\nbody`` into (meta_dict, body)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find(_FRONTMATTER_END, 4)
    if end == -1:
        return {}, text
    meta_raw = text[4:end]
    try:
        meta = yaml.safe_load(meta_raw) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    body = text[end + len(_FRONTMATTER_END):]
    return meta, body