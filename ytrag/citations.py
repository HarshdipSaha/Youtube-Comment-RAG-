"""Verify that an answer's claims trace back to the evidence it was given.

RAG rarely fails by inventing whole sentences. It fails by producing a fluent,
plausibly-cited paragraph containing a number that was never in the context --
"73% of viewers disliked the pacing" when the context said 31 of 120. Those are
exactly the claims a reader trusts and does not check.

Two mechanical checks run on every generated answer:

1. **Citations resolve.** Every ``[c12]`` must name a comment that was actually
   retrieved. An invented citation is worse than no citation, because it
   manufactures the appearance of grounding, so invalid ones are stripped from
   the text rather than merely reported.
2. **Quantities were computed.** Percentages, like counts and comment counts are
   checked against the set of figures the pipeline actually derived. Numbers are
   only checked when they appear *as quantitative claims* -- "chapter 1089" and
   "season 2" are not statistics and are left alone.

Both are conservative by design. The guard annotates and warns; it does not
silently rewrite the substance of an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[\s*(c\d+)\s*\]", re.IGNORECASE)

#: A bare number counts as a claim only in these shapes.
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_COUNTED_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s+(likes?|upvotes?|comments?|commenters?|people|users|viewers|replies)\b",
    re.IGNORECASE,
)

#: Percentages may be rounded; a claim within this many points of a computed
#: figure is treated as an honest restatement rather than an invention.
_PERCENT_TOLERANCE = 1.0
_COUNT_TOLERANCE = 0.51


@dataclass(slots=True)
class GuardReport:
    """What the guard found, and the text it is willing to stand behind."""

    text: str
    citations: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.invalid and not self.unsupported


class CitationGuard:
    """Checks an answer against the ids and figures the pipeline supplied."""

    def __init__(self, allowed_ids: set[str], allowed_numbers: set[float] | None = None):
        self.allowed_ids = {str(i).lower() for i in allowed_ids}
        self.allowed_numbers = {float(n) for n in (allowed_numbers or set())}

    def extract(self, text: str) -> list[str]:
        """Citation ids in order of first appearance, deduplicated."""
        seen: dict[str, None] = {}
        for match in _CITATION_RE.finditer(text or ""):
            seen.setdefault(match.group(1).lower(), None)
        return list(seen)

    def _supported(self, value: float, tolerance: float) -> bool:
        return any(abs(value - allowed) <= tolerance for allowed in self.allowed_numbers)

    def _check_numbers(self, text: str) -> list[str]:
        """Find quantitative claims with no corresponding computed figure.

        Skipped entirely when the pipeline supplied no figures: with nothing to
        check against, every number would be reported as unsupported, which is
        noise rather than a finding.
        """
        if not self.allowed_numbers:
            return []

        unsupported: list[str] = []
        for match in _PERCENT_RE.finditer(text):
            value = float(match.group(1))
            if not self._supported(value, _PERCENT_TOLERANCE):
                unsupported.append(match.group(0))

        for match in _COUNTED_RE.finditer(text):
            value = float(match.group(1).replace(",", ""))
            if not self._supported(value, _COUNT_TOLERANCE):
                unsupported.append(match.group(0))
        return unsupported

    def verify(self, text: str) -> GuardReport:
        """Check ``text`` and return a report with invalid citations removed."""
        text = text or ""
        cited = self.extract(text)
        valid = [c for c in cited if c in self.allowed_ids]
        invalid = [c for c in cited if c not in self.allowed_ids]

        cleaned = text
        for bad in invalid:
            cleaned = re.sub(rf"\s*\[\s*{re.escape(bad)}\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()

        unsupported = self._check_numbers(cleaned)

        warnings: list[str] = []
        if invalid:
            warnings.append(
                f"Removed {len(invalid)} citation(s) to comments that were not "
                f"retrieved: {', '.join(invalid)}."
            )
        if unsupported:
            warnings.append(
                "These figures do not match anything the pipeline computed and may "
                f"be invented: {', '.join(unsupported)}."
            )
        if not valid and not invalid:
            warnings.append(
                "The answer carries no citation, so it cannot be traced to any comment."
            )

        return GuardReport(
            text=cleaned,
            citations=valid,
            invalid=invalid,
            unsupported=unsupported,
            warnings=warnings,
        )
