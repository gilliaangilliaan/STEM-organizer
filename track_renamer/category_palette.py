"""Category color palette — reds through grays to black (24 swatches)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from track_renamer.engine.models import Rule

# Ordered: red → warm → cool → green → yellow → orange → brown → gray → black
CATEGORY_PALETTE_COLORS: tuple[str, ...] = (
    "#EF4444",
    "#F59E0B",
    "#A855F7",
    "#10B981",
    "#8D95A4",
    "#C1090B",
    "#C41D63",
    "#8351A1",
    "#5C4EA0",
    "#485FAB",
    "#398BCB",
    "#00B8D3",
    "#25BAA2",
    "#3FB655",
    "#76C043",
    "#B0D236",
    "#FED600",
    "#A44F0D",
    "#F36E21",
    "#A45C7A",
    "#5E4138",
    "#626262",
    "#455A64",
    "#000000",
)

CATEGORY_BADGE_TEXT = "#ffffff"

CATEGORY_BADGE_LABELS: dict[str, str] = {
    "Percussion": "PERC",
    "Orchestral": "ORCHEST",
}

# Legacy / alternate category names → canonical key in DEFAULT_CATEGORY_COLORS
CATEGORY_NAME_ALIASES: dict[str, str] = {
    "Synths": "Synth",
}

# Default color per category name (includes legacy names used in older presets)
DEFAULT_CATEGORY_COLORS: dict[str, str] = {
    "Bass": "#EF4444",
    "Drums": "#F59E0B",
    "Percussion": "#F36E21",
    "Synth": "#10B981",
    "Synths": "#10B981",
    "Pads": "#8351A1",
    "Wind": "#00B8D3",
    "Keys": "#485FAB",
    "Guitar": "#C1090B",
    "FX": "#000000",
    "Strings": "#76C043",
    "Vocals": "#A855F7",
    "Mallet": "#C41D63",
    "Orchestral": "#626262",
}

_PALETTE_LOWER = {color.lower(): color for color in CATEGORY_PALETTE_COLORS}
_PALETTE_LOWER["#fbaa19"] = "#A44F0D"
_PALETTE_LOWER["#d41f26"] = "#C1090B"
_PALETTE_LOWER["#636b7a"] = "#8D95A4"
_PALETTE_LOWER["#dd3226"] = "#A45C7A"


def category_badge_label(name: str) -> str:
    return CATEGORY_BADGE_LABELS.get(name, name.upper())


def _prefix_token_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in DEFAULT_CATEGORY_COLORS:
        lookup[name.upper()] = name
        lookup[category_badge_label(name).upper()] = name
    for alias, canonical in CATEGORY_NAME_ALIASES.items():
        lookup[alias.upper()] = canonical
    return lookup


_PREFIX_TOKEN_TO_CATEGORY = _prefix_token_lookup()


def parse_category_prefix_display(display: str) -> tuple[str, str] | None:
    """Split 'SYNTH - file.wav' into (canonical category name, remainder filename)."""
    if " - " not in display:
        return None
    head, tail = display.split(" - ", 1)
    token = head.strip().upper()
    category = _PREFIX_TOKEN_TO_CATEGORY.get(token)
    if not category:
        return None
    return (category, tail)


def resolve_category_name(name: str) -> str:
    stripped = (name or "").strip()
    return CATEGORY_NAME_ALIASES.get(stripped, stripped)


def default_category_color(name: str) -> str:
    stripped = (name or "").strip()
    if stripped in DEFAULT_CATEGORY_COLORS:
        return DEFAULT_CATEGORY_COLORS[stripped]
    canonical = resolve_category_name(stripped)
    return DEFAULT_CATEGORY_COLORS.get(canonical, CATEGORY_PALETTE_COLORS[0])


def category_color(name: str, stored: str = "", *, override: bool = False) -> str:
    """Return the display color for a category — defaults are tied to the category name."""
    default = default_category_color(name)
    if override and stored:
        normalized = _PALETTE_LOWER.get(stored.lower())
        if normalized:
            return normalized
    return default


def normalize_category_dict(cat: dict) -> None:
    name = (cat.get("name") or "").strip()
    if cat.get("colorOverride"):
        cat["color"] = category_color(name, cat.get("color", ""), override=True)
    else:
        cat["color"] = default_category_color(name)


def normalize_rules_category_colors(rules: list[Rule]) -> None:
    from track_renamer.engine.models import ConditionGroup, OpRule

    def walk(rule: Rule) -> None:
        if isinstance(rule, OpRule) and rule.op == "categoryBundle":
            for cat in rule.params.get("categories", []):
                if isinstance(cat, dict):
                    normalize_category_dict(cat)
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)

    for rule in rules:
        walk(rule)


def applied_category_colors(rules: list[Rule]) -> dict[str, str]:
    """Resolve the effective badge palette from the currently applied rules."""
    from track_renamer.engine.models import CategoryRule, ConditionGroup, OpRule

    colors: dict[str, str] = {}

    def add_category(cat: CategoryRule) -> None:
        name = resolve_category_name(cat.name)
        colors[name] = category_color(
            name,
            cat.color,
            override=cat.color_override,
        )

    def walk(rule: Rule) -> None:
        if isinstance(rule, CategoryRule):
            add_category(rule)
        elif isinstance(rule, OpRule) and rule.op == "categoryBundle":
            for raw in rule.params.get("categories", []):
                add_category(
                    CategoryRule.from_dict(raw) if isinstance(raw, dict) else raw
                )
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)
            for branch in rule.branches:
                walk(branch)

    for rule in rules:
        walk(rule)
    return colors


def sort_rule_category_keywords(rules: list[Rule]) -> bool:
    """Alphabetize and de-duplicate every category keyword list in-place."""
    from track_renamer.engine.models import CategoryRule, ConditionGroup, OpRule

    changed = False

    def sorted_keywords(value: str) -> str:
        unique: dict[str, str] = {}
        for raw in value.split(","):
            keyword = raw.strip()
            if keyword:
                unique.setdefault(keyword.casefold(), keyword)
        return ", ".join(sorted(unique.values(), key=str.casefold))

    def update_category(cat: CategoryRule) -> None:
        nonlocal changed
        normalized = sorted_keywords(cat.keywords)
        if normalized != cat.keywords:
            cat.keywords = normalized
            changed = True

    def walk(rule: Rule) -> None:
        nonlocal changed
        if isinstance(rule, CategoryRule):
            update_category(rule)
        elif isinstance(rule, OpRule) and rule.op == "categoryBundle":
            categories = rule.params.get("categories", [])
            for index, raw in enumerate(categories):
                if isinstance(raw, dict):
                    normalized = sorted_keywords(raw.get("keywords", ""))
                    if normalized != raw.get("keywords", ""):
                        raw["keywords"] = normalized
                        changed = True
                else:
                    update_category(raw)
        elif isinstance(rule, ConditionGroup):
            for child in rule.children:
                walk(child)
            for branch in rule.branches:
                walk(branch)

    for rule in rules:
        walk(rule)
    return changed
