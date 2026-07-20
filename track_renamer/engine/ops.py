"""Rule operation implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .defaults import (
    DEFAULT_CATEGORY_SOURCE,
    TITLE_CASE_ACRONYMS,
    map_instrument_to_category,
)
from .models import CategoryRule, Track
from .tokens import resolve_tokens

# Imported lazily in _apply_ml_category to avoid cycles if enrich imports ops later.
def _ml_should_apply(track: Track) -> str:
    from track_renamer.instrument_enrich import classify_decision

    action, _category = classify_decision(
        track.instrument,
        float(getattr(track, "instrument_score", 0.0) or 0.0),
        second_score=float(getattr(track, "instrument_second", 0.0) or 0.0),
    )
    return action

# Words ending in "s" that should not be singularized (bass -> bas).
_NO_SINGULARIZE = frozenset(
    {
        "bass",
        "brass",
        "glass",
        "class",
        "pass",
        "mass",
        "grass",
        "cross",
        "boss",
        "loss",
        "moss",
        "plus",
        "bus",
        "gas",
        "yes",
        "dos",
        "gis",
        "fx",
        "808",
        "909",
        "303",
    }
)

_FILENAME_CHUNK_RE = re.compile(r"[^a-zA-Z0-9]+")
_LETTER_OR_DIGIT_RE = re.compile(r"[A-Za-z]+|\d+")
_CAMEL_PART_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
_LEADING_NUMBER_PREFIX_RE = re.compile(r"^\d+\s*[-_.:)]\s*")
_LEADING_DASHES_RE = re.compile(r"^\s*(?:-\s*)+")
_TRAILING_NUMBER_RE = re.compile(r"[\s_-]+\d+\s*$")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMERIC_SUFFIX_RE = re.compile(r"(\D)(\d+)\s*$")


@dataclass(frozen=True, slots=True)
class CompiledKeyword:
    variants: frozenset[str]
    boundary_patterns: tuple[re.Pattern[str], ...]
    length: int


@dataclass(frozen=True, slots=True)
class CompiledCategory:
    name: str
    enabled: bool
    affix: str
    affix_position: str
    existing_affix_policy: str
    match_mode: str
    keywords: tuple[CompiledKeyword, ...]


@dataclass(frozen=True, slots=True)
class CompiledCategoryBundle:
    categories: tuple[CompiledCategory, ...]
    token_candidates: dict[str, tuple[tuple[int, int, int], ...]]
    boundary_candidates: tuple[tuple[re.Pattern[str], int, int, int], ...]
    contains_candidates: tuple[tuple[str, int, int, int], ...]


def _keyword_variants(keyword: str) -> frozenset[str]:
    """Singular/plural forms for category keyword matching."""
    kw = keyword.strip().lower()
    if not kw:
        return frozenset()
    variants: set[str] = {kw}
    if not any(ch.isalpha() for ch in kw):
        return frozenset(variants)

    if not kw.endswith("s"):
        if len(kw) > 1 and kw.endswith("y") and kw[-2] not in "aeiou":
            variants.add(kw[:-1] + "ies")
        elif re.search(r"(ch|sh|s|x|z)$", kw):
            variants.add(kw + "es")
        else:
            variants.add(kw + "s")
    else:
        if kw.endswith("ies") and len(kw) > 3:
            variants.add(kw[:-3] + "y")
        elif kw.endswith("es") and len(kw) > 4:
            root = kw[:-2]
            if re.search(r"(ch|sh|s|x|z)$", root):
                variants.add(root)
        if kw.endswith("s") and not kw.endswith("ss") and kw not in _NO_SINGULARIZE:
            variants.add(kw[:-1])

    return frozenset(variants)


def _filename_tokens(name: str) -> frozenset[str]:
    """Tokens from delimiters, CamelCase, and letter/digit boundaries (HiStrings → strings)."""
    tokens: set[str] = set()
    for chunk in _FILENAME_CHUNK_RE.split(name):
        if not chunk:
            continue
        for part in _LETTER_OR_DIGIT_RE.findall(chunk):
            tokens.add(part.lower())
        for part in _CAMEL_PART_RE.findall(chunk):
            if part:
                tokens.add(part.lower())
    return frozenset(tokens)


def _match_keyword(name: str, keyword: str, mode: str) -> bool:
    variants = _keyword_variants(keyword)
    if not variants:
        return False
    hay = name.lower()
    tokens = _filename_tokens(name)
    if mode == "wholeWord":
        for variant in variants:
            if variant in tokens:
                return True
            if re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", hay):
                return True
            parts = variant.split()
            if len(parts) > 1:
                pattern = r"(?<![a-z0-9])" + r"[\s_\-]+".join(re.escape(p) for p in parts) + r"(?![a-z0-9])"
                if re.search(pattern, hay):
                    return True
        return False
    return any(variant in hay for variant in variants)


def _keyword_match_length(keyword: str) -> int:
    return len(keyword.strip())


def compile_category_bundle(raw_categories: Any) -> CompiledCategoryBundle:
    """Compile category parsing and regex work once per preview generation."""
    if isinstance(raw_categories, CompiledCategoryBundle):
        return raw_categories

    compiled: list[CompiledCategory] = []
    token_candidates: dict[str, list[tuple[int, int, int]]] = {}
    boundary_candidates: list[tuple[re.Pattern[str], int, int, int]] = []
    contains_candidates: list[tuple[str, int, int, int]] = []
    keyword_order = 0
    for raw in raw_categories or ():
        cat = CategoryRule.from_dict(raw) if isinstance(raw, dict) else raw
        category_index = len(compiled)
        keywords: list[CompiledKeyword] = []
        for keyword in (k.strip() for k in cat.keywords.split(",")):
            if not keyword:
                continue
            variants = _keyword_variants(keyword)
            keyword_length = _keyword_match_length(keyword)
            patterns: list[re.Pattern[str]] = []
            if cat.match_mode == "wholeWord":
                for variant in variants:
                    parts = variant.split()
                    body = (
                        r"[\s_\-]+".join(re.escape(part) for part in parts)
                        if len(parts) > 1
                        else re.escape(variant)
                    )
                    patterns.append(
                        re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")
                    )
                    candidate = (keyword_length, keyword_order, category_index)
                    if variant.isalnum():
                        token_candidates.setdefault(variant, []).append(candidate)
                    else:
                        boundary_candidates.append(
                            (patterns[-1], keyword_length, keyword_order, category_index)
                        )
            else:
                for variant in variants:
                    contains_candidates.append(
                        (variant, keyword_length, keyword_order, category_index)
                    )
            keywords.append(
                CompiledKeyword(
                    variants=variants,
                    boundary_patterns=tuple(patterns),
                    length=keyword_length,
                )
            )
            keyword_order += 1
        compiled.append(
            CompiledCategory(
                name=(cat.name or "").strip(),
                enabled=cat.enabled,
                affix=cat.affix,
                affix_position=cat.affix_position,
                existing_affix_policy=cat.existing_affix_policy,
                match_mode=cat.match_mode,
                keywords=tuple(keywords),
            )
        )
    return CompiledCategoryBundle(
        categories=tuple(compiled),
        token_candidates={
            token: tuple(candidates)
            for token, candidates in token_candidates.items()
        },
        boundary_candidates=tuple(boundary_candidates),
        contains_candidates=tuple(contains_candidates),
    )


def _category_index_by_name(bundle: CompiledCategoryBundle, name: str) -> int:
    key = (name or "").strip().casefold()
    if not key:
        return -1
    for index, category in enumerate(bundle.categories):
        if category.enabled and category.name.casefold() == key:
            return index
    return -1


def _apply_category_affix(name: str, category: CompiledCategory) -> str:
    affix = category.affix
    if (
        category.existing_affix_policy == "skip"
        and affix
        and name.lower().startswith(affix.lower())
    ):
        return name
    if category.affix_position == "prefix":
        return f"{affix}{name}"
    return f"{name}{affix}"


def _find_keyword_category_index(name: str, bundle: CompiledCategoryBundle) -> int:
    haystack = name.lower()
    tokens = _filename_tokens(name)
    best_length = 0
    best_order = 1 << 30
    best_category_index = -1

    def consider(length: int, order: int, category_index: int) -> None:
        nonlocal best_length, best_order, best_category_index
        category = bundle.categories[category_index]
        if not category.enabled:
            return
        if length > best_length or (length == best_length and order < best_order):
            best_length = length
            best_order = order
            best_category_index = category_index

    for token in tokens:
        for candidate in bundle.token_candidates.get(token, ()):
            consider(*candidate)
    for pattern, length, order, category_index in bundle.boundary_candidates:
        if length >= best_length and pattern.search(haystack):
            consider(length, order, category_index)
    for variant, length, order, category_index in bundle.contains_candidates:
        if length >= best_length and variant in haystack:
            consider(length, order, category_index)
    return best_category_index


def _compiled_keyword_matches(
    keyword: CompiledKeyword,
    mode: str,
    haystack: str,
    tokens: frozenset[str],
) -> bool:
    if mode != "wholeWord":
        return any(variant in haystack for variant in keyword.variants)
    if any(variant in tokens for variant in keyword.variants):
        return True
    return any(pattern.search(haystack) for pattern in keyword.boundary_patterns)


def _apply_ml_category(
    name: str,
    bundle: CompiledCategoryBundle,
    track: Track,
) -> str:
    if _ml_should_apply(track) != "apply":
        return name
    mapped = (track.category or "").strip() or map_instrument_to_category(
        track.instrument
    )
    if not mapped:
        return name
    index = _category_index_by_name(bundle, mapped)
    if index < 0:
        return name
    return _apply_category_affix(name, bundle.categories[index])


def _apply_compiled_category(
    name: str,
    bundle: CompiledCategoryBundle,
    *,
    track: Track | None = None,
    source: str = DEFAULT_CATEGORY_SOURCE,
) -> str:
    mode = (source or DEFAULT_CATEGORY_SOURCE).strip().lower()
    if mode not in ("filename", "model", "combo"):
        mode = DEFAULT_CATEGORY_SOURCE

    if mode in ("filename", "combo"):
        keyword_index = _find_keyword_category_index(name, bundle)
        if keyword_index >= 0:
            return _apply_category_affix(name, bundle.categories[keyword_index])
        if mode == "filename":
            return name

    if mode in ("model", "combo") and track is not None:
        return _apply_ml_category(name, bundle, track)
    return name


def _resolve(text: str, ctx: dict[str, Any]) -> str:
    return resolve_tokens(
        text,
        track=ctx["track"],
        original_name=ctx["original_name"],
        current_name=ctx["current_name"],
        index=ctx["index"],
        counter=ctx.get("counter", ctx["index"]),
        variables=ctx.get("variables"),
    )


def _apply_category(name: str, categories: list[CategoryRule]) -> str:
    best_cat: CategoryRule | None = None
    best_kw_len = 0

    for cat in categories:
        if not cat.enabled:
            continue
        keywords = [k.strip() for k in cat.keywords.split(",") if k.strip()]
        for kw in keywords:
            if not _match_keyword(name, kw, cat.match_mode):
                continue
            kw_len = _keyword_match_length(kw)
            if kw_len > best_kw_len:
                best_kw_len = kw_len
                best_cat = cat

    if best_cat is None:
        return name

    affix = best_cat.affix
    if best_cat.existing_affix_policy == "skip" and affix and name.lower().startswith(affix.lower()):
        return name
    if best_cat.affix_position == "prefix":
        return f"{affix}{name}"
    return f"{name}{affix}"


def _title_case(name: str, acronyms: str) -> str:
    words = re.split(r"(\s+)", name)
    acronym_set = {a.strip().upper() for a in acronyms.split(",") if a.strip()}
    out: list[str] = []
    for part in words:
        if part.isspace():
            out.append(part)
            continue
        upper = part.upper()
        if upper in acronym_set:
            out.append(upper)
        else:
            out.append(part[:1].upper() + part[1:].lower() if part else part)
    return "".join(out)


def _remove_text(name: str, text: str, *, regex: bool, case_sensitive: bool) -> str:
    if not text:
        return name
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.sub(text, "", name, flags=flags)
    if case_sensitive:
        return name.replace(text, "")
    pattern = re.compile(re.escape(text), re.IGNORECASE)
    return pattern.sub("", name)


def _position_index(name: str, spec: dict[str, Any]) -> int | None:
    mode = spec.get("mode", "position")
    if mode == "position":
        pos = int(spec.get("position", 0))
        direction = spec.get("direction", "right")
        if direction == "left":
            return max(0, len(name) - pos)
        return min(pos, len(name))
    return None


def apply_op(name: str, op: str, params: dict[str, Any], ctx: dict[str, Any]) -> str:
    p = params or {}

    if op == "stripLeadingNumberPrefix" or op == "removeLeadingNumbers":
        return _LEADING_NUMBER_PREFIX_RE.sub("", name)

    if op == "stripLeadingDashes":
        return _LEADING_DASHES_RE.sub("", name)

    if op == "stripTrailingNumber":
        return _TRAILING_NUMBER_RE.sub("", name)

    if op == "collapseWhitespace":
        return _WHITESPACE_RE.sub(" ", name)

    if op == "trim":
        return name.strip()

    if op == "titleCase":
        return _title_case(name, p.get("acronyms", TITLE_CASE_ACRONYMS))

    if op == "addTextAtBeginning":
        text = _resolve(p.get("text", ""), ctx)
        return f"{text}{name}" if text else name

    if op == "addTextAtEnd":
        text = _resolve(p.get("text", ""), ctx)
        return f"{name}{text}" if text else name

    if op == "replaceText":
        find = _resolve(p.get("find", ""), ctx)
        repl = _resolve(p.get("replace", ""), ctx)
        if not find:
            return name
        if p.get("regex"):
            flags = 0 if p.get("caseSensitive") else re.IGNORECASE
            return re.sub(find, repl, name, flags=flags)
        if p.get("caseSensitive"):
            return name.replace(find, repl)
        return re.sub(re.escape(find), repl, name, flags=re.IGNORECASE)

    if op == "removeText":
        text = p.get("text", "")
        return _remove_text(name, text, regex=bool(p.get("regex")), case_sensitive=bool(p.get("caseSensitive")))

    if op == "removeTextFromBeginning":
        text = p.get("text", "")
        if not text:
            return name
        if name.lower().startswith(text.lower()):
            return name[len(text):]
        return name

    if op == "removeTextFromEnd":
        text = p.get("text", "")
        if not text:
            return name
        if name.lower().endswith(text.lower()):
            return name[: -len(text)]
        return name

    if op == "removeCharRange":
        start = _position_index(name, p.get("from", {})) or 0
        end = _position_index(name, p.get("to", {})) or start
        if start > end:
            start, end = end, start
        return name[:start] + name[end:]

    if op in ("categoryBundle", "renameGroupsByCategory"):
        raw = p.get("categories", [])
        return _apply_compiled_category(
            name,
            compile_category_bundle(raw),
            track=ctx.get("track"),
            source=str(p.get("source", DEFAULT_CATEGORY_SOURCE)),
        )

    if op == "padNumericSuffix":
        def repl(m: re.Match[str]) -> str:
            return f"{m.group(1)}{int(m.group(2)):02d}"
        return _NUMERIC_SUFFIX_RE.sub(repl, name)

    return name
