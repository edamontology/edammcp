import logging
import re
import threading
from collections.abc import Callable
from time import sleep

import requests

from edam_mcp.ontology.loader import OntologyLoader

from ..models.biotools import BiotoolsMatch

logger = logging.getLogger(__name__)
EDAM_LOADER = OntologyLoader()
_load_lock = threading.Lock()
_loaded = False

API_URL = "https://bio.tools/api/t/"
EDAM_URI = re.compile(r"https?://edamontology\.org/((topic|operation|data|format)_\d+)")
FILTER_KEYS = {"topic": "topicID", "operation": "operationID", "data": "dataTypeID", "format": "dataFormatID"}
CATEGORY_FIELDS = {
    "topic": ("topic_terms",),
    "operation": ("operation_terms",),
    "data": ("input_data_terms", "output_data_terms"),
    "format": ("input_format_terms", "output_format_terms"),
}
MAX_CALLS = 20  # bio.tools requests per query when terms are expanded


def _is_obsolete(uri: str) -> bool:
    """True if uri is obsolete or unknown to EDAM. Loads the ontology on first use, retrying after a failure."""
    global _loaded
    with _load_lock:  # tools run in worker threads; readers wait until the load is complete
        if not _loaded:
            if not EDAM_LOADER.load_ontology():
                raise RuntimeError("EDAM ontology could not be loaded")
            _loaded = True
    return (EDAM_LOADER.concepts.get(uri) or {}).get("deprecated", True)


def _terms(items) -> dict[str, dict]:
    """Map EDAM term URI to its label and obsolete flag, deduplicated."""
    return {x["uri"]: {"label": x.get("term"), "obsolete": _is_obsolete(x["uri"])} for x in items or []}


def _io_terms(functions: list[dict], key: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Data and format terms of the inputs or outputs (key) of all functions."""
    ios = [io for fn in functions for io in fn.get(key) or []]
    formats = [f for io in ios for f in io.get("format") or []]
    return _terms([io["data"] for io in ios if io.get("data")]), _terms(formats)


def _to_match(tool: dict) -> BiotoolsMatch:
    """Build a match from a bio.tools entry; term fields stay empty for unannotated tools."""
    functions = tool.get("function") or []
    terms = {
        "topic_terms": _terms(tool.get("topic")),
        "operation_terms": _terms([op for fn in functions for op in fn.get("operation") or []]),
    }
    terms["input_data_terms"], terms["input_format_terms"] = _io_terms(functions, "input")
    terms["output_data_terms"], terms["output_format_terms"] = _io_terms(functions, "output")

    return BiotoolsMatch(
        name=tool["name"],
        description=tool.get("description") or "",
        biotools_link=f"https://bio.tools/{tool['biotoolsID']}",
        **terms,
    )


def _search(
    params: dict,
    max_results: int,
    matches: list[BiotoolsMatch],
    seen: set[str],
    keep: Callable[[BiotoolsMatch], dict | None] | None = None,
    budget: list[int] | None = None,
) -> int:
    """Page through bio.tools results for params, appending new matches to matches.

    keep, if given, returns the matched_via of a tool, or None to drop it. budget, if given, is a one-item
    list of requests left; it is decremented per request and paging stops at zero.

    Returns the total number of tools bio.tools reports for the query.
    """
    page, total = 1, 0
    while budget is None or budget[0] > 0:
        if budget is not None:
            budget[0] -= 1
        r = requests.get(
            API_URL,
            params={**params, "format": "json", "per_page": 100, "page": page},
            timeout=(3.05, 30),
        )
        r.raise_for_status()
        data = r.json()
        total = data["count"]

        for tool in data["list"]:
            if tool["biotoolsID"] in seen:
                continue
            seen.add(tool["biotoolsID"])
            match = _to_match(tool)
            if keep is not None:
                via = keep(match)
                if via is None:
                    continue
                match.matched_via = via
            matches.append(match)

        if len(matches) >= max_results or not data.get("next"):
            break
        page += 1
        sleep(1)  # be polite to the bio.tools API between pages
    return total


def _search_by_id(biotools_id: str, matches: list[BiotoolsMatch]) -> int:
    r = requests.get(f"{API_URL}{biotools_id}", params={"format": "json"}, timeout=(3.05, 30))
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    matches.append(_to_match(r.json()))
    return 1


def query_by_text(
    seed: str,
    max_results: int = 10,
) -> tuple[list[BiotoolsMatch], int]:
    """Search for tools in bio.tools matching the keywords in seed, ranked by bio.tools relevance.

    Tools without EDAM annotations are returned with empty term fields.

    Args:
        seed: Keywords or tool name to search in bio.tools.
        max_results: Maximum number of matches to return.

    Returns:
        Matched bio.tools tools, and the total number of tools bio.tools reports for the query
    """
    matches: list[BiotoolsMatch] = []
    total = _search({"q": seed, "sort": "score"}, max_results, matches, set())
    return matches[:max_results], total


def query_by_text_exact_match(
    name: str | None = None, biotools_id: str | None = None
) -> tuple[list[BiotoolsMatch], int]:
    """Search for tools in bio.tools matching the exact name and/or the biotools_id identifier.

    Args:
        name: tool name to search in bio.tools.
        biotools_id: biotools tool identifier to search in bio.tools.

    Returns:
        Matched bio.tools tools, and the total number of tools bio.tools reports for the query
    """
    if name is None and biotools_id is None:
        raise ValueError("You must specify either the tool name or its database identifier")

    matches: list[BiotoolsMatch] = []
    if name is None:
        total = _search_by_id(re.sub(r"['\"]", "", biotools_id), matches)
    else:
        params = {"name": '"' + re.sub(r"['\"]", "", name) + '"'}
        if biotools_id is not None:
            params["biotoolsID"] = '"' + re.sub(r"['\"]", "", biotools_id) + '"'
        total = _search(params, 10, matches, set())
    return matches, total


def _split(uri: str) -> tuple[str, str]:
    """(term ID, category) of an EDAM topic/operation/data/format URI."""
    m = EDAM_URI.fullmatch(uri)
    if not m:
        raise ValueError(f"Not an EDAM topic/operation/data/format URI: {uri}")
    return m.group(1), m.group(2)


def _filter(uri: str) -> tuple[str, str]:
    """bio.tools filter key and quoted ID for an EDAM URI. Quoted for an exact match; unquoted, bio.tools does a loose text match."""
    term, category = _split(uri)
    return FILTER_KEYS[category], f'"{term}"'


def _expand(uri: str, descendant_depth: int, ancestor_depth: int) -> dict[str, tuple[str, int]]:
    """EDAM candidates for uri as URI to (direction, distance), nearest first and narrower before broader on ties."""
    related = EDAM_LOADER.get_related(uri, descendant_depth, ancestor_depth)
    ranked = sorted(related.items(), key=lambda kv: (kv[1][1], kv[1][0] == "ancestor"))
    return {u: v for u, v in ranked if EDAM_URI.fullmatch(u)}


def _via(match: BiotoolsMatch, expanded: dict[str, dict[str, tuple[str, int]]]) -> dict | None:
    """For each requested term, the closest candidate the tool is annotated with; None if any term has none."""
    via = {}
    for uri, candidates in expanded.items():
        category = _split(uri)[1]
        annotated = {t for field in CATEGORY_FIELDS[category] for t in getattr(match, field)}
        hit = min(
            (c for c in candidates if c in annotated),
            key=lambda c: (candidates[c][1], candidates[c][0] == "ancestor"),
            default=None,
        )
        if hit is None:
            return None
        direction, distance = candidates[hit]
        via[uri] = {"term": hit, "direction": direction, "distance": distance}
    return via


def _rank(match: BiotoolsMatch) -> tuple[int, int]:
    """Sort key: total distance of the matched terms, then how many of them are broader (ancestor) terms."""
    via = (match.matched_via or {}).values()
    return sum(v["distance"] for v in via), sum(v["direction"] == "ancestor" for v in via)


def query_by_edam_terms(
    edam_terms: list[str],
    max_results: int = 10,
    descendant_depth: int = 0,
    ancestor_depth: int = 0,
) -> tuple[list[BiotoolsMatch], int]:
    """Search for tools in bio.tools annotated with all of the given EDAM terms, optionally widened by hierarchy.

    bio.tools only combines filters of different categories, so at most one term per category is allowed.
    With depths, the term with the fewest candidates is queried once per candidate (bio.tools has no OR);
    the other terms and the depths are then checked on each tool's annotations. Queries are capped at MAX_CALLS.
    Candidates are queried nearest first, and the search stops once max_results tools are found, so with one
    widened term the result is the nearest matches; with two or more, a closer match may be missed.

    Args:
        edam_terms: EDAM concept URIs (topic, operation, data or format), at most one per category.
        max_results: Maximum number of matches to return.
        descendant_depth: Also match tools annotated with narrower terms up to this many steps down.
        ancestor_depth: Also match tools annotated with broader terms up to this many steps up.

    Returns:
        Matched bio.tools tools annotated with all of the terms (or their candidates), and the total number
        bio.tools reports. With depths, the total sums the queries and is an upper bound.

    Raises:
        ValueError: If a term is obsolete or unknown to EDAM, is not a topic, operation, data or format
            concept, or shares a category with another term.
        RuntimeError: If the EDAM ontology cannot be loaded.
    """
    bad = [u for u in edam_terms if _is_obsolete(u)]
    if bad:
        raise ValueError(f"Unknown or obsolete EDAM terms: {', '.join(bad)}")

    used: dict[str, str] = {}
    for uri in edam_terms:
        term, category = _split(uri)
        if used.setdefault(category, term) != term:
            raise ValueError(
                f'bio.tools can only combine one {category} term per query, got "{used[category]}" and "{term}"'
            )

    expand = descendant_depth > 0 or ancestor_depth > 0
    expanded = {u: _expand(u, descendant_depth, ancestor_depth) for u in edam_terms}
    multi = [u for u in edam_terms if len(expanded[u]) > 1]
    pivot = min(multi, key=lambda u: len(expanded[u]), default=None)
    fixed = dict(_filter(u) for u in edam_terms if u not in multi)
    queries = [fixed]
    if pivot is not None:
        pivot_key = _filter(pivot)[0]
        queries = [{**fixed, pivot_key: _filter(c)[1]} for c in expanded[pivot]]

    matches: list[BiotoolsMatch] = []
    seen: set[str] = set()
    total = 0
    budget = [MAX_CALLS] if expand else None
    keep = (lambda m: _via(m, expanded)) if expand else None
    for params in queries:
        if len(matches) >= max_results or (budget is not None and budget[0] <= 0):
            break
        total += _search(params, max_results, matches, seen, keep, budget)

    if budget is not None and budget[0] <= 0:
        logger.warning(
            f"EDAM term expansion reached the limit of {MAX_CALLS} bio.tools requests; results may be incomplete"
        )
    if expand:
        matches.sort(key=_rank)
    return matches[:max_results], total
