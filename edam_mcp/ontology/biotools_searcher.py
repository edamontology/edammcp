import re
from time import sleep

import requests

from ..models.biotools import BiotoolsMatch
from edam_mcp.ontology.loader import OntologyLoader

EDAM_LOADER = OntologyLoader()
EDAM_LOADER.load_ontology()

API_URL = "https://bio.tools/api/t/"
EDAM_URI = re.compile(r"https?://edamontology\.org/((topic|operation|data|format)_\d+)")
FILTER_KEYS = {"topic": "topicID", "operation": "operationID", "data": "dataTypeID", "format": "dataFormatID"}

def _is_obsolete(uri: str) -> bool:
    return (EDAM_LOADER.concepts.get(uri) or {}).get("obsolete", True)

def _terms(items) -> dict[str, str]:
    """Map EDAM term URI to label, deduplicated."""
    return {x["uri"]: { "label": x.get("term"), "obsolete": _is_obsolete(x.get("uri")) } or "" for x in items or []}

def _io_terms(functions: list[dict], key: str) -> tuple[dict[str, str], dict[str, str]]:
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


def _search(params: dict, max_results: int, matches: list[BiotoolsMatch], seen: set[str]) -> int:
    """Page through bio.tools results for params, appending new matches to matches.

    Returns the total number of tools bio.tools reports for the query.
    """
    page = 1
    while True:
        r = requests.get(
            API_URL,
            params={**params, "format": "json", "per_page": 100, "page": page},
            timeout=(3.05, 30),
        )
        r.raise_for_status()
        data = r.json()

        for tool in data["list"]:
            if tool["biotoolsID"] in seen:
                continue
            seen.add(tool["biotoolsID"])
            matches.append(_to_match(tool))

        if len(matches) >= max_results or not data.get("next"):
            return data["count"]
        page += 1
        sleep(1)  # be polite to the bio.tools API between pages

def _search_by_id(biotools_id: str, matches: list[BiotoolsMatch]) -> int:
    r = requests.get(
            f"{API_URL}/{biotools_id}",
            params={ "format": "json" },
            timeout=(3.05, 30),
        )
    r.raise_for_status()
    tool = r.json()
    if( "biotoolsID" in tool ):
        matches.append(_to_match(tool))
        return 1
    return 0

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

def _sanitize_input(text):
    return re.sub(r'[\'"]', '', text)

def query_by_text_exact_match(
    name: str = None,
    biotools_id: str = None
) -> tuple[list[BiotoolsMatch], int]:
    """Search for tools in bio.tools matching the name or the biotools_id identifier in the repository.

    Args:
        name: tool name to search in bio.tools.
        biotools_id: biotools tool identifier to search in bio.tools.

    Returns:
        Matched bio.tools tools, and the total number of tools bio.tools reports for the query
    """
    max_results = 10

    params = {}
    if( name is not None ):
        name = _sanitize_input(name)
        params["name"] = f'"{name}"'
    if( biotools_id is not None ):
        biotools_id = _sanitize_input(biotools_id)
        params["biotoolsID"] = f'"{biotools_id}"'
    
    if( len(params) == 0 ):
        raise ValueError("You must specify either the tool name or its database identifier")

    matches: list[BiotoolsMatch] = []
    if("biotoolsID" in params and (len(params) == 1) ):
        total = _search_by_id(biotools_id, matches)
    else:
        total = _search( params, max_results, matches, set())

    return matches[:max_results], total


def _filter_valid_edam_terms(user_uris):
    bad = [u for u in user_uris if _is_obsolete(u)]
    if bad:
        raise ValueError(f"Unknown or obsolete EDAM terms: { ', '.join(bad) }")

    expanded = set(user_uris)
    # Children expansion
    for u in user_uris:
        expanded.update(c for c in EDAM_LOADER.concepts[u]["children"] if not _is_obsolete(c))
    return expanded

def query_by_edam_terms(
    edam_terms: list[str],
    max_results: int = 10,
) -> tuple[list[BiotoolsMatch], int]:
    """Search for tools in bio.tools annotated with all of the given EDAM terms.

    bio.tools only combines filters of different categories, so at most one term per category is allowed.

    Args:
        edam_terms: EDAM concept URIs (topic, operation, data or format), at most one per category.
        max_results: Maximum number of matches to return.

    Returns:
        Matched bio.tools tools annotated with all of the terms, and the total number bio.tools reports

    Raises:
        ValueError: If a URI is not an EDAM topic, operation, data or format concept,
            or two terms share a category.
    """
    edam_terms = _filter_valid_edam_terms(edam_terms)
    if( len(edam_terms) == 0 ):
        raise ValueError("The input URIs do not correspond to valid terms in EDAM ontology")

    # Version combining each term of a distinct category to refine results, but bio.tools api does not accept multipe item of each
    """
    params: dict[str, str] = {}
    for uri in edam_terms:
        m = EDAM_URI.fullmatch(uri)
        if not m:
            raise ValueError(f"Not an EDAM topic/operation/data/format URI: {uri}")
        term, category = m.groups()
        key = FILTER_KEYS[category]
        # Quoted for an exact match; unquoted, bio.tools does a loose text match on the ID
        if params.setdefault(key, f'"{term}"') != f'"{term}"':
            raise ValueError(
                f'bio.tools can only combine one {category} term per query, got {params[key]} and "{term}"'
            )
    """

    # Version overcoming limitation in bio.tools API and amplifying coverage
    seen: set[str] = set()
    matches: list[BiotoolsMatch] = []
    for uri in edam_terms:
        m = EDAM_URI.fullmatch(uri)
        if not m:
            raise ValueError(f"Not an EDAM topic/operation/data/format URI: {uri}")
        term, category = m.groups()
        key = FILTER_KEYS[category]
        params: dict[str, str] = {key: f'"{term}"' } 
        total = _search(params, max_results, matches, seen)

    return matches[:max_results], total
