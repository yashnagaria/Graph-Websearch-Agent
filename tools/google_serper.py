"""Web search tool for the research graph.

Two providers are supported:

* ``serper``     - google.serper.dev, needs ``SERPER_API_KEY``. Best quality.
* ``duckduckgo`` - no API key at all. Used automatically when no Serper key is
                   present, so the graph still runs with nothing but a Gemini key.

The node entry point is :func:`get_google_serper`; the name is kept for backwards
compatibility with ``agent_graph/graph.py`` and the original CLI.
"""

import os
import json
import requests
from bs4 import BeautifulSoup

from utils.helper_functions import load_config
from states.state import AgentGraphState

config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

SEARCH_RESULT_LIMIT = 10
REQUEST_TIMEOUT = 20

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def format_results(organic_results):
    """Render a list of ``{title, link, snippet}`` dicts as the SERP text the selector reads."""
    result_strings = []
    for result in organic_results:
        title = result.get('title', 'No Title')
        link = result.get('link', '#')
        snippet = result.get('snippet', 'No snippet available.')
        result_strings.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n---")

    return '\n'.join(result_strings)


def resolve_provider():
    """Pick a provider from SEARCH_PROVIDER, falling back to whichever key is available."""
    requested = (os.environ.get("SEARCH_PROVIDER") or "auto").strip().lower()
    has_serper = bool(os.environ.get("SERPER_API_KEY"))

    if requested == "serper":
        return "serper" if has_serper else "duckduckgo"
    if requested in ("duckduckgo", "ddg"):
        return "duckduckgo"
    return "serper" if has_serper else "duckduckgo"


def search_serper(query):
    """Query google.serper.dev. Raises on any non-2xx response."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            'Content-Type': 'application/json',
            'X-API-KEY': os.environ['SERPER_API_KEY'],
        },
        data=json.dumps({"q": query}),
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    results = response.json()

    organic = results.get('organic') or []
    return [
        {
            "title": item.get("title", "No Title"),
            "link": item.get("link", "#"),
            "snippet": item.get("snippet", "No snippet available."),
        }
        for item in organic[:SEARCH_RESULT_LIMIT]
    ]


def _search_duckduckgo_package(query):
    """Use the ``ddgs`` package if it is installed. Returns None when unavailable."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older name of the same package
        except ImportError:
            return None

    with DDGS() as ddgs:
        hits = list(ddgs.text(query, max_results=SEARCH_RESULT_LIMIT))

    return [
        {
            "title": hit.get("title", "No Title"),
            "link": hit.get("href") or hit.get("url") or "#",
            "snippet": hit.get("body", "No snippet available."),
        }
        for hit in hits
    ]


def _search_duckduckgo_html(query):
    """Last-resort fallback: parse DuckDuckGo's no-JavaScript HTML endpoint."""
    response = requests.post(
        "https://html.duckduckgo.com/html/",
        headers=_BROWSER_HEADERS,
        data={"q": query},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    for node in soup.select("div.result")[:SEARCH_RESULT_LIMIT]:
        link_node = node.select_one("a.result__a")
        if not link_node:
            continue
        snippet_node = node.select_one(".result__snippet")
        results.append({
            "title": link_node.get_text(strip=True) or "No Title",
            "link": link_node.get("href") or "#",
            "snippet": snippet_node.get_text(" ", strip=True) if snippet_node else "No snippet available.",
        })

    return results


def search_duckduckgo(query):
    """DuckDuckGo search with no API key. Tries the package first, then raw HTML."""
    results = _search_duckduckgo_package(query)
    if results:
        return results
    return _search_duckduckgo_html(query)


def run_search(query):
    """Run a search with the resolved provider, falling back to DuckDuckGo if Serper fails."""
    provider = resolve_provider()

    if provider == "serper":
        try:
            results = search_serper(query)
            if results:
                return results, "serper"
        except Exception as serper_error:  # quota, bad key, network - keep going
            print(f"Serper search failed ({serper_error}); falling back to DuckDuckGo.")

    return search_duckduckgo(query), "duckduckgo"


def get_google_serper(state: AgentGraphState, plan):
    """Graph node: turn the planner's search term into a formatted SERP."""
    load_config(config_path)

    plan_data = plan().content
    plan_data = json.loads(plan_data)
    search = plan_data.get("search_term")

    if not search:
        return {**state, "serper_response": "No search term was provided by the planner."}

    try:
        results, provider = run_search(search)
    except Exception as search_error:
        return {**state, "serper_response": f"Search error occurred: {search_error}"}

    if not results:
        return {**state, "serper_response": "No organic results found."}

    header = f"Search provider: {provider} | Query: {search}\n---"
    return {**state, "serper_response": f"{header}\n{format_results(results)}"}
