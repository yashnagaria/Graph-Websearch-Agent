"""Streamlit front-end for the Graph Websearch Agent.

Runs the LangGraph research workflow (planner -> search -> selector -> scraper ->
reporter -> reviewer -> router) and streams every agent hand-off into the page as
it happens, so you can watch the loop think rather than staring at a spinner.

Deployed on Streamlit Community Cloud this file is the entry point; locally:

    streamlit run streamlit_app.py
"""

import os
import ast
import json
import traceback

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Graph Websearch Agent",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "claude": "claude-3-5-sonnet-20240620",
    "ollama": "llama3:instruct",
    "vllm": "meta-llama/Meta-Llama-3-70B-Instruct",
}

# Google retires model ids without much notice - gemini-2.0-flash and gemini-2.5-flash
# are both already refused for new keys. These were verified against a live AI Studio key.
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
]
# Groq retires ids aggressively too - llama3-70b-8192 and llama3-8b-8192 are both gone
# and the API answers with an explicit "has been decommissioned" message.
# Current list: https://console.groq.com/docs/models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
]

MODEL_CHOICES = {"gemini": GEMINI_MODELS, "groq": GROQ_MODELS}
CUSTOM_MODEL = "Custom…"

PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "claude": "CLAUD_API_KEY",
}

AGENT_LABELS = {
    "planner": ("🧠 Planner", "Turned the question into a search strategy"),
    "serper_tool": ("🔎 Search", "Fetched the search engine results page"),
    "selector": ("🎯 Selector", "Picked the most promising result to read"),
    "scraper_tool": ("📄 Scraper", "Pulled the text off the selected page"),
    "reporter": ("✍️ Reporter", "Drafted an answer from the scraped source"),
    "reviewer": ("⚖️ Reviewer", "Critiqued the draft against the question"),
    "router": ("🧭 Router", "Decided who works next"),
    "final_report": ("✅ Final report", "Approved the answer"),
    "end": ("🏁 End", "Workflow finished"),
}

EXAMPLE_QUESTIONS = [
    "What is the current population of Tokyo and how has it changed in the last decade?",
    "Who won the most recent FIFA World Cup, and what was the final score?",
    "What are the headline features of Python 3.13?",
    "What is Anthropic's Model Context Protocol and what problem does it solve?",
]


# ---------------------------------------------------------------------------
# Secrets / settings helpers
# ---------------------------------------------------------------------------

def secret(name, default=""):
    """Read a Streamlit secret, tolerating the common case of no secrets file at all."""
    try:
        return st.secrets.get(name, default) or default
    except Exception:
        return default


@st.cache_data(ttl=600, show_spinner=False)
def fetch_groq_models(api_key):
    """Ask Groq which models this key can actually use.

    Free and paid tiers serve different model sets, and ids get retired without
    notice, so a hardcoded list goes stale. Returns [] on any failure and the
    caller falls back to the static list.
    """
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        ids = [entry.get("id", "") for entry in response.json().get("data", [])]
    except Exception:
        return []

    # Drop the non-chat models: speech, safety classifiers, embeddings.
    skip = ("whisper", "tts", "guard", "embed")
    return sorted(i for i in ids if i and not any(word in i.lower() for word in skip))


def apply_environment(keys):
    """Push the collected keys into the process environment for the graph to pick up."""
    for name, value in keys.items():
        if value:
            os.environ[name] = str(value)
        else:
            os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# Reading node updates out of the stream
# ---------------------------------------------------------------------------

def to_text(value):
    """Nodes return plain strings, LangChain messages, or lists of them. Normalise to text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return to_text(value[-1]) if value else ""
    content = getattr(value, "content", None)
    return content if isinstance(content, str) else str(value)


def latest(update, key):
    if not isinstance(update, dict):
        return ""
    return to_text(update.get(key))


def try_json(text):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def error_in(text):
    """Agents swallow API failures and return {"error": ...}. Return that message, if any."""
    data = try_json(text)
    if isinstance(data, dict) and "error" in data:
        return str(data["error"])
    return None


def render_planner(container, text):
    data = try_json(text)
    if not data:
        container.markdown(text or "_no output_")
        return
    if "error" in data:
        container.error(data["error"])
        return
    container.markdown(f"**Search term**\n\n`{data.get('search_term', '')}`")
    if data.get("overall_strategy"):
        container.markdown(f"**Strategy**\n\n{data['overall_strategy']}")
    if data.get("additional_information"):
        container.markdown(f"**Notes**\n\n{data['additional_information']}")


def render_selector(container, text):
    data = try_json(text)
    if not data:
        container.markdown(text or "_no output_")
        return
    if "error" in data:
        container.error(data["error"])
        return
    url = data.get("selected_page_url", "")
    if url:
        container.markdown(f"**Selected page**\n\n{url}")
    if data.get("description"):
        container.markdown(f"**Description**\n\n{data['description']}")
    if data.get("reason_for_selection"):
        container.markdown(f"**Why this one**\n\n{data['reason_for_selection']}")


def render_scraper(container, text):
    """The scraper stores ``str({"source": url, "content": text})`` - unpack it if we can."""
    payload = None
    try:
        payload = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass

    if isinstance(payload, dict):
        container.markdown(f"**Source**\n\n{payload.get('source', 'unknown')}")
        body = payload.get("content", "")
        container.text(body[:1500] + ("…" if len(body) > 1500 else ""))
    else:
        container.text(text[:1500])


def render_reviewer(container, text):
    data = try_json(text)
    if not data:
        container.markdown(text or "_no output_")
        return
    if "error" in data:
        container.error(data["error"])
        return
    for key, value in data.items():
        container.markdown(f"**{key.replace('_', ' ').title()}**\n\n{value}")


def render_router(container, text):
    data = try_json(text)
    if not data:
        container.markdown(text or "_no output_")
        return
    if "error" in data:
        container.error(data["error"])
        return
    next_agent = data.get("next_agent", "unknown")
    if isinstance(next_agent, list):
        next_agent = next_agent[-1] if next_agent else "unknown"
    container.markdown(f"**Next agent** → `{next_agent}`")


def render_search(container, text):
    container.text(text[:3000] + ("…" if len(text) > 3000 else ""))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    server = st.selectbox(
        "Model provider",
        list(PROVIDER_DEFAULT_MODELS.keys()),
        index=0,
        help="Gemini works with a free key from Google AI Studio.",
    )

    # The key comes before the model picker so the model list can be fetched with it.
    key_env = PROVIDER_KEY_ENV.get(server)
    provider_key = ""
    if key_env:
        provider_key = st.text_input(
            f"{server.title()} API key",
            value=secret(key_env),
            type="password",
            help="Stored only for this session. On Streamlit Cloud, set it once under "
                 "App settings → Secrets.",
        )
    else:
        st.caption(f"`{server}` runs locally and needs no API key.")

    live_models = fetch_groq_models(provider_key) if server == "groq" and provider_key else []
    if live_models:
        # Show the models we would have recommended first, then everything else the key allows.
        preferred = [m for m in MODEL_CHOICES.get(server, []) if m in live_models]
        options = preferred + [m for m in live_models if m not in preferred]
    else:
        options = MODEL_CHOICES.get(server, [])

    if options:
        choice = st.selectbox(
            "Model",
            options + [CUSTOM_MODEL],
            index=0,
            help="Providers retire model ids without notice. If one stops working, the error "
                 "message names it — pick another here.",
        )
        model = (
            st.text_input("Custom model id", value=PROVIDER_DEFAULT_MODELS[server])
            if choice == CUSTOM_MODEL
            else choice
        )
        if live_models:
            st.caption(f"✅ {len(live_models)} chat models available to this key.")
        elif server == "groq" and provider_key:
            st.caption("⚠️ Could not list models from Groq — showing defaults.")
    else:
        model = st.text_input("Model name", value=PROVIDER_DEFAULT_MODELS[server])

    ollama_base_url = ""
    if server == "ollama":
        ollama_base_url = st.text_input(
            "Ollama server URL",
            value=secret("OLLAMA_BASE_URL") or "http://localhost:11434",
            help="Leave as localhost when running the app on the same machine as Ollama. "
                 "A deployed app cannot reach your localhost — point this at a tunnel instead.",
        )

    model_endpoint = None
    stop_token = None
    if server == "vllm":
        model_endpoint = st.text_input("vLLM endpoint", value="")
        stop_token = st.text_input("Stop token", value="<|end_of_text|>")

    st.divider()
    st.subheader("🔎 Search")

    search_provider = st.selectbox(
        "Search provider",
        ["auto", "serper", "duckduckgo"],
        index=0,
        help="`auto` uses Serper when a key is present and falls back to DuckDuckGo, which needs no key.",
    )

    serper_key = ""
    if search_provider in ("auto", "serper"):
        serper_key = st.text_input(
            "Serper API key (optional)",
            value=secret("SERPER_API_KEY"),
            type="password",
            help="Free tier at serper.dev. Leave blank to use DuckDuckGo instead.",
        )

    st.divider()
    st.subheader("🎛️ Run controls")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    recursion_limit = st.number_input(
        "Recursion limit",
        min_value=1,
        max_value=100,
        value=2,
        step=1,
        help="Maximum node executions before LangGraph stops the run. A clean pass needs "
             "about 9 (planner, search, selector, scraper, reporter, reviewer, router, "
             "final report, end), and each reviewer rejection adds 3 more.",
    )
    st.caption(
        "A full pass = **9 steps / 5 LLM calls**. Each reviewer rejection adds 3 steps "
        "(3 more calls). Below 9, the run stops before producing a report."
    )
    show_trace = st.toggle("Show live agent trace", value=True)

    st.divider()
    st.caption(
        "Search falls back to DuckDuckGo automatically, so a Gemini key alone is enough to run this app."
    )


# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------

st.title("🕸️ Graph Websearch Agent")
st.markdown(
    "A team of LLM agents — **planner, selector, reporter, reviewer, router** — wired together as a "
    "LangGraph state machine. They search the live web, read one page at a time, draft an answer, "
    "critique it, and loop until the reviewer is satisfied."
)

if "question" not in st.session_state:
    st.session_state.question = ""

with st.expander("💡 Example questions", expanded=False):
    cols = st.columns(2)
    for index, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[index % 2].button(example, key=f"example_{index}", use_container_width=True):
            st.session_state.question = example

question = st.text_area(
    "Research question",
    key="question",
    height=100,
    placeholder="Ask anything that needs a look at the live web…",
)

run = st.button("🚀 Run research", type="primary", use_container_width=True)


def build_workflow():
    """Import and compile the graph. Imported lazily so the page renders before the heavy imports."""
    from agent_graph.graph import create_graph, compile_workflow

    graph = create_graph(
        server=server,
        model=model,
        model_endpoint=model_endpoint or None,
        temperature=temperature,
        stop=stop_token or None,
    )
    return compile_workflow(graph)


if run:
    if not question.strip():
        st.warning("Enter a research question first.")
        st.stop()

    if key_env and not provider_key:
        st.error(f"A {server.title()} API key is required. Add it in the sidebar or in Streamlit secrets.")
        st.stop()

    apply_environment({
        "GEMINI_API_KEY": provider_key if server == "gemini" else secret("GEMINI_API_KEY"),
        "OPENAI_API_KEY": provider_key if server == "openai" else secret("OPENAI_API_KEY"),
        "GROQ_API_KEY": provider_key if server == "groq" else secret("GROQ_API_KEY"),
        "CLAUD_API_KEY": provider_key if server == "claude" else secret("CLAUD_API_KEY"),
        "SERPER_API_KEY": serper_key,
        "SEARCH_PROVIDER": search_provider,
        "OLLAMA_BASE_URL": ollama_base_url,
    })

    trace_area = st.container()
    step = 0
    final_report = ""
    last_report = ""
    errors = []
    stopped_early = False

    try:
        with st.spinner("Compiling the agent graph…"):
            workflow = build_workflow()

        stream = workflow.stream(
            {"research_question": question.strip()},
            {"recursion_limit": int(recursion_limit)},
        )

        progress = st.status("Researching…", expanded=True)

        for event in stream:
            for node, update in event.items():
                label, caption = AGENT_LABELS.get(node, (node, ""))
                step += 1
                progress.update(label=f"{label} — step {step}")

                if node == "end":
                    continue

                # An agent that hit an API failure returns an {"error": ...} payload.
                # Collect those, and never let one end up rendered as the report.
                for key in (f"{node}_response", "final_reports", "scraper_response"):
                    message = error_in(latest(update, key))
                    if message and message not in errors:
                        errors.append(message)

                if node == "reporter":
                    text = latest(update, "reporter_response")
                    if text and not error_in(text):
                        last_report = text
                if node == "final_report":
                    text = latest(update, "final_reports")
                    if text and not error_in(text):
                        final_report = text

                if not show_trace:
                    continue

                with trace_area.expander(f"{step}. {label} — {caption}", expanded=False):
                    box = st.container()
                    if node == "planner":
                        render_planner(box, latest(update, "planner_response"))
                    elif node == "serper_tool":
                        render_search(box, latest(update, "serper_response"))
                    elif node == "selector":
                        render_selector(box, latest(update, "selector_response"))
                    elif node == "scraper_tool":
                        render_scraper(box, latest(update, "scraper_response"))
                    elif node == "reporter":
                        box.markdown(latest(update, "reporter_response") or "_no output_")
                    elif node == "reviewer":
                        render_reviewer(box, latest(update, "reviewer_response"))
                    elif node == "router":
                        render_router(box, latest(update, "router_response"))
                    elif node == "final_report":
                        box.markdown(latest(update, "final_reports") or "_no output_")

        progress.update(label=f"Done in {step} steps", state="complete", expanded=False)

    except Exception as error:  # surface the failure in the UI instead of a blank page
        # Running out of recursion budget is an expected outcome when the limit is set
        # low to save API calls, so keep whatever the reporter already wrote rather than
        # throwing the whole run away.
        if "recursion" in str(error).lower():
            stopped_early = True
            progress.update(
                label=f"Stopped at the recursion limit after {step} steps",
                state="complete",
                expanded=False,
            )
        else:
            st.error(f"The workflow stopped: {error}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()

    report = final_report or last_report
    st.divider()

    if errors:
        with st.expander(f"⚠️ {len(errors)} agent call(s) failed during this run", expanded=not report):
            for message in errors:
                st.warning(message)
            st.caption(
                "Free-tier Gemini returns 503/429 under load. Each call already retries with "
                "backoff — if these persist, wait a minute or switch model in the sidebar."
            )

    if report:
        st.subheader("📋 Final report")
        if stopped_early and not final_report:
            st.info(
                f"Recursion limit ({int(recursion_limit)}) reached before the reviewer approved "
                "this draft — it is the reporter's latest output, shown as-is. Raise the limit "
                "for a reviewed answer."
            )
        st.markdown(report)
        st.download_button(
            "⬇️ Download report (Markdown)",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
        )
    else:
        st.warning(
            f"No report was produced. The recursion limit is {int(recursion_limit)}; a full pass "
            "needs 9 steps (5 of them LLM calls), so the run stopped before the reporter wrote "
            "anything. Raise the limit to at least 9."
        )
