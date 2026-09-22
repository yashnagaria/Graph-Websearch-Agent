# 🕸️ Graph Websearch Agent

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=yashnagaria/Graph-Websearch-Agent&branch=main&mainModule=streamlit_app.py)

A multi-agent web research system built as a **LangGraph state machine**. Five LLM agents —
planner, selector, reporter, reviewer and router — pass a shared state between each other,
search the live web, read one page at a time, draft an answer, critique it, and loop until the
reviewer is satisfied.

It ships with a **Streamlit UI** that streams every agent hand-off to the page as it happens, so
you can watch the loop reason instead of staring at a spinner.

> 📖 Curious how this was built and what it is good for? Read [`interview.md`](interview.md).

---

## How it works

```
                 ┌──────────┐
   question ───► │ planner  │  picks the search term + strategy
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │  search  │  Serper (or keyless DuckDuckGo)
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │ selector │  picks the single best URL to read
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │ scraper  │  pulls the page text
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │ reporter │  drafts an answer, cites the source
                 └────┬─────┘
                      ▼
                 ┌──────────┐
                 │ reviewer │  critiques the draft
                 └────┬─────┘
                      ▼
                 ┌──────────┐      ┌──────────────┐
                 │  router  │ ───► │ final report │ ──► end
                 └────┬─────┘      └──────────────┘
                      │
     back to planner / selector / reporter if the draft is not good enough
```

Every agent writes into a shared `AgentGraphState`, so later agents see the full history of what
the team has already tried. The router is the only node with a conditional edge — it is what turns
a straight pipeline into a **loop that can retry**.

---

## Quick start — run it locally

```bash
git clone https://github.com/yashnagaria/Graph-Websearch-Agent.git
cd Graph-Websearch-Agent

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Add your key (a free one from [Google AI Studio](https://aistudio.google.com/app/apikey) is enough):

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit the file and paste your GEMINI_API_KEY
```

Run it:

```bash
streamlit run streamlit_app.py
```

The app opens at <http://localhost:8501>. You can also paste keys straight into the sidebar
without creating a secrets file at all.

---

## Deploy to Streamlit Community Cloud

The repo is already laid out the way Streamlit Cloud expects — `streamlit_app.py` at the root,
`requirements.txt` next to it — so deploying is three clicks:

1. Go to **<https://share.streamlit.io>** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub** and fill in:
   - **Repository:** `yashnagaria/Graph-Websearch-Agent`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Open **Advanced settings** and:
   - set **Python version** to **3.11**
   - paste your keys into the **Secrets** box:

     ```toml
     GEMINI_API_KEY = "your-google-ai-studio-key"
     # optional, better search results:
     # SERPER_API_KEY = "your-serper-key"
     ```
4. Click **Deploy**.

Or just click the **Open in Streamlit** badge at the top of this README — it pre-fills the form.

> Secrets can be added or changed later from **App settings → Secrets**; the app reboots itself.

---

## Configuration

| Setting | Where | Notes |
|---|---|---|
| `GEMINI_API_KEY` | secrets / sidebar | Free from Google AI Studio. **The only key you need.** |
| `SERPER_API_KEY` | secrets / sidebar | Optional. Free tier at [serper.dev](https://serper.dev/). Better results than the fallback. |
| `OPENAI_API_KEY` | secrets / sidebar | Only if you pick the `openai` provider. |
| `GROQ_API_KEY` | secrets / sidebar | Only if you pick the `groq` provider. |
| `CLAUD_API_KEY` | secrets / sidebar | Only if you pick the `claude` provider. |
| `SEARCH_PROVIDER` | sidebar | `auto` (default), `serper`, or `duckduckgo`. |

Key resolution order: **environment / Streamlit secrets → `config/config.yaml`**. An empty value in
the YAML never overwrites a key that is already set, which is what lets the cloud deployment work
without editing any files.

**You do not need a search API key.** With no `SERPER_API_KEY` present the app searches DuckDuckGo,
which requires no signup. If a Serper key is present but the call fails (quota, bad key), it falls
back to DuckDuckGo automatically rather than crashing the run.

### Supported model providers

| Provider | Default model | Key needed |
|---|---|---|
| `gemini` | `gemini-3.6-flash` | `GEMINI_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `groq` | `llama3-70b-8192` | `GROQ_API_KEY` |
| `claude` | `claude-3-5-sonnet-20240620` | `CLAUD_API_KEY` |
| `ollama` | `llama3:instruct` | none (local) |
| `vllm` | any HF path | none (your endpoint) |

---

## Other ways to run it

**Command line:**

```bash
python -m app.app
```

(edit the `server` / `model` variables at the top of [`app/app.py`](app/app.py) first)

**Original Chainlit UI:**

```bash
pip install -r requirements-chainlit.txt
chainlit run app/chat.py
```

---

## Project layout

```
streamlit_app.py        Streamlit UI - the Cloud entry point
agent_graph/graph.py    LangGraph wiring: nodes, edges, the router's conditional edge
agents/agents.py        The five agents + the final-report and end nodes
prompts/prompts.py      System prompts and the JSON schemas each agent must return
models/                 One thin client per provider (gemini, openai, groq, claude, ollama, vllm)
tools/google_serper.py  Web search: Serper with a keyless DuckDuckGo fallback
tools/basic_scraper.py  Fetches a URL and strips it to text
states/state.py         The shared AgentGraphState and its accessors
config/config.yaml      Optional local key file
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Paste the key in the sidebar, or add it under App settings → Secrets. |
| Workflow ends with no report | Raise the recursion limit in the sidebar, or rephrase the question. |
| `error in scraping website, 403 Forbidden` | Normal — the site blocks bots. The router sends the selector back to pick a different page. |
| Search returns nothing | DuckDuckGo rate-limits busy shared IPs. Add a free Serper key for reliable results. |
| `503 This model is currently experiencing high demand` | Free-tier load. Every call already retries 4× with backoff; if it persists, pick another model in the sidebar or wait a few minutes. |
| `429 You exceeded your current quota` | Free-tier daily/per-minute quota is used up. Wait, or enable billing on the Google Cloud project behind the key. |
| `404 This model is no longer available to new users` | Google retired that model id. Pick one from the sidebar dropdown — `gemini-2.0-flash` and `gemini-2.5-flash` are both already retired. |
| Build fails on Streamlit Cloud | Set the Python version to **3.11** in the app's advanced settings. |

---

## Credits

Built on the LangGraph agent pattern popularised by
[John Adeojo's `graph_websearch_agent`](https://github.com/john-adeojo/graph_websearch_agent).
Licensed under the terms in [`LICENSE`](LICENSE).

Further reference docs live in [`docs/INDEX.md`](docs/INDEX.md).
