# The story behind Graph Websearch Agent

*How a single LLM call turned into a five-agent state machine that argues with itself until the answer is good enough.*

---

## 1. The problem that started it

Ask a language model "what is the population of Tokyo right now?" and you get a confident number.
It is often wrong, and it is always stale — the model is answering from training data that froze
months or years ago. The obvious fix is to let it search the web. So you bolt on a search API,
paste the top ten snippets into the prompt, and ask again.

That works, right up until it doesn't:

- **Snippets lie by omission.** A search snippet is 160 characters chosen by a ranking algorithm,
  not by relevance to *your* question. The model confidently summarises a fragment.
- **One shot, no recovery.** If the search term was bad, the whole answer is bad, and nothing in
  the pipeline notices.
- **No self-awareness.** The model has no mechanism to say "this source didn't actually answer the
  question, let me look somewhere else."

The classic RAG chain is a straight line: `query → search → stuff context → answer`. A straight
line cannot go back. And *going back* is exactly what research is.

So the question became: **what if the pipeline could loop?**

---

## 2. Why a graph and not a chain

The moment you want retries, you are no longer describing a chain — you are describing a **state
machine**. Nodes do work. Edges decide who works next. One of those edges has to be conditional,
or nothing can ever loop.

That is precisely what [LangGraph](https://github.com/langchain-ai/langgraph) models, so the
architecture fell out of the problem statement:

| Node | Job | Output |
|---|---|---|
| **planner** | Read the question, decide what to actually search for | JSON: `search_term`, `overall_strategy` |
| **search** | Hit the search engine | A formatted SERP |
| **selector** | Read the whole SERP and pick *one* URL worth reading | JSON: `selected_page_url` + why |
| **scraper** | Fetch that page, strip it to text | `{source, content}` |
| **reporter** | Write an answer grounded in that page | Markdown |
| **reviewer** | Critique the draft against the original question | JSON: structured feedback |
| **router** | Decide: ship it, or send someone back to work | JSON: `next_agent` |

The whole design hinges on one edge. In [`agent_graph/graph.py`](agent_graph/graph.py):

```python
graph.add_conditional_edges(
    "router",
    lambda state: pass_review(state=state),
)
```

Every other edge is fixed. This one reads the router's JSON and returns a node name — which may be
`planner` (the search term was wrong), `selector` (wrong page), `reporter` (right source, bad
write-up), or `final_report` (we're done). That single line is what separates "a RAG script" from
"a research loop."

---

## 3. The design decisions, and why

### Separate agents instead of one big prompt

It is tempting to write one giant prompt: *"search, pick a source, write an answer, check it."*
It fails, for a reason that is easy to miss: a model that writes an answer and then grades its own
answer in the same context window is grading its own homework with the answer sheet open. It
almost always says "looks good."

Splitting reporter and reviewer into **separate LLM calls with different system prompts and no
shared reasoning** means the reviewer sees the draft as an artefact, not as its own work. The
critiques get noticeably sharper.

### One page at a time, not ten

The selector deliberately picks **a single URL**. This felt wrong at first — why throw away nine
results? But it makes the system honest. Every claim in the final report traces to one named
source you can click. If that source is bad, the reviewer says so and the loop picks another. Ten
sources blended into one answer gives you a report nobody can audit.

### Force JSON everywhere it matters

Four of the five agents return JSON against a declared schema in
[`prompts/prompts.py`](prompts/prompts.py). The router in particular *must* return
`{"next_agent": "..."}` — a conditional edge that receives prose instead of a node name crashes
the graph. Gemini's `response_mime_type: application/json` and OpenAI's `json_object` mode do the
heavy lifting; a code-fence stripper catches the rest.

### A shared state, not passed messages

Every node reads and writes one `AgentGraphState` ([`states/state.py`](states/state.py)). Each
field is annotated with LangGraph's `add_messages` reducer, so writes **append** rather than
overwrite:

```python
planner_response: Annotated[list, add_messages]
```

This is what makes the loop intelligent rather than amnesiac. When the router sends work back to
the planner, the planner receives the reviewer's feedback *and* can see what it already tried. It
does not propose the same failed search term twice.

---

## 4. The parts that fought back

An honest build log is more useful than a highlight reel. These are the things that actually broke.

**The config loader ate its own keys.** The original `load_config` looped over `config.yaml` and
wrote every value into the environment — including empty ones, which it replaced with placeholder
strings like `'default_openai_api_key'`. On a laptop with a filled-in YAML, fine. On Streamlit
Cloud, where keys arrive through `st.secrets` and the YAML is empty, it silently overwrote every
real key with garbage a split second before the first API call. The fix inverts the precedence:
**an environment variable that already has a value always wins**, and an empty YAML entry never
overwrites anything.

**Gemini 2.5 Flash answered with silence.** Switching to a 2.5 model produced responses with a
`candidates` array, a valid `finishReason`, and *no text*. The model was spending its entire output
budget on internal reasoning tokens before it ever started writing. The fix is one line in
[`models/gemini_models.py`](models/gemini_models.py) — `"thinkingConfig": {"thinkingBudget": 0}` —
plus error handling that reports `finishReason` instead of throwing an opaque `KeyError`, because
"no text (finishReason: MAX_TOKENS)" is a debuggable message and `KeyError: 'parts'` is not.

**Half the web returns 403 to Python.** The scraper called `requests.get(url)` with the default
`python-requests/2.x` user agent. A large share of news sites and docs pages reject that outright.
Sending a browser user agent and a timeout fixed most of it — and crucially, a 403 is *handled*,
not fatal: it goes into the state as scraper output, the reviewer sees an empty source, and the
router sends the selector back to pick a different page. **The failure becomes an input to the
loop**, which is the whole point of the architecture.

**Search needed a credit card.** The original build required a Serper key. Anyone who cloned the
repo hit a `KeyError: 'SERPER_API_KEY'` on the first run. Now `tools/google_serper.py` resolves a
provider at runtime: use Serper if a key exists, fall back to DuckDuckGo (no signup, no key) if it
does not — and fall back again if a Serper call fails mid-run. **One free Gemini key is now enough
to run the entire system.**

**A hardcoded path to somebody's Google Drive.** `app/chat.py` pointed at
`G:/My Drive/Data-Centric Solutions/…`. It ran on exactly one machine on Earth. Now it resolves
relative to the file.

---

## 5. From terminal to a URL anyone can open

The original interface was `input()` in a terminal with `termcolor` output. That is a fine way to
debug an agent and a terrible way to show one to anybody else.

The Streamlit rewrite ([`streamlit_app.py`](streamlit_app.py)) exists because of one insight:
**the intermediate steps are the product.** A research agent that prints an answer after ninety
seconds of silence feels broken and untrustworthy. The same agent that shows you *"🧠 Planner chose
the search term 'Tokyo population 2025' → 🎯 Selector picked the Statistics Bureau page → ⚖️ Reviewer
says the figure is stale, try again"* feels like watching a colleague work.

So the app consumes `workflow.stream(...)` and renders each node's update as it arrives — the
planner's strategy, the raw SERP, the chosen URL, the scraped text, each draft, each critique, each
routing decision. The final report lands at the bottom with a download button. Failures render as
a visible error with a traceback in an expander, rather than a blank page.

Deployment shaped the code too: the entry point moved to the repo root, Chainlit moved out to
`requirements-chainlit.txt` so the Cloud build stays small, and every key became overridable from
the sidebar so the app is usable by someone who has never seen the repo.

---

## 6. What it is good for

### Use case A — Time-sensitive factual lookup

**When the answer changed after the model's training cutoff.** Prices, populations, standings,
release notes, "who currently holds X." A plain LLM answers confidently and wrongly; this loop goes
and reads a page from this week, and the reviewer specifically checks the answer's recency.

### Use case B — Cited research briefs

**When "the model said so" is not good enough.** Every report names the URL it was written from,
because the reporter only ever sees one scraped page. That makes the output checkable — you can
open the source and disagree with it, which you cannot do with a model's recollection.

### Use case C — Question triage on unfamiliar topics

**When you do not know the right search term yet.** The planner's job is to translate a vague human
question into a good query, and the loop rewrites that query when the results disappoint. It is a
better first pass than a human guessing keywords cold.

### Use case D — A teaching harness for agent design

**When you want to see multi-agent orchestration without a framework hiding it.** Every prompt is
in one file, every model client is ~60 lines of `requests`, and the graph is twenty lines of edges.
You can break one agent and watch exactly how the system compensates.

### Where it is the wrong tool

Not for questions the model already knows (slow and expensive for no benefit), not for questions
needing twenty sources synthesised (it reads one page per pass by design), and not for anything
behind a login or a hard paywall.

---

## 7. A worked example, end to end

**Question:** *"What are the headline features of Python 3.13?"*

**Step 1 — 🧠 Planner.** Reads the question and returns:

```json
{
  "search_term": "Python 3.13 new features release notes",
  "overall_strategy": "Target the official python.org 'What's New' document rather than blog summaries, which often repeat each other and get details wrong.",
  "additional_information": "Fallback terms: 'What's New In Python 3.13', 'PEP 703 free-threading'"
}
```

Note what happened: the user's casual phrasing became a query with the words a release-notes page
actually contains.

**Step 2 — 🔎 Search.** The term goes to Serper, or to DuckDuckGo if no key is set. Ten results come
back as `Title / Link / Snippet` triples.

**Step 3 — 🎯 Selector.** Reads all ten and commits to one:

```json
{
  "selected_page_url": "https://docs.python.org/3/whatsnew/3.13.html",
  "description": "The official What's New in Python 3.13 document.",
  "reason_for_selection": "Primary source, maintained by CPython core developers, complete rather than a summary."
}
```

It passed over higher-ranked blog posts in favour of the primary source. That preference is written
into the selector's system prompt.

**Step 4 — 📄 Scraper.** Fetches the page with a browser user agent, strips the HTML to text, caps
it, and stores `{"source": "https://docs.python.org/...", "content": "..."}`.

**Step 5 — ✍️ Reporter.** Sees *only* the scraped text and the question, and drafts:

> **Python 3.13 — headline features**
> - A new interactive interpreter (REPL) with multiline editing, colour output and block history.
> - **Free-threaded mode (PEP 703):** an experimental build that runs without the GIL.
> - **Experimental JIT compiler (PEP 744).**
> - Substantially improved error messages…
>
> *Source: https://docs.python.org/3/whatsnew/3.13.html*

**Step 6 — ⚖️ Reviewer.** Grades the draft against the question, not against its own reasoning. A
typical first-pass critique:

> *"The free-threading and JIT entries do not state that both are experimental opt-in builds, which
> a reader would reasonably misinterpret as defaults. Recommend the reporter revise."*

**Step 7 — 🧭 Router.** Turns that critique into a routing decision:

```json
{"next_agent": "reporter"}
```

The conditional edge fires and control goes **back** to the reporter — not to the planner, because
the source was right and only the write-up was wrong. The reporter revises with the critique in its
context, the reviewer reads it again, and this time the router returns:

```json
{"next_agent": "final_report"}
```

**Step 8 — ✅ Final report.** The approved draft is written to `final_reports`, the graph reaches
`end`, and the Streamlit page renders it with a download button — above a full, expandable trace of
all eight steps.

**Total:** roughly 7–9 LLM calls, one search, one page fetched, one self-correction. The
self-correction is the part a straight-line RAG chain can never do.

---

## 8. What I would build next

- **Parallel selectors.** Read the top three sources concurrently and let the reporter cross-check
  them. LangGraph supports fan-out; the state reducers are already append-only, so the shape is
  right — the reporter prompt is what needs rethinking.
- **A budget, not just a recursion limit.** Cap the run by tokens and wall-clock, not only by step
  count, and surface the cost in the UI.
- **Persistent checkpoints.** LangGraph's `SqliteSaver` would let a run be paused, inspected and
  resumed — and would turn the trace into something you can replay.
- **Reviewer scoring.** Have the reviewer emit a numeric confidence alongside its prose, so the
  router can stop on "good enough" instead of "the reviewer ran out of objections."

---

## 9. The one-paragraph version

*I built a web research agent as a LangGraph state machine rather than a RAG chain, because
research needs to be able to go back. Five agents — planner, selector, reporter, reviewer, router —
share an append-only state; the router owns the only conditional edge, and it can send work back to
any earlier agent when the reviewer rejects a draft. Keeping the reviewer in a separate LLM call
from the reporter is what makes the critiques real. Along the way I fixed a config loader that
overwrote live API keys with placeholders, a Gemini 2.5 model that spent its whole output budget
thinking, and a scraper that half the internet rejected for its user agent — and I added a keyless
DuckDuckGo fallback so the whole thing runs on one free API key. The Streamlit front-end streams
every agent hand-off live, because with an agent, the reasoning is the product.*
