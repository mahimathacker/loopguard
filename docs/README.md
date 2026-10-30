# Maintaining the LoopGuard docs

The Mintlify site lives in this directory. Start reading at [Introduction](index.mdx).
This file is a maintainer handoff, not a published docs page; `.mintignore` excludes it.

## Preview

Use Node.js 20.17+ and run the current official `mint` CLI from `docs/`:

```bash
cd docs
npx --yes mint dev --port 3001 --no-open
```

Open http://localhost:3001. Port 3001 leaves port 3000 free for the LoopGuard demo UI.
The preview stays local. Mintlify may require a login to activate local search;
reading the preview does not require deploying it.

## Validate

From `docs/`:

```bash
npx --yes mint validate
npx --yes mint broken-links --check-anchors
npx --yes mint a11y
```

Use the `mint` package, not the legacy `mintlify` npm package. The CLI follows
[Mintlify's current command reference](https://www.mintlify.com/docs/cli/commands).
Its generated cache and installed tooling are not source files to commit.

From the repository root, also run:

```bash
python -m unittest discover -v
python scripts/check_docs.py
python scripts/verify_doc_examples.py
python examples/first_run.py
python -m loopguard.ingest examples/sample_trace.json --json
python -m loopguard.check examples/sample_trace.json --json
```

Install the framework extra to run `first_run.py`. The sample trace check should
exit `1`; it deliberately contains a loop. Provider-backed scenarios need separate
credentials and model access. Do not interpret offline fixture scores as production
accuracy.

## Navigation

```text
Start here
  Introduction
  Getting started
  How LoopGuard works
Concepts
  AI agents and loops
  Tests, traces, and evals
  Agent trajectories and convergence
Use LoopGuard
  Detecting stuck agents
  LangGraph and LangChain
  Configure LoopGuard
  Analyze saved traces
  Add a trace regression check
Explore the demo
  Demo scenarios
  Run the demo UI
Test and troubleshoot
  Evaluation and tests
  Troubleshooting
  Project status
  Contributing and docs maintenance
Reference
  Python API
  CLI reference
  Configuration reference
```

Every public `.mdx` page belongs in `docs.json`. Use root-relative site links such
as `/guides/configuration`. README links at the repository root use real file paths
so they work on GitHub before hosting exists.

## Automated maintenance

See [the contributing guide](development/contributing.mdx) for the CI behavior
and local equivalents. The three workflows live in `.github/workflows/`:

- `documentation.yml`: tests, offline docs checks, README example execution,
  Mintlify build, and internal link checks on PRs and pushes.
- `docs-drift.yml`: a file-based PR warning and summary for developer-facing changes
  without Markdown updates. It does not rewrite docs or prove coverage.
- `external-links.yml`: a weekly/manual Lychee check of public documentation URLs.
  Its schedule takes effect after the workflow reaches the default branch.

`python scripts/check_docs.py --release` fails on the current hosted-URL placeholder
and two screenshot placeholders. Normal checks warn so in-progress docs can be
reviewed. Version tags and manual release-readiness runs use the strict check.
These checks do not publish the docs. Require the relevant status checks in branch
protection if they should block merging.

## Agent discovery

Mintlify generates `/llms.txt`, `/llms-full.txt`, and `.md` page exports from the
site content. We keep these generated so links and page contents follow the
navigation. SDK guidance in `docs.json` under `markdown.instructions` is included
in the agent-facing exports. See [Mintlify's export documentation](https://www.mintlify.com/docs/ai/markdown-export).

The checked-in `robots.txt` allows crawling. `seo.metatags.robots` permits page
indexing. Mintlify generates `/sitemap.xml`; after confirming the public domain,
add its absolute URL as a `Sitemap:` directive to `robots.txt`.

After deployment, verify `/robots.txt`, `/sitemap.xml`, `/llms.txt`,
`/llms-full.txt`, and `/getting-started.md` on the actual docs domain. Confirm that
the index covers all navigation pages and that Markdown exports include the SDK
guidance. These generated routes return 404 in the local preview used here.
The dashboard's **Don't index project** setting must be off for public discovery.
Discovery files help tools find content; they do not guarantee indexing or citation.

## Ask in an external AI tool

`docs.json` enables Mintlify's built-in `chatgpt`, `claude`, `aistudio`, `perplexity`,
`grok`, `copy`, and `view` context-menu actions in the page header. Google AI Studio
is the supported entry point for Gemini. The first action is the primary button;
the others appear in its dropdown. This setup uses external AI tools rather than
Mintlify's paid embedded assistant. No model key or chat backend is added here.

These actions provide the current page as context, not an automatically indexed
copy of every repository file. The introduction explains how to include the
published `llms.txt` or `llms-full.txt` for broader questions and how to copy a page
when fetching is unavailable. The user's external AI account and its limits apply.

After deployment, check the links on the real docs URL and confirm that each
provider can retrieve the page. Local preview links can be inspected, but external
tools cannot fetch `localhost`. See
[Mintlify's contextual menu documentation](https://www.mintlify.com/docs/ai/contextual-menu)
for the supported actions.

## Screenshot provenance

All four images were copied unchanged from `ui/public/`; no screenshots were generated.

| Original | Docs image | Used on |
| --- | --- | --- |
| `realagentloop.png` | `images/stuck-agent.png` | Introduction and demo scenarios; the root README uses the original. |
| `scriptedtool.png` | `images/exact-loop.png` | Exact scenario and local demo setup. |
| `scriptedopenai.png` | `images/semantic-loop.png` | Semantic scenario. |
| `realagentmath.png` | `images/healthy-calculation.png` | Healthy calculation scenario. |

These captures show an older UI layout. Captions disclose that; they do not claim
that current thresholds always stop at the counts displayed in those screenshots.
Each of the four visual scenarios includes a runnable Python SDK example and an
explanation of the messages behind the UI. The screenshots illustrate SDK behavior;
the UI is one way to inspect it.

## Screenshot capture checklist

Two new captures complete the healthy-recovery versus permanent-failure comparison.
Both placeholders are on `demo/scenarios.mdx`.

First follow [local demo setup](demo/running-locally.mdx). With the Python environment
active, run the backend from the repository root:

```bash
uvicorn server:app --reload --port 8000
```

In a second terminal, from the repository root:

```bash
cd ui
npm install
npm run dev
```

Configure one chat provider in the root `.env`. Neither capture needs embeddings.
Open http://localhost:3000 and use the scenario selector; CLI commands alone do not
populate the browser UI.

1. **503 recovery:** choose **Controlled ReAct: 503 -> success**, select **Run**,
   and wait for completion. Capture the scenario name, decision/verdict, graph,
   and event log containing both `503 temporarily unavailable` results and the
   later success. Save the real capture as `images/controlled-503-recovery.png`.
2. **401 failure:** choose **Controlled ReAct: auth failure**, select **Run**,
   and wait for `STOP`. Capture the selected scenario, first 401 result, stop
   reason/decision, and red graph together. The demo stops on the first observed
   permanent failure, so do not force additional failures. Save the real capture
   as `images/controlled-401-failure.png`.

Replace each visible screenshot placeholder with a `Frame` and an image with clear
alt text after the file exists. Use a viewport wide enough to show the event log;
avoid clipping the later success out of the recovery capture.

## Items to verify before release

- Provider model access: defaults come from source; no credential-backed run is
  implied by the offline checks. Gemini chat and embedding defaults are separate.
- Your agent's error handling must propagate `LoopGuardInterrupt`; a recorded
  `STOP` alone does not prove execution stopped.
- The published 0.2.1 wheel was inspected: Gemini support, controlled demo scenarios,
  and additional serialized decision display fields require the source checkout.
- Add the two captures and verify local search after Mintlify login if needed.
- Replace `[DOCS URL TO ADD AFTER DEPLOYMENT]` in the root README only when a real
  hosted URL exists; point the Docs button to it at the same time.

## Verification completed

- All 72 runtime and documentation automation tests passed with framework dependencies installed.
- The first-run example passed against both the source checkout and the actual
  published 0.2.1 wheel.
- A real LangGraph `ToolNode` run propagated the stop exception when configured
  to let tool errors escape.
- Mintlify full validation/build and internal link checks passed.
- All 20 pages rendered successfully in Chrome with no broken images. The
  introduction, getting-started page, demo scenarios, config reference, and all
  three concept pages, plus the two new practical guides, were checked at a 390px
  viewport without document overflow.
- Mintlify found alt text on every image. Brand colors meet the check's AA/minimum
  contrast thresholds; its optional AAA suggestion remains a warning.
- Python code blocks parsed, JSON/YAML samples loaded, and nine trace/example
  command variants returned the expected exit codes.
- The convergence example printed `0.58`; the equal-score and nonpositive-count
  limitations described in the concept guide were checked against the helper.
- The local backend's runtime and graph routes and the exact WebSocket run were
  checked without making provider calls.
- Workflow syntax passed actionlint. The external-link checker passed against
  the actual public links. Strict release checking correctly failed on the three
  remaining reader-facing publication placeholders.
- ChatGPT, Claude, Google AI Studio, Perplexity, and Grok context-menu actions
  were checked at desktop and mobile widths, including the generated page URLs
  and copy/Markdown alternatives.
  External chat navigation was intercepted during verification; retrieval of a
  published page still needs a deployment check.

Provider-backed scenarios were inspected in source, not run against paid APIs.
Local search requires Mintlify login and was not activated.

## Files in this change

```text
README.md
CONTRIBUTING.md
.lychee.toml
.github/workflows/
  documentation.yml
  docs-drift.yml
  external-links.yml
scripts/
  check_docs.py
  check_docs_drift.py
  verify_doc_examples.py
tests/
  test_docs_automation.py
examples/
  first_run.py
  langgraph_callback_demo.py
docs/
  .mintignore
  README.md
  docs.json
  robots.txt
  index.mdx
  getting-started.mdx
  how-it-works.mdx
  detection.mdx
  troubleshooting.mdx
  concepts/
    agents-and-loops.mdx
    tests-traces-and-evals.mdx
    trajectory-and-convergence.mdx
  guides/
    integration.mdx
    configuration.mdx
    saved-traces.mdx
    regression-checks.mdx
  demo/
    scenarios.mdx
    running-locally.mdx
  development/
    evaluation.mdx
    status.mdx
    contributing.mdx
  reference/
    python.mdx
    cli.mdx
    configuration.mdx
  images/
    stuck-agent.png
    exact-loop.png
    semantic-loop.png
    healthy-calculation.png
```

## Publish later

No hosting setup or deployment is part of this change. After content review and
approval to publish:

1. Connect `mahimathacker/loopguard` to your Mintlify project.
2. Choose the reviewed branch and set the documentation/content path to `/docs`.
3. Preview the build, then publish using the Mintlify dashboard.
4. Add the resulting public URL to the root README.

See [Mintlify monorepo setup](https://www.mintlify.com/docs/deploy/monorepo) for the
content-path setting. Do not invent a `loopguard.mintlify.app` address before the
project has a confirmed domain.
