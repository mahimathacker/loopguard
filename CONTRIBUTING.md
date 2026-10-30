# Contributing to LoopGuard

For a bug report, include the smallest reproducible trace, the settings you used,
and the decision you expected. Remove credentials and private data from examples.

## Test a change

From a source checkout in your Python environment:

```bash
python -m pip install -e ".[langgraph]"
python -m unittest discover -v
python scripts/check_docs.py
python scripts/verify_doc_examples.py
```

For a documentation change, use Node.js 24 and the tested Mintlify version:

```bash
npm install --global mint@4.2.1178
cd docs
mint validate
mint broken-links --check-anchors
```

## Keeping the docs accurate

Pull requests check the tests, documentation structure, internal links, images,
and runnable callback examples. A weekly workflow checks external URLs. Changes
to developer-facing Python APIs, CLI commands, configuration, scenarios, providers,
or package metadata should include the relevant README or docs updates.

The drift check is a file-based reminder for human review. It does not prove that
a documentation update covers a code change, and it does not rewrite content.

Unresolved publication placeholders warn during normal checks. Before a release,
run `python scripts/check_docs.py --release` to fail if any remain in reader-facing
content. Version-tag builds and the workflow's release-readiness option run the
same strict check. The current missing hosted URL and two screenshots must be
resolved before that check can pass.

Read the [contributing guide](docs/development/contributing.mdx) for CI behavior,
or [add a trace regression case](docs/guides/regression-checks.mdx) to document
expected healthy and stuck behavior. See [the maintainer handoff](docs/README.md)
for local preview and later publishing.
