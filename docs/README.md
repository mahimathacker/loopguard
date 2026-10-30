# LoopGuard documentation

The published site is [loopguard.mintlify.site](https://loopguard.mintlify.site/).
Mintlify builds from this repository's `docs/` folder on `master`. Update the SDK
and its documentation in the same pull request when behavior changes.

## Preview

Use Node.js 24 and the Mintlify version pinned in CI. From the repository root:

```bash
cd docs
npx --yes mint@4.2.1178 dev --port 3001 --no-open
```

Open http://localhost:3001. Port 3000 remains available for the demo UI.

## Validate

With the framework dependencies installed, run from the repository root:

```bash
python -m unittest discover -v
python scripts/check_docs.py --release
python scripts/verify_doc_examples.py
cd docs
npx --yes mint@4.2.1178 validate
npx --yes mint@4.2.1178 broken-links --check-anchors
```

See [Contributing](../CONTRIBUTING.md) for dependency installation and CI details.
The strict documentation check rejects unresolved publication placeholders.

## Documentation maintenance

- Pull requests and pushes run Python tests, documentation validation, internal
  link and image checks, and executable callback examples.
- The drift workflow flags developer-facing changes without related documentation
  changes for human review. It does not rewrite documentation.
- The weekly external-link workflow checks public HTTP URLs with retries and
  reports failures in the job summary. It can also run manually.

The workflows are in `.github/workflows/`. Required-check settings in GitHub
control whether a failing check blocks merging. See
[documentation maintenance](development/contributing.mdx) for the commands,
workflow behavior, and failure investigation.

## Publishing and discovery

Mintlify's Git source must be `mahimathacker/loopguard`, branch `master`, content
directory `docs`. After changes merge, verify the deployment and public pages.
Use local preview and CI for review; hosted PR previews depend on the Mintlify plan.

The Docs button and hosted link in the root README, the canonical URL in
`docs.json`, and the sitemap directive in `robots.txt` use
`https://loopguard.mintlify.site`. Keep them aligned if the domain changes.

Mintlify generates `llms.txt`, `llms-full.txt`, Markdown page exports, and the
sitemap from the published content. Keep agent guidance in
`docs.json` under `markdown.instructions` rather than maintaining separate copies.
After deploying, verify:

- [Documentation index](https://loopguard.mintlify.site/llms.txt).
- [Full documentation export](https://loopguard.mintlify.site/llms-full.txt).
- [Getting-started Markdown](https://loopguard.mintlify.site/getting-started.md).
- [Sitemap](https://loopguard.mintlify.site/sitemap.xml).
- [Crawler rules](https://loopguard.mintlify.site/robots.txt).

The exported documentation links must use the current public domain.

## Screenshots

The screenshots are real demo captures from `ui/public/`:

| Source | Documentation asset | Example |
| --- | --- | --- |
| `scriptedtool.png` | `images/exact-loop.png` | Repeated identical calls with empty results. |
| `scriptedopenai.png` | `images/semantic-loop.png` | Repeated intent across different search phrases. |
| `realagentmath.png` | `images/healthy-calculation.png` | A successful calculation. |
| `realagentloop.png` | `images/stuck-agent.png` | An interrupted search for an impossible goal. |

Keep captions tied to the behavior shown. Add descriptive alt text when replacing
an image, and verify it in local preview.

## Editing pages

Every public page belongs in `docs.json`. Use root-relative links such as
`/guides/integration` in MDX. Keep source links in repository Markdown relative
to the file containing them. `.mintignore` excludes this tooling guide from the
published navigation.
