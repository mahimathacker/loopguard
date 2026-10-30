import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_docs import check_repository, heading_ids
from scripts.check_docs_drift import review_changes


class DocumentationChecksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        (self.root / "README.md").write_text("# Project\n\n[Docs](docs/index.mdx)\n", encoding="utf-8")
        (self.docs / "docs.json").write_text(json.dumps({
            "navigation": {"groups": [{"group": "Start", "pages": ["index"]}]}
        }), encoding="utf-8")
        self.write_page("## First step\n\nA working page.\n")

    def write_page(self, body):
        (self.docs / "index.mdx").write_text(
            "---\ntitle: Introduction\ndescription: Get started.\n---\n\n" + body,
            encoding="utf-8",
        )

    def messages(self, *, release=False):
        return [item.message for item in check_repository(self.root, release=release)]

    def test_valid_site_and_readme_pass(self):
        self.assertEqual(self.messages(), [])

    def test_navigation_to_missing_page_fails(self):
        (self.docs / "index.mdx").unlink()
        self.assertTrue(any("Navigation page does not exist" in message for message in self.messages()))

    def test_public_page_missing_from_navigation_fails(self):
        (self.docs / "extra.mdx").write_text("---\ntitle: Extra\ndescription: Extra page.\n---\n", encoding="utf-8")
        self.assertTrue(any("Add this public page" in message for message in self.messages()))

    def test_ignored_maintainer_readme_does_not_need_navigation(self):
        (self.docs / ".mintignore").write_text("README.md\n", encoding="utf-8")
        (self.docs / "README.md").write_text("# Maintainers\n", encoding="utf-8")
        self.assertEqual(self.messages(), [])

    def test_duplicate_navigation_is_reported(self):
        path = self.docs / "docs.json"
        value = json.loads(path.read_text())
        value["navigation"]["groups"][0]["pages"].append("index")
        path.write_text(json.dumps(value), encoding="utf-8")
        self.assertTrue(any("Duplicate navigation" in message for message in self.messages()))

    def test_root_relative_links_and_heading_fragments_pass(self):
        self.write_page("## First step\n\n[Step](/index#first-step)\n")
        self.assertEqual(self.messages(), [])

    def test_missing_heading_fails(self):
        self.write_page("[Wrong heading](/index#missing)\n")
        self.assertTrue(any("Missing heading" in message for message in self.messages()))

    def test_missing_image_fails(self):
        self.write_page('<img src="/images/missing.png" alt="A real capture" />\n')
        self.assertTrue(any("Missing local link or image" in message for message in self.messages()))

    def test_images_need_alt_text(self):
        (self.docs / "capture.png").write_bytes(b"image fixture")
        for image in ('![](/capture.png)\n', '<img src="/capture.png" />\n'):
            with self.subTest(image=image):
                self.write_page(image)
                self.assertTrue(any("image alt text" in message for message in self.messages()))

    def test_valid_image_and_markdown_export_link_pass(self):
        (self.docs / "capture.png").write_bytes(b"image fixture")
        self.write_page('![Capture](/capture.png)\n\n[Markdown](/index.md)\n')
        self.assertEqual(self.messages(), [])

    def test_invalid_python_and_json_are_reported(self):
        self.write_page('```python\nif:\n```\n\n```json\n{"broken":}\n```\n')
        messages = self.messages()
        self.assertTrue(any("Invalid python" in message for message in messages))
        self.assertTrue(any("Invalid json" in message for message in messages))

    def test_code_fences_need_language_and_closing_marker(self):
        self.write_page("```\nUnclosed example\n")
        self.assertTrue(any("language" in message for message in self.messages()))
        self.assertTrue(any("Close the code fence" in message for message in self.messages()))

    def test_placeholders_warn_on_pr_and_fail_for_release(self):
        self.write_page("# [SCREENSHOT NEEDED: EXAMPLE]\n")
        normal = check_repository(self.root)
        strict = check_repository(self.root, release=True)
        self.assertEqual([item.severity for item in normal], ["warning"])
        self.assertEqual([item.severity for item in strict], ["error"])

    def test_documented_marker_inside_code_does_not_fail_release(self):
        self.write_page("```text\n[SCREENSHOT NEEDED: EXAMPLE]\n```\n")
        self.assertEqual(self.messages(release=True), [])

    def test_duplicate_heading_anchor_matches_numbered_slug(self):
        self.assertEqual(heading_ids("## Steps\n\n## Steps\n"), {"steps", "steps-1"})

    def test_link_cannot_escape_repository(self):
        self.write_page("[Outside](../../outside.md)\n")
        self.assertTrue(any("Missing local link" in message for message in self.messages()))

    def test_frontmatter_requires_description(self):
        (self.docs / "index.mdx").write_text("---\ntitle: Intro\n---\n", encoding="utf-8")
        self.assertTrue(any("frontmatter" in message for message in self.messages()))


class DocumentationDriftTests(unittest.TestCase):
    def test_cli_change_without_docs_needs_review(self):
        surfaces, docs_changed = review_changes(["loopguard/check.py"])
        self.assertIn("CLI commands", surfaces)
        self.assertFalse(docs_changed)

    def test_version_and_provider_changes_are_watched(self):
        surfaces, _ = review_changes(["pyproject.toml", ".env.example"])
        self.assertIn("Package version, Python support, and extras", surfaces)
        self.assertIn("Scenarios and providers", surfaces)

    def test_backend_and_frontend_contract_changes_are_watched(self):
        surfaces, docs_changed = review_changes(["server.py", "ui/app/types.ts"])
        self.assertIn("Demo server and frontend integration", surfaces)
        self.assertFalse(docs_changed)

    def test_docs_change_is_recorded_without_claiming_coverage(self):
        surfaces, docs_changed = review_changes(["loopguard/guard.py", "docs/guides/configuration.mdx"])
        self.assertTrue(surfaces)
        self.assertTrue(docs_changed)

    def test_tests_and_images_do_not_count_as_documentation_updates(self):
        _, docs_changed = review_changes(["tests/test_loopguard.py", "docs/images/capture.png"])
        self.assertFalse(docs_changed)

    def test_docs_only_change_has_no_api_warning(self):
        surfaces, docs_changed = review_changes(["README.md"])
        self.assertEqual(surfaces, {})
        self.assertTrue(docs_changed)


if __name__ == "__main__":
    unittest.main()
