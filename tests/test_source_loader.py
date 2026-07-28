"""Tests for corpus file discovery and offset-preserving chunking."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import warnings

from src.source_loader import SourceLoader


class SourceLoaderTest(TestCase):
    """Verify relevant text files are included without binary ingestion."""

    def test_includes_supported_source_document_and_build_files(self) -> None:
        """Discover relevant languages and named build files."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                ".gitignore": "*.generated",
                "CMakeLists.txt": "set(CUDA_VERSION 12.8)",
                "CODEOWNERS": "* @maintainer",
                "Dockerfile.cpu": "FROM python:3.12",
                "config.yaml": "engine: cpu",
                "docs.md": "# Documentation",
                "fix.patch": "diff --git a/old b/new",
                "kernel.cu": "void kernel() {}",
                "module.py": "def function():\n    return 42",
                "runtime.env": "DEVICE=cpu",
                "settings.json": '{"enabled": true}',
            }
            for name, content in expected.items():
                (root / name).write_text(content)
            (root / "image.png").write_bytes(b"\x89PNG\r\n")
            (root / "archive.bin").write_bytes(b"\x00\xff")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("repository metadata")

            sources = SourceLoader(root).getSources(chunk_size=2000)
            indexed_paths = {source.file_path.name for source in sources}

            self.assertEqual(indexed_paths, set(expected))
            for source in sources:
                original = expected[source.file_path.name]
                self.assertEqual(
                    source.content,
                    original[
                        source.first_character_index:
                        source.last_character_index
                    ],
                )

    def test_records_correct_offsets_for_repeated_content(self) -> None:
        """Use splitter offsets instead of locating repeated text globally."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = "repeated block\n\nrepeated block\n\nfinal block"
            path = root / "repeated.md"
            path.write_text(content)

            sources = SourceLoader(root).getDocs(chunk_size=16)

            self.assertGreater(len(sources), 1)
            self.assertEqual(
                [source.first_character_index for source in sources],
                sorted(source.first_character_index for source in sources),
            )
            for source in sources:
                self.assertEqual(
                    source.content,
                    content[
                        source.first_character_index:
                        source.last_character_index
                    ],
                )

    def test_skips_unreadable_allowlisted_text_file(self) -> None:
        """Warn and continue when an allowlisted file is not UTF-8 text."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "invalid.txt").write_bytes(b"\xff\xfe")
            (root / "valid.md").write_text("valid")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sources = SourceLoader(root).getSources(chunk_size=2000)

            self.assertEqual([source.content for source in sources], ["valid"])
            self.assertEqual(len(caught), 1)
            self.assertIn("Skipping unreadable source", str(caught[0].message))

    def test_returns_paths_in_deterministic_order(self) -> None:
        """Sort source paths so persisted indexes are reproducible."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z.py").write_text("z = 1")
            (root / "a.py").write_text("a = 1")

            sources = SourceLoader(root).getCode(chunk_size=2000)

            self.assertEqual(
                [source.file_path.name for source in sources],
                ["a.py", "z.py"],
            )
