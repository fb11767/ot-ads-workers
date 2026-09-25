#!/usr/bin/env python3
"""Tests hors réseau du QA : schéma, incertitude, taille, flou, index."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qa  # noqa: E402
import run_queue  # noqa: E402


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _text_image(lines: list[str], size: tuple[int, int] = (320, 400)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 28)
    y = 40
    for line in lines:
        draw.text((24, y), line, fill="black", font=font)
        y += 48
    return image


class ChecklistTests(unittest.TestCase):
    def test_fifteen_items(self) -> None:
        text = qa.load_checklist()
        self.assertEqual(qa.ITEM_RE.findall(text), [str(n) for n in range(1, 16)])


class ReportTests(unittest.TestCase):
    def test_pass_json(self) -> None:
        report = qa.parse_model_output(
            '```json\n{"country":"de","src_ad":"a","verdict":"PASS","uncertain":false,"fails":[]}\n```',
            "de",
            "a",
        )
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["fails"], [])
        self.assertEqual(set(report), {"country", "src_ad", "verdict", "fails"})

    def test_exact_line_fail(self) -> None:
        report = qa.parse_model_output(
            json.dumps(
                {
                    "verdict": "FAIL",
                    "fails": [{"item": "4", "détail": "wrote fur", "correction": "Render für."}],
                }
            ),
            "de",
            "ad",
        )
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["fails"][0]["item"], 4)
        self.assertIn("fur", report["fails"][0]["detail"])

    def test_garbage_is_fail(self) -> None:
        report = qa.parse_model_output("not json", "it", "x")
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["fails"][0]["item"], "model")

    def test_uncertainty_never_passes(self) -> None:
        report = qa.parse_model_output(
            '{"verdict":"PASS","uncertain":true,"fails":[]}',
            "de",
            "a",
        )
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["fails"][0]["item"], "model")
        report = qa.parse_model_output(
            '{"verdict":"PASS","fails":[]} but I am not sure about the accent',
            "de",
            "a",
        )
        self.assertEqual(report["verdict"], "FAIL")

    def test_invented_criterion_is_dropped(self) -> None:
        report = qa.parse_model_output(
            json.dumps({"verdict": "FAIL", "fails": [{"item": 99, "detail": "logo style", "correction": "change it"}]}),
            "de",
            "a",
        )
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["fails"], [])

    def test_pass_with_real_fails_stays_fail(self) -> None:
        report = qa.parse_model_output(
            json.dumps(
                {"verdict": "PASS", "fails": [{"item": 10, "detail": "wrong price", "correction": "Use 29,90 €."}]}
            ),
            "de",
            "a",
        )
        self.assertEqual(report["verdict"], "FAIL")
        self.assertEqual(report["fails"][0]["item"], 10)

    def test_merge_keeps_deterministic_fail(self) -> None:
        model = qa.parse_model_output('{"verdict":"PASS","fails":[]}', "de", "a")
        merged = qa.merge_fails(
            [{"item": 2, "detail": "800x600", "correction": "Use 320x400."}],
            model,
        )
        self.assertEqual(merged["verdict"], "FAIL")
        self.assertEqual(merged["fails"][0]["item"], 2)


class DeterministicTests(unittest.TestCase):
    def test_size_mismatch(self) -> None:
        image = _text_image(["bis zu 73 % Rabatt"], (300, 300))
        source = _text_image(["up to 73% off"], (320, 400))
        brief = qa.normalize_brief(
            {
                "country": "de",
                "src_ad": "a",
                "expected_lines": ["bis zu 73 % Rabatt"],
                "target_size": [320, 400],
                "prices": ["29,90 €"],
                "percentage": 73,
            }
        )
        fails = qa.deterministic_fails(image, brief, source)
        self.assertIn(2, [fail["item"] for fail in fails])

    def test_sharp_copy_is_not_blurry(self) -> None:
        image = _text_image(["bis zu 73 % Rabatt", "29,90 €"])
        brief = qa.normalize_brief({"country": "de", "src_ad": "a", "target_size": list(image.size)})
        fails = qa.deterministic_fails(image, brief, image)
        self.assertNotIn(15, [fail["item"] for fail in fails])

    def test_heavy_blur_fails(self) -> None:
        sharp = Image.new("RGB", (120, 120), "white")
        pixels = sharp.load()
        for y in range(120):
            for x in range(120):
                if (x // 3 + y // 3) % 2 == 0:
                    pixels[x, y] = (0, 0, 0)
        blurred = sharp.filter(ImageFilter.GaussianBlur(radius=8))
        brief = qa.normalize_brief({"country": "de", "src_ad": "a", "target_size": [120, 120]})
        fails = qa.deterministic_fails(blurred, brief, sharp)
        self.assertIn(15, [fail["item"] for fail in fails])

    def test_invalid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jpg"
            path.write_bytes(b"this is not an image at all, really")
            with self.assertRaises(ValueError):
                qa.open_rgb(path)


class LoopHelpersTests(unittest.TestCase):
    def test_best_attempt_prefers_pass_then_fewer_fails(self) -> None:
        passed = {"verdict": "PASS", "fails": []}
        one = {"verdict": "FAIL", "fails": [{"item": 4}]}
        three = {"verdict": "FAIL", "fails": [{}, {}, {}]}
        self.assertLess(qa.attempt_rank(passed, 10), qa.attempt_rank(one, 500))
        self.assertLess(qa.attempt_rank(one, 10), qa.attempt_rank(three, 10))
        self.assertLess(qa.attempt_rank(one, 50), qa.attempt_rank(one, 10))

    def test_corrections_are_added_to_the_next_prompt(self) -> None:
        brief = qa.brief_from_fix_item(
            {
                "country": "de",
                "src_ad": "ad",
                "instruction": "Remplacer le titre par « bis zu 73 % Rabatt ».",
                "target_size": [320, 400],
                "prices": ["29,90 €"],
                "percentage": 73,
            }
        )
        self.assertFalse(brief["expected_lines_exhaustive"])
        self.assertEqual(brief["expected_lines"], ["bis zu 73 % Rabatt"])
        text = qa.generation_addon(
            brief,
            [{"item": 4, "detail": "accent missing", "correction": "Write Rabatt exactly."}],
        )
        self.assertIn("bis zu 73 % Rabatt", text)
        self.assertIn("Item 4", text)
        self.assertIn("29,90 €", text)

    def test_prompt_contains_full_checklist(self) -> None:
        checklist = qa.load_checklist()
        _system, user = qa.build_vision_prompt(
            checklist,
            qa.normalize_brief({"country": "de", "src_ad": "a", "expected_lines": ["Rabatt"], "target_size": [10, 10]}),
        )
        self.assertIn("1. Mise en page", user)
        self.assertIn("15. Net", user)
        self.assertIn("Rabatt", user)
        self.assertIn("exact", user.lower())

    def test_indices(self) -> None:
        self.assertEqual(qa.select_indices(10, ""), list(range(10)))
        self.assertEqual(qa.select_indices(10, "0-2"), [0, 1, 2])
        self.assertEqual(qa.select_indices(10, "0:3"), [0, 1, 2])
        self.assertEqual(qa.select_indices(10, "1,4,8"), [1, 4, 8])
        with self.assertRaises(ValueError):
            qa.select_indices(3, "0-5")

    def test_folder_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "de-one"
            item.mkdir()
            _text_image(["Rabatt"]).save(item / "adapted.jpg", quality=90)
            _text_image(["discount"]).save(item / "ca_source.jpg", quality=90)
            (item / "brief.json").write_text(
                json.dumps(
                    {
                        "country": "de",
                        "src_ad": "one",
                        "expected_lines": ["Rabatt"],
                        "prices": ["29,90 €"],
                        "percentage": 73,
                        "target_size": [320, 400],
                        "pitfalls": [],
                    }
                ),
                encoding="utf-8",
            )
            jobs = qa.discover_folder(root)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["src_ad"], "one")
            self.assertEqual(jobs[0]["brief"]["percentage"], 73)


class CliDryRunTests(unittest.TestCase):
    def test_fix_dry_run_index_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "in"
            country = inputs / "de"
            country.mkdir(parents=True)
            current = country / "current.jpg"
            source = country / "ca.jpg"
            _text_image(["rabais"]).save(current, quality=90)
            _text_image(["discount"]).save(source, quality=90)
            queue = [
                {
                    "country": "de",
                    "src_ad": f"ad{i}",
                    "instruction": "Corriger le mot « Rabatt ».",
                    "current": f"de/current.jpg",
                    "ca_source": "de/ca.jpg",
                    "target_size": [320, 400],
                    "expected_lines": ["Rabatt"],
                    "prices": ["29,90 €"],
                    "percentage": 73,
                }
                for i in range(4)
            ]
            queue_path = root / "fix_queue.json"
            queue_path.write_text(json.dumps(queue), encoding="utf-8")
            artifacts = root / "out"
            code = run_queue.main(
                [
                    "--mode",
                    "fix",
                    "--queue",
                    str(queue_path),
                    "--inputs",
                    str(inputs),
                    "--artifacts",
                    str(artifacts),
                    "--indices",
                    "1-2",
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads((artifacts / "report_1-2.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["items"]), 2)
            self.assertEqual(report["items"][0]["src_ad"], "ad1")


if __name__ == "__main__":
    unittest.main()
