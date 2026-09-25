#!/usr/bin/env python3
"""Contrôle qualité des visuels Ortho Terre.

Les items déterministes (fichier, taille, flou) sont décidés dans ce module.
Le reste de la checklist est jugé par un modèle vision hébergé sur Replicate,
appelé uniquement avec REPLICATE_API_TOKEN. Toute erreur ou incertitude du
modèle produit un verdict FAIL, jamais PASS.

Le rapport public est exactement :
{country, src_ad, verdict, fails:[{item, detail, correction}]}.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps

CHECKLIST_PATH = Path(__file__).resolve().parents[1] / "QA_CHECKLIST.md"
QA_MODEL_DEFAULT = "openai/gpt-5"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
ADAPTED_STEMS = ("adapted", "image", "out", "final", "current")
CA_STEMS = ("ca_source", "ca", "source", "reference")
# Prix publics Replicate, septembre 2026, pour l'estimation du batch.
QA_TOKEN_USD_PER_MILLION = {
    "openai/gpt-5": (1.25, 10.0),
    "openai/gpt-5-mini": (0.25, 2.0),
    "openai/gpt-4.1": (2.0, 8.0),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4o": (2.50, 10.0),
}
IMAGE_USD = {
    "low": 0.012,
    "medium": 0.047,
    "high": 0.128,
    "xhigh": 0.20,
    "max": 0.30,
    "auto": 0.25,
}
UNCERTAIN_RE = re.compile(
    r"\b(uncertain|unsure|not sure|cannot tell|can't tell|cannot read|can't read|"
    r"illegible|hard to read|difficult to read|incertain|incertaine|pas s[uû]r|"
    r"illisible|peut-être|impossible à lire)\b",
    re.IGNORECASE,
)
ITEM_RE = re.compile(r"(?m)^(\d+)\.\s")


def load_checklist(path: Path | None = None) -> str:
    checklist = path or CHECKLIST_PATH
    if not checklist.is_file():
        raise FileNotFoundError(f"checklist introuvable : {checklist}")
    text = checklist.read_text(encoding="utf-8")
    found = [int(number) for number in ITEM_RE.findall(text)]
    if found != list(range(1, 16)):
        raise ValueError(f"la checklist doit contenir les items 1 à 15 dans l'ordre, trouvé {found}")
    return text


def sharpness_score(image: Image.Image) -> float:
    """Variance du laplacien : plus la valeur est basse, plus l'image est floue."""
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype="float32")
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(lap.var())


def _as_lines(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _target_size(value: Any) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and "x" in value.lower():
        left, right = re.split(r"[xX]", value, maxsplit=1)
        value = [left, right]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
        if width < 1 or height < 1:
            raise ValueError("target_size invalide")
        return width, height
    raise ValueError("target_size doit être [largeur, hauteur]")


def normalize_brief(data: dict[str, Any], country: str = "", src_ad: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("le brief doit être un objet JSON")

    def pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in data and data[key] not in (None, ""):
                return data[key]
        return default

    exhaustive = pick("expected_lines_exhaustive", "exhaustive", default=None)
    return {
        "country": str(pick("country", default=country) or country or "").strip(),
        "src_ad": str(pick("src_ad", "id", default=src_ad) or src_ad or "").strip(),
        "expected_lines": _as_lines(pick("expected_lines", "expected_text", "text_lines", "lignes", "lines")),
        "prices": _as_lines(pick("prices", "prix")),
        "percentage": pick("percentage", "percent", "pourcentage", "pct"),
        "target_size": _target_size(pick("target_size", "taille")),
        "pitfalls": _as_lines(pick("pitfalls", "known_pitfalls", "pieges", "pièges")),
        "expected_lines_exhaustive": True if exhaustive is None else bool(exhaustive),
    }


def brief_from_fix_item(item: dict[str, Any]) -> dict[str, Any]:
    """Construit un brief QA depuis un item de fix_queue.

    Les guillemets « » de l'instruction sont des lignes obligatoires, pas
    forcément l'intégralité du texte. Un champ expected_lines fourni par
    l'item est exhaustif, sauf expected_lines_exhaustive à false.
    """
    instruction = str(item.get("instruction") or "").strip()
    quoted = [part.strip() for part in re.findall(r"«\s*(.*?)\s*»", instruction)]
    explicit = item.get("expected_lines")
    if explicit:
        lines = explicit
        exhaustive = item.get("expected_lines_exhaustive", True)
    else:
        lines = quoted
        exhaustive = False
    payload: dict[str, Any] = {
        "country": item.get("country") or "",
        "src_ad": item.get("src_ad") or "",
        "expected_lines": lines,
        "expected_lines_exhaustive": exhaustive,
        "prices": item.get("prices") or [],
        "percentage": item.get("percentage", item.get("percent")),
        "target_size": item.get("target_size"),
        "pitfalls": item.get("pitfalls") or ([instruction] if instruction else []),
    }
    nested = item.get("brief")
    if isinstance(nested, dict):
        merged = dict(nested)
        for key, value in payload.items():
            merged.setdefault(key, value)
        payload = merged
    return normalize_brief(payload, str(item.get("country") or ""), str(item.get("src_ad") or ""))


def public_report(country: str, src_ad: str, fails: list[dict[str, Any]]) -> dict[str, Any]:
    cleaned: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for fail in fails:
        item = fail.get("item")
        detail = str(fail.get("detail") or "").strip()
        correction = str(fail.get("correction") or "").strip() or detail or "Regenerate the image."
        if not detail:
            detail = "QA failure without a description."
        key = item
        if key in seen:
            current = next(entry for entry in cleaned if entry["item"] == key)
            if detail not in current["detail"]:
                current["detail"] += " | " + detail
            if len(correction) > len(current["correction"]):
                current["correction"] = correction
            continue
        seen.add(key)
        cleaned.append({"item": item, "detail": detail, "correction": correction})
    verdict = "PASS" if not cleaned else "FAIL"
    return {"country": country, "src_ad": src_ad, "verdict": verdict, "fails": cleaned}


def model_fail(country: str, src_ad: str, detail: str, correction: str, extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    fails = list(extra or [])
    fails.append(
        {
            "item": "model",
            "detail": detail.strip() or "QA model error",
            "correction": correction.strip() or "Regenerate the whole image and run QA again.",
        }
    )
    return public_report(country, src_ad, fails)


def deterministic_fails(
    image: Image.Image,
    brief: dict[str, Any],
    source: Image.Image | None = None,
) -> list[dict[str, Any]]:
    """Taille, validité déjà assurée par l'ouverture, et heuristique de flou."""
    fails: list[dict[str, Any]] = []
    width, height = image.size
    if width < 2 or height < 2:
        fails.append(
            {
                "item": 2,
                "detail": f"Image is {width}x{height}, which is not a usable final size.",
                "correction": "Regenerate a valid image at the brief target_size, with nothing cropped.",
            }
        )
        return fails
    target = brief.get("target_size")
    if target is None:
        fails.append(
            {
                "item": 2,
                "detail": "target_size is missing from the brief, so the final size cannot be certified.",
                "correction": "Set target_size to the CA source size and regenerate at that exact size.",
            }
        )
    elif (width, height) != tuple(target):
        fails.append(
            {
                "item": 2,
                "detail": f"Final size is {width}x{height}, expected {target[0]}x{target[1]}.",
                "correction": (
                    f"Output the whole image at exactly {target[0]}x{target[1]} pixels. "
                    "Do not crop text, the product, or banners."
                ),
            }
        )
    score = sharpness_score(image)
    source_score = None
    if source is not None:
        reference = source.convert("RGB")
        if reference.size != image.size:
            reference = reference.resize(image.size, Image.Resampling.LANCZOS)
        source_score = sharpness_score(reference)
    blur = score < 8.0 or (source_score is not None and source_score > 80.0 and score < source_score * 0.30)
    if blur:
        compared = f" (source {source_score:.0f})" if source_score is not None else ""
        fails.append(
            {
                "item": 15,
                "detail": f"Image looks blurry (sharpness {score:.0f}{compared}).",
                "correction": "Regenerate a sharp image with crisp text, no stretched edges and no artifacts.",
            }
        )
    return fails


def _fail_entry(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    item = raw.get("item", raw.get("numero", raw.get("n")))
    if isinstance(item, str) and item.isdigit():
        item = int(item)
    if item == "model":
        pass
    elif isinstance(item, int) and 1 <= item <= 15:
        pass
    else:
        return None
    detail = raw.get("detail", raw.get("détail", raw.get("dettail", "")))
    correction = raw.get("correction", raw.get("fix", ""))
    detail = str(detail or "").strip()
    correction = str(correction or "").strip()
    if not detail and not correction:
        return None
    return {
        "item": item,
        "detail": detail or correction,
        "correction": correction or detail,
    }


def parse_model_output(text: str, country: str, src_ad: str) -> dict[str, Any]:
    """Transforme la réponse du modèle en rapport. Invalide ou incertain => FAIL."""
    raw_text = text or ""
    stripped = raw_text.strip()
    if not stripped:
        return model_fail(country, src_ad, "QA model returned an empty response.", "Regenerate the image; QA could not certify it.")
    fenced = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced)
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start < 0 or end <= start:
        return model_fail(
            country,
            src_ad,
            "QA model did not return JSON.",
            "Regenerate the image; QA could not certify it.",
        )
    try:
        payload = json.loads(fenced[start : end + 1])
    except json.JSONDecodeError:
        return model_fail(
            country,
            src_ad,
            "QA model returned invalid JSON.",
            "Regenerate the image; QA could not certify it.",
        )
    if not isinstance(payload, dict):
        return model_fail(country, src_ad, "QA model JSON was not an object.", "Regenerate the image; QA could not certify it.")

    fails: list[dict[str, Any]] = []
    raw_fails = payload.get("fails", payload.get("failures", []))
    if isinstance(raw_fails, dict):
        raw_fails = [raw_fails]
    if not isinstance(raw_fails, list):
        return model_fail(country, src_ad, "QA model fails field was not a list.", "Regenerate the image; QA could not certify it.")
    for raw in raw_fails:
        entry = _fail_entry(raw)
        if entry is not None:
            fails.append(entry)

    verdict = str(payload.get("verdict") or "").strip().upper()
    uncertain = payload.get("uncertain")
    uncertain_flag = uncertain is True or str(uncertain).strip().lower() in {"true", "yes", "1"}
    outside = fenced[:start] + fenced[end + 1 :]
    if UNCERTAIN_RE.search(outside):
        uncertain_flag = True
    invented_only = bool(raw_fails) and not fails
    if verdict not in {"PASS", "FAIL"} or uncertain_flag:
        fails.append(
            {
                "item": "model",
                "detail": "QA model was uncertain or did not return a strict PASS/FAIL verdict.",
                "correction": "Regenerate the whole image with exact text, then run QA again. Do not treat an uncertain read as a pass.",
            }
        )
    elif verdict == "FAIL" and not fails and not invented_only:
        fails.append(
            {
                "item": "model",
                "detail": "QA model returned FAIL without a checklist item.",
                "correction": "Regenerate the image and address every checklist miss explicitly.",
            }
        )
    report_country = str(payload.get("country") or country).strip() or country
    report_src = str(payload.get("src_ad") or src_ad).strip() or src_ad
    return public_report(report_country, report_src, fails)


def merge_fails(deterministic: list[dict[str, Any]], model_report: dict[str, Any]) -> dict[str, Any]:
    """Le code gagne sur les faits mesurés. Un PASS modèle ne peut pas effacer un FAIL déterministe."""
    combined: list[dict[str, Any]] = []
    index: dict[Any, dict[str, Any]] = {}
    for fail in list(deterministic) + list(model_report.get("fails") or []):
        item = fail["item"]
        if item not in index:
            copied = {"item": item, "detail": fail["detail"], "correction": fail["correction"]}
            index[item] = copied
            combined.append(copied)
            continue
        current = index[item]
        if fail["detail"] and fail["detail"] not in current["detail"]:
            current["detail"] = current["detail"] + " | " + fail["detail"]
        if len(fail.get("correction") or "") > len(current.get("correction") or ""):
            current["correction"] = fail["correction"]
    return public_report(str(model_report.get("country") or ""), str(model_report.get("src_ad") or ""), combined)


def attempt_rank(report: dict[str, Any], sharpness: float) -> tuple[int, int, float]:
    """Plus petit = meilleur. Un PASS bat tout. Sinon moins d'échecs, puis plus de netteté."""
    fails = list(report.get("fails") or [])
    passed = report.get("verdict") == "PASS" and not fails
    return (0 if passed else 1, len(fails), -float(sharpness))


def corrections_text(fails: list[dict[str, Any]]) -> str:
    lines = [
        "The previous attempt FAILED strict QA. Apply every correction below on the whole image. Change nothing else."
    ]
    for fail in fails:
        lines.append(f"- Item {fail['item']}: {fail['detail']} Correction: {fail['correction']}")
    return "\n".join(lines)


def generation_addon(brief: dict[str, Any], corrections: list[dict[str, Any]] | None = None) -> str:
    parts: list[str] = []
    lines = brief.get("expected_lines") or []
    if lines:
        scope = (
            "These lines are the complete text of the ad. Do not add any other word."
            if brief.get("expected_lines_exhaustive")
            else "Each of these lines must appear exactly. Do not invent extra words."
        )
        rendered = "\n".join(f"- {line}" for line in lines)
        parts.append("Required text, same spelling, accents and punctuation:\n" + scope + "\n" + rendered)
    prices = brief.get("prices") or []
    if prices:
        parts.append("Exact local prices: " + "; ".join(str(price) for price in prices))
    if brief.get("percentage") is not None:
        parts.append(f"Exact percentage from the brief: {brief['percentage']}")
    pitfalls = brief.get("pitfalls") or []
    if pitfalls:
        parts.append("Known pitfalls:\n" + "\n".join(f"- {item}" for item in pitfalls))
    if corrections:
        parts.append(corrections_text(corrections))
    return "\n\n".join(parts)


def build_vision_prompt(checklist: str, brief: dict[str, Any]) -> tuple[str, str]:
    system = (
        "You are a strict visual QA inspector for Ortho Terre advertisement adaptations. "
        "You compare the adapted image against the Canadian source and against the brief. "
        "Return only one JSON object, with no markdown. "
        "Verdict PASS is allowed only when every checklist item passes and you are certain. "
        "If you are uncertain about any item, including a single letter or accent, verdict must be FAIL. "
        "Never guess a PASS. Do not invent criteria that are not in the checklist or the brief. "
        "Ignore defects that the checklist does not cover."
    )
    brief_json = json.dumps(
        {
            "country": brief.get("country"),
            "src_ad": brief.get("src_ad"),
            "expected_lines": brief.get("expected_lines") or [],
            "expected_lines_exhaustive": bool(brief.get("expected_lines_exhaustive")),
            "prices": brief.get("prices") or [],
            "percentage": brief.get("percentage"),
            "target_size": brief.get("target_size"),
            "pitfalls": brief.get("pitfalls") or [],
        },
        ensure_ascii=False,
        indent=2,
    )
    user = f"""Checklist (the only quality criteria, 15 items, one miss = FAIL, no soft pass):

{checklist.strip()}

Brief:
{brief_json}

Images:
- Image 1, the first image, is the adapted advertisement to judge.
- Image 2, the second image, is the Canadian source (CA). Compare layout, product, people and persona to it.
- If you instead receive one side-by-side image, LEFT is adapted and RIGHT is the CA source.

OCR rules, apply them to every text line:
- Read every visible text line on the adapted image. Preserve accents exactly (ä ö ü ß å æ ø é è à ç, and Greek).
- Compare each expected line to the image as an exact string: same words, same accents, same punctuation, no extra word, no missing word, no invented letter.
- If expected_lines_exhaustive is true, any word on the image that is not part of expected_lines is a failure (checklist item 5).
- If it is false, every expected line must still appear exactly, and the other visible lines must still satisfy the checklist.
- Prices must match the brief exactly, including currency and local format.
- The percentage must match the brief and checklist item 11.
- A line you cannot read with certainty is a FAIL, never a PASS.

Return this JSON object and nothing else:
{{
  "country": "{brief.get("country") or ""}",
  "src_ad": "{brief.get("src_ad") or ""}",
  "verdict": "PASS or FAIL",
  "uncertain": false,
  "fails": [{{"item": 4, "detail": "what is wrong", "correction": "imperative edit instruction"}}]
}}
item is an integer from 1 to 15. verdict is PASS only when fails is empty and uncertain is false.
Write detail and correction in English."""
    return system, user


def _save_rgb_jpeg(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=95, subsampling=0)


def _stitch(adapted: Image.Image, source: Image.Image, path: Path) -> None:
    left = adapted.convert("RGB")
    right = source.convert("RGB")
    height = max(left.height, right.height)
    if left.height != height:
        left = left.resize((max(1, round(left.width * height / left.height)), height), Image.Resampling.LANCZOS)
    if right.height != height:
        right = right.resize((max(1, round(right.width * height / right.height)), height), Image.Resampling.LANCZOS)
    banner = 36
    canvas = Image.new("RGB", (left.width + right.width, height + banner), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "LEFT = ADAPTED", fill="black")
    draw.text((left.width + 8, 8), "RIGHT = CA SOURCE", fill="black")
    canvas.paste(left, (0, banner))
    canvas.paste(right, (left.width, banner))
    canvas.save(path, format="JPEG", quality=95, subsampling=0)


def vision_payload(
    model: str,
    system: str,
    user: str,
    adapted: Path,
    source: Path,
    work: Path,
) -> tuple[dict[str, Any], list[Any]]:
    """Construit l'entrée Replicate. Les fichiers retournés doivent rester ouverts jusqu'à l'upload."""
    handles: list[Any] = []
    if model.startswith("anthropic/"):
        stitched = work / "side_by_side.jpg"
        with Image.open(adapted) as left, Image.open(source) as right:
            _stitch(left, right, stitched)
        handle = stitched.open("rb")
        handles.append(handle)
        note = "\n\nThe attached image is a side-by-side: LEFT is the adapted ad, RIGHT is the CA source."
        return (
            {
                "prompt": user + note,
                "system_prompt": system,
                "image": handle,
                "max_tokens": 2500,
                "max_image_resolution": 2,
            },
            handles,
        )
    first = adapted.open("rb")
    second = source.open("rb")
    handles.extend([first, second])
    payload: dict[str, Any] = {
        "prompt": user,
        "system_prompt": system,
        "image_input": [first, second],
        "max_completion_tokens": 2500,
    }
    if "gpt-5" in model or "/o1" in model or "/o3" in model or "/o4" in model:
        payload["reasoning_effort"] = "low"
        payload["verbosity"] = "low"
    else:
        payload["temperature"] = 0
    return payload, handles


def prediction_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    if isinstance(output, list):
        return "".join(prediction_text(part) for part in output)
    url = getattr(output, "url", None)
    if isinstance(url, str):
        return url
    read = getattr(output, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)
    return str(output)


def prediction_metrics(prediction: Any) -> dict[str, Any]:
    raw = getattr(prediction, "metrics", None) or {}
    if not isinstance(raw, dict):
        try:
            raw = dict(raw)
        except Exception:
            raw = {}

    def pick(*keys: str) -> Any:
        for key in keys:
            if raw.get(key) is not None:
                return raw[key]
        return None

    return {
        "predict_time": pick("predict_time", "total_time"),
        "input_tokens": pick("input_token_count", "input_tokens"),
        "output_tokens": pick("output_token_count", "output_tokens"),
    }


def open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise ValueError(f"fichier introuvable : {path}")
    if path.stat().st_size < 32:
        raise ValueError(f"fichier image vide ou trop petit : {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            transposed = ImageOps.exif_transpose(image) or image
            rgb = transposed.convert("RGB")
            rgb.load()
            return rgb
    except Exception as exc:
        raise ValueError(f"fichier image illisible : {path.name} ({exc.__class__.__name__})") from exc


def evaluate_image(
    adapted: Image.Image | Path,
    source: Image.Image | Path,
    brief: dict[str, Any],
    *,
    checklist: str,
    model: str,
    predict: Callable[[str, dict[str, Any]], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Juge une image. predict(model, payload) exécute l'appel Replicate et retourne la prédiction."""
    country = str(brief.get("country") or "")
    src_ad = str(brief.get("src_ad") or "")
    metrics: dict[str, Any] = {"predict_time": None, "input_tokens": None, "output_tokens": None, "model": model}
    try:
        adapted_image = open_rgb(adapted) if isinstance(adapted, Path) else adapted.convert("RGB")
        source_image = open_rgb(source) if isinstance(source, Path) else source.convert("RGB")
    except Exception as exc:
        report = model_fail(
            country,
            src_ad,
            f"Adapted or CA file is invalid ({exc}).",
            "Provide a readable adapted image and the CA source, then rerun QA.",
        )
        return report, metrics

    det = deterministic_fails(adapted_image, brief, source_image)
    system, user = build_vision_prompt(checklist, brief)
    try:
        with tempfile.TemporaryDirectory(prefix="ot-qa-") as tmp:
            work = Path(tmp)
            adapted_path = work / "adapted.jpg"
            source_path = work / "ca.jpg"
            _save_rgb_jpeg(adapted_image, adapted_path)
            _save_rgb_jpeg(source_image, source_path)
            payload, handles = vision_payload(model, system, user, adapted_path, source_path, work)
            try:
                prediction = predict(model, payload)
            finally:
                for handle in handles:
                    handle.close()
        status = getattr(prediction, "status", "succeeded")
        if status != "succeeded":
            detail = getattr(prediction, "error", None) or status
            report = model_fail(country, src_ad, f"QA model error: {detail}", "Rerun QA. A model error is a FAIL.", det)
            return report, metrics
        metrics.update(prediction_metrics(prediction))
        parsed = parse_model_output(prediction_text(getattr(prediction, "output", None)), country, src_ad)
        parsed["country"] = country or parsed["country"]
        parsed["src_ad"] = src_ad or parsed["src_ad"]
        return merge_fails(det, parsed), metrics
    except Exception as exc:
        report = model_fail(
            country,
            src_ad,
            f"QA model error: {exc.__class__.__name__}: {exc}",
            "Rerun QA. A model error is a FAIL, never a PASS.",
            det,
        )
        return report, metrics


def _first_image(folder: Path, stems: tuple[str, ...]) -> Path | None:
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    for stem in stems:
        for path in files:
            if path.stem.lower() == stem:
                return path
    return None


def _job_from_directory(folder: Path) -> dict[str, Any]:
    brief_path = folder / "brief.json"
    if not brief_path.is_file():
        raise ValueError(f"brief.json manquant dans {folder}")
    brief = json.loads(brief_path.read_text(encoding="utf-8-sig"))
    if not isinstance(brief, dict):
        raise ValueError(f"brief invalide : {brief_path}")
    adapted = _first_image(folder, ADAPTED_STEMS)
    source = _first_image(folder, CA_STEMS)
    images = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    if adapted is None or source is None:
        remaining = [path for path in images if path not in {adapted, source}]
        if adapted is None and remaining:
            adapted = remaining.pop(0)
        if source is None and remaining:
            source = remaining.pop(0)
    if adapted is None or source is None:
        raise ValueError(f"images adapted et ca_source introuvables dans {folder}")
    if adapted.resolve() == source.resolve():
        raise ValueError(f"adapted et ca_source sont le même fichier dans {folder}")
    normalized = normalize_brief(brief, folder.name, str(brief.get("src_ad") or folder.name))
    return {"adapted": adapted, "ca_source": source, "brief": normalized, "country": normalized["country"], "src_ad": normalized["src_ad"]}


def discover_folder(folder: Path) -> list[dict[str, Any]]:
    if not folder.is_dir():
        raise ValueError(f"dossier QA introuvable : {folder}")
    manifest = folder / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("manifest.json doit être une liste")
        return [_job_from_record(item, folder, folder) for item in data]
    children = sorted(path for path in folder.iterdir() if path.is_dir())
    if children:
        return [_job_from_directory(child) for child in children]
    return [_job_from_directory(folder)]


def _resolve_existing(raw: str, roots: list[Path]) -> Path:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate
    for root in roots:
        nested = (root / raw).resolve()
        if nested.is_file():
            return nested
    raise ValueError(f"fichier introuvable : {raw}")


def _job_from_record(item: Any, inputs: Path | None, folder: Path | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("item QA invalide")
    roots = [path for path in (inputs, folder) if path is not None]
    adapted = _resolve_existing(str(item.get("adapted") or item.get("image") or ""), roots)
    source = _resolve_existing(str(item.get("ca_source") or item.get("source") or item.get("ca") or ""), roots)
    brief_raw = item.get("brief")
    if isinstance(brief_raw, str):
        brief_path = _resolve_existing(brief_raw, roots)
        brief_data = json.loads(brief_path.read_text(encoding="utf-8-sig"))
        if not isinstance(brief_data, dict):
            raise ValueError(f"brief invalide : {brief_path}")
    elif isinstance(brief_raw, dict):
        brief_data = brief_raw
    else:
        brief_data = {key: item[key] for key in item if key not in {"adapted", "ca_source", "source", "ca", "image"}}
    country = str(item.get("country") or brief_data.get("country") or "")
    src_ad = str(item.get("src_ad") or brief_data.get("src_ad") or adapted.stem)
    brief = normalize_brief(brief_data, country, src_ad)
    if item.get("country"):
        brief["country"] = str(item["country"])
    if item.get("src_ad"):
        brief["src_ad"] = str(item["src_ad"])
    return {
        "adapted": adapted,
        "ca_source": source,
        "brief": brief,
        "country": brief["country"],
        "src_ad": brief["src_ad"],
    }


def load_job_list(path: Path, inputs: Path | None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("la liste QA doit être un tableau JSON")
    return [_job_from_record(item, inputs, path.parent) for item in data]


def estimate_cost(
    qa_metrics: list[dict[str, Any]],
    *,
    qa_model: str,
    generated_images: int,
    image_quality: str,
) -> dict[str, Any]:
    rates = QA_TOKEN_USD_PER_MILLION.get(qa_model)
    input_tokens = 0
    output_tokens = 0
    have_tokens = False
    qa_seconds = 0.0
    for metrics in qa_metrics:
        if metrics.get("input_tokens") is not None and metrics.get("output_tokens") is not None:
            have_tokens = True
            input_tokens += int(metrics["input_tokens"])
            output_tokens += int(metrics["output_tokens"])
        if metrics.get("predict_time") is not None:
            qa_seconds += float(metrics["predict_time"])
    qa_usd = None
    if have_tokens and rates is not None:
        qa_usd = input_tokens * rates[0] / 1_000_000 + output_tokens * rates[1] / 1_000_000
    per_image = IMAGE_USD.get(image_quality, IMAGE_USD["auto"])
    image_usd = generated_images * per_image
    total = image_usd + (qa_usd or 0.0)
    return {
        "qa_model": qa_model,
        "qa_calls": len(qa_metrics),
        "input_tokens": input_tokens if have_tokens else None,
        "output_tokens": output_tokens if have_tokens else None,
        "qa_predict_seconds": round(qa_seconds, 3),
        "qa_usd": None if qa_usd is None else round(qa_usd, 6),
        "generated_images": generated_images,
        "image_quality": image_quality,
        "image_usd": round(image_usd, 6),
        "estimated_usd": round(total, 6) if qa_usd is not None or generated_images else (None if not generated_images else round(image_usd, 6)),
        "rates": {
            "qa_usd_per_million_input_output": rates,
            "image_usd_each": per_image,
            "source": "Replicate public prices, September 2026",
        },
    }


def select_indices(count: int, spec: str) -> list[int]:
    text = spec.strip()
    if not text:
        return list(range(count))
    selected: list[int] = []
    for part in text.split(","):
        piece = part.strip()
        if not piece:
            continue
        if re.fullmatch(r"\d+", piece):
            selected.append(int(piece))
            continue
        if re.fullmatch(r"\d+:\d+", piece):
            start_s, end_s = piece.split(":", 1)
            selected.extend(range(int(start_s), int(end_s)))
            continue
        if re.fullmatch(r"\d*-\d*", piece) and piece != "-":
            start_s, end_s = piece.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else count - 1
            if end < start:
                raise ValueError(f"intervalle d'index invalide : {piece}")
            selected.extend(range(start, end + 1))
            continue
        raise ValueError(f"index invalide : {piece}")
    unique: list[int] = []
    for index in selected:
        if index < 0 or index >= count:
            raise ValueError(f"index hors file : {index} (taille {count})")
        if index not in unique:
            unique.append(index)
    return unique
