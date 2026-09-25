#!/usr/bin/env python3
"""Worker de file : adapte des visuels Ortho Terre via Replicate.

Lit une tranche JSON (country, src_ad, brief, copy, out, class), résout les
chemins à partir du segment <country>/ dans le dossier d'entrées, et écrit
./artifacts/<country>/<src_ad>.jpg plus ./artifacts/report.json.

Le jeton est lu uniquement depuis REPLICATE_API_TOKEN. Ce programme ne
l'affiche pas, ne l'écrit pas et ne le journalise pas.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

try:
    import qa as qa_mod
except ImportError:  # import depuis la racine du dépôt
    from worker import qa as qa_mod

DEFAULT_MODEL = "google/nano-banana-pro"
SUPPORTED_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 4),
    (9, 16),
    (16, 9),
    (21, 9),
)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
REQUIRED_FIELDS = ("country", "src_ad", "brief", "copy", "out", "class")
COUNTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
SRC_AD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TERMINAL_STATUSES = ("succeeded", "failed", "canceled")
JPG_QUALITY = 92


class RetryableError(Exception):
    """Erreur HTTP transitoire (429 ou 5xx) à réessayer."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        super().__init__(detail or f"HTTP {status}")


class WorkerError(Exception):
    """Échec d'un item, avec l'identifiant de prédiction lorsqu'il existe."""

    def __init__(self, message: str, prediction_id: str | None = None) -> None:
        super().__init__(message)
        self.prediction_id = prediction_id


@dataclass(frozen=True)
class PadMeta:
    width: int
    height: int
    pad_w: int
    pad_h: int
    pad_x: int
    pad_y: int
    ratio: tuple[int, int]


class RateLimiter:
    """Au plus N acquisitions par fenêtre glissante (défaut : 60 secondes)."""

    def __init__(
        self,
        max_per_minute: int,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_minute < 1:
            raise ValueError("max_per_minute doit être >= 1")
        self.max_per_minute = max_per_minute
        self.window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._stamps: list[float] = []

    def acquire(self) -> None:
        while True:
            now = self._clock()
            self._stamps = [stamp for stamp in self._stamps if now - stamp < self.window_s]
            if len(self._stamps) < self.max_per_minute:
                self._stamps.append(now)
                return
            wait = self.window_s - (now - self._stamps[0])
            self._sleep(max(wait, 0.01))


def http_status(exc: BaseException) -> int | None:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def is_retryable(exc: BaseException) -> bool:
    status = http_status(exc)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    try:
        import httpx
        import requests
    except ImportError:
        httpx = None  # type: ignore[assignment]
        requests = None  # type: ignore[assignment]
    if requests is not None and isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if httpx is not None and isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    return False


def with_retry(
    fn: Callable[[], Any],
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    delay = base_delay
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            sleep(delay)
            delay = min(delay * 2, max_delay)
    assert last is not None
    raise last


def redact(text: str) -> str:
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if token:
        text = text.replace(token, "[REDACTED]")
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"r8_[A-Za-z0-9_\-]+", "r8_[REDACTED]", text)
    text = re.sub(r"https?://\S+", lambda match: match.group(0).split("?", 1)[0], text)
    if len(text) > 500:
        text = text[:500] + "…"
    return text


def token_present() -> bool:
    return bool(os.environ.get("REPLICATE_API_TOKEN", "").strip())


def missing_token_message() -> str:
    return (
        "REPLICATE_API_TOKEN manquant. "
        "Exportez cette variable d'environnement, puis relancez le worker."
    )


def country_relative(path: str, country: str) -> str:
    """Garde le chemin à partir du dernier segment <country>/."""
    raw = str(path).replace("\\", "/").strip()
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError(f"chemin refusé : {path}")
    indexes = [index for index, part in enumerate(parts) if part == country]
    if not indexes:
        raise ValueError(f"segment {country}/ introuvable dans {path}")
    return "/".join(parts[indexes[-1] :])


def safe_join(root: Path, relative: str) -> Path:
    base = root.resolve()
    candidate = (base / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"chemin hors du dossier d'entrées : {relative}")
    return candidate


def resolve_input(path: str, country: str, inputs: Path) -> Path:
    return safe_join(inputs, country_relative(path, country))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        transposed = ImageOps.exif_transpose(image) or image
        rgb = transposed.convert("RGB")
        rgb.load()
        return rgb


def nearest_ratio(width: int, height: int) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("image de taille nulle")
    ratio = width / height
    return min(SUPPORTED_RATIOS, key=lambda item: abs(math.log((item[0] / item[1]) / ratio)))


def reflect_index(index: int, length: int) -> int:
    """Index replié comme numpy.pad(..., mode='reflect')."""
    if length <= 0:
        raise ValueError("dimension nulle")
    if length == 1:
        return 0
    period = 2 * (length - 1)
    index %= period
    if index < 0:
        index += period
    if index >= length:
        return period - index
    return index


def _reflect_pad_slow(image: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    width, height = image.size
    source = image.load()
    canvas = Image.new("RGB", (width + left + right, height + top + bottom))
    pixels = canvas.load()
    for y in range(canvas.height):
        sy = reflect_index(y - top, height)
        for x in range(canvas.width):
            sx = reflect_index(x - left, width)
            pixels[x, y] = source[sx, sy]
    return canvas


def reflect_pad(image: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    if min(left, top, right, bottom) < 0:
        raise ValueError("marge négative")
    width, height = image.size
    if left == 0 and top == 0 and right == 0 and bottom == 0:
        return image.copy()
    if left >= width or right >= width or top >= height or bottom >= height or width < 2 or height < 2:
        return _reflect_pad_slow(image, left, top, right, bottom)

    canvas = Image.new("RGB", (width + left + right, height + top + bottom))
    canvas.paste(image, (left, top))
    if left:
        strip = image.crop((1, 0, left + 1, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        canvas.paste(strip, (0, top))
    if right:
        strip = image.crop((width - 1 - right, 0, width - 1, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        canvas.paste(strip, (left + width, top))
    if top:
        strip = image.crop((0, 1, width, top + 1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        canvas.paste(strip, (left, 0))
    if bottom:
        strip = image.crop((0, height - 1 - bottom, width, height - 1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        canvas.paste(strip, (left, top + height))
    if left and top:
        corner = image.crop((1, 1, left + 1, top + 1))
        corner = corner.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        canvas.paste(corner, (0, 0))
    if right and top:
        corner = image.crop((width - 1 - right, 1, width - 1, top + 1))
        corner = corner.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        canvas.paste(corner, (left + width, 0))
    if left and bottom:
        corner = image.crop((1, height - 1 - bottom, left + 1, height - 1))
        corner = corner.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        canvas.paste(corner, (0, top + height))
    if right and bottom:
        corner = image.crop((width - 1 - right, height - 1 - bottom, width - 1, height - 1))
        corner = corner.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        canvas.paste(corner, (left + width, top + height))
    return canvas


def pad_to_supported(image: Image.Image) -> tuple[Image.Image, PadMeta]:
    """Complète l'image jusqu'au ratio Replicate le plus proche (reflet)."""
    width, height = image.size
    ratio_w, ratio_h = nearest_ratio(width, height)
    target = ratio_w / ratio_h
    current = width / height
    if target > current:
        pad_w, pad_h = round(height * target), height
    else:
        pad_w, pad_h = width, round(width / target)
    pad_x = (pad_w - width) // 2
    pad_y = (pad_h - height) // 2
    padded = reflect_pad(image, pad_x, pad_y, pad_w - width - pad_x, pad_h - height - pad_y)
    meta = PadMeta(width, height, pad_w, pad_h, pad_x, pad_y, (ratio_w, ratio_h))
    return padded, meta


def fit_cover(raw: Image.Image, meta: PadMeta) -> Image.Image:
    """Ramène la sortie modèle aux dimensions exactes de la source."""
    image = raw.convert("RGB")
    scale = max(meta.pad_w / image.width, meta.pad_h / image.height)
    resized_w = max(meta.pad_w, round(image.width * scale))
    resized_h = max(meta.pad_h, round(image.height * scale))
    if image.size != (resized_w, resized_h):
        image = image.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
    offset_x = (resized_w - meta.pad_w) // 2
    offset_y = (resized_h - meta.pad_h) // 2
    box = (
        offset_x + meta.pad_x,
        offset_y + meta.pad_y,
        offset_x + meta.pad_x + meta.width,
        offset_y + meta.pad_y + meta.height,
    )
    if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
        raise ValueError(f"recadrage hors image : {box} pour {image.size}")
    cropped = image.crop(box)
    if cropped.size != (meta.width, meta.height):
        raise ValueError(f"recadrage {cropped.size} au lieu de {(meta.width, meta.height)}")
    return cropped


def save_jpg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    image.save(temporary, format="JPEG", quality=quality, subsampling=0, optimize=True)
    temporary.replace(path)


def build_prompt(brief: dict[str, Any], copy_text: str, classification: str) -> str:
    changes = [str(item) for item in brief.get("what_to_change") or []]
    rules = [str(item) for item in brief.get("rules") or []]
    sources = [str(item) for item in brief.get("source_text_to_translate") or []]
    parts = [
        "Edit the attached advertisement image. Keep the same product, layout, framing and call to action. "
        "Apply only the changes below. Use the real local prices and currency given in the brief, "
        "with native spelling and accents. Do not add shipping information, a country name, or a description. "
        "Do not invent prices.",
        (
            f"Brand: {brief.get('brand') or 'Ortho Terre'}\n"
            f"Market: {brief.get('market') or ''}\n"
            f"Language: {brief.get('language') or ''}\n"
            f"Classification: {classification}"
        ),
    ]
    if changes:
        parts.append("Changes:\n" + "\n".join(f"- {item}" for item in changes))
    if sources:
        parts.append("Source text to translate:\n" + "\n".join(f"- {item}" for item in sources))
    if copy_text.strip():
        parts.append("Localized copy (render exactly):\n" + copy_text.strip())
    if rules:
        parts.append("Rules:\n" + "\n".join(f"- {item}" for item in rules))
    return "\n\n".join(parts)


GPT_IMAGE_RATIOS: tuple[tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("3:2", 3 / 2),
    ("2:3", 2 / 3),
    ("4:3", 4 / 3),
    ("3:4", 3 / 4),
    ("16:9", 16 / 9),
    ("9:16", 9 / 16),
)
ENGINES = {
    "gpt-image": "openai/gpt-image-2.5-sunburst",
    "nano-banana": "google/nano-banana-pro",
}


def is_gpt_image(model: str) -> bool:
    return "gpt-image" in model


def gpt_aspect_ratio(width: int, height: int) -> str:
    ratio = width / height
    name, _value = min(GPT_IMAGE_RATIOS, key=lambda item: abs(math.log(item[1] / ratio)))
    return name


def model_input(
    model: str,
    prompt: str,
    image_path: Path | list[Path],
    resolution: str = "2K",
    aspect_ratio: str | None = None,
    quality: str | None = None,
) -> dict[str, Any]:
    paths = image_path if isinstance(image_path, list) else [image_path]
    if is_gpt_image(model):
        # openai_api_key reste absent : le proxy Replicate suffit avec REPLICATE_API_TOKEN.
        return {
            "prompt": prompt,
            "input_images": paths,
            "output_format": "jpeg",
            "quality": quality or "auto",
            "number_of_images": 1,
            "aspect_ratio": aspect_ratio or "auto",
        }
    if "kontext" in model:
        return {
            "prompt": prompt,
            "input_image": paths[0],
            "output_format": "jpg",
            "aspect_ratio": "match_input_image",
            "prompt_upsampling": False,
            "safety_tolerance": 2,
        }
    payload: dict[str, Any] = {
        "prompt": prompt,
        "image_input": paths,
        "output_format": "jpg",
        "aspect_ratio": "match_input_image",
    }
    if model.endswith("pro"):
        payload["resolution"] = resolution
        payload["allow_fallback_model"] = False
    return payload


def files_named(root: Path, name: str) -> list[Path]:
    base = root.resolve()
    matches: list[Path] = []
    if not base.is_dir():
        return matches
    for path in base.rglob("*"):
        if path.is_file() and path.name == name and base in path.resolve().parents:
            matches.append(path)
    return matches


def _unique(paths: list[Path]) -> Path | None:
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if len(unique) == 1:
        return unique[0]
    return None


def find_source(brief: dict[str, Any], country: str, src_ad: str, inputs: Path) -> Path:
    raw = str(brief.get("source_image_path") or "").strip()
    if raw:
        try:
            resolved = resolve_input(raw, country, inputs)
        except ValueError:
            resolved = None
        if resolved is not None and resolved.is_file():
            return resolved
        direct = Path(raw)
        if direct.is_file():
            return direct
        basename = Path(raw.replace("\\", "/")).name
        if basename and basename not in (".", ".."):
            matches = files_named(inputs, basename)
            found = _unique(matches)
            if found is not None:
                return found
            if len({path.resolve() for path in matches}) > 1:
                raise ValueError(f"plusieurs images sources nommées {basename}")

    candidates: list[Path] = []
    for folder in (
        inputs / country / "sources",
        inputs / country,
        inputs / "sources",
        inputs,
    ):
        for ext in IMAGE_EXTS:
            candidates.append(folder / f"{src_ad}{ext}")
    found = _unique(candidates)
    if found is not None:
        return found
    matches = []
    for ext in IMAGE_EXTS:
        matches.extend(files_named(inputs, f"{src_ad}{ext}"))
    found = _unique(matches)
    if found is not None:
        return found
    if matches:
        raise ValueError(f"plusieurs images sources pour {src_ad}")
    raise ValueError(f"image source introuvable pour {country}/{src_ad}")


def output_path(artifacts: Path, country: str, src_ad: str) -> Path:
    return artifacts / country / f"{src_ad}.jpg"


def relative_output(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def blank_report(
    country: str,
    src_ad: str,
    classification: str,
    model: str,
    out: str | None,
    status: str,
    error: str | None = None,
    prediction_id: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    return {
        "country": country,
        "src_ad": src_ad,
        "class": classification,
        "status": status,
        "model": model,
        "prediction_id": prediction_id,
        "out": out,
        "error": error,
        "width": width,
        "height": height,
    }


def validate_identity(item: dict[str, Any]) -> tuple[str, str, str]:
    missing = [field for field in REQUIRED_FIELDS if field not in item or str(item[field]).strip() == ""]
    if missing:
        raise ValueError("champs manquants : " + ", ".join(missing))
    country = str(item["country"]).strip()
    src_ad = str(item["src_ad"]).strip()
    classification = str(item["class"]).strip()
    if not COUNTRY_RE.match(country):
        raise ValueError(f"country invalide : {country}")
    if not SRC_AD_RE.match(src_ad) or ".." in src_ad:
        raise ValueError(f"src_ad invalide : {src_ad}")
    return country, src_ad, classification


def prepare_item(item: dict[str, Any], inputs: Path) -> dict[str, Any]:
    country, src_ad, classification = validate_identity(item)
    brief_path = resolve_input(str(item["brief"]), country, inputs)
    country_relative(str(item["out"]), country)
    if not brief_path.is_file():
        raise ValueError(f"brief introuvable : {country_relative(str(item['brief']), country)}")
    brief = load_json(brief_path)
    if not isinstance(brief, dict):
        raise ValueError("le brief doit être un objet JSON")
    if brief.get("hold"):
        return {
            "country": country,
            "src_ad": src_ad,
            "class": classification,
            "brief": brief,
            "copy_text": "",
            "image": None,
            "source": None,
            "hold": True,
        }
    copy_path = resolve_input(str(item["copy"]), country, inputs)
    if not copy_path.is_file():
        raise ValueError(f"copy introuvable : {country_relative(str(item['copy']), country)}")
    copy_text = read_text(copy_path)
    source = find_source(brief, country, src_ad, inputs)
    image = open_rgb(source)
    return {
        "country": country,
        "src_ad": src_ad,
        "class": classification,
        "brief": brief,
        "copy_text": copy_text,
        "image": image,
        "source": source,
        "hold": False,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _checked_response(response: Any, auth: bool) -> Any:
    status = response.status_code
    if status == 429 or 500 <= status <= 599:
        response.close()
        raise RetryableError(status)
    if auth and status in (400, 401, 403):
        return response
    if status >= 400:
        response.raise_for_status()
    return response


def download_output(url: str) -> bytes:
    import requests

    if not url.startswith("https://"):
        raise ValueError("URL de sortie refusée")
    token = os.environ.get("REPLICATE_API_TOKEN", "").strip()

    def fetch(auth: bool) -> Any:
        headers = {"Authorization": f"Bearer {token}"} if auth and token else {}

        def once() -> Any:
            response = requests.get(url, headers=headers, timeout=300)
            return _checked_response(response, auth)

        return with_retry(once)

    response = fetch(True)
    if response.status_code in (400, 401, 403):
        response.close()
        response = fetch(False)
    try:
        if response.status_code >= 400:
            response.raise_for_status()
        return response.content
    finally:
        response.close()


def output_url(output: Any) -> str:
    item = output[0] if isinstance(output, list) else output
    if item is None:
        raise ValueError("prédiction sans image")
    url = item if isinstance(item, str) else getattr(item, "url", None)
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("sortie Replicate illisible")
    return url


def run_prediction(client: Any, model: str, payload: dict[str, Any], limiter: RateLimiter, timeout_s: int) -> Any:
    def create() -> Any:
        limiter.acquire()
        return client.predictions.create(model=model, input=payload, wait=60)

    prediction = with_retry(create)
    deadline = time.monotonic() + timeout_s
    while prediction.status not in TERMINAL_STATUSES:
        if time.monotonic() > deadline:
            raise WorkerError(
                f"prédiction {prediction.id} au-delà de {timeout_s}s",
                str(prediction.id),
            )
        time.sleep(4)

        def reload() -> None:
            prediction.reload()

        with_retry(reload)
    return prediction


def adapt_image(
    image: Image.Image,
    prompt: str,
    model: str,
    limiter: RateLimiter,
    timeout_s: int,
) -> tuple[Image.Image, str]:
    import replicate

    padded, meta = pad_to_supported(image)
    client = replicate.Client()
    with tempfile.TemporaryDirectory(prefix="ot-adapt-") as tmp:
        pad_path = Path(tmp) / "pad.jpg"
        raw_path = Path(tmp) / "raw.jpg"
        save_jpg(padded, pad_path, quality=95)
        prediction = run_prediction(client, model, model_input(model, prompt, pad_path), limiter, timeout_s)
        prediction_id = str(prediction.id)
        if prediction.status != "succeeded":
            detail = prediction.error or prediction.status
            raise WorkerError(f"{prediction_id}: {detail}", prediction_id)
        raw_path.write_bytes(download_output(output_url(prediction.output)))
        fitted = fit_cover(open_rgb(raw_path), meta)
    if fitted.size != (meta.width, meta.height):
        raise RuntimeError(f"dimensions {fitted.size} au lieu de {(meta.width, meta.height)}")
    return fitted, prediction_id


# Boîte du tableau blanc sur les visuels 1254×1254 (phrase manuscrite seule).
WHITEBOARD_BOX = (470, 88, 1100, 292)
LOCAL_GOLF = {
    "be-nl": "a Belgian golf course in Flanders or the Ardennes",
    "pt": "a Portuguese golf course on the coast near Sintra",
    "gr": "a Greek golf course with Mediterranean hills and dry light",
    "no": "a Norwegian golf course with pines and a fjord in the distance",
    "es": "a Spanish golf course in Andalusia or the foothills of the Pyrenees",
}


def is_fix_queue(queue: list[Any]) -> bool:
    item = queue[0] if queue else None
    return isinstance(item, dict) and "instruction" in item and "current" in item and "target_size" in item


def quoted_phrases(instruction: str) -> list[str]:
    return [part.strip() for part in re.findall(r"«\s*(.*?)\s*»", instruction)]


def task_kind(instruction: str) -> str:
    text = instruction.lower()
    if text.startswith("tableau blanc"):
        return "whiteboard"
    if "pack 1" in text:
        return "pack"
    if "golf" in text:
        return "golf"
    if "voiture" in text:
        return "car"
    if "badge" in text:
        return "badge"
    if "bâtiment" in text or "batiment" in text or "tremblant" in text:
        return "buildings"
    if "cycliste" in text or "panneau" in text:
        return "sign"
    return "edit"


def resize_exact(image: Image.Image, width: int, height: int) -> Image.Image:
    """Échelle l'image entière à la taille cible, sans rien recadrer ni recomposer."""
    if width < 1 or height < 1:
        raise ValueError("target_size invalide")
    image = image.convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def center_crop_resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """Recadrage centré au ratio cible, puis resize exact."""
    if width < 1 or height < 1:
        raise ValueError("target_size invalide")
    image = image.convert("RGB")
    src_w, src_h = image.size
    target_ratio = width / height
    current_ratio = src_w / src_h
    if current_ratio > target_ratio:
        crop_w = max(1, round(src_h * target_ratio))
        left = max(0, (src_w - crop_w) // 2)
        image = image.crop((left, 0, left + crop_w, src_h))
    elif current_ratio < target_ratio:
        crop_h = max(1, round(src_w / target_ratio))
        top = max(0, (src_h - crop_h) // 2)
        image = image.crop((0, top, src_w, top + crop_h))
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if image.size != (width, height):
        raise ValueError(f"taille {image.size} au lieu de {(width, height)}")
    return image


def sharpness_score(image: Image.Image) -> float:
    import numpy as np

    gray = np.asarray(image.convert("L"), dtype="float32")
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * center
    return float(lap.var())


def paste_feather(base: Image.Image, overlay: Image.Image, box: tuple[int, int, int, int], radius: int = 14) -> Image.Image:
    x0, y0, x1, y1 = box
    canvas = base.convert("RGB").copy()
    piece = overlay.convert("RGB")
    if piece.size != (x1 - x0, y1 - y0):
        piece = piece.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    mask = Image.new("L", piece.size, 0)
    pixels = mask.load()
    width, height = piece.size
    for y in range(height):
        for x in range(width):
            edge = min(x, y, width - 1 - x, height - 1 - y)
            if edge >= radius:
                pixels[x, y] = 255
            else:
                pixels[x, y] = int(255 * edge / radius)
    canvas.paste(piece, (x0, y0), mask)
    return canvas


def build_fix_prompt(item: dict[str, Any], kind: str, attempt: int) -> str:
    instruction = str(item["instruction"]).strip()
    phrases = quoted_phrases(instruction)
    country = str(item["country"])
    retry = ""
    if attempt >= 2:
        retry = (
            " The previous attempt was rejected. Change only the requested detail. "
            "Do not introduce any extra word, letter, logo, watermark or translation."
        )
    if attempt >= 3:
        retry += " Make the requested text larger, sharper, and spelled exactly as given."
    if kind == "whiteboard":
        lines = "\n".join(f"Line {index}: {phrase}" for index, phrase in enumerate(phrases, start=1))
        extra = ""
        if "pack 3" in instruction.lower():
            extra = (
                " Also, on pack 3 only, set the title to exactly EQUILIBRIO TERRESTRE "
                "with a thin small subtitle, and leave the cable unchanged. "
                "Do not alter pack 1, pack 2, prices, or any other text."
            )
        return (
            "Edit the first image. The second image is the Canadian reference: "
            "copy only its handwritten marker style, slant, size and position. "
            "Rewrite the handwritten whiteboard sentence so it reads EXACTLY, "
            "with the same spelling, accents, punctuation and line breaks:\n"
            f"{lines}\n"
            "Do not add any other word, quote, translation, caption or decoration. "
            "Keep the whiteboard frame and everything outside the handwriting unchanged. "
            "Black marker handwriting only."
            f"{extra}{retry}"
        )
    if kind == "golf":
        place = LOCAL_GOLF.get(country, "a local golf course for this country")
        return (
            "Edit the first image. The second image is the Canadian reference for layout only. "
            f"Replace only the background with {place}: a green fairway, exactly two golf carts, "
            "and one or two golfers, composed like the golf background of the reference. "
            "Keep the man, his face, clothes, pose, the table, the product and every text pixel-identical. "
            "Do not add words, logos or watermarks."
            f"{retry}"
        )
    if kind == "pack":
        rendered = " / ".join(phrases) if phrases else instruction
        return (
            "Edit the first image. The second image shows the Canadian pack layout to follow. "
            "Change only pack 1 so its layout matches the reference: "
            f"{rendered}. "
            "Keep pack 2, pack 3, the whiteboard, prices, people and every other text unchanged. "
            "Do not add extra words."
            f"{retry}"
        )
    if kind == "car":
        return (
            "Edit the first image. The second image shows the intact grey car to restore. "
            "Replace the wrecked car with an intact grey car like the reference. "
            "Only the local Greek background may change. Keep the framing. "
            "Do not add any text, logo or watermark."
            f"{retry}"
        )
    if kind == "badge":
        return (
            "Edit the first image. Replace only the Tremblant badge or sign with a plain blank white panel. "
            "Change nothing else: same buildings, people, framing, colors and any other text. "
            "The white panel must contain no letters."
            f"{retry}"
        )
    if kind == "buildings":
        return (
            "Edit the first image. Replace the Tremblant resort buildings with hotels of South Tyrol "
            "(Alto Adige / Haut-Adige), keeping the same framing and crop. "
            "Do not add the word Tremblant. Do not add extra text, logos or watermarks. "
            "Keep people and the overall composition."
            f"{retry}"
        )
    if kind == "sign":
        return (
            "Edit the first image. In the top-right cyclists panel only, replace the Quebec lake scenery "
            "with the Haute-Sûre or the Moselle in Luxembourg. "
            "Keep the same cyclists, their poses, the panel framing, and every other panel unchanged. "
            "Do not change any text."
            f"{retry}"
        )
    return (
        "Edit the first image using the second image only as a layout reference. "
        f"Apply only this correction: {instruction}. "
        "Keep product, people, prices and already validated text unchanged. Do not add stray text."
        f"{retry}"
    )


def edit_with_references(
    images: list[Image.Image],
    prompt: str,
    model: str,
    limiter: RateLimiter,
    timeout_s: int,
    resolution: str = "2K",
    fit: str = "cover",
    target: tuple[int, int] | None = None,
    quality: str | None = None,
    metrics_sink: list[dict[str, Any]] | None = None,
) -> tuple[Image.Image, str]:
    """Édite la première image ; les suivantes sont des références.

    fit="resize" met à l'échelle la sortie entière (pas de recadrage, pas de collage).
    C'est le seul ajustement utilisé par la boucle QA gpt-image.
    """
    import replicate

    if not images:
        raise ValueError("aucune image à éditer")
    client = replicate.Client()
    if is_gpt_image(model):
        primary = images[0].convert("RGB")
        out_w, out_h = target or primary.size
        with tempfile.TemporaryDirectory(prefix="ot-fix-") as tmp:
            root = Path(tmp)
            paths: list[Path] = []
            for index, image in enumerate(images):
                path = root / f"in_{index}.jpg"
                save_jpg(image.convert("RGB"), path, quality=95)
                paths.append(path)
            raw_path = root / "raw.jpg"
            prediction = run_prediction(
                client,
                model,
                model_input(
                    model,
                    prompt,
                    paths,
                    aspect_ratio=gpt_aspect_ratio(out_w, out_h),
                    quality=quality,
                ),
                limiter,
                timeout_s,
            )
            prediction_id = str(prediction.id)
            if prediction.status != "succeeded":
                detail = prediction.error or prediction.status
                raise WorkerError(f"{prediction_id}: {detail}", prediction_id)
            if metrics_sink is not None:
                metrics_sink.append(qa_mod.prediction_metrics(prediction))
            raw_path.write_bytes(download_output(output_url(prediction.output)))
            raw = open_rgb(raw_path)
            if fit == "resize":
                fitted = resize_exact(raw, out_w, out_h)
            else:
                fitted = center_crop_resize(raw, primary.width, primary.height)
        return fitted, prediction_id
    padded_primary, meta = pad_to_supported(images[0])
    with tempfile.TemporaryDirectory(prefix="ot-fix-") as tmp:
        paths: list[Path] = []
        root = Path(tmp)
        right = meta.pad_w - meta.width - meta.pad_x
        bottom = meta.pad_h - meta.height - meta.pad_y
        for index, image in enumerate(images):
            source = image.convert("RGB")
            if source.size != (meta.width, meta.height):
                source = center_crop_resize(source, meta.width, meta.height)
            padded = reflect_pad(source, meta.pad_x, meta.pad_y, right, bottom)
            path = root / f"in_{index}.jpg"
            save_jpg(padded, path, quality=95)
            paths.append(path)
        raw_path = root / "raw.jpg"
        prediction = run_prediction(
            client,
            model,
            model_input(model, prompt, paths, resolution=resolution),
            limiter,
            timeout_s,
        )
        prediction_id = str(prediction.id)
        if prediction.status != "succeeded":
            detail = prediction.error or prediction.status
            raise WorkerError(f"{prediction_id}: {detail}", prediction_id)
        raw_path.write_bytes(download_output(output_url(prediction.output)))
        raw = open_rgb(raw_path)
        if fit == "resize":
            out_w, out_h = target or (meta.width, meta.height)
            fitted = resize_exact(raw, out_w, out_h)
        else:
            fitted = fit_cover(raw, meta)
    return fitted, prediction_id


def resolve_fix_path(relative: str, inputs: Path) -> Path:
    path = safe_join(inputs, relative.replace("\\", "/"))
    if not path.is_file():
        raise ValueError(f"fichier introuvable : {relative}")
    return path


def parse_target_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("target_size doit être [largeur, hauteur]")
    width, height = int(value[0]), int(value[1])
    if width < 1 or height < 1:
        raise ValueError("target_size invalide")
    return width, height


def load_report_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    data = load_json(path)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get("country") and item.get("src_ad"):
            index[(str(item["country"]), str(item["src_ad"]))] = item
    return index


def image_report_path(artifacts: Path, country: str, src_ad: str) -> Path:
    return artifacts / country / src_ad / "report.json"


def summary_path(artifacts: Path, indices: str) -> Path:
    text = indices.strip()
    if not text:
        return artifacts / "batch_summary.json"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return artifacts / f"batch_summary_{safe}.json"


def qa_brief_for_fix(item: dict[str, Any], inputs: Path) -> dict[str, Any]:
    payload = dict(item)
    raw = item.get("brief")
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if not path.is_file():
            path = resolve_fix_path(raw, inputs)
        loaded = load_json(path)
        if not isinstance(loaded, dict):
            raise ValueError("brief QA invalide")
        payload["brief"] = loaded
    return qa_mod.brief_from_fix_item(payload)


def _predict(model: str, payload: dict[str, Any], limiter: RateLimiter, timeout_s: int) -> Any:
    import replicate

    client = replicate.Client()
    return run_prediction(client, model, payload, limiter, timeout_s)


def process_fix_item_qa(
    item: dict[str, Any],
    inputs: Path,
    artifacts: Path,
    limiter: RateLimiter,
    timeout_s: int,
    max_attempts: int,
    model: str,
    qa_model: str,
    checklist: str,
    image_quality: str | None,
    current: Image.Image,
    reference: Image.Image,
    target: tuple[int, int],
    kind: str,
    destination: Path,
) -> dict[str, Any]:
    """Génère l'image entière, lance le QA, et réinjecte les corrections. Au plus max_attempts."""
    country = str(item["country"])
    src_ad = str(item["src_ad"])
    brief = qa_brief_for_fix(item, inputs)
    brief["country"] = country
    brief["src_ad"] = src_ad
    if brief.get("target_size") is None:
        brief["target_size"] = target
    out_label = destination.as_posix()
    report_path = image_report_path(artifacts, country, src_ad)
    corrections: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    prediction_ids: list[str] = []
    qa_metrics: list[dict[str, Any]] = []
    gen_metrics: list[dict[str, Any]] = []
    generated = 0
    started = time.perf_counter()
    used_attempts = 0
    for attempt in range(1, max_attempts + 1):
        used_attempts = attempt
        prompt = build_fix_prompt(item, kind, attempt)
        addon = qa_mod.generation_addon(brief, corrections)
        if addon:
            prompt = prompt + "\n\n" + addon
        try:
            edited, prediction_id = edit_with_references(
                [current, reference],
                prompt,
                model,
                limiter,
                timeout_s,
                fit="resize",
                target=target,
                quality=image_quality,
                metrics_sink=gen_metrics,
            )
        except Exception as exc:
            message = redact(str(exc) or exc.__class__.__name__)
            report = qa_mod.model_fail(
                country,
                src_ad,
                f"Generation error: {message}",
                "Regenerate the whole image. Do not crop, paste or composite.",
            )
            rank = qa_mod.attempt_rank(report, 0.0)
            if best is None or rank < best["rank"]:
                best = {"rank": rank, "image": None, "report": report, "attempt": used_attempts}
            corrections = report["fails"]
            print(f"[retry] {country} {src_ad} essai {attempt} : {message}", flush=True)
            continue
        prediction_ids.append(prediction_id)
        generated += 1
        if edited.size != target:
            edited = resize_exact(edited, target[0], target[1])
        report, metrics = qa_mod.evaluate_image(
            edited,
            reference,
            brief,
            checklist=checklist,
            model=qa_model,
            predict=lambda qa_model_name, payload: _predict(qa_model_name, payload, limiter, timeout_s),
        )
        qa_metrics.append(metrics)
        rank = qa_mod.attempt_rank(report, qa_mod.sharpness_score(edited))
        if best is None or rank < best["rank"]:
            best = {"rank": rank, "image": edited, "report": report, "attempt": used_attempts}
        if report["verdict"] == "PASS":
            print(f"[ok] {country} {src_ad} QA PASS essai {attempt}", flush=True)
            break
        corrections = report["fails"]
        items = ", ".join(str(fail["item"]) for fail in corrections)
        print(f"[retry] {country} {src_ad} essai {attempt} QA FAIL items {items}", flush=True)
    assert best is not None
    report = redact_report(best["report"])
    if best["image"] is not None:
        save_jpg(best["image"], destination, quality=JPG_QUALITY)
    write_report(report_path, report)
    elapsed = time.perf_counter() - started
    verdict = report["verdict"]
    status = "ok" if verdict == "PASS" else "échec"
    width = best["image"].size[0] if best["image"] is not None else None
    height = best["image"].size[1] if best["image"] is not None else None
    entry = fix_report_entry(
        country,
        src_ad,
        status,
        model,
        int(best["attempt"]),
        f"qa {verdict}, essai {best['attempt']}",
        out_label if destination.is_file() else None,
        width,
        height,
        prediction_ids,
    )
    entry["verdict"] = report["verdict"]
    entry["fails"] = report["fails"]
    entry["report"] = report_path.resolve().as_posix()
    if destination.is_file():
        entry["out"] = destination.resolve().as_posix()
    entry["attempts_this_run"] = used_attempts
    entry["seconds"] = round(elapsed, 3)
    entry["generated_images"] = generated
    entry["qa_metrics"] = qa_metrics
    entry["gen_metrics"] = gen_metrics
    entry["qa_model"] = qa_model
    print(f"[{verdict}] {country} {src_ad} -> {out_label}", flush=True)
    return entry


def fix_report_entry(
    country: str,
    src_ad: str,
    status: str,
    model: str,
    attempts: int,
    remark: str,
    out: str | None = None,
    width: int | None = None,
    height: int | None = None,
    prediction_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "country": country,
        "src_ad": src_ad,
        "status": status,
        "model": model,
        "attempts": attempts,
        "remark": remark,
        "out": out,
        "width": width,
        "height": height,
        "prediction_ids": prediction_ids or [],
    }


def apply_fix_once(
    current: Image.Image,
    reference: Image.Image,
    item: dict[str, Any],
    kind: str,
    model: str,
    attempt: int,
    limiter: RateLimiter,
    timeout_s: int,
) -> tuple[Image.Image, list[str]]:
    prompt = build_fix_prompt(item, kind, attempt)
    # Les grands visuels photo passent en 4K pour limiter le flou au resize final.
    resolution = "4K" if kind != "whiteboard" and max(current.size) >= 1600 else "2K"
    prediction_ids: list[str] = []
    if kind == "whiteboard" and current.size == (1254, 1254) and reference.size == current.size:
        box = WHITEBOARD_BOX
        crop = current.crop(box)
        ref_crop = reference.crop(box)
        edited, prediction_id = edit_with_references(
            [crop, ref_crop], prompt, model, limiter, timeout_s, resolution=resolution
        )
        prediction_ids.append(prediction_id)
        merged = paste_feather(current, edited, box)
        if "pack 3" in str(item["instruction"]).lower():
            # Deuxième passe ciblée : le pack 3 est le volet de droite.
            width, height = current.size
            pack_box = (width * 2 // 3, int(height * 0.42), width, int(height * 0.92))
            pack_prompt = (
                "Edit the first image, which is pack 3 of the ad. "
                "Set the product title to exactly EQUILIBRIO TERRESTRE in the same place as the reference title, "
                "with one thin small subtitle under it. Leave the cable unchanged. "
                "Do not add any other words. Keep the product photo, colors and framing."
            )
            pack_edit, pack_id = edit_with_references(
                [merged.crop(pack_box), reference.crop(pack_box)],
                pack_prompt,
                model,
                limiter,
                timeout_s,
                resolution=resolution,
            )
            prediction_ids.append(pack_id)
            merged = paste_feather(merged, pack_edit, pack_box, radius=10)
        return merged, prediction_ids
    edited, prediction_id = edit_with_references(
        [current, reference], prompt, model, limiter, timeout_s, resolution=resolution
    )
    prediction_ids.append(prediction_id)
    return edited, prediction_ids


def assess_fix(
    image: Image.Image,
    original: Image.Image,
    target: tuple[int, int],
    kind: str,
    ignore_boxes: list[tuple[int, int, int, int]] | None = None,
) -> str | None:
    if image.size != target:
        return f"taille {image.size[0]}x{image.size[1]} au lieu de {target[0]}x{target[1]}"
    reference = original
    if original.size != image.size:
        reference = center_crop_resize(original, image.size[0], image.size[1])
    score = sharpness_score(image)
    base = sharpness_score(reference)
    if base > 1 and score < base * 0.22:
        return f"image trop floue (netteté {score:.0f} contre {base:.0f})"
    if kind == "whiteboard" and image.size == original.size:
        import numpy as np

        before = np.asarray(original.convert("RGB"), dtype="int16")
        after = np.asarray(image.convert("RGB"), dtype="int16")
        if before.shape != after.shape:
            return "dimensions internes incohérentes"
        delta = np.abs(before - after).max(axis=2)
        mask = delta > 18
        for box in ignore_boxes or [WHITEBOARD_BOX]:
            x0, y0, x1, y1 = box
            mask[y0:y1, x0:x1] = False
        changed = int(mask.sum())
        if changed > 2500:
            return f"zones hors retouche modifiées ({changed} px)"
    return None


def process_fix_item(
    item: Any,
    inputs: Path,
    artifacts: Path,
    limiter: RateLimiter | None,
    timeout_s: int,
    dry_run: bool,
    force: bool,
    max_attempts: int,
    previous: dict[str, Any] | None,
    model: str,
    qa_enabled: bool = False,
    qa_model: str = qa_mod.QA_MODEL_DEFAULT,
    checklist: str = "",
    image_quality: str | None = None,
) -> dict[str, Any]:
    country = ""
    src_ad = ""
    try:
        if not isinstance(item, dict):
            raise ValueError("item de file invalide")
        country = str(item.get("country", "")).strip()
        src_ad = str(item.get("src_ad", "")).strip()
        if not COUNTRY_RE.match(country):
            raise ValueError(f"country invalide : {country}")
        if not SRC_AD_RE.match(src_ad) or ".." in src_ad:
            raise ValueError(f"src_ad invalide : {src_ad}")
        instruction = str(item.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("instruction vide")
        kind = task_kind(instruction)
        target = parse_target_size(item.get("target_size"))
        destination = output_path(artifacts, country, src_ad)
        out_label = destination.as_posix()
        prior_attempts = int(previous.get("attempts") or 0) if previous else 0
        if destination.is_file() and destination.stat().st_size > 0 and not force and not dry_run:
            existing = open_rgb(destination)
            passed_before = previous and previous.get("status") == "ok"
            if qa_enabled:
                passed_before = bool(
                    passed_before
                    and previous
                    and previous.get("verdict") == "PASS"
                    and image_report_path(artifacts, country, src_ad).is_file()
                )
            if existing.size == target and passed_before:
                print(f"[skipped] {country} {src_ad}", flush=True)
                kept = dict(previous or {})
                kept["out"] = out_label
                return kept
        if dry_run:
            current_path = resolve_fix_path(str(item["current"]), inputs)
            resolve_fix_path(str(item["ca_source"]), inputs)
            current = open_rgb(current_path)
            print(
                f"[dry_run] {country} {src_ad} {kind} {model} {current.size} -> {target[0]}x{target[1]}",
                flush=True,
            )
            return fix_report_entry(
                country,
                src_ad,
                "ok",
                model,
                0,
                f"dry-run {kind}",
                out_label,
                target[0],
                target[1],
            )
        assert limiter is not None
        current = open_rgb(resolve_fix_path(str(item["current"]), inputs))
        reference = open_rgb(resolve_fix_path(str(item["ca_source"]), inputs))
        if qa_enabled:
            if not checklist:
                checklist = qa_mod.load_checklist()
            return process_fix_item_qa(
                item,
                inputs,
                artifacts,
                limiter,
                timeout_s,
                max_attempts,
                model,
                qa_model,
                checklist,
                image_quality,
                current,
                reference,
                target,
                kind,
                destination,
            )
        start_attempt = prior_attempts + 1 if force else 1
        last_remark = "échec"
        last_final: Image.Image | None = None
        prediction_ids: list[str] = []
        used_attempts = prior_attempts
        for attempt in range(start_attempt, max_attempts + 1):
            used_attempts = attempt
            edited, ids = apply_fix_once(
                current, reference, item, kind, model, attempt, limiter, timeout_s
            )
            prediction_ids.extend(ids)
            final = center_crop_resize(edited, target[0], target[1])
            last_final = final
            ignore = [WHITEBOARD_BOX]
            if "pack 3" in instruction.lower() and current.size == (1254, 1254):
                width, height = current.size
                ignore.append((width * 2 // 3, int(height * 0.42), width, int(height * 0.92)))
            problem = assess_fix(final, current, target, kind, ignore)
            if problem:
                last_remark = problem
                print(f"[retry] {country} {src_ad} essai {attempt} : {problem}", flush=True)
                continue
            save_jpg(final, destination, quality=JPG_QUALITY)
            remark = f"{kind}, essai {attempt}"
            print(f"[ok] {country} {src_ad} {model} essai {attempt} -> {out_label}", flush=True)
            return fix_report_entry(
                country,
                src_ad,
                "ok",
                model,
                attempt,
                remark,
                out_label,
                final.size[0],
                final.size[1],
                prediction_ids,
            )
        print(f"[échec] {country} {src_ad} : {last_remark}", file=sys.stderr, flush=True)
        if last_final is not None:
            save_jpg(last_final, destination, quality=JPG_QUALITY)
        return fix_report_entry(
            country,
            src_ad,
            "échec",
            model,
            used_attempts,
            last_remark,
            out_label if destination.is_file() else None,
            last_final.size[0] if last_final is not None else None,
            last_final.size[1] if last_final is not None else None,
            prediction_ids,
        )
    except Exception as exc:
        message = redact(str(exc) or exc.__class__.__name__)
        label = f"{country} {src_ad}".strip() or "?"
        print(f"[échec] {label} : {message}", file=sys.stderr, flush=True)
        attempts = int(previous.get("attempts") or 0) if previous else 0
        entry = fix_report_entry(country, src_ad, "échec", model, attempts, message)
        if qa_enabled and country and src_ad:
            report = redact_report(
                qa_mod.model_fail(country, src_ad, message, "Fix the error and rerun the whole-image generation.")
            )
            report_file = image_report_path(artifacts, country, src_ad)
            write_report(report_file, report)
            entry["verdict"] = "FAIL"
            entry["fails"] = report["fails"]
            entry["report"] = report_file.resolve().as_posix()
        return entry


def parse_only(raw: str) -> set[tuple[str, str]] | None:
    text = raw.strip()
    if not text:
        return None
    selected: set[tuple[str, str]] = set()
    for part in text.split(","):
        if ":" not in part:
            raise SystemExit(f"--only invalide : {part}")
        country, src_ad = part.split(":", 1)
        selected.add((country.strip(), src_ad.strip()))
    return selected


def redact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "country": report.get("country") or "",
        "src_ad": report.get("src_ad") or "",
        "verdict": report.get("verdict") or "FAIL",
        "fails": [
            {
                "item": fail.get("item"),
                "detail": redact(str(fail.get("detail") or "")),
                "correction": redact(str(fail.get("correction") or "")),
            }
            for fail in report.get("fails") or []
        ],
    }


def _safe_name(value: str, pattern: re.Pattern[str], fallback: str) -> str:
    text = value.strip()
    if pattern.fullmatch(text) and ".." not in text:
        return text
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or fallback
    return cleaned[:64]


def write_batch_summary(
    artifacts: Path,
    indices: str,
    mode: str,
    ordered: list[dict[str, Any]],
    qa_model: str,
    image_quality: str,
) -> Path:
    qa_metrics: list[dict[str, Any]] = []
    generated = 0
    gen_seconds = 0.0
    items: list[dict[str, Any]] = []
    for entry in ordered:
        qa_metrics.extend(entry.get("qa_metrics") or [])
        generated += int(entry.get("generated_images") or 0)
        for metrics in entry.get("gen_metrics") or []:
            if metrics.get("predict_time") is not None:
                gen_seconds += float(metrics["predict_time"])
        items.append(
            {
                "country": entry.get("country"),
                "src_ad": entry.get("src_ad"),
                "verdict": entry.get("verdict"),
                "status": entry.get("status"),
                "attempts": entry.get("attempts"),
                "attempts_run": entry.get("attempts_this_run"),
                "seconds": entry.get("seconds"),
                "image": entry.get("out"),
                "report": entry.get("report"),
            }
        )
    passed = sum(1 for entry in ordered if entry.get("verdict") == "PASS")
    failed = sum(1 for entry in ordered if entry.get("verdict") == "FAIL")
    payload = {
        "mode": mode,
        "indices": indices or None,
        "pass": passed,
        "fail": failed,
        "total": len(ordered),
        "items": items,
        "cost": qa_mod.estimate_cost(
            qa_metrics,
            qa_model=qa_model,
            generated_images=generated,
            image_quality=image_quality or "auto",
        ),
    }
    payload["cost"]["gen_predict_seconds"] = round(gen_seconds, 3)
    path = summary_path(artifacts, indices)
    write_report(path, payload)
    print(f"batch: {passed} PASS, {failed} FAIL, {len(ordered)} total", flush=True)
    print(f"résumé: {path.resolve().as_posix()}", flush=True)
    for item in items:
        if item.get("image"):
            print(f"image: {item['image']}", flush=True)
        if item.get("report"):
            print(f"rapport: {item['report']}", flush=True)
    return path


def run_fix_queue(args: argparse.Namespace, queue: list[Any]) -> int:
    if not args.dry_run and not token_present():
        print(missing_token_message(), file=sys.stderr)
        return 1
    try:
        indexes = qa_mod.select_indices(len(queue), args.indices)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    selected = parse_only(args.only)
    qa_enabled = not args.no_qa
    if args.indices.strip():
        report_path = args.artifacts / f"report_{summary_path(args.artifacts, args.indices).stem.removeprefix('batch_summary_')}.json"
    else:
        report_path = args.artifacts / "report.json"
    previous = load_report_index(report_path)
    limiter = None if args.dry_run else RateLimiter(args.max_per_minute)
    checklist = ""
    if qa_enabled and not args.dry_run:
        try:
            checklist = qa_mod.load_checklist(args.checklist)
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    ordered: list[dict[str, Any]] = []
    for index in indexes:
        item = queue[index]
        if not isinstance(item, dict):
            entry = fix_report_entry("", "", "échec", ENGINES[args.engine], 0, "item de file invalide")
            ordered.append(entry)
            continue
        key = (str(item.get("country", "")).strip(), str(item.get("src_ad", "")).strip())
        if selected is not None and key not in selected:
            if not args.indices.strip() and key in previous:
                ordered.append(previous[key])
            continue
        entry = process_fix_item(
            item,
            args.inputs,
            args.artifacts,
            limiter,
            args.timeout,
            args.dry_run,
            args.force,
            args.max_attempts,
            previous.get(key),
            ENGINES[args.engine],
            qa_enabled=qa_enabled,
            qa_model=args.qa_model,
            checklist=checklist,
            image_quality=args.image_quality,
        )
        if entry.get("verdict"):
            strict = redact_report(
                {
                    "country": entry.get("country"),
                    "src_ad": entry.get("src_ad"),
                    "verdict": entry.get("verdict"),
                    "fails": entry.get("fails") or [],
                }
            )
            report_file = image_report_path(args.artifacts, str(entry.get("country") or ""), str(entry.get("src_ad") or ""))
            write_report(report_file, strict)
            entry["report"] = report_file.resolve().as_posix()
            entry["fails"] = strict["fails"]
            entry["verdict"] = strict["verdict"]
        ordered.append(entry)
        payload = {
            "max_per_minute": args.max_per_minute,
            "dry_run": bool(args.dry_run),
            "indices": args.indices or None,
            "qa": qa_enabled,
            "items": ordered,
        }
        write_report(report_path, payload)
    if qa_enabled and not args.dry_run:
        write_batch_summary(
            args.artifacts,
            args.indices,
            "fix",
            ordered,
            args.qa_model,
            args.image_quality or "auto",
        )
    counts: dict[str, int] = {}
    for item in ordered:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    print(f"{len(ordered)} item(s): {summary}", flush=True)
    print(f"rapport: {report_path.resolve().as_posix()}", flush=True)
    if any(item["status"] == "échec" for item in ordered):
        return 1
    if qa_enabled and any(item.get("verdict") == "FAIL" for item in ordered):
        return 1
    return 0


def run_qa_mode(args: argparse.Namespace) -> int:
    if not args.dry_run and not token_present():
        print(missing_token_message(), file=sys.stderr)
        return 1
    checklist = qa_mod.load_checklist(args.checklist)
    try:
        if args.qa_folder is not None:
            jobs = qa_mod.discover_folder(args.qa_folder)
        elif args.queue is not None:
            jobs = qa_mod.load_job_list(args.queue, args.inputs)
        else:
            print("mode qa : fournir --qa-folder ou --queue", file=sys.stderr)
            return 1
        indexes = qa_mod.select_indices(len(jobs), args.indices)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    jobs = [jobs[index] for index in indexes]
    limiter = None if args.dry_run else RateLimiter(args.max_per_minute)
    ordered: list[dict[str, Any]] = []
    file_error = False
    for job in jobs:
        started = time.perf_counter()
        country = _safe_name(str(job.get("country") or ""), COUNTRY_RE, "qa")
        src_ad = _safe_name(str(job.get("src_ad") or ""), SRC_AD_RE, "item")
        brief = dict(job["brief"])
        brief["country"] = country
        brief["src_ad"] = src_ad
        report_file = image_report_path(args.artifacts, country, src_ad)
        metrics: dict[str, Any] = {}
        try:
            if args.dry_run:
                image = qa_mod.open_rgb(job["adapted"])
                source = qa_mod.open_rgb(job["ca_source"])
                det = qa_mod.deterministic_fails(image, brief, source)
                report = qa_mod.model_fail(
                    country,
                    src_ad,
                    "Dry run: the vision model was not called, so this item cannot PASS.",
                    "Rerun without --dry-run to certify the image.",
                    det,
                )
            else:
                assert limiter is not None
                report, metrics = qa_mod.evaluate_image(
                    job["adapted"],
                    job["ca_source"],
                    brief,
                    checklist=checklist,
                    model=args.qa_model,
                    predict=lambda model, payload: _predict(model, payload, limiter, args.timeout),
                )
        except Exception as exc:
            file_error = True
            report = qa_mod.model_fail(
                country,
                src_ad,
                f"QA error: {exc}",
                "Fix the input files and rerun QA.",
            )
        report = redact_report(report)
        write_report(report_file, report)
        elapsed = round(time.perf_counter() - started, 3)
        print(f"[{report['verdict']}] {country} {src_ad} ({elapsed}s)", flush=True)
        ordered.append(
            {
                "country": country,
                "src_ad": src_ad,
                "verdict": report["verdict"],
                "status": "ok" if report["verdict"] == "PASS" else "échec",
                "attempts": 0,
                "seconds": elapsed,
                "out": job["adapted"].resolve().as_posix(),
                "report": report_file.resolve().as_posix(),
                "qa_metrics": [metrics] if metrics else [],
                "generated_images": 0,
            }
        )
    write_batch_summary(args.artifacts, args.indices, "qa", ordered, args.qa_model, args.image_quality or "auto")
    if file_error and not args.dry_run:
        return 1
    if args.dry_run:
        return 1 if file_error else 0
    if any(entry["verdict"] == "FAIL" for entry in ordered):
        return 1
    return 0


def process_item(
    item: Any,
    inputs: Path,
    artifacts: Path,
    model: str,
    dry_run: bool,
    force: bool,
    limiter: RateLimiter | None,
    timeout_s: int,
) -> dict[str, Any]:
    country = ""
    src_ad = ""
    classification = ""
    destination: Path | None = None
    try:
        if not isinstance(item, dict):
            raise ValueError("item de file invalide")
        country, src_ad, classification = validate_identity(item)
        destination = output_path(artifacts, country, src_ad)
        out_label = relative_output(destination)
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            width = height = None
            try:
                existing = open_rgb(destination)
                width, height = existing.size
            except Exception:
                width = height = None
            print(f"[skipped] {country} {src_ad} -> {out_label}", flush=True)
            return blank_report(
                country, src_ad, classification, model, out_label, "skipped", width=width, height=height
            )
        prepared = prepare_item(item, inputs)
        if prepared["hold"]:
            reason = prepared["brief"].get("hold_reason") or "hold"
            print(f"[hold] {country} {src_ad} {reason}", flush=True)
            return blank_report(country, src_ad, classification, model, out_label, "hold")
        image = prepared["image"]
        width, height = image.size
        if dry_run:
            print(f"[dry_run] {country} {src_ad} {width}x{height} -> {out_label}", flush=True)
            return blank_report(
                country, src_ad, classification, model, out_label, "dry_run", width=width, height=height
            )
        assert limiter is not None
        prompt = build_prompt(prepared["brief"], prepared["copy_text"], classification)
        fitted, prediction_id = adapt_image(image, prompt, model, limiter, timeout_s)
        save_jpg(fitted, destination, quality=JPG_QUALITY)
        print(f"[ok] {country} {src_ad} {prediction_id} -> {out_label}", flush=True)
        return blank_report(
            country,
            src_ad,
            classification,
            model,
            out_label,
            "ok",
            prediction_id=prediction_id,
            width=fitted.size[0],
            height=fitted.size[1],
        )
    except Exception as exc:
        message = redact(str(exc) or exc.__class__.__name__)
        label = f"{country} {src_ad}".strip() or "?"
        print(f"[error] {label} : {message}", file=sys.stderr, flush=True)
        out_label = relative_output(destination) if destination is not None else None
        prediction_id = getattr(exc, "prediction_id", None)
        return blank_report(
            country,
            src_ad,
            classification,
            model,
            out_label,
            "error",
            error=message,
            prediction_id=prediction_id if isinstance(prediction_id, str) else None,
        )


def load_queue(path: Path) -> list[Any]:
    if not path.is_file():
        raise SystemExit(f"file introuvable : {path}")
    data = load_json(path)
    if not isinstance(data, list):
        raise SystemExit("la file doit être une liste JSON d'items")
    return data


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapte une file d'images publicitaires Ortho Terre via Replicate."
    )
    parser.add_argument("--queue", type=Path, default=None, help="File JSON (liste d'items)")
    parser.add_argument("--inputs", type=Path, default=None, help="Dossier des sources, briefs et copies")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Dossier de sortie")
    parser.add_argument("--max-per-minute", type=int, default=6, help="Prédictions max par minute (défaut 6)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Modèle Replicate du mode adapt (défaut {DEFAULT_MODEL})")
    parser.add_argument(
        "--engine",
        choices=tuple(ENGINES),
        default="gpt-image",
        help="Moteur du mode fix : gpt-image (défaut, openai/gpt-image-2.5-sunburst) ou nano-banana",
    )
    parser.add_argument("--timeout", type=int, default=900, help="Attente max d'une prédiction, en secondes")
    parser.add_argument("--dry-run", action="store_true", help="Valide la file et les chemins, sans Replicate")
    parser.add_argument("--force", action="store_true", help="Régénère un item même si le JPG existe")
    parser.add_argument(
        "--mode",
        choices=("auto", "adapt", "fix", "qa"),
        default="auto",
        help="auto détecte une file de corrections (instruction + current). qa juge des images déjà adaptées.",
    )
    parser.add_argument("--max-attempts", type=int, default=3, help="Essais max par image en mode fix (boucle QA)")
    parser.add_argument(
        "--only",
        default="",
        help="Sous-ensemble country:src_ad,country:src_ad (mode fix)",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="Sous-ensemble d'index de la file : 0-4 (inclus), 0:5 (fin exclue), 1,4,8. "
        "Plusieurs workers prennent des plages différentes.",
    )
    parser.add_argument(
        "--qa-folder",
        type=Path,
        default=None,
        help="Dossier QA (sous-dossiers ou manifest.json). Mode qa uniquement.",
    )
    parser.add_argument(
        "--qa-model",
        default=qa_mod.QA_MODEL_DEFAULT,
        help=f"Modèle vision Replicate pour le QA (défaut {qa_mod.QA_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Mode fix sans boucle QA (ancien contrôle taille/flou, avec collage ciblé)",
    )
    parser.add_argument(
        "--image-quality",
        choices=("low", "medium", "high", "xhigh", "max", "auto"),
        default=None,
        help="Qualité gpt-image. Défaut du modèle : auto.",
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=None,
        help="Checklist QA. Défaut : QA_CHECKLIST.md à la racine du dépôt.",
    )
    args = parser.parse_args(argv)
    if args.max_per_minute < 1:
        parser.error("--max-per-minute doit être >= 1")
    if args.timeout < 1:
        parser.error("--timeout doit être >= 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts doit être >= 1")
    if args.mode == "qa":
        if args.qa_folder is None and args.queue is None:
            parser.error("mode qa : fournir --qa-folder ou --queue")
        if args.qa_folder is not None and not args.qa_folder.is_dir():
            parser.error(f"dossier QA introuvable : {args.qa_folder}")
        if args.inputs is not None and not args.inputs.is_dir():
            parser.error(f"dossier d'entrées introuvable : {args.inputs}")
    else:
        if args.queue is None:
            parser.error("--queue est requis hors mode qa sur dossier")
        if args.inputs is None or not args.inputs.is_dir():
            parser.error(f"dossier d'entrées introuvable : {args.inputs}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode == "qa":
        return run_qa_mode(args)
    queue = load_queue(args.queue)
    use_fix = args.mode == "fix" or (args.mode == "auto" and is_fix_queue(queue))
    if use_fix:
        return run_fix_queue(args, queue)
    try:
        queue = [queue[index] for index in qa_mod.select_indices(len(queue), args.indices)]
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not args.dry_run and not token_present():
        print(missing_token_message(), file=sys.stderr)
        return 1
    limiter = None if args.dry_run else RateLimiter(args.max_per_minute)
    report_path = args.artifacts / "report.json"
    items: list[dict[str, Any]] = []
    payload = {
        "model": args.model,
        "max_per_minute": args.max_per_minute,
        "dry_run": bool(args.dry_run),
        "items": items,
    }
    for item in queue:
        items.append(
            process_item(
                item,
                args.inputs,
                args.artifacts,
                args.model,
                args.dry_run,
                args.force,
                limiter,
                args.timeout,
            )
        )
        write_report(report_path, payload)
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    print(f"{len(items)} item(s): {summary}", flush=True)
    print(f"rapport: {relative_output(report_path)}", flush=True)
    if any(item["status"] == "error" for item in items):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
