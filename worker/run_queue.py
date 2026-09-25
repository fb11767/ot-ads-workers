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


def model_input(model: str, prompt: str, image_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "image_input": [image_path],
        "output_format": "jpg",
        "aspect_ratio": "match_input_image",
    }
    if model.endswith("pro"):
        payload["resolution"] = "2K"
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
    parser.add_argument("--queue", required=True, type=Path, help="File JSON (liste d'items)")
    parser.add_argument("--inputs", required=True, type=Path, help="Dossier des sources, briefs et copies")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Dossier de sortie")
    parser.add_argument("--max-per-minute", type=int, default=6, help="Prédictions max par minute (défaut 6)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Modèle Replicate (défaut {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=900, help="Attente max d'une prédiction, en secondes")
    parser.add_argument("--dry-run", action="store_true", help="Valide la file et les chemins, sans Replicate")
    parser.add_argument("--force", action="store_true", help="Régénère un item même si le JPG existe")
    args = parser.parse_args(argv)
    if args.max_per_minute < 1:
        parser.error("--max-per-minute doit être >= 1")
    if args.timeout < 1:
        parser.error("--timeout doit être >= 1")
    if not args.inputs.is_dir():
        parser.error(f"dossier d'entrées introuvable : {args.inputs}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.dry_run and not token_present():
        print(missing_token_message(), file=sys.stderr)
        return 1
    queue = load_queue(args.queue)
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
