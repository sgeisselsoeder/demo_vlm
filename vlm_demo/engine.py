"""Inferenz-Schleife: holt das neueste Kamerabild, führt die aktuelle Aufgabe aus, verteilt Ergebnisse."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import threading
import time
import traceback

import torch
from PIL import Image

from .camera import Camera
from .models import MODELS_BY_ID, VLMBackend, create_backend, parse_yes_no
from .tasks import PRESETS_BY_ID, DEFAULT_PRESET, YES_NO, Task

log = logging.getLogger("vlm_demo.engine")


class Hub:
    """Verteilt Nachrichten aus beliebigen Threads an alle verbundenen WebSocket-Clients."""

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()

    def publish(self, msg: dict) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self._fanout, msg)

    def _fanout(self, msg: dict) -> None:
        for q in self.clients:
            if q.qsize() < 200:  # langsame Clients verlieren lieber Nachrichten als Speicher
                q.put_nowait(msg)


def _jpeg_b64(img: Image.Image, max_side: int = 480, quality: int = 75) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _crop(full: Image.Image, box, margin: float = 0.12, max_side: int = 448) -> Image.Image:
    """Schneidet eine (normierte) Box mit etwas Rand aus dem hochaufgelösten Originalbild aus."""
    x1, y1, x2, y2 = box
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    W, H = full.size
    crop = full.crop((max(0, int((x1 - mw) * W)), max(0, int((y1 - mh) * H)),
                      min(W, int((x2 + mw) * W)), min(H, int((y2 + mh) * H))))
    if min(crop.size) < 64:  # winzige Ausschnitte hochskalieren, sonst sieht das Modell nichts
        s = 64 / min(crop.size)
        crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.BICUBIC)
    crop.thumbnail((max_side, max_side))
    return crop


class Engine:
    def __init__(self, camera: Camera, hub: Hub):
        self.camera, self.hub = camera, hub
        self.backend: VLMBackend | None = None
        self.loading: str | None = None
        self.error: str | None = None
        self.task: Task = PRESETS_BY_ID[DEFAULT_PRESET].task
        self.preset_id: str | None = DEFAULT_PRESET
        self.continuous = False
        self.interval = 0.0       # Pause zwischen zwei Läufen in Sekunden
        self.max_side = 640       # Bildgröße fürs Modell (bestimmt Anzahl Bild-Tokens)
        self.busy = False
        self.runs_per_sec = 0.0
        self._once = False
        self._run_gen = 0         # wird erhöht, um laufende Generierung abzubrechen
        self._pending_model: str | None = None
        self._watch_state: bool | None = None
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)

    # ---------------------------------------------------------------- Steuerung (aus API-Threads)
    def start(self) -> None:
        self._thread.start()
        threading.Thread(target=self._status_loop, name="status", daemon=True).start()

    def request_model(self, model_id: str) -> None:
        if model_id not in MODELS_BY_ID:
            raise KeyError(model_id)
        self._pending_model = model_id
        self._abort()

    def set_task(self, task: Task, preset_id: str | None = None) -> None:
        self.task, self.preset_id = task, preset_id
        self._watch_state = None
        self._abort()
        self.publish_status()

    def set_running(self, action: str) -> None:
        if action == "start":
            self.continuous = True
        elif action == "stop":
            self.continuous = False
            self._abort()
        elif action == "once":
            self._once = True
        self._wake.set()
        self.publish_status()

    def _abort(self) -> None:
        self._run_gen += 1
        self._wake.set()

    # ---------------------------------------------------------------- Status
    def status(self) -> dict:
        free, total = torch.cuda.mem_get_info()
        return {
            "type": "status",
            "model_id": self.backend.spec.id if self.backend else None,
            "loading": self.loading,
            "error": self.error or self.camera.error,
            "continuous": self.continuous,
            "busy": self.busy,
            "interval": self.interval,
            "max_side": self.max_side,
            "camera": self.camera.index,
            "mirror": self.camera.mirror,
            "preset_id": self.preset_id,
            "task": self.task.model_dump(),
            "runs_per_sec": round(self.runs_per_sec, 2),
            "vram_used": round((total - free) / 2**30, 2),
            "vram_total": round(total / 2**30, 2),
            "vram_model": round(torch.cuda.memory_allocated() / 2**30, 2),
        }

    def publish_status(self) -> None:
        self.hub.publish(self.status())

    def _status_loop(self) -> None:
        while True:
            time.sleep(1.0)
            self.publish_status()

    # ---------------------------------------------------------------- Worker
    def _load_model(self, model_id: str) -> None:
        spec = MODELS_BY_ID[model_id]
        self.loading, self.error = spec.name, None
        self.publish_status()
        if self.backend:
            self.backend.unload()
            self.backend = None
        try:
            t0 = time.time()
            backend = create_backend(spec)
            backend.load()
            self.backend = backend
            log.info("%s geladen in %.1f s", spec.name, time.time() - t0)
        except Exception as e:  # noqa: BLE001 - Fehler soll in der GUI landen
            traceback.print_exc()
            self.error = f"Laden von {spec.name} fehlgeschlagen: {e}"
            torch.cuda.empty_cache()
        finally:
            self.loading = None
            self.publish_status()

    def _loop(self) -> None:
        last_t = None
        while True:
            if self._pending_model:
                model_id, self._pending_model = self._pending_model, None
                self._load_model(model_id)
                continue
            if not self.backend or not (self.continuous or self._once):
                self._wake.wait(0.2)
                self._wake.clear()
                last_t = None
                continue
            self._once = False
            frame = self.camera.latest()
            if frame is None:
                time.sleep(0.1)
                continue

            gen = self._run_gen
            self.busy = True
            try:
                result = self._run(self.task, frame, gen)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self.hub.publish({"type": "error", "message": f"{type(e).__name__}: {e}"})
                time.sleep(1.0)
                continue
            finally:
                self.busy = False
            if result is not None and gen == self._run_gen:
                self.hub.publish(result)
                now = time.time()
                if last_t:
                    self.runs_per_sec = 0.7 * self.runs_per_sec + 0.3 / max(now - last_t, 1e-3)
                last_t = now
            if self.continuous and self.interval > 0 and gen == self._run_gen:
                self._wake.clear()
                self._wake.wait(self.interval)

    def _run(self, task: Task, frame, gen: int) -> dict | None:
        backend = self.backend
        full = Image.fromarray(frame)
        img = full.copy()
        img.thumbnail((self.max_side, self.max_side))
        should_stop = lambda: gen != self._run_gen  # noqa: E731
        prompt = task.prompt_en if (task.prompt_en and backend.spec.family == "moondream") else task.prompt

        t0 = time.perf_counter()
        res: dict = {"type": "result", "mode": task.mode, "title": task.title,
                     "model": backend.spec.name, "prompt": prompt, "ts": time.time()}

        if task.mode == "detect":
            queries = [c.query.strip() for c in task.classes if c.query.strip()]
            boxes, raw = backend.detect(img, queries)
            names = {c.query.strip().lower(): (c.label or c.query) for c in task.classes}
            items = [{"label": names.get(b.label.strip().lower(), b.label or "Objekt"),
                      "box": b.to_dict()["box"]} for b in boxes]
            t_detect = time.perf_counter() - t0
            if task.attribute and items and not should_stop():
                crops = [_crop(full, it["box"]) for it in items]
                answers = backend.ask_many(crops, task.attribute.question, max_new_tokens=4)
                for it, ans in zip(items, answers):
                    yes = parse_yes_no(ans)
                    it["attr"] = yes
                    if yes:
                        it["label"] = task.attribute.yes_label
                    elif task.attribute.no_label:
                        it["label"] = task.attribute.no_label
                raw += "\n\n# Stufe 2 – " + task.attribute.question + "\n" + "\n".join(
                    f"Box {i + 1}: {a}" for i, a in enumerate(answers))
                res["stage_ms"] = {"detect": round(t_detect * 1000), "attribute": round((time.perf_counter() - t0 - t_detect) * 1000)}
            counts: dict[str, int] = {}
            for it in items:
                counts[it["label"]] = counts.get(it["label"], 0) + 1
            res.update(boxes=items, counts=counts, raw=raw)

        elif task.mode == "ask":
            self.hub.publish({"type": "stream_start", "title": task.title})
            text = backend.ask(img, prompt, task.max_new_tokens,
                               on_text=lambda s: self.hub.publish({"type": "token", "text": s}),
                               should_stop=should_stop)
            res.update(text=text, raw=text)

        elif task.mode == "watch":
            raw = backend.ask(img, prompt.strip() + YES_NO, max_new_tokens=4)
            answer = parse_yes_no(raw)
            res.update(answer=answer, raw=raw, triggered=bool(answer) and self._watch_state is not True)
            self._watch_state = answer

        if should_stop():
            return None
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        res["image_size"] = list(img.size)
        res["thumb"] = _jpeg_b64(img)
        return res
