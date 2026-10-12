"""VLM-Backends mit einheitlicher Schnittstelle: ask() für Freitext, detect() für Bounding-Boxes."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from PIL import Image

TextCallback = Callable[[str], None]
StopFlag = Callable[[], bool]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    hf_id: str
    family: str  # "qwen3vl" | "moondream"
    description: str
    vram_gb: float
    quant: str | None = None
    revision: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: list[ModelSpec] = [
    ModelSpec(
        id="qwen3-vl-2b",
        name="Qwen3-VL 2B",
        hf_id="Qwen/Qwen3-VL-2B-Instruct",
        family="qwen3vl",
        description="Schnellstes Modell, versteht Deutsch, gutes Grounding (bf16)",
        vram_gb=4.5,
    ),
    ModelSpec(
        id="qwen3-vl-4b-nf4",
        name="Qwen3-VL 4B (4-bit)",
        hf_id="Qwen/Qwen3-VL-4B-Instruct",
        family="qwen3vl",
        description="Genauer und sprachlich stärker, per bitsandbytes NF4 quantisiert",
        vram_gb=3.8,
        quant="nf4",
    ),
    ModelSpec(
        id="moondream2",
        name="Moondream 2",
        hf_id="vikhyatk/moondream2",
        family="moondream",
        description="Kompaktes VLM mit nativer Objekterkennung; Prompts am besten auf Englisch",
        vram_gb=4.0,
        revision="2025-06-21",
    ),
]
MODELS_BY_ID = {m.id: m for m in MODELS}


@dataclass
class Box:
    label: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, normiert auf 0..1

    def to_dict(self) -> dict:
        return {"label": self.label, "box": [round(v, 4) for v in self.box]}


def _clip_box(x1, y1, x2, y2) -> tuple[float, float, float, float] | None:
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return x1, y1, x2, y2


_BBOX_RE = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?'
)


def parse_qwen_boxes(text: str, scale: float = 1000.0) -> list[tuple[str, tuple]]:
    """Parst Qwen3-VL-Grounding-Ausgaben ([{"bbox_2d": [...], "label": ...}], Koordinaten 0..1000)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    items: list = []
    try:
        data = json.loads(cleaned)
        items = data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        # Abgeschnittene oder leicht kaputte Ausgabe: Boxen einzeln herausfischen
        for m in _BBOX_RE.finditer(cleaned):
            items.append({"bbox_2d": [float(v) for v in m.groups()[:4]], "label": m.group(5) or ""})
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bb = it.get("bbox_2d") or it.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            coords = [float(v) / scale for v in bb]
        except (TypeError, ValueError):
            continue
        out.append((str(it.get("label", "")), tuple(coords)))
    return out


def parse_yes_no(text: str) -> bool | None:
    t = text.strip().lower()
    if re.match(r"^\W*(yes|ja|true|y\b|correct|richtig)", t):
        return True
    if re.match(r"^\W*(no|nein|false|n\b|incorrect|falsch)", t):
        return False
    return None


class VLMBackend:
    spec: ModelSpec

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def load(self) -> None: ...

    def unload(self) -> None:
        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def ask(self, image: Image.Image, prompt: str, max_new_tokens: int = 256,
            on_text: TextCallback | None = None, should_stop: StopFlag | None = None) -> str:
        raise NotImplementedError

    def ask_many(self, images: list[Image.Image], prompt: str, max_new_tokens: int = 8) -> list[str]:
        """Dieselbe Frage zu mehreren Bildausschnitten (z. B. Attributprüfung pro Person)."""
        return [self.ask(img, prompt, max_new_tokens) for img in images]

    def detect(self, image: Image.Image, labels: list[str]) -> tuple[list[Box], str]:
        """Liefert (Boxen, Rohausgabe des Modells)."""
        raise NotImplementedError


class _CallbackStreamer:
    """Minimaler Streamer für generate(): dekodiert inkrementell und ruft on_text mit neuem Text auf."""

    def __init__(self, tokenizer, on_text: TextCallback):
        self.tokenizer, self.on_text = tokenizer, on_text
        self.tokens: list[int] = []
        self.printed = 0
        self.skip_prompt = True

    def put(self, value):
        if self.skip_prompt:  # erster Aufruf enthält den Prompt
            self.skip_prompt = False
            return
        self.tokens.extend(value.flatten().tolist())
        text = self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        if text.endswith("�"):  # unvollständiges UTF-8-Zeichen abwarten
            return
        if len(text) > self.printed:
            self.on_text(text[self.printed:])
            self.printed = len(text)

    def end(self):
        pass


class _FlagStopping:
    def __init__(self, should_stop: StopFlag):
        self.should_stop = should_stop

    def __call__(self, input_ids, scores, **kwargs):
        stop = self.should_stop()
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


class QwenVLBackend(VLMBackend):
    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        kwargs: dict = {"dtype": torch.bfloat16, "device_map": "cuda", "attn_implementation": "sdpa"}
        if self.spec.quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                # Vision-Encoder in bf16 lassen: kleiner Speichergewinn, aber spürbar bessere Boxen
                llm_int8_skip_modules=["visual", "lm_head"],
            )
        self.model = AutoModelForImageTextToText.from_pretrained(self.spec.hf_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(self.spec.hf_id)
        self.processor.tokenizer.padding_side = "left"

    def _inputs(self, images: list[Image.Image], prompts: list[str]):
        conversations = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": p}]}]
            for img, p in zip(images, prompts)
        ]
        return self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
        ).to(self.model.device)

    @torch.inference_mode()
    def _generate(self, images, prompts, max_new_tokens, on_text=None, should_stop=None) -> list[str]:
        inputs = self._inputs(images, prompts)
        kwargs = {}
        if on_text:
            kwargs["streamer"] = _CallbackStreamer(self.processor.tokenizer, on_text)
        if should_stop:
            from transformers import StoppingCriteriaList
            kwargs["stopping_criteria"] = StoppingCriteriaList([_FlagStopping(should_stop)])
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None, **kwargs,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]

    def ask(self, image, prompt, max_new_tokens=256, on_text=None, should_stop=None) -> str:
        return self._generate([image], [prompt], max_new_tokens, on_text, should_stop)[0]

    def ask_many(self, images, prompt, max_new_tokens=8) -> list[str]:
        if not images:
            return []
        return self._generate(images, [prompt] * len(images), max_new_tokens)

    def detect(self, image, labels):
        if labels:
            cats = ", ".join(labels)
            prompt = (f'Locate every instance that belongs to the following categories: "{cats}". '
                      "Report bbox coordinates in JSON format.")
        else:
            prompt = "Locate every distinct object in the image. Report bbox coordinates in JSON format."
        raw = self.ask(image, prompt, max_new_tokens=1024)
        boxes = []
        for label, coords in parse_qwen_boxes(raw):
            if (c := _clip_box(*coords)) is not None:
                boxes.append(Box(label, c))
        return boxes, raw


class MoondreamBackend(VLMBackend):
    def load(self) -> None:
        # Moondreams Remote-Code ist nicht mit from_pretrained() aus transformers v5 kompatibel,
        # daher Klasse direkt instanziieren und Gewichte selbst laden.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        rev = self.spec.revision
        config = AutoConfig.from_pretrained(self.spec.hf_id, revision=rev, trust_remote_code=True)
        cls = get_class_from_dynamic_module("hf_moondream.HfMoondream", self.spec.hf_id, revision=rev)
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                model = cls(config)
        finally:
            torch.set_default_dtype(default_dtype)
        weights = load_file(hf_hub_download(self.spec.hf_id, "model.safetensors", revision=rev), device="cpu")
        model.load_state_dict(weights, strict=True)
        self.model = model.eval()

    @torch.inference_mode()
    def ask(self, image, prompt, max_new_tokens=256, on_text=None, should_stop=None) -> str:
        settings = {"max_tokens": max_new_tokens, "variant": None}
        if not on_text:
            return self.model.query(image, prompt, settings=settings)["answer"].strip()
        parts = []
        for chunk in self.model.query(image, prompt, stream=True, settings=settings)["answer"]:
            parts.append(chunk)
            on_text(chunk)
            if should_stop and should_stop():
                break
        return "".join(parts).strip()

    @torch.inference_mode()
    def detect(self, image, labels):
        boxes, raw = [], {}
        for label in labels or ["object"]:
            objs = self.model.detect(image, label)["objects"]
            raw[label] = objs
            for o in objs:
                if (c := _clip_box(o["x_min"], o["y_min"], o["x_max"], o["y_max"])) is not None:
                    boxes.append(Box(label, c))
        return boxes, json.dumps(raw, indent=1)


def create_backend(spec: ModelSpec) -> VLMBackend:
    return {"qwen3vl": QwenVLBackend, "moondream": MoondreamBackend}[spec.family](spec)
