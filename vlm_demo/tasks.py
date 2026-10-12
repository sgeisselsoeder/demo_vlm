"""Aufgabendefinitionen und vorgefertigte Prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DetectClass(BaseModel):
    query: str                 # was das Modell suchen soll (englisch funktioniert am zuverlässigsten)
    label: str | None = None   # Anzeigename in der GUI


class Attribute(BaseModel):
    """Zweite Stufe: Jede gefundene Box wird ausgeschnitten und mit einer Ja/Nein-Frage geprüft."""
    question: str
    yes_label: str
    no_label: str | None = None  # None = Klassenname behalten


class Task(BaseModel):
    mode: Literal["detect", "ask", "watch"]
    title: str = "Eigener Prompt"
    prompt: str = ""
    prompt_en: str | None = None  # Alternative für Modelle, die nur Englisch gut können
    classes: list[DetectClass] = Field(default_factory=list)
    attribute: Attribute | None = None
    max_new_tokens: int = Field(default=200, ge=1, le=1024)


class Preset(BaseModel):
    id: str
    icon: str
    task: Task
    description: str = ""


YES_NO = " Answer only with yes or no."


def _p(id, icon, description="", **task) -> Preset:
    return Preset(id=id, icon=icon, description=description, task=Task(**task))


PRESETS: list[Preset] = [
    # ---- Erkennen --------------------------------------------------------------------------
    _p("glasses", "👓", "Zweistufig: erst alle Personen lokalisieren, dann jeden Ausschnitt fragen: Brille ja/nein?",
       mode="detect", title="Person / Person mit Brille",
       classes=[DetectClass(query="person", label="Person")],
       attribute=Attribute(question="Is this person wearing glasses or sunglasses?" + YES_NO,
                           yes_label="Person mit Brille", no_label="Person")),
    _p("persons", "🧍", "Alle Personen finden und zählen",
       mode="detect", title="Personen", classes=[DetectClass(query="person", label="Person")]),
    _p("face_hand", "🖐️", "Gesichter und Hände lokalisieren",
       mode="detect", title="Gesichter & Hände",
       classes=[DetectClass(query="face", label="Gesicht"), DetectClass(query="hand", label="Hand")]),
    _p("smile", "😄", "Gesichter finden und prüfen, ob gelächelt wird",
       mode="detect", title="Lächel-Detektor",
       classes=[DetectClass(query="face", label="Gesicht")],
       attribute=Attribute(question="Is this person smiling?" + YES_NO,
                           yes_label="Lächelt 😄", no_label="Ernst 😐")),
    _p("helmet", "⛑️", "Arbeitsschutz-Check: Trägt jede Person einen Helm?",
       mode="detect", title="Helm-Check (PSA)",
       classes=[DetectClass(query="person", label="Person")],
       attribute=Attribute(question="Is this person wearing a helmet or hard hat on their head?" + YES_NO,
                           yes_label="Mit Helm ✅", no_label="Ohne Helm ⚠️")),
    _p("desk", "☕", "Typische Gegenstände am Arbeitsplatz",
       mode="detect", title="Schreibtisch-Objekte",
       classes=[DetectClass(query=q, label=l) for q, l in [
           ("cup", "Tasse"), ("bottle", "Flasche"), ("cell phone", "Handy"), ("pen", "Stift"),
           ("keyboard", "Tastatur"), ("laptop", "Laptop"), ("book", "Buch")]]),
    _p("everything", "🔎", "Offenes Vokabular: das Modell entscheidet selbst, was es benennt",
       mode="detect", title="Alles erkennen", classes=[]),
    # ---- Fragen -----------------------------------------------------------------------------
    _p("describe", "📝", mode="ask", title="Szene beschreiben",
       prompt="Beschreibe kurz und präzise, was auf dem Bild zu sehen ist.",
       prompt_en="Describe briefly and precisely what you see in this image."),
    _p("people", "🧑", mode="ask", title="Personen beschreiben",
       prompt="Beschreibe jede Person im Bild: Kleidung, Haltung, Gesichtsausdruck und Accessoires (z. B. Brille).",
       prompt_en="Describe each person in the image: clothing, posture, facial expression and accessories such as glasses."),
    _p("emotion", "🎭", mode="ask", title="Emotion", max_new_tokens=20,
       prompt="Welche Emotion zeigt die Person? Antworte mit einem Wort und einem passenden Emoji.",
       prompt_en="Which emotion does the person show? Answer with one word."),
    _p("gesture", "✌️", mode="ask", title="Handgeste", max_new_tokens=40,
       prompt="Welche Handgeste zeigt die Person (z. B. Daumen hoch, Peace-Zeichen, Winken, Faust, OK)? Antworte kurz.",
       prompt_en="Which hand gesture is the person making (e.g. thumbs up, peace sign, waving, fist, OK)? Answer briefly."),
    _p("fingers", "🔢", mode="ask", title="Finger zählen", max_new_tokens=10,
       prompt="Wie viele Finger hält die Person insgesamt hoch? Antworte nur mit der Zahl.",
       prompt_en="How many fingers is the person holding up in total? Answer with a number only."),
    _p("ocr", "🔤", mode="ask", title="Text lesen (OCR)",
       prompt="Lies allen sichtbaren Text im Bild wortgetreu vor. Wenn kein Text sichtbar ist, antworte „kein Text“.",
       prompt_en="Transcribe all visible text in the image verbatim."),
    _p("whatisit", "🧐", mode="ask", title="Was halte ich?", max_new_tokens=80,
       prompt="Was hält die Person in der Hand? Benenne den Gegenstand und erkläre in einem Satz, wofür man ihn benutzt.",
       prompt_en="What is the person holding? Name the object and explain in one sentence what it is used for."),
    _p("compliment", "💐", mode="ask", title="Kompliment", max_new_tokens=80,
       prompt="Mache der Person im Bild ein freundliches, originelles Kompliment in einem Satz, passend zu dem, was du siehst.",
       prompt_en="Give the person in the image a kind, original one-sentence compliment based on what you see."),
    _p("poem", "🎤", mode="ask", title="Vierzeiler", max_new_tokens=120,
       prompt="Schreibe einen kurzen, lustigen Vierzeiler über diese Szene.",
       prompt_en="Write a short, funny four-line poem about this scene."),
    # ---- Überwachen -------------------------------------------------------------------------
    _p("w_thumbs", "👍", "Alarm, sobald jemand den Daumen hebt",
       mode="watch", title="Daumen hoch?", prompt="Is someone in the image giving a thumbs up?"),
    _p("w_hand", "🙋", "Wie im Hörsaal: Meldet sich jemand?",
       mode="watch", title="Hand gehoben?", prompt="Is a person raising their hand above their head?"),
    _p("w_glasses", "👓", mode="watch", title="Brille im Bild?",
       prompt="Is any person in the image wearing glasses?"),
    _p("w_phone", "📱", mode="watch", title="Handy in der Hand?",
       prompt="Is someone holding a mobile phone in their hand?"),
    _p("w_empty", "🚪", "Schlägt an, wenn der Platz verlassen wurde",
       mode="watch", title="Niemand da?", prompt="Is the image free of people, with no person visible at all?"),
]
PRESETS_BY_ID = {p.id: p for p in PRESETS}
DEFAULT_PRESET = "glasses"
