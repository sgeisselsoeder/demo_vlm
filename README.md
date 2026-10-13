# VLM Live-Demo

Live-Auswertung der Webcam mit modernen Vision-Language-Modellen, lokal auf der GPU
(getestet: RTX 5070 Laptop, 8 GB VRAM). Die Bedienung läuft über eine Web-GUI mit fertigen
und freien Prompts.

```bash
./run.sh                                   # Kamera /dev/video0, Qwen3-VL 2B
./run.sh --camera 4 --model qwen3-vl-4b-nf4
./run.sh --host 0.0.0.0                    # auch von anderen Geräten im Netz erreichbar
```

Danach im Browser **http://localhost:8000** öffnen.

## Modelle

Es ist immer nur ein Modell im VRAM. Der Wechsel läuft über das Dropdown oben rechts und dauert 4–12 s.

| Modell | VRAM | Detektion | Antwort (Szene) | Ja/Nein | Bemerkung |
|---|---|---|---|---|---|
| Qwen3-VL 2B (bf16) | ~4,0 GB | ~0,3–1,2 s | ~1,2 s | ~80 ms | Standard, schnell, gutes Deutsch |
| Qwen3-VL 4B (NF4, 4-bit) | ~2,7 GB | ~0,6–2,4 s | ~2,7 s | ~170 ms | genauer, sprachlich stärker |
| Moondream 2 (2025-06-21) | ~3,6 GB | ~0,3 s pro Klasse | ~0,9 s | ~160 ms | natives `detect()`, antwortet auf Englisch |

Die Zeiten wurden bei 640 px Bildgröße gemessen. Bei der Detektion hängt die Dauer von der Anzahl
der gefundenen Objekte ab, weil Qwen jede Box als JSON-Text ausgibt.

## Modi

- **🎯 Erkennen:** Das Modell liefert Bounding-Boxes für frei wählbare Klassen (offenes Vokabular).
  Optional gibt es eine **zweite Stufe**: Jede Box wird aus dem hochaufgelösten Kamerabild
  ausgeschnitten und dem Modell mit einer Ja/Nein-Frage vorgelegt. Davon hängt das Label ab.
  Beispiel-Preset **„Person / Person mit Brille“**: Stufe 1 sucht `person`, Stufe 2 fragt
  „Is this person wearing glasses?“. Nach demselben Muster arbeiten der Lächel-Detektor und der
  Helm-Check (Arbeitsschutz).
- **💬 Fragen:** Freie Fragen zum Kamerabild. Die Antwort wird Token für Token in die GUI gestreamt.
  Presets: Szene beschreiben, Emotion, Handgeste, Finger zählen, OCR, „Was halte ich?“, Kompliment,
  Vierzeiler …
- **🚨 Überwachen:** Eine Ja/Nein-Bedingung wird dauernd geprüft, z. B. „Hand gehoben?“,
  „Daumen hoch?“, „Niemand da?“. Wechselt die Antwort auf *Ja*, blinkt der Rahmen rot, es ertönt ein
  Alarmton und der Treffer landet markiert im Verlauf.

## Weitere Funktionen

- **Live-Schleife** (Taste `L`) oder **Einzelanalyse** (Leertaste); Tasten `1`/`2`/`3` wechseln den Modus
- Läuft gerade eine Generierung, bricht ein Wechsel von Aufgabe oder Modell sie sofort ab
- **Analysebild**: zeigt genau das Bild, das das Modell gesehen hat, statt des Livebilds
- **Verlauf** mit Vorschaubildern samt Boxen; ein Klick öffnet den Eintrag erneut
- **Rohausgabe**: zeigt den Prompt und die unbearbeitete Modellantwort (gut für die Lehre: wie sieht Grounding-JSON aus?)
- **Vorlesen** der Antworten per Browser-Sprachausgabe
- **Schnappschuss** als PNG mit eingezeichneten Boxen
- Anzeige von VRAM-Nutzung, Latenz und Analysen pro Sekunde
- Einstellbar: Kamera, Spiegelung, Bildgröße fürs Modell (bestimmt die Zahl der Bild-Tokens, also das Verhältnis von Tempo zu Genauigkeit), Pause zwischen den Analysen

## Aufbau

```
vlm_demo/
  app.py      FastAPI: GUI, MJPEG-Stream (/video.mjpg), REST-API, WebSocket (/ws)
  engine.py   Inferenz-Thread: neuestes Kamerabild → Aufgabe → Ergebnis an alle Clients
  models.py   Backends (Qwen3-VL, Moondream) mit einheitlichem ask()/detect(), Parser für Grounding-JSON
  tasks.py    Aufgabenschema und alle Presets (hier eigene hinzufügen)
  camera.py   Kamera-Thread (OpenCV/V4L2), hält nur das jeweils neueste Bild
static/       index.html, app.js, style.css (ohne Build-Schritt)
```

Ein neues Modell bekommt einen Eintrag in `MODELS` in `models.py`. Gehört es zu einer neuen Familie,
braucht es außerdem eine `VLMBackend`-Unterklasse mit `load()`, `ask()` und `detect()`.

## Hinweise zur Installation

- Die Blackwell-GPU (sm_120) braucht PyTorch mit CUDA ≥ 12.8. Deshalb bezieht `pyproject.toml`
  `torch` aus dem Index `cu128`. `uv sync` installiert alles. Python 3.12 wird verwendet, weil das
  System-Python 3.14 noch nicht für alle Pakete Wheels hat.
- Die Modelle landen im Hugging-Face-Cache (`~/.cache/huggingface`). `run.sh` schaltet die
  Xet-Downloads ab, weil diese in diesem Netz abbrachen.
- Der Remote-Code von Moondream ist nicht mit `from_pretrained()` aus transformers v5 kompatibel.
  `MoondreamBackend.load()` lädt die Gewichte deshalb selbst.

## Modell-Cache

Die Gewichte liegen dauerhaft unter `~/.cache/huggingface/hub/` (zusammen rund 16 GB) und werden nur
beim ersten Start heruntergeladen. Liegen alle drei Modelle im Cache, startet `run.sh` im
Offline-Modus (`HF_HUB_OFFLINE=1`). Dann gibt es keine Update-Abfragen bei Hugging Face, die
Gewichte bleiben unverändert und die Demo läuft ohne Internet. Mit `VLM_ONLINE=1 ./run.sh` lässt
sich der Online-Modus erzwingen, etwa für neu hinzugefügte Modelle.
