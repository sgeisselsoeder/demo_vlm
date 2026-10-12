"""Web-Server: liefert GUI, Kamerastream, REST-API und WebSocket für Live-Ergebnisse."""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .camera import Camera, list_cameras
from .engine import Engine, Hub
from .models import MODELS, MODELS_BY_ID
from .tasks import PRESETS, PRESETS_BY_ID, Task

STATIC = Path(__file__).resolve().parent.parent / "static"

camera = Camera()
hub = Hub()
engine = Engine(camera, hub)
startup = {"camera": 0, "model": "qwen3-vl-2b"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    try:
        camera.open(startup["camera"])
    except RuntimeError as e:
        logging.error(e)
    engine.start()
    engine.request_model(startup["model"])
    yield
    camera.close()


app = FastAPI(title="VLM Live-Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/video.mjpg")
def video():
    return StreamingResponse(camera.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/config")
def config():
    return {
        "models": [m.to_dict() for m in MODELS],
        "presets": [p.model_dump() for p in PRESETS],
        "cameras": list_cameras(),
        "status": engine.status(),
    }


class ModelReq(BaseModel):
    id: str


@app.post("/api/model")
def set_model(req: ModelReq):
    if req.id not in MODELS_BY_ID:
        raise HTTPException(404, "Unbekanntes Modell")
    engine.request_model(req.id)
    return {"ok": True}


class TaskReq(BaseModel):
    preset_id: str | None = None
    task: Task | None = None
    run: bool = False  # direkt einmal ausführen


@app.post("/api/task")
def set_task(req: TaskReq):
    if req.preset_id:
        if req.preset_id not in PRESETS_BY_ID:
            raise HTTPException(404, "Unbekannter Preset")
        engine.set_task(PRESETS_BY_ID[req.preset_id].task, req.preset_id)
    elif req.task:
        engine.set_task(req.task)
    else:
        raise HTTPException(400, "preset_id oder task angeben")
    if req.run and not engine.continuous:
        engine.set_running("once")
    return {"ok": True}


class RunReq(BaseModel):
    action: str = Field(pattern="^(start|stop|once)$")


@app.post("/api/run")
def run(req: RunReq):
    engine.set_running(req.action)
    return {"ok": True}


class SettingsReq(BaseModel):
    interval: float | None = Field(default=None, ge=0, le=60)
    max_side: int | None = Field(default=None, ge=224, le=1600)
    camera: int | None = None
    mirror: bool | None = None


@app.post("/api/settings")
def settings(req: SettingsReq):
    if req.interval is not None:
        engine.interval = req.interval
    if req.max_side is not None:
        engine.max_side = req.max_side
    if req.mirror is not None:
        camera.mirror = req.mirror
    if req.camera is not None and req.camera != camera.index:
        try:
            camera.open(req.camera)
        except RuntimeError as e:
            raise HTTPException(500, str(e))
    engine.publish_status()
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()
    hub.clients.add(queue)
    await ws.send_json(engine.status())

    async def sender():
        while True:
            await ws.send_json(await queue.get())

    task = asyncio.create_task(sender())
    try:
        while True:
            await ws.receive_text()  # nur zum Erkennen eines Verbindungsabbruchs
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        hub.clients.discard(queue)


def main():
    parser = argparse.ArgumentParser(description="Live-Webcam-Demo für Vision-Language-Modelle")
    parser.add_argument("--camera", type=int, default=0, help="V4L2-Index der Kamera (/dev/videoN)")
    parser.add_argument("--model", default="qwen3-vl-2b", choices=list(MODELS_BY_ID))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    startup.update(camera=args.camera, model=args.model)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"\n  ➜  GUI: http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
