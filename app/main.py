import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup so first frame isn't slow
    detector.load_model()
    yield


app = FastAPI(title="Sushi Counter API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Expect binary JPEG frame from client
            frame_bytes = await websocket.receive_bytes()
            result = detector.predict(frame_bytes)
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws] Error: {e}")
        await websocket.close()


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    data = await file.read()
    return detector.predict(data)


@app.get("/health")
async def health():
    model_path = detector.CUSTOM_MODEL
    return {
        "status": "ok",
        "custom_model_loaded": model_path.exists(),
        "model_path": str(model_path),
    }


# Serve built React app in production (after `npm run build`)
frontend_build = Path(__file__).parent / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="static")
