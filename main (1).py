import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

RUNWAY_BASE = "https://api.dev.runwayml.com"
RUNWAY_VERSION = "2024-11-06"

app = FastAPI(title="NARROW AI API", version="1.0.0")

allowed = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed == ["*"] else allowed,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=1000)
    model: str = "gen4.5"
    duration: int = Field(default=5, ge=4, le=10)
    ratio: str = "1280:720"
    image_url: HttpUrl | None = None

def runway_headers():
    key = os.getenv("RUNWAYML_API_SECRET") or os.getenv("RUNWAY_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Runway API key is not configured on the backend.")
    return {
        "Authorization": f"Bearer {key}",
        "X-Runway-Version": RUNWAY_VERSION,
        "Content-Type": "application/json",
    }

@app.get("/")
def root():
    return {"name": "NARROW AI API", "status": "online"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "runway_key_configured": bool(
            os.getenv("RUNWAYML_API_SECRET") or os.getenv("RUNWAY_API_KEY")
        ),
    }

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    headers = runway_headers()
    if req.image_url:
        endpoint = f"{RUNWAY_BASE}/v1/image_to_video"
        payload = {
            "model": req.model,
            "promptImage": str(req.image_url),
            "promptText": req.prompt,
            "ratio": req.ratio,
            "duration": req.duration,
        }
    else:
        endpoint = f"{RUNWAY_BASE}/v1/text_to_video"
        payload = {
            "model": req.model,
            "promptText": req.prompt,
            "ratio": req.ratio,
            "duration": req.duration,
        }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not reach Runway. Please try again.")
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()

@app.get("/api/tasks/{task_id}")
async def task_status(task_id: str):
    headers = runway_headers()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{RUNWAY_BASE}/v1/tasks/{task_id}", headers=headers
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="Could not reach Runway to check task status.")
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()
