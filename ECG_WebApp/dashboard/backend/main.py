"""
CardioScan AI — FastAPI Backend
Serves the dashboard frontend with model predictions and metrics.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from engine import analyze_patient, get_gallery, compute_insights

app = FastAPI(title="CardioScan AI API", version="1.0.0")

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend ──────────────────────────────────────────────────
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


# ── API Routes ──────────────────────────────────────────────────────
@app.get("/api/gallery")
async def gallery():
    """Return representative case per class for the validation gallery."""
    return get_gallery()


@app.get("/api/analyze/{patient_idx}")
async def analyze(patient_idx: int):
    """Full diagnostic analysis for a patient record."""
    if patient_idx < 0 or patient_idx > 49:
        return {"error": "Patient index must be between 0 and 49"}
    return analyze_patient(patient_idx)


@app.get("/api/insights")
async def insights():
    """Aggregate model performance metrics across the test set."""
    return compute_insights()


# ── Entry Point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
