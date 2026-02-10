import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from "app"
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from fastapi import FastAPI, UploadFile, File, Request, HTTPException # pyre-ignore-all-errors
from fastapi.responses import HTMLResponse # pyre-ignore-all-errors
from fastapi.staticfiles import StaticFiles # pyre-ignore-all-errors
from fastapi.templating import Jinja2Templates # pyre-ignore-all-errors
from pydantic import BaseModel
from app.ml.model_manager import manager # pyre-ignore-all-errors

app = FastAPI(title="Fingerprint Analysis API")

class AnalysisResult(BaseModel):
    blood_group: str
    sex: str
    age: str
    diabetic_status: str
    filename: str

# Setup templates and static files
templates = Jinja2Templates(directory="app/templates")
# Ensure static dir exists
if not os.path.exists("app/static"):
    os.makedirs("app/static")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_fingerprint(file: UploadFile = File(...)):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform analysis
        results = manager.predict(file_path)
        
        # Add original filename to results
        results["filename"] = file.filename
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Optionally clean up file
        # os.remove(file_path)
        pass

if __name__ == "__main__":
    import uvicorn # pyre-ignore-all-errors
    uvicorn.run(app, host="127.0.0.1", port=8000)
