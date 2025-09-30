from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Routes for multi-pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request):
    # Call recommender logic here
    recs = ["Recommended: Rugrats"]  # Placeholder, integrate full
    return templates.TemplateResponse("recommend.html", {"request": request, "recs": recs})

@app.get("/generate", response_class=HTMLResponse)
async def generate(request: Request):
    # Trigger script generation
    os.system("python scripts/generate_video_script.py")  # For demo
    return templates.TemplateResponse("generate.html", {"request": request, "message": "Script Generated!"})

# More routes for other features