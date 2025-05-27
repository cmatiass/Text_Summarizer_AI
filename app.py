from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
import sys
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from pydantic import BaseModel
from src.textSummarizer.pipeline.predicition_pipeline import PredictionPipeline


app = FastAPI(
    title="AI Text Summarizer",
    description="Intelligent text summarization using advanced NLP models",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs-redirect", tags=["documentation"])
async def docs_redirect():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(input_data: TextInput):
    try:
        obj = PredictionPipeline()
        result = obj.predict(input_data.text)
        return {"prediction": result}
    except Exception as e:
        raise e

@app.get("/predict")
async def predict_route_get(text: str):
    try:
        obj = PredictionPipeline()
        result = obj.predict(text)
        return {"prediction": result}
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
