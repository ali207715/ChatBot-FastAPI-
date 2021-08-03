from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import chat_bot as cb
import uvicorn
from typing import Optional

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def mainpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/get_response")
def get_response(msg: str):
    ints = cb.predict_class(msg)
    res = cb.get_response(ints, cb.intents)
    return str(res)
