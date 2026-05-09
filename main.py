from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent.agent import run_agent

app = FastAPI(title="Smart Technical Knowledge Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(query: Query):
    result = run_agent(query.question)
    return {"response": result}

@app.get("/health")
def health():
    return {"status": "running"}