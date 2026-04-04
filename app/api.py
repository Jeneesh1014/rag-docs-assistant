import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import build_generator, build_reranker, build_retriever, run_generation
from rag_docs.config.settings import GROQ_MODEL
from rag_docs.entity import RAGAnswer
from rag_docs.logging.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class QuestionRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up retriever, reranker, and generator")
    app.state.retriever = build_retriever()
    app.state.reranker = build_reranker()
    app.state.generator = build_generator()
    logger.info("Startup finished")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Ask My Docs",
    description="RAG API over ingested AI/ML research PDFs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.post("/ask", response_model=RAGAnswer)
async def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    logger.info("Query: %s", question[:200] + ("..." if len(question) > 200 else ""))

    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(
                run_generation,
                question,
                app.state.retriever,
                app.state.reranker,
                app.state.generator,
            ),
            timeout=60,
        )
    except asyncio.TimeoutError:
        logger.error("Pipeline timed out after 60s")
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

    return answer
