import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from main import build_retriever, build_reranker, build_generator, run_generation
from rag_docs.entity import RAGAnswer
from rag_docs.logging.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


class QuestionRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building pipeline...")
    app.state.retriever = build_retriever()
    app.state.reranker = build_reranker()
    app.state.generator = build_generator()
    logger.info("Pipeline ready — API is live")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Ask My Docs",
    description="RAG API for AI & ML research papers",
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
    return {
        "status": "ok",
        "pipeline": {
            "retriever": app.state.retriever is not None,
            "reranker": app.state.reranker is not None,
            "generator": app.state.generator is not None,
        },
    }


@app.post("/ask", response_model=RAGAnswer)
async def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    logger.info(f"Query received: {question}")

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
        logger.error("Pipeline timed out")
        raise HTTPException(status_code=504, detail="Request timed out.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return answer