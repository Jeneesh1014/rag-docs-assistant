import os
import pytest
from unittest.mock import patch
from dotenv import load_dotenv

from rag_docs.entity import (
    GenerationConfig,
    RerankingArtifact,
    RerankingResult,
    RAGAnswer,
    Citation,
)
from rag_docs.core.generation import Generator

load_dotenv()


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def generator():
    config = GenerationConfig(groq_api_key=os.getenv("GROQ_API_KEY", "fake-key"))
    g = Generator(config)
    g.load_client()
    return g


@pytest.fixture
def fake_artifact():
    return RerankingArtifact(
        query="How does attention work in transformers?",
        results=[
            RerankingResult(
                content="An attention function maps a query and key-value pairs to an output.",
                source="attention_paper.pdf",
                relevance_score=0.953,
                original_rank=1,
                reranked_rank=1,
            ),
            RerankingResult(
                content="BERT uses bidirectional self-attention across all layers.",
                source="bert.pdf",
                relevance_score=0.847,
                original_rank=3,
                reranked_rank=2,
            ),
            RerankingResult(
                content="Vision transformers apply attention to image patches.",
                source="vit_paper.pdf",
                relevance_score=0.772,
                original_rank=2,
                reranked_rank=3,
            ),
        ],
        total_results=3,
        model="rerank-english-v3.0",
        reranking_time_seconds=0.4,
    )


# ─────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────

def test_client_loads(generator):
    """Groq client must not be None after load_client()."""
    assert generator.client is not None


# ─────────────────────────────────────────────────────────────
# 2. build_prompt
# ─────────────────────────────────────────────────────────────

def test_prompt_contains_query(generator, fake_artifact):
    """Query must appear in the user prompt."""
    _, user_prompt = generator.build_prompt(fake_artifact)
    assert fake_artifact.query in user_prompt


def test_prompt_contains_all_sources(generator, fake_artifact):
    """All source filenames must appear in the prompt."""
    _, user_prompt = generator.build_prompt(fake_artifact)
    for result in fake_artifact.results:
        assert result.source in user_prompt


def test_prompt_has_citation_markers(generator, fake_artifact):
    """[1] [2] [3] markers must be in prompt so LLM knows how to cite."""
    _, user_prompt = generator.build_prompt(fake_artifact)
    assert "[1]" in user_prompt
    assert "[2]" in user_prompt
    assert "[3]" in user_prompt


# ─────────────────────────────────────────────────────────────
# 3. build_answer
# ─────────────────────────────────────────────────────────────

def test_build_answer_returns_rag_answer(generator, fake_artifact):
    """Must return a RAGAnswer instance."""
    answer = generator.build_answer(
        query=fake_artifact.query,
        raw_answer="Attention maps queries to outputs [1].",
        artifact=fake_artifact,
        elapsed=1.0,
    )
    assert isinstance(answer, RAGAnswer)


def test_build_answer_citation_count(generator, fake_artifact):
    """Citation count must match number of reranked results."""
    answer = generator.build_answer(
        query=fake_artifact.query,
        raw_answer="Answer [1][2][3].",
        artifact=fake_artifact,
        elapsed=0.5,
    )
    assert len(answer.citations) == 3


def test_build_answer_citation_sources(generator, fake_artifact):
    """Citation sources must match reranked result sources."""
    answer = generator.build_answer(
        query=fake_artifact.query,
        raw_answer="Answer [1][2][3].",
        artifact=fake_artifact,
        elapsed=0.5,
    )
    expected = [r.source for r in fake_artifact.results]
    actual = [c.source for c in answer.citations]
    assert actual == expected


def test_build_answer_chunk_indexes(generator, fake_artifact):
    """chunk_index must be 1, 2, 3 matching reranked_rank."""
    answer = generator.build_answer(
        query=fake_artifact.query,
        raw_answer="Answer [1][2][3].",
        artifact=fake_artifact,
        elapsed=0.5,
    )
    assert [c.chunk_index for c in answer.citations] == [1, 2, 3]


def test_build_answer_strips_whitespace(generator, fake_artifact):
    """LLM response whitespace should be stripped."""
    answer = generator.build_answer(
        query=fake_artifact.query,
        raw_answer="   Answer.   ",
        artifact=fake_artifact,
        elapsed=0.5,
    )
    assert answer.answer == "Answer."


# ─────────────────────────────────────────────────────────────
# 4. initiate_generation — mocked
# ─────────────────────────────────────────────────────────────

def test_initiate_generation_mocked(generator, fake_artifact):
    """Full pipeline returns RAGAnswer — Groq call mocked."""
    with patch.object(generator, "call_groq", return_value="Mocked answer [1][2][3]."):
        answer = generator.initiate_generation(fake_artifact)

    assert isinstance(answer, RAGAnswer)
    assert len(answer.answer) > 0
    assert len(answer.citations) == 3
    assert answer.question == fake_artifact.query
    assert answer.total_chunks_used == 3


# ─────────────────────────────────────────────────────────────
# 5. real Groq API call — one live test
# ─────────────────────────────────────────────────────────────

def test_real_groq_call(generator, fake_artifact):
    """One real API call — confirms live integration works."""
    answer = generator.initiate_generation(fake_artifact)

    assert isinstance(answer, RAGAnswer)
    assert len(answer.answer) > 0
    assert len(answer.citations) == 3
    assert answer.model_used == "llama-3.1-8b-instant"
    assert answer.generation_time_seconds > 0