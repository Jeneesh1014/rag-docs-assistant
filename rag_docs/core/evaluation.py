import json
import time
from pathlib import Path
from typing import List

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag_docs.config.settings import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
)
from rag_docs.entity import (
    EvaluationConfig,
    EvaluationArtifact,
    EvaluationSample,
    RetrievalConfig,
    RerankingConfig,
    GenerationConfig,
)
from rag_docs.core.retrieval import Retriever
from rag_docs.core.reranking import Reranker
from rag_docs.core.generation import Generator
from rag_docs.logging.logger import get_logger

logger = get_logger(__name__)


class Evaluator:

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.retriever = None
        self.reranker = None
        self.generator = None

    def build_pipeline(self) -> None:
        logger.info("Setting up pipeline components")

        retrieval_config = RetrievalConfig(
            chroma_db_path=CHROMA_DB_PATH,
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
            embedding_device=EMBEDDING_DEVICE,
        )
        self.retriever = Retriever(retrieval_config)
        self.retriever.load_vector_store()
        self.retriever.load_bm25_index()

        reranking_config = RerankingConfig(cohere_api_key=self.config.cohere_api_key)
        self.reranker = Reranker(reranking_config)

        generation_config = GenerationConfig(
            groq_api_key=self.config.groq_api_key,
            model=self.config.groq_model,
        )
        self.generator = Generator(generation_config)
        self.generator.load_client()

        logger.info("Pipeline ready")

    def build_ragas_llm(self) -> LangchainLLMWrapper:
        llm = ChatGroq(
            api_key=self.config.groq_api_key,
            model=self.config.groq_model,
            temperature=0.0,
        )
        return LangchainLLMWrapper(llm)

    def build_ragas_embeddings(self) -> LangchainEmbeddingsWrapper:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
        )
        return LangchainEmbeddingsWrapper(embeddings)

    def load_questions(self) -> List[dict]:
        path = Path(self.config.questions_path)
        if not path.exists():
            raise FileNotFoundError(f"Questions file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        logger.info(f"Loaded {len(questions)} questions from {path}")
        return questions

    def run_single_question(self, question: str, ground_truth: str) -> EvaluationSample:
        retrieval_artifact = self.retriever.retrieve(question)
        reranking_artifact = self.reranker.rerank(retrieval_artifact)
        rag_answer = self.generator.initiate_generation(reranking_artifact)

        contexts = [r.content for r in reranking_artifact.results]

        return EvaluationSample(
            question=question,
            answer=rag_answer.answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    def collect_samples(self, questions: List[dict]) -> List[EvaluationSample]:
        samples = []
        total = len(questions)

        for i, item in enumerate(questions, start=1):
            logger.info(f"Question {i}/{total}: {item['question'][:60]}...")

            try:
                sample = self.run_single_question(item["question"], item["ground_truth"])
                samples.append(sample)
            except Exception as e:
                logger.error(f"Question {i} failed: {e}")
                samples.append(
                    EvaluationSample(
                        question=item["question"],
                        answer="",
                        contexts=[],
                        ground_truth=item["ground_truth"],
                    )
                )

            if i < total:
                logger.info(f"Waiting {self.config.sleep_between_questions}s")
                time.sleep(self.config.sleep_between_questions)

        return samples

    def score_with_ragas(
        self,
        samples: List[EvaluationSample],
        ragas_llm: LangchainLLMWrapper,
        ragas_embeddings: LangchainEmbeddingsWrapper,
    ) -> dict:
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict(
            {
                "question":     [s.question for s in samples],
                "answer":       [s.answer for s in samples],
                "contexts":     [s.contexts for s in samples],
                "ground_truth": [s.ground_truth for s in samples],
            }
        )

        logger.info(f"Scoring {len(samples)} samples with Ragas — this takes a few minutes")

        # max_workers=1 forces sequential calls — Groq rejects n>1 parallel requests
        run_config = RunConfig(max_workers=1, timeout=120)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )

        # newer Ragas returns per-sample lists — average them manually
        def safe_mean(val):
            if isinstance(val, list):
                valid = [v for v in val if v is not None]
                return sum(valid) / len(valid) if valid else 0.0
            return float(val)

        scores = {
            "faithfulness":      safe_mean(result["faithfulness"]),
            "context_precision": safe_mean(result["context_precision"]),
            "context_recall":    safe_mean(result["context_recall"]),
        }

        logger.info(
            f"faithfulness={scores['faithfulness']:.3f} | "
            f"context_precision={scores['context_precision']:.3f} | "
            f"context_recall={scores['context_recall']:.3f}"
        )

        return scores

  

    def save_results(self, scores: dict, samples: List[EvaluationSample]) -> str:
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(samples),
            "scores": scores,
            "quality_gate": {
                "min_faithfulness": 0.7,
                "passes": scores["faithfulness"] >= 0.7,
            },
            "samples": [
                {
                    "question":     s.question,
                    "answer":       s.answer,
                    "contexts":     s.contexts,
                    "ground_truth": s.ground_truth,
                }
                for s in samples
            ],
        }

        path = Path(self.config.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {path}")
        return str(path)

    def initiate_evaluation(self) -> EvaluationArtifact:
        logger.info("Starting evaluation")

        self.build_pipeline()
        ragas_llm = self.build_ragas_llm()
        ragas_embeddings = self.build_ragas_embeddings()

        questions = self.load_questions()
        samples = self.collect_samples(questions)
        scores = self.score_with_ragas(samples, ragas_llm, ragas_embeddings)
        results_path = self.save_results(scores, samples)

        artifact = EvaluationArtifact(
            total_questions=len(samples),
            faithfulness=scores["faithfulness"],
            answer_relevancy=None,
            context_precision=scores["context_precision"],
            context_recall=scores["context_recall"],
            results_path=results_path,
            samples=samples,
        )

        if artifact.passes_quality_gate():
            logger.info("Quality gate passed")
        else:
            logger.warning(f"Quality gate failed — faithfulness {artifact.faithfulness:.3f} is below 0.7")

        return artifact