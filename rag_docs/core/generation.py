import time
from groq import Groq
from rag_docs.config.settings import GROQ_MODEL, MAX_TOKENS, TEMPERATURE
from rag_docs.entity import GenerationConfig, RAGAnswer, Citation, RerankingArtifact
from rag_docs.logging.logger import get_logger

logger = get_logger(__name__)


class Generator:
    """
    Sends reranked chunks + query to Groq LLM.
    Returns a structured RAGAnswer with citations.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = None


    def load_client(self):
        """Connect to Groq API."""
        self.client = Groq(api_key=self.config.groq_api_key)
        logger.info(f"Groq client loaded — model: {self.config.model}")


    def build_prompt(self, artifact: RerankingArtifact) -> tuple[str, str]:
        """
        Format the 3 reranked chunks into a prompt.
        Returns (system_prompt, user_prompt) as a tuple.
        We keep them separate because Groq expects separate
        system and user messages.
        """

        system_prompt = (
            "You are an AI research assistant. "
            "Answer questions using ONLY the provided context. "
            "Always cite sources using [1], [2], [3] inline. "
            "Be precise and academic in tone. "
            "If the context does not contain enough information, say so clearly."
        )

        # Build the numbered context block from each reranked chunk
        context_blocks = []
        for i, result in enumerate(artifact.results, start=1):
            block = (
                f"[{i}] Source: {result.source}\n"
                f"    {result.content}"
            )
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        user_prompt = (
            f"Context:\n\n"
            f"{context_text}\n\n"
            f"Question: {artifact.query}\n\n"
            f"Instructions:\n"
            f"- Use ONLY the context above\n"
            f"- Cite sources inline like this: [1] or [2] or [3]\n"
            f"- Be concise but complete"
        )

        return system_prompt, user_prompt


    def call_groq(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send prompts to Groq and return the raw text response.
        Separated from generate() so it is easy to mock in tests.
        """
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        # response.choices[0].message.content is where Groq puts the answer
        return response.choices[0].message.content


    def build_answer(
        self,
        query: str,
        raw_answer: str,
        artifact: RerankingArtifact,
        elapsed: float,
    ) -> RAGAnswer:
        """
        Wrap the raw LLM text into a structured RAGAnswer.
        Citations come from the RerankingArtifact — we already
        know the source, chunk index and relevance score from Cohere.
        We don't need to parse citations from the LLM text.
        """

        citations = [
            Citation(
                source=result.source,
                chunk_index=result.reranked_rank,   # 1, 2, or 3
                relevance_score=result.relevance_score,
            )
            for result in artifact.results
        ]

        return RAGAnswer(
            question=query,
            answer=raw_answer.strip(),
            citations=citations,
            model_used=self.config.model,
            generation_time_seconds=round(elapsed, 2),
            total_chunks_used=len(artifact.results),
        )


    def log_results(self, answer: RAGAnswer):
        """Log the answer and citations in a readable format."""
        logger.info("─" * 50)
        logger.info(f"Question: {answer.question}")
        logger.info(f"Answer:\n{answer.answer}")
        logger.info("Citations:")
        for c in answer.citations:
            logger.info(
                f"  [{c.chunk_index}] {c.source} "
                f"— relevance: {c.relevance_score:.3f}"
            )
        logger.info(
            f"Model: {answer.model_used} | "
            f"Time: {answer.generation_time_seconds}s | "
            f"Chunks: {answer.total_chunks_used}"
        )
        logger.info("─" * 50)


    def initiate_generation(self, artifact: RerankingArtifact) -> RAGAnswer:
        """
        Run the full generation pipeline:
        build prompt → call Groq → build answer → log
        """
        logger.info(f"Starting generation for: {artifact.query}")

        system_prompt, user_prompt = self.build_prompt(artifact)

        start = time.time()
        raw_answer = self.call_groq(system_prompt, user_prompt)
        elapsed = time.time() - start

        answer = self.build_answer(
            query=artifact.query,
            raw_answer=raw_answer,
            artifact=artifact,
            elapsed=elapsed,
        )

        self.log_results(answer)
        return answer