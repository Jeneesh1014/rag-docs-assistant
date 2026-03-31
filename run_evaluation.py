import os
import sys
from dotenv import load_dotenv

from rag_docs.entity import EvaluationConfig
from rag_docs.core.evaluation import Evaluator

load_dotenv()


def main():
    config = EvaluationConfig(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        questions_path="data/eval_questions.json",
        results_path="evaluation_results.json",
        sleep_between_questions=10,
    )

    evaluator = Evaluator(config)
    artifact = evaluator.initiate_evaluation()

    print("\n" + "─" * 50)
    print(f"Questions evaluated : {artifact.total_questions}")
    print(f"Faithfulness        : {artifact.faithfulness:.3f}")
    print(f"Answer relevancy    : skipped (Groq n=1 limitation)")
    print(f"Context precision   : {artifact.context_precision:.3f}")
    print(f"Context recall      : {artifact.context_recall:.3f}")
    print(f"Quality gate        : {'PASS' if artifact.passes_quality_gate() else 'FAIL'}")
    print(f"Results saved to    : {artifact.results_path}")
    print("─" * 50 + "\n")

    # exit 1 so GitHub Actions fails the build when quality drops
    if not artifact.passes_quality_gate():
        sys.exit(1)


if __name__ == "__main__":
    main()