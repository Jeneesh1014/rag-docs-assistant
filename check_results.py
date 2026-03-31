import json
import sys
from pathlib import Path


def main():
    path = Path("evaluation_results.json")

    if not path.exists():
        print("evaluation_results.json not found — run run_evaluation.py first")
        sys.exit(1)

    with open(path, "r") as f:
        results = json.load(f)

    scores = results["scores"]
    gate = results["quality_gate"]

    print("\n" + "─" * 50)
    print(f"Timestamp           : {results['timestamp']}")
    print(f"Questions evaluated : {results['total_questions']}")
    print(f"Faithfulness        : {scores['faithfulness']:.3f}")
    print(f"Context precision   : {scores['context_precision']:.3f}")
    print(f"Context recall      : {scores['context_recall']:.3f}")
    print(f"Min faithfulness    : {gate['min_faithfulness']}")
    print(f"Quality gate        : {'PASS' if gate['passes'] else 'FAIL'}")
    print("─" * 50 + "\n")

    if not gate["passes"]:
        print(f"FAILED — faithfulness {scores['faithfulness']:.3f} is below {gate['min_faithfulness']}")
        sys.exit(1)

    print("All quality gates passed")
    sys.exit(0)


if __name__ == "__main__":
    main()