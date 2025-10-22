#!/usr/bin/env python3
"""Run full evaluation using the LLM judge and judgy for corrected metrics.

=== OVERVIEW ===
This is the final step in the LLM-as-Judge evaluation pipeline. It evaluates all
conversation traces using the finalized judge prompt and computes statistically
corrected success rates using the judgy library.

=== WHAT THIS SCRIPT DOES ===
1. Loads the finalized judge prompt (developed in earlier steps)
2. Loads judge performance data from the test set (TPR/TNR measurements)
3. Loads all conversation traces to evaluate
4. Runs the LLM judge on all traces in parallel (for efficiency)
5. Computes corrected success rate using judgy (accounts for judge errors)
6. Saves results with confidence intervals

=== KEY CONCEPTS ===

**Raw Success Rate vs Corrected Success Rate:**
- Raw: Simple proportion of traces the judge labeled as PASS
- Corrected: Statistically adjusted rate that accounts for judge errors
  - If the judge is too lenient (high false positive rate), raw rate is inflated
  - If the judge is too strict (high false negative rate), raw rate is deflated
  - Judgy uses TPR (True Positive Rate) and TNR (True Negative Rate) to correct

**Why We Need Judgy:**
LLM judges aren't perfect. They make mistakes:
- False Positives: Judge says PASS when it should be FAIL
- False Negatives: Judge says FAIL when it should be PASS

Without correction, our evaluation metrics would be biased. Judgy uses the
judge's measured performance on a labeled test set to estimate the true success rate.

**Parallel Processing:**
This script uses ThreadPoolExecutor with 32 workers to evaluate traces in parallel,
significantly speeding up evaluation when processing hundreds of traces.

=== PREREQUISITES ===
Before running this script, you must have run:
1. generate_traces.py - Creates the conversation traces to evaluate
2. develop_judge.py - Develops and finalizes the judge prompt
3. evaluate_judge.py - Measures judge performance on a test set

=== OUTPUT ===
- results/final_evaluation.json - Complete results with metrics and interpretation
- Console output showing raw vs corrected success rates with confidence intervals
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

import litellm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from judgy import estimate_success_rate
from rich.console import Console

load_dotenv()

console = Console()

# Model used for the LLM judge
MODEL_NAME_JUDGE: Final[str] = os.environ.get("MODEL_NAME_JUDGE", "gpt-4.1-nano")

# Maximum number of parallel workers for trace evaluation
# 32 workers allows us to make 32 concurrent LLM API calls, significantly
# speeding up evaluation of large trace sets (e.g., 500+ traces)
# Adjust this based on API rate limits and available system resources
MAX_WORKERS = 32


def load_traces(csv_path: str) -> List[Dict[str, Any]]:
    """Load traces from CSV file.

    Each trace represents one conversation with the recipe bot and contains:
    - query: The user's recipe request (e.g., "I want a pasta dish")
    - dietary_restriction: The dietary constraint (e.g., "vegan", "gluten-free")
    - response: The recipe bot's generated response
    """
    df = pd.read_csv(csv_path)
    # Convert DataFrame to list of dictionaries, one dict per trace
    return df.to_dict("records")


def load_judge_prompt(prompt_path: str) -> str:
    """Load the judge prompt from file.

    The judge prompt is a template with placeholders (__QUERY__, __DIETARY_RESTRICTION__,
    __RESPONSE__) that get replaced with actual values when evaluating each trace.
    """
    with open(prompt_path, "r") as f:
        return f.read()


def load_test_data(judgy_path: str) -> Tuple[List[int], List[int]]:
    """Load test labels and predictions for judgy.

    Returns:
        test_labels: Ground truth labels (1=PASS, 0=FAIL) from manual annotation
        test_preds: Judge's predictions on the same test set (1=PASS, 0=FAIL)

    These are used to measure the judge's accuracy (TPR and TNR), which judgy
    uses to correct the predictions on unlabeled data.
    """
    with open(judgy_path, "r") as f:
        data = json.load(f)
    return data["test_labels"], data["test_preds"]


def evaluate_single_trace_for_binary(args: tuple) -> int:
    """Evaluate a single trace and return binary prediction (1 for PASS, 0 for FAIL).

    This function is designed to be called in parallel by ThreadPoolExecutor.
    It takes a conversation trace and uses the LLM judge to determine if the
    recipe bot's response properly adheres to dietary restrictions.
    """
    # Unpack arguments (tuple format required for parallel processing)
    trace, judge_prompt = args

    # Extract components from the trace
    query = trace["query"]  # User's original recipe request
    dietary_restriction = trace["dietary_restriction"]  # e.g., "vegan", "gluten-free"
    response = trace["response"]  # Recipe bot's generated response

    # Format the judge prompt by replacing placeholders with actual values
    # The judge prompt template contains __QUERY__, __DIETARY_RESTRICTION__, __RESPONSE__
    formatted_prompt = judge_prompt.replace("__QUERY__", query)
    formatted_prompt = formatted_prompt.replace(
        "__DIETARY_RESTRICTION__", dietary_restriction
    )
    formatted_prompt = formatted_prompt.replace("__RESPONSE__", response)

    try:
        # Call the LLM judge to evaluate if the response follows dietary restrictions
        completion = litellm.completion(
            model=MODEL_NAME_JUDGE,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse the LLM's JSON response
        # The judge is expected to return JSON with a "label" field (PASS or FAIL)
        try:
            # Handle case where LLM wraps JSON in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            # Handle case where JSON is embedded in other text
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            # Handle case where response is pure JSON
            else:
                json_text = response_text

            # Parse the extracted JSON text
            result = json.loads(json_text)
            predicted_label = result.get("label", "UNKNOWN")

            # Convert the label to binary format
            # 1 = PASS (recipe follows dietary restrictions)
            # 0 = FAIL (recipe violates dietary restrictions)
            if predicted_label == "PASS":
                return 1
            elif predicted_label == "FAIL":
                return 0
            else:
                # If label is neither PASS nor FAIL (e.g., UNKNOWN), treat as failure
                # Conservative approach: when in doubt, mark as failure
                return 0

        except json.JSONDecodeError:
            # If we can't parse the JSON response, treat as failure
            # This handles cases where the LLM doesn't follow the expected format
            return 0

    except Exception as e:
        # Catch all other errors (API failures, network issues, etc.)
        # Conservative approach: treat errors as failures
        return 0


def run_judge_on_traces(
    judge_prompt: str, traces: List[Dict[str, Any]], max_workers: int = MAX_WORKERS
) -> List[int]:
    """Run the judge on all traces and return binary predictions using parallel processing.

    This function orchestrates parallel evaluation of all traces using ThreadPoolExecutor.
    It distributes the work across multiple threads to speed up evaluation when processing
    large numbers of traces (e.g., 500+ traces).
    """

    console.print(
        f"[yellow]Running judge on {len(traces)} traces with {max_workers} workers..."
    )

    # Prepare tasks for parallel processing
    # Each task is a tuple of (trace, judge_prompt) that will be evaluated
    tasks = [(trace, judge_prompt) for trace in traces]

    predictions = []

    # Use ThreadPoolExecutor for parallel evaluation
    # This allows multiple LLM API calls to run concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        # future_to_task maps each Future object to its original task for tracking
        future_to_task = {
            executor.submit(evaluate_single_trace_for_binary, task): task
            for task in tasks
        }

        # Process completed tasks with progress tracking
        # as_completed() yields futures as they finish (in completion order, not submission order)
        with console.status("[yellow]Evaluating traces in parallel...") as status:
            completed = 0
            total = len(tasks)

            for future in as_completed(future_to_task):
                # Get the result (0 or 1) from the completed evaluation
                result = future.result()
                predictions.append(result)
                completed += 1

                # Update the progress status in the terminal
                status.update(
                    f"[yellow]Evaluated {completed}/{total} traces ({completed/total*100:.1f}%)"
                )

    console.print(f"[green]Completed parallel evaluation of {len(predictions)} traces")
    return predictions


def compute_metrics_with_judgy(
    test_labels: List[int], test_preds: List[int], unlabeled_preds: List[int]
) -> Tuple[float, float, float, float]:
    """Compute corrected success rate and confidence interval using judgy.

    Judgy is a statistical library that corrects for judge errors (false positives/negatives).
    It uses the judge's performance on a labeled test set (where we know the true labels)
    to estimate the true success rate from predictions on unlabeled data.

    The correction accounts for:
    - True Positive Rate (TPR): How often the judge correctly identifies PASS cases
    - True Negative Rate (TNR): How often the judge correctly identifies FAIL cases

    Without this correction, we'd just use the raw observed success rate, which could be
    biased if the judge has systematic errors (e.g., too lenient or too strict).
    """

    # Estimate true success rate with judgy
    # test_labels: Ground truth labels from manually annotated test set
    # test_preds: Judge's predictions on the test set (used to measure judge accuracy)
    # unlabeled_preds: Judge's predictions on all traces (what we want to correct)
    theta_hat, lower_bound, upper_bound = estimate_success_rate(
        test_labels=test_labels, test_preds=test_preds, unlabeled_preds=unlabeled_preds
    )

    # Also compute raw observed success rate (without correction)
    # This is simply the proportion of traces the judge labeled as PASS
    raw_success_rate = np.mean(unlabeled_preds)

    return theta_hat, lower_bound, upper_bound, raw_success_rate


def save_final_results(
    theta_hat: float,
    lower_bound: float,
    upper_bound: float,
    raw_success_rate: float,
    total_traces: int,
    results_dir: Path,
) -> None:
    """Save final evaluation results to JSON file.

    Creates a comprehensive results file containing both raw and corrected metrics,
    along with confidence intervals and interpretation guidance.
    """

    results = {
        "final_evaluation": {
            "total_traces_evaluated": total_traces,
            # Raw rate: simple proportion of PASS predictions (may be biased)
            "raw_observed_success_rate": raw_success_rate,
            # Corrected rate: statistically adjusted for judge errors (more accurate)
            "corrected_success_rate": theta_hat,
            # 95% confidence interval for the corrected rate
            "confidence_interval_95": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
            "interpretation": {
                "description": "Corrected success rate accounts for judge errors (TPR/TNR)",
                "raw_vs_corrected": f"Raw rate: {raw_success_rate:.3f}, Corrected rate: {theta_hat:.3f}",
            },
        }
    }

    results_path = results_dir / "final_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Saved final results to {results_path}")


def print_interpretation(
    theta_hat: float, lower_bound: float, upper_bound: float, raw_success_rate: float
) -> None:
    """Print interpretation of results to console.

    Displays both raw and corrected success rates, along with the confidence interval
    and the magnitude of the correction applied by judgy.

    The correction magnitude shows how much the judge's errors affected the raw rate.
    A large correction suggests the judge had systematic biases (e.g., too lenient/strict).
    """

    console.print("\n[bold]Final Results:")
    console.print("=" * 30)

    # Raw rate: what we observe directly from judge predictions
    console.print(
        f"[blue]Raw Observed Success Rate: {raw_success_rate:.3f} ({raw_success_rate*100:.1f}%)"
    )
    # Corrected rate: statistically adjusted estimate of the true rate
    console.print(
        f"[green]Corrected Success Rate: {theta_hat:.3f} ({theta_hat*100:.1f}%)"
    )
    # Confidence interval: we're 95% confident the true rate is in this range
    console.print(
        f"[yellow]95% Confidence Interval: [{lower_bound:.3f}, {upper_bound:.3f}]"
    )
    console.print(
        f"[yellow]                        [{lower_bound*100:.1f}%, {upper_bound*100:.1f}%]"
    )

    # Show how much the correction changed the estimate
    correction_magnitude = abs(raw_success_rate - theta_hat)
    console.print(
        f"[cyan]Correction Applied: {correction_magnitude:.3f} ({correction_magnitude*100:.1f} percentage points)"
    )


def main():
    """Main function for full evaluation.

    This orchestrates the complete evaluation pipeline:
    1. Load the finalized judge prompt (developed in earlier steps)
    2. Load judge performance data from test set (for error correction)
    3. Load all conversation traces to evaluate
    4. Run judge on all traces in parallel
    5. Compute corrected success rate using judgy
    6. Save and display results
    """
    console.print("[bold blue]Full Recipe Bot Dietary Adherence Evaluation")
    console.print("=" * 60)

    # Set up paths
    script_dir = Path(__file__).parent  # scripts/
    hw3_dir = script_dir.parent  # homeworks/hw3/
    data_dir = hw3_dir / "data"  # homeworks/hw3/data/
    results_dir = hw3_dir / "results"  # homeworks/hw3/results/

    # ===== STEP 1: Load the finalized judge prompt =====
    # This prompt was developed and validated in earlier steps (develop_judge.py)
    prompt_path = results_dir / "judge_prompt.txt"
    if not prompt_path.exists():
        console.print("[red]Error: Judge prompt not found!")
        console.print("[yellow]Please run develop_judge.py first.")
        return

    judge_prompt = load_judge_prompt(str(prompt_path))
    console.print("[green]Loaded judge prompt")

    # ===== STEP 2: Load test set performance data for judgy =====
    # This data contains the judge's predictions on a labeled test set
    # We need this to measure the judge's TPR and TNR for error correction
    judgy_path = results_dir / "judgy_test_data.json"
    if not judgy_path.exists():
        console.print("[red]Error: Test set performance data not found!")
        console.print("[yellow]Please run evaluate_judge.py first.")
        return

    test_labels, test_preds = load_test_data(str(judgy_path))
    console.print(f"[green]Loaded test set performance: {len(test_labels)} examples")

    # ===== STEP 3: Load all raw traces for evaluation =====
    # These are the conversation traces we want to evaluate
    # Each trace contains: query, dietary_restriction, response
    traces_path = data_dir / "raw_traces.csv"
    if not traces_path.exists():
        console.print("[red]Error: Raw traces not found!")
        console.print("[yellow]Please run generate_traces.py first.")
        return

    all_traces = load_traces(str(traces_path))
    console.print(f"[green]Loaded {len(all_traces)} traces for evaluation")

    # ===== STEP 4: Run judge on all traces =====
    # This uses parallel processing to evaluate all traces efficiently
    # Each trace gets a binary prediction: 1 (PASS) or 0 (FAIL)
    console.print("[yellow]Running judge on all traces... This will take a while.")
    predictions = run_judge_on_traces(judge_prompt, all_traces)

    console.print(f"[green]Completed evaluation of {len(predictions)} traces")
    console.print(f"[blue]Raw success rate: {np.mean(predictions):.3f}")

    # ===== STEP 5: Compute corrected metrics with judgy =====
    # This applies statistical correction to account for judge errors
    # The corrected rate is more accurate than the raw observed rate
    console.print("[yellow]Computing corrected success rate with judgy...")
    theta_hat, lower_bound, upper_bound, raw_success_rate = compute_metrics_with_judgy(
        test_labels, test_preds, predictions
    )

    # ===== STEP 6: Print and save results =====
    print_interpretation(theta_hat, lower_bound, upper_bound, raw_success_rate)
    save_final_results(
        theta_hat,
        lower_bound,
        upper_bound,
        raw_success_rate,
        len(all_traces),
        results_dir,
    )

    console.print("\n[bold green]Full evaluation completed successfully!")
    console.print("[blue]Check the results/ directory for detailed outputs.")


if __name__ == "__main__":
    main()
