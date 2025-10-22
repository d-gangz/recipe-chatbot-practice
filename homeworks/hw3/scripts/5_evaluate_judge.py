#!/usr/bin/env python3
"""
Final test set evaluation script to measure LLM judge performance and prepare data for judgy bias correction.

This script evaluates the finalized LLM judge (developed in 4_develop_judge.py) on the
held-out test set to get unbiased estimates of True Positive Rate (TPR) and True Negative
Rate (TNR). These metrics are critical for the judgy package to mathematically correct
for judge bias when applied to unlabeled production data.

Input data sources:
  - homeworks/hw3/data/test_set.csv (held-out test set, ground truth labels)
  - homeworks/hw3/results/judge_prompt.txt (finalized judge prompt from development)

Output destinations:
  - homeworks/hw3/results/judge_performance.json (TPR/TNR metrics)
  - homeworks/hw3/results/test_predictions.json (detailed predictions for error analysis)
  - homeworks/hw3/results/judgy_test_data.json (formatted confusion matrix for judgy package)

Dependencies:
  - MODEL_NAME_JUDGE environment variable (defaults to gpt-4o-nano)
  - litellm for LLM API calls
  - pandas for CSV data loading

Key exports:
  - evaluate_judge_on_test(): Main evaluation function that returns TPR, TNR, predictions
  - evaluate_single_trace(): Evaluates a single recipe trace with the judge
  - save_results(): Saves metrics in judgy-compatible format

Side effects:
  - Makes LLM API calls for evaluation (costs money)
  - Creates judge_performance.json, test_predictions.json, judgy_test_data.json files

=== CRITICAL WORKFLOW POSITION ===

This is Step 5 in the evaluation pipeline:

Step 4 (develop_judge.py) → Iterate on dev set → Finalize judge prompt
    ↓
Step 5 (THIS SCRIPT) → Run ONCE on test set → Get unbiased TPR/TNR
    ↓
Step 6 (run_full_evaluation.py) → Apply judge to unlabeled data → Use judgy to correct bias

=== WHY THIS SCRIPT EXISTS ===

1. **Separation of Concerns**: You iterate freely on the dev set (step 4) without
   contaminating your test set. Only when satisfied with your judge do you run this script.

2. **Unbiased Metrics**: The test set has never been seen during development, so the
   TPR/TNR measured here are unbiased estimates of real-world judge performance.

3. **judgy Preparation**: The judgy package needs to know your judge's error patterns
   (TPR/TNR) to mathematically correct biased predictions on unlabeled production data.

   Example: If your judge has TNR=0.75 (misses 25% of FAIL cases), and it says 85%
   of production data passes, the real pass rate is likely higher (~92.6% after correction).

4. **Avoid Overfitting**: You should run this script SPARINGLY (ideally once). Every time
   you look at test set results and adjust your prompt, you leak information and reduce
   the test set's ability to predict real-world performance.

=== WHAT GETS SAVED FOR JUDGY ===

The script saves three files, but judgy_test_data.json is the key output:

{
  "test_labels": [1, 0, 1, 1, 0, ...],  # Ground truth (1=PASS, 0=FAIL)
  "test_preds": [1, 0, 0, 1, 0, ...],   # Judge predictions
  "description": "Confusion matrix for judgy bias correction"
}

This gives judgy the confusion matrix components (TP, FP, TN, FN) needed to compute
correction factors. When you later apply your judge to 2400 unlabeled production traces,
judgy uses these factors to adjust for known judge bias.

=== KEY DIFFERENCES FROM 4_develop_judge.py ===

| Aspect           | 4_develop_judge.py (DEV)      | 5_evaluate_judge.py (TEST)     |
|------------------|-------------------------------|--------------------------------|
| Dataset          | dev_set.csv                   | test_set.csv                   |
| Sample Size      | 50 traces (to save costs)     | ALL test traces (no sampling)  |
| Purpose          | Iterate and refine judge      | Get unbiased final metrics     |
| Run Frequency    | Many times during development | ONCE when judge is finalized   |
| Output           | dev_predictions.json          | judgy_test_data.json + metrics |
| Prompt Creation  | Can auto-generate w/ examples | Uses existing judge_prompt.txt |
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

import litellm
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

MAX_WORKERS = 32

console = Console()

# Model used for the LLM judge
MODEL_NAME_JUDGE: Final[str] = os.environ.get("MODEL_NAME_JUDGE", "gpt-4.1-nano")


def load_data_split(csv_path: str) -> List[Dict[str, Any]]:
    """Load a data split from CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def load_judge_prompt(prompt_path: str) -> str:
    """Load the judge prompt from file."""
    with open(prompt_path, "r") as f:
        return f.read()


def evaluate_single_trace(args: tuple) -> Dict[str, Any]:
    """
    Evaluate a single recipe trace with the judge - designed for parallel processing.

    This function is called by ThreadPoolExecutor for each test trace. It replaces
    the placeholders in the judge prompt with actual recipe data, calls the LLM,
    and compares the predicted label against the ground truth.

    Args:
        args: Tuple of (trace_dict, judge_prompt_str) for parallel processing compatibility

    Returns:
        Dict containing true_label, predicted_label, reasoning, and success status
    """
    trace, judge_prompt = args

    # Extract recipe data from this test trace
    query = trace["query"]  # User's recipe request
    dietary_restriction = trace["dietary_restriction"]  # e.g., "vegan", "gluten-free"
    response = trace["response"]  # Recipe generated by the bot
    true_label = trace["label"]  # Ground truth: should this PASS or FAIL?

    # Replace placeholders with actual recipe data to create the evaluation prompt
    # The judge prompt template contains __QUERY__, __DIETARY_RESTRICTION__, __RESPONSE__
    # which we replace with the specific values for this test case
    formatted_prompt = judge_prompt.replace("__QUERY__", query)
    formatted_prompt = formatted_prompt.replace(
        "__DIETARY_RESTRICTION__", dietary_restriction
    )
    formatted_prompt = formatted_prompt.replace("__RESPONSE__", response)

    try:
        # Call the LLM judge to evaluate this recipe
        # Using the same model as in development (gpt-4o-nano by default for cost efficiency)
        completion = litellm.completion(
            model=MODEL_NAME_JUDGE,
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse JSON response from the judge
        # Expected format: {"reasoning": "...", "label": "PASS" or "FAIL"}
        # The LLM might wrap it in markdown code blocks or add extra text
        try:
            if "```json" in response_text:
                # LLM wrapped response in markdown code block (```json ... ```)
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # LLM returned plain JSON (possibly with surrounding text)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Assume entire response is JSON
                json_text = response_text

            result = json.loads(json_text)
            predicted_label = result.get("label", "UNKNOWN")
            reasoning = result.get("reasoning", "")
        except json.JSONDecodeError:
            # Failed to parse JSON - mark as UNKNOWN rather than crashing
            predicted_label = "UNKNOWN"
            reasoning = "Failed to parse JSON response"

        # Return the evaluation result comparing true vs predicted labels
        # This will be used to compute the confusion matrix (TP, FP, TN, FN)
        return {
            "trace_id": trace.get("trace_id", "unknown"),
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "true_label": true_label,  # Ground truth from human labeling
            "predicted_label": predicted_label,  # What the judge predicted
            "reasoning": reasoning,  # Judge's explanation
            "success": True,
        }

    except Exception as e:
        # Handle LLM API errors or other exceptions gracefully
        # Don't crash the entire evaluation if one trace fails
        return {
            "trace_id": trace.get("trace_id", "unknown"),
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "true_label": true_label,
            "predicted_label": "ERROR",
            "reasoning": f"Error: {str(e)}",
            "success": False,
        }


def evaluate_judge_on_test(
    judge_prompt: str, test_traces: List[Dict[str, Any]], max_workers: int = MAX_WORKERS
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate the judge prompt on ALL test traces using parallel processing.

    Unlike 4_develop_judge.py which samples 50 traces from dev set, this evaluates
    ALL test traces to get the most accurate unbiased estimates of TPR and TNR.

    The resulting TPR/TNR are critical for judgy to correct bias when applying the
    judge to unlabeled production data.

    Args:
        judge_prompt: The finalized judge prompt from development phase
        test_traces: All traces from test_set.csv (with ground truth labels)
        max_workers: Number of parallel threads (default 32 for speed)

    Returns:
        Tuple of (TPR, TNR, predictions_list)
    """

    console.print(
        f"[yellow]Evaluating judge on {len(test_traces)} test traces with {max_workers} workers..."
    )

    # Prepare tasks for parallel processing
    # Each task is a tuple of (trace, judge_prompt) for evaluate_single_trace()
    tasks = [(trace, judge_prompt) for trace in test_traces]

    predictions = []

    # Use ThreadPoolExecutor to evaluate all traces in parallel (much faster!)
    # With 32 workers, we can evaluate 32 traces simultaneously
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool at once
        future_to_task = {
            executor.submit(evaluate_single_trace, task): task for task in tasks
        }

        # Process completed tasks as they finish (not necessarily in order)
        with console.status("[yellow]Evaluating traces in parallel...") as status:
            completed = 0
            total = len(tasks)

            for future in as_completed(future_to_task):
                result = future.result()
                predictions.append(result)
                completed += 1

                # Log any failures but continue processing
                if not result["success"]:
                    console.print(
                        f"[yellow]Warning: Failed to evaluate trace {result['trace_id']}: {result.get('reasoning', 'Unknown error')}"
                    )

                # Update progress display
                status.update(
                    f"[yellow]Evaluated {completed}/{total} traces ({completed/total*100:.1f}%)"
                )

    console.print(f"[green]Completed parallel evaluation of {len(predictions)} traces")

    # Calculate confusion matrix components for TPR/TNR computation
    # These metrics tell us about the judge's error patterns, which judgy needs!

    # TP = True Positive: Judge correctly identified a PASS (correct acceptance)
    tp = sum(
        1
        for p in predictions
        if p["true_label"] == "PASS" and p["predicted_label"] == "PASS"
    )

    # FN = False Negative: Judge said FAIL but should have been PASS (incorrect rejection)
    fn = sum(
        1
        for p in predictions
        if p["true_label"] == "PASS" and p["predicted_label"] == "FAIL"
    )

    # TN = True Negative: Judge correctly identified a FAIL (correct rejection)
    tn = sum(
        1
        for p in predictions
        if p["true_label"] == "FAIL" and p["predicted_label"] == "FAIL"
    )

    # FP = False Positive: Judge said PASS but should have been FAIL (missed violation!)
    fp = sum(
        1
        for p in predictions
        if p["true_label"] == "FAIL" and p["predicted_label"] == "PASS"
    )

    # TPR (True Positive Rate) = Sensitivity = Recall for PASS class
    # "Of all recipes that should PASS, what % did the judge correctly identify?"
    # TPR = TP / (TP + FN) = TP / (all actual positives)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # TNR (True Negative Rate) = Specificity = Recall for FAIL class
    # "Of all recipes that should FAIL, what % did the judge correctly identify?"
    # TNR = TN / (TN + FP) = TN / (all actual negatives)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # These TPR/TNR values are what judgy uses to correct bias!
    # Example: If TNR=0.75, the judge misses 25% of failures. When applied to
    # production data showing 85% pass rate, judgy adjusts it to ~92.6% (true rate).
    return tpr, tnr, predictions


def analyze_errors(predictions: List[Dict[str, Any]]) -> None:
    """
    Analyze prediction errors to understand judge failure modes.

    WARNING: Looking at test set errors compromises the test set's objectivity!
    Every time you look at these errors and adjust your prompt, you're effectively
    "training" on the test set. Use this information sparingly.

    Better approach: If you see patterns in test errors, add more diverse examples
    to your training set and create a new test set, rather than directly fixing
    the issues you see here.
    """

    # False Positives (FP): Judge said PASS, but should have been FAIL
    # These are DANGEROUS errors - the judge missed a dietary restriction violation!
    # Example: Recipe has dairy but judge said it's dairy-free
    false_positives = [
        p
        for p in predictions
        if p["true_label"] == "FAIL" and p["predicted_label"] == "PASS"
    ]

    # False Negatives (FN): Judge said FAIL, but should have been PASS
    # These are ANNOYING errors - the judge rejected a valid recipe
    # Example: Recipe is vegan but judge incorrectly flagged it as non-vegan
    false_negatives = [
        p
        for p in predictions
        if p["true_label"] == "PASS" and p["predicted_label"] == "FAIL"
    ]

    console.print("\n[bold]Error Analysis:")
    console.print(f"False Positives: {len(false_positives)}")
    console.print(f"False Negatives: {len(false_negatives)}")

    # Show sample false positives (missed violations)
    if false_positives:
        console.print(
            "\n[red]Sample False Positives (Judge said PASS, should be FAIL):"
        )
        for i, fp in enumerate(false_positives[:3], 1):
            console.print(f"{i}. {fp['dietary_restriction']}: {fp['query']}")
            console.print(f"   Reasoning: {fp['reasoning'][:100]}...")

    # Show sample false negatives (incorrect rejections)
    if false_negatives:
        console.print(
            "\n[yellow]Sample False Negatives (Judge said FAIL, should be PASS):"
        )
        for i, fn in enumerate(false_negatives[:3], 1):
            console.print(f"{i}. {fn['dietary_restriction']}: {fn['query']}")
            console.print(f"   Reasoning: {fn['reasoning'][:100]}...")


def save_results(
    tpr: float, tnr: float, predictions: List[Dict[str, Any]], results_dir: Path
) -> None:
    """
    Save evaluation results in three formats for different purposes.

    This function saves:
    1. judge_performance.json - Human-readable metrics (TPR, TNR, accuracy)
    2. test_predictions.json - Detailed predictions for error analysis
    3. judgy_test_data.json - Confusion matrix for judgy bias correction (CRITICAL!)

    The judgy_test_data.json file is what enables step 6 (run_full_evaluation.py)
    to mathematically correct for judge bias on unlabeled production data.
    """

    # === FILE 1: Save human-readable performance metrics ===
    # This helps you understand how well your judge performs on the test set
    performance = {
        "test_set_performance": {
            "true_positive_rate": tpr,  # % of PASSes correctly identified
            "true_negative_rate": tnr,  # % of FAILs correctly identified
            "balanced_accuracy": (tpr + tnr) / 2,  # Average of TPR and TNR
            "total_predictions": len(predictions),
            "correct_predictions": sum(
                1 for p in predictions if p["true_label"] == p["predicted_label"]
            ),
            "accuracy": sum(
                1 for p in predictions if p["true_label"] == p["predicted_label"]
            )
            / len(predictions),
        }
    }

    performance_path = results_dir / "judge_performance.json"
    with open(performance_path, "w") as f:
        json.dump(performance, f, indent=2)
    console.print(f"[green]Saved performance metrics to {performance_path}")

    # === FILE 2: Save detailed predictions for error analysis ===
    # You can review this file to see which specific recipes the judge got wrong
    # This is helpful for understanding failure modes, but remember: looking at
    # test set errors compromises future test set objectivity!
    predictions_path = results_dir / "test_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(predictions, f, indent=2)
    console.print(f"[green]Saved test predictions to {predictions_path}")

    # === FILE 3: Save confusion matrix in judgy-compatible format ===
    # This is the MOST IMPORTANT output for the next step (run_full_evaluation.py)!
    #
    # judgy needs to know your judge's error patterns to correct bias on unlabeled data.
    # We convert PASS/FAIL labels to 1/0 format that judgy expects:
    #   - 1 = PASS (positive class)
    #   - 0 = FAIL (negative class)
    test_labels = [1 if p["true_label"] == "PASS" else 0 for p in predictions]
    test_preds = [1 if p["predicted_label"] == "PASS" else 0 for p in predictions]

    # This creates parallel arrays where index i contains:
    # - test_labels[i]: ground truth for trace i (0 or 1)
    # - test_preds[i]: judge's prediction for trace i (0 or 1)
    #
    # judgy uses these arrays to compute the confusion matrix:
    #   TP = sum((test_labels == 1) & (test_preds == 1))
    #   FP = sum((test_labels == 0) & (test_preds == 1))
    #   TN = sum((test_labels == 0) & (test_preds == 0))
    #   FN = sum((test_labels == 1) & (test_preds == 0))
    #
    # Then it calculates TPR = TP/(TP+FN) and TNR = TN/(TN+FP)
    # These rates are used to correct bias when the judge evaluates unlabeled data!
    judgy_data = {
        "test_labels": test_labels,  # Ground truth: what should the judge say?
        "test_preds": test_preds,  # Predictions: what did the judge actually say?
        "description": "Test set labels and predictions for judgy evaluation",
    }

    judgy_path = results_dir / "judgy_test_data.json"
    with open(judgy_path, "w") as f:
        json.dump(judgy_data, f, indent=2)
    console.print(f"[green]Saved judgy test data to {judgy_path}")
    console.print(
        "[blue]→ This judgy_test_data.json will be used in step 6 to correct bias!"
    )


def main():
    """
    Main function to evaluate the finalized judge on the test set.

    This orchestrates the complete test evaluation workflow:
    1. Load the held-out test set (with ground truth labels)
    2. Load the finalized judge prompt (from 4_develop_judge.py)
    3. Evaluate ALL test traces in parallel
    4. Calculate TPR and TNR from the confusion matrix
    5. Save results in judgy-compatible format for step 6

    CRITICAL: This should only be run ONCE when you're satisfied with your judge!
    Running it multiple times and looking at test errors leaks information and
    reduces the test set's ability to predict real-world performance.
    """
    console.print("[bold blue]LLM Judge Test Set Evaluation")
    console.print("=" * 50)

    # === STEP 0: Set up file paths ===
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"
    results_dir = hw3_dir / "results"

    # === STEP 1: Load test set ===
    # The test set is the held-out data that was NOT used during judge development
    # It provides unbiased estimates of how the judge will perform in production
    test_path = data_dir / "test_set.csv"
    if not test_path.exists():
        console.print("[red]Error: Test set not found!")
        console.print("[yellow]Please run split_data.py first.")
        return

    test_traces = load_data_split(str(test_path))
    console.print(f"[green]Loaded {len(test_traces)} test traces")

    # === STEP 2: Load the finalized judge prompt ===
    # This prompt was created/refined in 4_develop_judge.py using the dev set
    # Now we're testing it on completely unseen test data
    prompt_path = results_dir / "judge_prompt.txt"
    if not prompt_path.exists():
        console.print("[red]Error: Judge prompt not found!")
        console.print("[yellow]Please run develop_judge.py first.")
        return

    judge_prompt = load_judge_prompt(str(prompt_path))
    console.print("[green]Loaded judge prompt")

    # === STEP 3: Evaluate judge on ALL test traces ===
    # Unlike dev set evaluation which samples 50 traces, we evaluate ALL test traces
    # to get the most accurate TPR/TNR estimates for judgy bias correction
    console.print("[yellow]Evaluating judge on test set... This may take a while.")
    tpr, tnr, predictions = evaluate_judge_on_test(judge_prompt, test_traces)

    # === STEP 4: Display performance metrics ===
    console.print("\n[bold]Judge Performance on Test Set:")
    console.print(f"True Positive Rate (TPR): {tpr:.3f}")
    console.print(f"True Negative Rate (TNR): {tnr:.3f}")
    console.print(f"Balanced Accuracy: {(tpr + tnr) / 2:.3f}")
    console.print(
        f"Overall Accuracy: {sum(1 for p in predictions if p['true_label'] == p['predicted_label']) / len(predictions):.3f}"
    )

    # === STEP 5: Analyze errors (optional, but beware!) ===
    # Looking at test set errors helps you understand failure modes, but it also
    # compromises the test set's objectivity. Use sparingly!
    analyze_errors(predictions)

    # === STEP 6: Save results for judgy ===
    # This saves three files, with judgy_test_data.json being the critical output
    # that enables bias correction in step 6 (run_full_evaluation.py)
    save_results(tpr, tnr, predictions, results_dir)

    console.print("\n[bold green]Test set evaluation completed!")
    console.print(
        "[blue]Results saved for use with judgy in the final evaluation step."
    )
    console.print(
        "\n[yellow]Next step: Run 6_run_full_evaluation.py to apply the judge to"
    )
    console.print("[yellow]unlabeled production data with judgy bias correction!")


if __name__ == "__main__":
    main()
