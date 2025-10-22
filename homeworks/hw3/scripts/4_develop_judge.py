#!/usr/bin/env python3
"""
LLM-as-Judge development script for evaluating recipe dietary adherence quality on the dev set.

Input data sources:
  - homeworks/hw3/data/train_set.csv (for few-shot examples)
  - homeworks/hw3/data/dev_set.csv (for judge evaluation)
  - homeworks/hw3/results/judge_prompt.txt (optional, when OWN_PROMPT=True)

Output destinations:
  - homeworks/hw3/results/judge_prompt.txt (generated judge prompt)
  - homeworks/hw3/results/dev_predictions.json (evaluation results)

Dependencies:
  - MODEL_NAME_JUDGE environment variable (defaults to gpt-4.1-nano)
  - litellm for LLM API calls
  - pandas for CSV data loading

Key exports:
  - create_judge_prompt(): Creates judge prompt with few-shot examples
  - evaluate_judge_on_dev(): Evaluates judge performance on dev set
  - evaluate_single_trace(): Evaluates a single recipe trace

Side effects:
  - Makes LLM API calls for evaluation
  - Creates judge_prompt.txt and dev_predictions.json files

=== WORKFLOW OVERVIEW ===

This script implements an LLM-as-Judge system that evaluates whether recipe
responses properly adhere to dietary restrictions (vegan, gluten-free, etc.).

STEP 1: Prompt Creation
┌─────────────────┐
│ Training Data   │
│ (train_set.csv) │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│ select_few_shot_examples()   │  ← Randomly select 1 PASS + 3 FAIL examples
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ create_judge_prompt()        │  ← Build prompt with base instructions + examples
└────────────┬─────────────────┘
             │
             ▼
        ┌────────────┐
        │Judge Prompt│  ← Saved to judge_prompt.txt
        └────┬───────┘
             │
STEP 2: Evaluation on Dev Set
             │
             ▼
┌──────────────────────────────┐
│ evaluate_judge_on_dev()      │  ← Sample 50 dev traces
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────┐
│ ThreadPoolExecutor (32 parallel workers)         │
│  ├─ evaluate_single_trace(trace1, judge_prompt) │
│  ├─ evaluate_single_trace(trace2, judge_prompt) │
│  ├─ evaluate_single_trace(trace3, judge_prompt) │
│  └─ ... (up to 50 traces in parallel)           │
└────────────┬─────────────────────────────────────┘
             │
             ▼
Each evaluate_single_trace():
  1. Replaces __QUERY__, __DIETARY_RESTRICTION__, __RESPONSE__ placeholders
  2. Calls LLM to get PASS/FAIL prediction
  3. Parses JSON response
  4. Returns {true_label, predicted_label, ...}
             │
             ▼
┌──────────────────────────────┐
│ Calculate Metrics            │
│  • TPR (True Positive Rate)  │  ← PASS recipes correctly identified
│  • TNR (True Negative Rate)  │  ← FAIL recipes correctly identified
│  • Balanced Accuracy         │  ← (TPR + TNR) / 2
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Save Results                 │
│  • judge_prompt.txt          │
│  • dev_predictions.json      │
└──────────────────────────────┘

=== USAGE MODES ===

MODE 1: Default (Auto-generated prompt with random examples)
  - Set SEED = None, OWN_PROMPT = False
  - Randomly selects 1 PASS + 3 FAIL examples from train set
  - Creates prompt and saves to judge_prompt.txt
  - Run: uv run python 4_develop_judge.py

MODE 2: Reproducible (Auto-generated with fixed seed)
  - Set SEED = 42, OWN_PROMPT = False
  - Same as Mode 1, but uses seed for reproducible example selection
  - Run: uv run python 4_develop_judge.py

MODE 3: Custom prompt (Manual refinement)
  - Set OWN_PROMPT = True
  - Manually edit homeworks/hw3/results/judge_prompt.txt
  - Must include placeholders: __QUERY__, __DIETARY_RESTRICTION__, __RESPONSE__
  - Run: uv run python 4_develop_judge.py

RECOMMENDED WORKFLOW:
  1. Run Mode 1 to generate baseline prompt
  2. Review judge_prompt.txt and dev_predictions.json
  3. Manually refine prompt based on errors
  4. Run Mode 3 to test your improved prompt
  5. Iterate until TPR and TNR are satisfactory
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import litellm
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console

# Global variables that can be set by user
SEED = None  # Set to an integer to use a seed for reproducibility of selected few-shot examples
OWN_PROMPT = False  # Set to True to use a base prompt of your own design

# Start script
load_dotenv()
MAX_WORKERS = 32

console = Console()

# Model used for the LLM judge
MODEL_NAME_JUDGE: Final[str] = os.environ.get("MODEL_NAME_JUDGE", "gpt-4.1-nano")


def load_data_split(csv_path: str) -> List[Dict[str, Any]]:
    """Load a data split from CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def select_few_shot_examples(
    train_traces: List[Dict[str, Any]],
    num_positive: int = 1,
    num_negative: int = 3,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Select few-shot examples randomly from train set for the judge prompt.

    This function is used in STEP 1 of the workflow to select examples that will
    teach the judge how to evaluate recipes. By default, it selects 1 PASS example
    (recipe that correctly follows dietary restrictions) and 3 FAIL examples
    (recipes that violate restrictions).

    The asymmetric ratio (1:3) helps the judge learn to be more critical, since
    catching violations (FAIL cases) is typically more important than confirming
    adherence (PASS cases).
    """

    console.print("[yellow]Selecting random few-shot examples...")

    # Separate by label - split training data into positive and negative examples
    train_pass = [trace for trace in train_traces if trace["label"] == "PASS"]
    train_fail = [trace for trace in train_traces if trace["label"] == "FAIL"]

    selected_examples = []

    # Select positive examples (PASS) randomly
    if seed is not None:
        random.seed(seed)  # Set seed for reproducibility if provided
    if train_pass and len(train_pass) >= num_positive:
        selected_examples.extend(random.sample(train_pass, num_positive))
    elif train_pass:
        selected_examples.extend(train_pass)  # Use all available if less than requested

    # Select negative examples (FAIL) randomly
    if train_fail and len(train_fail) >= num_negative:
        selected_examples.extend(random.sample(train_fail, num_negative))
    elif train_fail:
        selected_examples.extend(train_fail)  # Use all available if less than requested

    pass_c = len([e for e in selected_examples if e["label"] == "PASS"])
    fail_c = len([e for e in selected_examples if e["label"] == "FAIL"])
    console.print(
        f"[green]Selected {len(selected_examples)} "
        f"few-shot examples ({pass_c} PASS, {fail_c} FAIL)"
    )
    return selected_examples


def create_judge_prompt(few_shot_examples: List[Dict[str, Any]]) -> str:
    """
    Create the LLM judge prompt with few-shot examples.

    This function is used in STEP 1 of the workflow. It builds the complete
    prompt that will be used to evaluate recipes by:
    1. Starting with base instructions (dietary definitions, evaluation criteria)
    2. Adding the selected few-shot examples to teach by demonstration
    3. Adding placeholders (__QUERY__, __DIETARY_RESTRICTION__, __RESPONSE__)
       that will be replaced with actual recipe data during evaluation

    The resulting prompt serves as the "brain" of the judge system.
    """

    # Base prompt - defines the judge's role and evaluation criteria
    base_prompt = """You are an expert nutritionist and dietary specialist evaluating whether recipe responses properly adhere to specified dietary restrictions.

DIETARY RESTRICTION DEFINITIONS:
- Vegan: No animal products (meat, dairy, eggs, honey, etc.)
- Vegetarian: No meat or fish, but dairy and eggs are allowed
- Gluten-free: No wheat, barley, rye, or other gluten-containing grains
- Dairy-free: No milk, cheese, butter, yogurt, or other dairy products
- Keto: Very low carb (typically <20g net carbs), high fat, moderate protein
- Paleo: No grains, legumes, dairy, refined sugar, or processed foods
- Pescatarian: No meat except fish and seafood
- Kosher: Follows Jewish dietary laws (no pork, shellfish, mixing meat/dairy)
- Halal: Follows Islamic dietary laws (no pork, alcohol, proper slaughter)
- Nut-free: No tree nuts or peanuts
- Low-carb: Significantly reduced carbohydrates (typically <50g per day)
- Sugar-free: No added sugars or high-sugar ingredients
- Raw vegan: Vegan foods not heated above 118°F (48°C)
- Whole30: No grains, dairy, legumes, sugar, alcohol, or processed foods
- Diabetic-friendly: Low glycemic index, controlled carbohydrates
- Low-sodium: Reduced sodium content for heart health

EVALUATION CRITERIA:
- PASS: The recipe clearly adheres to the dietary preferences with appropriate ingredients and preparation methods
- FAIL: The recipe contains ingredients or methods that violate the dietary preferences
- Consider both explicit ingredients and cooking methods

Here are some examples of how to evaluate dietary adherence:

"""

    # Add few-shot examples - these teach the judge by showing example evaluations
    # Each example shows: query, response, reasoning, and correct label (PASS/FAIL)
    for i, example in enumerate(few_shot_examples, 1):
        base_prompt += f"\nExample {i}:\n"
        base_prompt += f"Query: {example['query']}\n"
        base_prompt += f"Recipe Response: {example['response']}\n"
        base_prompt += f"Reasoning: {example['reasoning']}\n"
        base_prompt += f"Label: {example['label']}\n"

    # Add evaluation template with placeholders
    # These placeholders (__QUERY__, etc.) will be replaced with actual data
    # in evaluate_single_trace() when evaluating new recipes
    base_prompt += """

Now evaluate the following recipe response:

Query: __QUERY__
Dietary Restriction: __DIETARY_RESTRICTION__
Recipe Response: __RESPONSE__

Provide your evaluation in the following JSON format:
{
    "reasoning": "Detailed explanation of your evaluation, citing specific ingredients or methods",
    "label": "PASS" or "FAIL"
}"""

    return base_prompt


def read_judge_prompt(f: Path) -> str:  # file path to the judge prompt
    """
    Read a custom judge prompt from file.

    This function is used when OWN_PROMPT=True in the workflow. It allows you to:
    1. First run the script to generate a baseline prompt
    2. Manually edit the saved prompt based on error analysis
    3. Re-run the script to test your improved prompt

    The custom prompt must contain the required placeholders:
    __QUERY__, __DIETARY_RESTRICTION__, __RESPONSE__
    """

    if not f.exists():
        raise FileNotFoundError(f"Judge prompt file {f} does not exist.")

    return f.read_text(encoding="utf-8")


def evaluate_single_trace(args: tuple) -> Dict[str, Any]:
    """
    Evaluate a single recipe trace with the judge - designed for parallel processing.

    This is the core evaluation function called by evaluate_judge_on_dev().
    It's designed to work with ThreadPoolExecutor for parallel evaluation.

    Process:
    1. Extract recipe data from the trace (query, dietary restriction, response)
    2. Replace placeholders in judge prompt with actual recipe data
    3. Call LLM to get a PASS/FAIL prediction with reasoning
    4. Parse the JSON response to extract the label
    5. Return comparison of true_label vs predicted_label

    Args:
        args: Tuple of (trace, judge_prompt) for parallel processing compatibility

    Returns:
        Dictionary with evaluation result including true vs predicted labels
    """
    trace, judge_prompt = args

    # Extract the recipe data from this trace
    query = trace["query"]
    dietary_restriction = trace["dietary_restriction"]
    response = trace["response"]
    true_label = trace["label"]  # Ground truth: should this recipe PASS or FAIL?

    # Validate that judge prompt contains required placeholders
    if "__QUERY__" not in judge_prompt:
        raise ValueError("Judge prompt does not contain __QUERY__ placeholder.")
    if "__DIETARY_RESTRICTION__" not in judge_prompt:
        raise ValueError(
            "Judge prompt does not contain __DIETARY_RESTRICTION__ placeholder."
        )
    if "__RESPONSE__" not in judge_prompt:
        raise ValueError("Judge prompt does not contain __RESPONSE__ placeholder.")

    # Replace placeholders with actual recipe data to create the evaluation prompt
    formatted_prompt = judge_prompt.replace("__QUERY__", query)
    formatted_prompt = formatted_prompt.replace(
        "__DIETARY_RESTRICTION__", dietary_restriction
    )
    formatted_prompt = formatted_prompt.replace("__RESPONSE__", response)

    try:
        # Call the LLM judge to evaluate this recipe
        completion = litellm.completion(
            model=MODEL_NAME_JUDGE,  # Use a cheaper model for judge evaluation
            messages=[{"role": "user", "content": formatted_prompt}],
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse JSON response to extract the predicted label
        # The LLM should return: {"reasoning": "...", "label": "PASS" or "FAIL"}
        try:
            # Handle different JSON formats the LLM might return
            if "```json" in response_text:
                # LLM wrapped response in markdown code block
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
        except json.JSONDecodeError:
            # Failed to parse JSON - mark as UNKNOWN
            predicted_label = "UNKNOWN"

        # Return the evaluation result
        return {
            "trace_id": trace.get("trace_id", "unknown"),
            "true_label": true_label,  # What it should be (ground truth)
            "predicted_label": predicted_label,  # What judge predicted
            "query": query,
            "dietary_restriction": dietary_restriction,
            "success": True,
        }

    except Exception as e:
        # Handle LLM API errors or other exceptions
        return {
            "trace_id": trace.get("trace_id", "unknown"),
            "true_label": true_label,
            "predicted_label": "ERROR",
            "query": query,
            "dietary_restriction": dietary_restriction,
            "success": False,
            "error": str(e),
        }


def evaluate_judge_on_dev(
    judge_prompt: str,
    dev_traces: List[Dict[str, Any]],
    sample_size: int = 50,
    max_workers: int = MAX_WORKERS,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Evaluate the judge prompt on a sample of the dev set using parallel processing.

    This is STEP 2 of the workflow. It tests how well the judge prompt performs
    by evaluating it on dev set recipes (data the judge has never seen).

    Process:
    1. Sample up to 50 traces from dev set (to save time/cost)
    2. Use ThreadPoolExecutor to evaluate traces in parallel (32 workers)
    3. Each worker calls evaluate_single_trace() for one recipe
    4. Collect all predictions (true_label vs predicted_label)
    5. Calculate performance metrics: TPR, TNR, Balanced Accuracy

    Metrics explained:
    - TPR (True Positive Rate): Of all PASS recipes, what % did judge correctly identify?
    - TNR (True Negative Rate): Of all FAIL recipes, what % did judge correctly identify?
    - Balanced Accuracy: (TPR + TNR) / 2 - overall performance score

    Returns:
        Tuple of (TPR, TNR, predictions_list)
    """

    # Sample dev traces for evaluation (limit to sample_size to save API costs)
    if len(dev_traces) > sample_size:
        sampled_traces = random.sample(dev_traces, sample_size)
    else:
        sampled_traces = dev_traces

    console.print(
        f"[yellow]Evaluating judge on {len(sampled_traces)} dev traces with {max_workers} workers..."
    )

    # Prepare tasks for parallel processing
    # Each task is a tuple of (trace, judge_prompt) for evaluate_single_trace()
    tasks = [(trace, judge_prompt) for trace in sampled_traces]

    predictions = []

    # Use ThreadPoolExecutor for parallel evaluation (much faster than sequential)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
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

                if not result["success"]:
                    console.print(
                        f"[yellow]Warning: Failed to evaluate trace {result['trace_id']}: {result.get('error', 'Unknown error')}"
                    )

                # Update progress display
                status.update(
                    f"[yellow]Evaluated {completed}/{total} traces ({completed/total*100:.1f}%)"
                )

    console.print(f"[green]Completed parallel evaluation of {len(predictions)} traces")

    # Calculate confusion matrix components
    # TP = True Positive: judge said PASS, and it was correct
    tp = sum(
        1
        for p in predictions
        if p["true_label"] == "PASS" and p["predicted_label"] == "PASS"
    )
    # FN = False Negative: judge said FAIL, but should have been PASS
    fn = sum(
        1
        for p in predictions
        if p["true_label"] == "PASS" and p["predicted_label"] == "FAIL"
    )
    # TN = True Negative: judge said FAIL, and it was correct
    tn = sum(
        1
        for p in predictions
        if p["true_label"] == "FAIL" and p["predicted_label"] == "FAIL"
    )
    # FP = False Positive: judge said PASS, but should have been FAIL
    fp = sum(
        1
        for p in predictions
        if p["true_label"] == "FAIL" and p["predicted_label"] == "PASS"
    )

    # Calculate TPR (sensitivity/recall for PASS) and TNR (specificity for FAIL)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # PASS accuracy
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # FAIL accuracy

    return tpr, tnr, predictions


def save_judge_prompt(prompt: str, output_path: str) -> None:
    """Save the judge prompt to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    console.print(f"[green]Saved judge prompt to {output_path}")


def main():
    """
    Main function to develop and evaluate the LLM judge.

    This orchestrates the complete workflow from STEP 1 (prompt creation) to
    STEP 2 (evaluation on dev set). See the workflow diagram in the module
    docstring for a visual representation.

    Workflow:
    1. Load training and dev data from CSV files
    2a. If OWN_PROMPT=False: Create judge prompt with random few-shot examples
    2b. If OWN_PROMPT=True: Load custom judge prompt from file
    3. Evaluate judge on dev set (50 samples, parallel processing)
    4. Calculate and display metrics (TPR, TNR, Balanced Accuracy)
    5. Save judge prompt and predictions for analysis
    """
    console.print("[bold blue]LLM Judge Development")
    console.print("=" * 50)

    # ========================================
    # SETUP: Configure paths
    # ========================================
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"
    results_dir = hw3_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # ========================================
    # LOAD DATA: Check that data splits exist
    # ========================================
    train_path = data_dir / "train_set.csv"
    dev_path = data_dir / "dev_set.csv"

    if not train_path.exists() or not dev_path.exists():
        console.print("[red]Error: Train or dev set not found!")
        console.print("[yellow]Please run split_data.py first.")
        return

    # ========================================
    # STEP 1: CREATE JUDGE PROMPT
    # ========================================
    # Two modes:
    # Mode A (OWN_PROMPT=False): Auto-generate with random examples from training data
    # Mode B (OWN_PROMPT=True): Use manually refined prompt from previous run

    if not OWN_PROMPT:
        # Mode A: Generate prompt automatically
        train_traces = load_data_split(str(train_path))
        console.print(f"[green]Loaded {len(train_traces)} train traces")

        # Select few-shot examples (1 PASS, 3 FAIL by default)
        few_shot_examples = select_few_shot_examples(train_traces, seed=SEED)

        if not few_shot_examples:
            console.print("[red]Failed to select few-shot examples!")
            return

    # Load dev set for evaluation (used in both modes)
    dev_traces = load_data_split(str(dev_path))
    console.print(f"[green]Loaded {len(dev_traces)} dev traces")

    # Create or load the judge prompt
    prompt_path = results_dir / "judge_prompt.txt"

    if OWN_PROMPT:
        # Mode B: Load manually refined prompt
        console.print("[yellow]Using custom judge prompt...")
        judge_prompt = read_judge_prompt(prompt_path)
    else:
        # Mode A: Create new prompt with selected examples
        console.print("[yellow]Using base judge prompt...")
        judge_prompt = create_judge_prompt(few_shot_examples)

    # ========================================
    # STEP 2: EVALUATE JUDGE ON DEV SET
    # ========================================
    # Test the judge on 50 dev traces using 32 parallel workers
    console.print("[yellow]Evaluating judge on dev set...")
    tpr, tnr, predictions = evaluate_judge_on_dev(judge_prompt, dev_traces)

    # ========================================
    # RESULTS: Display performance metrics
    # ========================================
    console.print("\n[bold]Judge Performance on Dev Set:")
    console.print(f"True Positive Rate (TPR): {tpr:.3f}")  # PASS accuracy
    console.print(f"True Negative Rate (TNR): {tnr:.3f}")  # FAIL accuracy
    console.print(f"Balanced Accuracy: {(tpr + tnr) / 2:.3f}")  # Overall score

    # ========================================
    # SAVE: Store prompt and predictions
    # ========================================
    # Save the judge prompt (only if auto-generated, not custom)
    if not OWN_PROMPT:
        save_judge_prompt(judge_prompt, str(prompt_path))

    # Save predictions for error analysis
    # You can review this file to see which recipes the judge got wrong
    predictions_path = results_dir / "dev_predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    console.print(f"[green]Saved dev predictions to {predictions_path}")

    console.print("\n[bold green]Judge development completed!")
    console.print(f"[blue]Judge prompt saved to: {prompt_path}")


if __name__ == "__main__":
    main()
