#!/usr/bin/env python3
"""Label Recipe Bot traces for dietary adherence using GPT-4o.

This script can be used as a powerful automatic labeler to create ground
truth labels for whether Recipe Bot responses properly adhere to dietary restrictions.
It uses GPT-4o as a powerful labeler and also comes with a well designed and tested
labeling prompt.

Remember that automatic creation of ground truth labels should never be used without
also looking extensively at the data. This script is intended as a fast way to give
you "ground truth" for this homework assignment.

HOW MANY TRACES ARE LABELED:
- Input: raw_traces.csv (~500 traces from generate_traces.py)
- Sampled for labeling: 200 traces (line 385: sample_size=200)
- Final output: ~150 traces (75 PASS + 75 FAIL, lines 391-392: target_positive=75, target_negative=75)
- Output file: labeled_traces.csv

TO CHANGE THE NUMBER OF TRACES:
- To label more/fewer traces total: Change sample_size parameter in line 385
- To change final balanced dataset size: Change target_positive and target_negative in lines 391-392
- To change parallelism: Change MAX_WORKERS constant (line 43) - higher = faster but more API load
"""

import pandas as pd  # For CSV file reading/writing
import json  # For parsing JSON responses from GPT-4o
import random  # For random sampling of traces
from pathlib import Path  # For file path operations
from typing import List, Dict, Any, Optional  # Type hints for better code documentation
from rich.console import Console  # For colored terminal output
from rich.progress import track  # For progress bars (not used in this version)
import litellm  # Unified API for calling different LLM providers (we use GPT-4o)
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)  # For parallel processing
from dotenv import load_dotenv  # For loading API keys from .env file

# Load environment variables (like OPENAI_API_KEY) from .env file
load_dotenv()

# Number of parallel threads for labeling (32 traces labeled simultaneously)
MAX_WORKERS = 32

# Rich console for formatted output
console = Console()

# Labeling prompt for GPT-4o
LABELING_PROMPT = """You are an expert nutritionist and dietary specialist. Your task is to evaluate whether a recipe response properly adheres to the specified dietary restriction.

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
- PASS: The recipe clearly adheres to the dietary restriction with appropriate ingredients and preparation methods
- FAIL: The recipe contains ingredients or methods that violate the dietary restriction
- Consider both explicit ingredients and cooking methods
- Be strict but reasonable in your evaluation

Please analyze the query, dietary restriction, and recipe response, then provide your evaluation.

Query: {query}
Dietary Restriction: {dietary_restriction}
Recipe Response: {response}

Provide your analysis in the following JSON format:
{{
    "reasoning": "Detailed explanation of your evaluation, citing specific ingredients or methods",
    "label": "PASS" or "FAIL",
    "confidence": "HIGH", "MEDIUM", or "LOW"
}}"""


def load_traces(csv_path: str) -> List[Dict[str, Any]]:
    """Load traces from CSV file.

    Args:
        csv_path: Path to raw_traces.csv file

    Returns:
        List of dictionaries, where each dict represents one row/trace
        Example: [{'query': '...', 'dietary_restriction': '...', 'response': '...'}, ...]
    """
    # Read CSV into pandas DataFrame
    df = pd.read_csv(csv_path)

    # Convert DataFrame to list of dictionaries
    # 'records' format: each row becomes a dict with column names as keys
    # Example: df with columns [query, dietary_restriction, response]
    #   becomes: [{'query': 'val1', 'dietary_restriction': 'val2', 'response': 'val3'}, ...]
    return df.to_dict("records")


def get_labeling_response(
    query: str, dietary_restriction: str, response: str
) -> Optional[Dict[str, Any]]:
    """Get labeling response from GPT-4o.

    Sends a trace to GPT-4o for evaluation and returns the parsed JSON response.

    Args:
        query: User's recipe query (e.g., "What's a good breakfast?")
        dietary_restriction: The dietary restriction to check (e.g., "vegan")
        response: Recipe Bot's response to evaluate

    Returns:
        Dict with keys: 'reasoning', 'label' (PASS/FAIL), 'confidence' (HIGH/MEDIUM/LOW)
        or None if API call fails or JSON parsing fails
    """
    try:
        # Fill in the template prompt with actual values
        prompt = LABELING_PROMPT.format(
            query=query, dietary_restriction=dietary_restriction, response=response
        )

        # Call GPT-4o API (temperature=0 for deterministic/consistent results)
        completion = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # No randomness - always get same answer for same input
        )

        # Extract the text response from GPT-4o
        response_text = completion.choices[0].message.content.strip()

        # Try to parse JSON response
        try:
            # GPT-4o sometimes wraps JSON in markdown code blocks like: ```json {...} ```
            # We need to extract just the JSON part
            if "```json" in response_text:
                # Find where ```json starts, then extract content until next ```
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                # JSON not in markdown, but might have extra text around it
                # Extract from first { to last }
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Assume entire response is JSON
                json_text = response_text

            # Parse the JSON string into a Python dictionary
            result = json.loads(json_text)
            return result
        except json.JSONDecodeError:
            console.print(
                f"[yellow]Warning: Could not parse JSON response: {response_text}"
            )
            return None

    except Exception as e:
        console.print(f"[red]Error getting labeling response: {str(e)}")
        return None


def label_single_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Label a single trace using GPT-4o.

    This function is called by each thread in the parallel processing pool.

    Args:
        trace: Dictionary with keys 'query', 'dietary_restriction', 'response'

    Returns:
        Original trace dict with added fields:
        - label: "PASS" or "FAIL"
        - reasoning: GPT-4o's explanation
        - confidence: "HIGH", "MEDIUM", or "LOW"
        - labeled: True if successful, False if labeling failed
    """
    # Extract the three fields we need from the trace
    query = trace["query"]
    dietary_restriction = trace["dietary_restriction"]
    response = trace["response"]

    # Send to GPT-4o for evaluation
    labeling_result = get_labeling_response(query, dietary_restriction, response)

    # Add the labeling results to the trace
    if labeling_result:
        # Successfully got a label from GPT-4o
        labeled_trace = trace.copy()  # Don't modify original, create a copy
        labeled_trace.update(
            {
                "label": labeling_result.get("label"),  # "PASS" or "FAIL"
                "reasoning": labeling_result.get("reasoning"),  # Explanation text
                "confidence": labeling_result.get(
                    "confidence"
                ),  # "HIGH"/"MEDIUM"/"LOW"
                "labeled": True,  # Flag indicating successful labeling
            }
        )
    else:
        # Labeling failed (API error or JSON parsing error)
        labeled_trace = trace.copy()
        labeled_trace.update(
            {"label": None, "reasoning": None, "confidence": None, "labeled": False}
        )

    return labeled_trace


def label_traces(
    traces: List[Dict[str, Any]], sample_size: int = 150, max_workers: int = MAX_WORKERS
) -> List[Dict[str, Any]]:
    """Label a sample of traces using GPT-4o with parallel processing.

    This is the KEY optimization: instead of labeling 200 traces sequentially (400 seconds),
    we label them in parallel with 32 workers (~12 seconds).

    Args:
        traces: Full list of traces to sample from
        sample_size: Number of traces to label (default 150)
        max_workers: Number of parallel threads (default 32)

    Returns:
        List of labeled traces (same size as sample_size or smaller if some failed)
    """
    # Step 1: Randomly sample traces to label
    # We sample more than needed (200) because some may fail to label
    if len(traces) > sample_size:
        sampled_traces = random.sample(traces, sample_size)
    else:
        sampled_traces = traces

    labeled_traces = []

    # Step 2: Create a thread pool for parallel processing
    # ThreadPoolExecutor manages a pool of worker threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all labeling tasks to the thread pool at once
        # This creates a dictionary mapping: {future_object: original_trace}
        # executor.submit() returns immediately without waiting for completion
        future_to_trace = {
            executor.submit(label_single_trace, trace): trace
            for trace in sampled_traces
        }

        # Step 3: Collect results as they complete (not in submission order!)
        # as_completed() yields futures as they finish (fastest first)
        with console.status(
            "[yellow]Labeling traces with GPT-4o in parallel..."
        ) as status:
            completed = 0
            total = len(sampled_traces)

            # Process each completed labeling task
            for future in as_completed(future_to_trace):
                labeled_trace = (
                    future.result()
                )  # Get the labeled trace from this thread
                labeled_traces.append(labeled_trace)
                completed += 1

                # Update progress display: "Labeled 47/200 traces (23.5%)"
                status.update(
                    f"[yellow]Labeled {completed}/{total} traces ({completed/total*100:.1f}%)"
                )

    console.print(f"[green]Completed parallel labeling of {len(labeled_traces)} traces")
    return labeled_traces


def balance_labels(
    labeled_traces: List[Dict[str, Any]],
    target_positive: int = 75,
    target_negative: int = 75,
) -> List[Dict[str, Any]]:
    """Balance the dataset to have roughly equal positive and negative examples.

    Machine learning models train better with balanced classes (50/50 PASS/FAIL).
    This prevents bias toward the majority class.

    Args:
        labeled_traces: All labeled traces (may include some with labeled=False)
        target_positive: Number of PASS examples to include (default 75)
        target_negative: Number of FAIL examples to include (default 75)

    Returns:
        Balanced list of ~150 traces (75 PASS + 75 FAIL), shuffled randomly
    """
    # Step 1: Filter out failed labeling attempts
    # Only keep traces where labeling succeeded AND label is valid
    valid_traces = [
        t for t in labeled_traces if t["labeled"] and t["label"] in ["PASS", "FAIL"]
    ]

    # Step 2: Separate by label type
    pass_traces = [t for t in valid_traces if t["label"] == "PASS"]
    fail_traces = [t for t in valid_traces if t["label"] == "FAIL"]

    console.print(
        f"[blue]Available traces: {len(pass_traces)} PASS, {len(fail_traces)} FAIL"
    )

    # Step 3: Sample equal numbers from each category
    # Use min() in case we don't have enough of one type
    # Example: if we only have 60 FAIL traces, we take all 60 (not 75)
    selected_pass = random.sample(pass_traces, min(target_positive, len(pass_traces)))
    selected_fail = random.sample(fail_traces, min(target_negative, len(fail_traces)))

    # Step 4: Combine and shuffle
    # Shuffling prevents all PASS traces from being first
    balanced_traces = selected_pass + selected_fail
    random.shuffle(balanced_traces)

    console.print(
        f"[green]Balanced dataset: {len(selected_pass)} PASS, {len(selected_fail)} FAIL"
    )

    return balanced_traces


def save_labeled_traces(traces: List[Dict[str, Any]], output_path: str) -> None:
    """Save labeled traces to CSV file.

    Args:
        traces: List of labeled trace dictionaries
        output_path: Path where CSV file should be saved

    Output CSV will have columns:
    - query, dietary_restriction, response (original fields)
    - label, reasoning, confidence, labeled (new fields from GPT-4o)
    """
    # Convert list of dictionaries back to DataFrame
    df = pd.DataFrame(traces)

    # Save to CSV without row index column
    df.to_csv(output_path, index=False)
    console.print(f"[green]Saved {len(traces)} labeled traces to {output_path}")


def main():
    """Main function to label traces.

    Pipeline:
    1. Load raw_traces.csv (500 traces from generate_traces.py)
    2. Sample and label 200 traces in parallel using GPT-4o (gets ~180-200 successful)
    3. Balance to 75 PASS + 75 FAIL = 150 total
    4. Save to labeled_traces.csv
    5. Print summary statistics
    """
    console.print("[bold blue]Recipe Bot Trace Labeling")
    console.print("=" * 50)

    # Set up file paths
    # Example: if script is at hw3/scripts/2_label_data.py
    # Then: script_dir = hw3/scripts/, hw3_dir = hw3/, data_dir = hw3/data/
    script_dir = Path(__file__).parent  # Directory containing this script
    hw3_dir = script_dir.parent  # Parent directory (hw3/)
    data_dir = hw3_dir / "data"  # Data directory (hw3/data/)

    # Load raw traces from CSV
    traces_path = data_dir / "raw_traces.csv"
    if not traces_path.exists():
        console.print(f"[red]Error: {traces_path} not found!")
        console.print("[yellow]Please run generate_traces.py first.")
        return

    traces = load_traces(str(traces_path))  # Returns list of ~500 trace dictionaries
    console.print(f"[green]Loaded {len(traces)} traces")

    # Label 200 traces in parallel (we label extra in case some fail)
    # With 32 workers, this takes ~12 seconds instead of ~400 seconds sequentially
    console.print("[yellow]Labeling traces with GPT-4o using parallel processing...")
    labeled_traces = label_traces(
        traces, sample_size=200, max_workers=MAX_WORKERS
    )  # Returns ~180-200 successfully labeled traces

    # Balance the dataset to 75 PASS + 75 FAIL for better ML training
    balanced_traces = balance_labels(
        labeled_traces, target_positive=75, target_negative=75
    )

    # Save the final balanced dataset to CSV
    output_path = data_dir / "labeled_traces.csv"
    save_labeled_traces(balanced_traces, str(output_path))

    # Print summary statistics about the labeled dataset
    console.print("\n[bold]Labeling Summary:")
    console.print(f"Total labeled traces: {len(balanced_traces)}")

    # Count how many of each label type (PASS/FAIL)
    label_counts = {}
    confidence_counts = {}
    for trace in balanced_traces:
        label = trace["label"]
        confidence = trace["confidence"]
        # .get() with default 0 handles first occurrence of each label/confidence
        label_counts[label] = label_counts.get(label, 0) + 1
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

    # Display label distribution (should be ~75 PASS, ~75 FAIL)
    console.print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        console.print(f"  {label}: {count}")

    # Display confidence distribution (shows how confident GPT-4o was)
    console.print("\nConfidence distribution:")
    for confidence, count in sorted(confidence_counts.items()):
        console.print(f"  {confidence}: {count}")


if __name__ == "__main__":
    main()
