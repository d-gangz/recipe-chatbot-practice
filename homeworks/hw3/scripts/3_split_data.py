#!/usr/bin/env python3
"""Split labeled traces into train, dev, and test sets.

This script splits the labeled traces into stratified train/dev/test sets
for developing and evaluating the LLM judge.

WHAT THIS SCRIPT DOES:
- Input: labeled_traces.csv (~150 traces with PASS/FAIL labels)
- Output: 3 CSV files split with stratification (maintains balanced PASS/FAIL ratios)
  * train_set.csv (15% = ~23 traces) - For few-shot examples in judge prompt
  * dev_set.csv (40% = ~60 traces) - For iterative judge development/tuning
  * test_set.csv (45% = ~67 traces) - For final unbiased evaluation

SPLIT RATIOS (lines 154-156):
- Train: 15% (small, just for few-shot examples)
- Dev: 40% (large, for iterative development)
- Test: 45% (largest, for final evaluation)

STRATIFICATION:
- Ensures each split maintains the same PASS/FAIL ratio as the original data
- Example: If original is 50% PASS / 50% FAIL, each split will also be ~50/50
- Controlled by stratify=df['label'] parameter in train_test_split calls
"""

import pandas as pd  # For CSV reading/writing and DataFrame operations
import random  # Not actively used but imported
from pathlib import Path  # For file path operations
from typing import List, Dict, Any, Tuple  # Type hints for better code documentation
from rich.console import Console  # For colored terminal output
from sklearn.model_selection import train_test_split  # For stratified data splitting

console = Console()


def load_labeled_traces(csv_path: str) -> List[Dict[str, Any]]:
    """Load labeled traces from CSV file.

    Args:
        csv_path: Path to labeled_traces.csv

    Returns:
        List of dictionaries, each representing one labeled trace
        Example: [{'query': '...', 'dietary_restriction': '...', 'response': '...',
                   'label': 'PASS', 'reasoning': '...', 'confidence': 'HIGH'}, ...]
    """
    # Read CSV into DataFrame
    df = pd.read_csv(csv_path)

    # Convert to list of dictionaries ('records' format)
    # Each row becomes a dict with column names as keys
    return df.to_dict("records")


def stratified_split(
    traces: List[Dict[str, Any]],
    train_ratio: float = 0.15,
    dev_ratio: float = 0.40,
    test_ratio: float = 0.45,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split traces into train/dev/test sets with stratification by label.

    STRATIFICATION EXPLANATION:
    - stratify=df["label"] ensures each split maintains the same PASS/FAIL ratio
    - Example: If input is 50% PASS / 50% FAIL, then:
      * Train will be ~50% PASS / 50% FAIL
      * Dev will be ~50% PASS / 50% FAIL
      * Test will be ~50% PASS / 50% FAIL
    - Without stratification, you might get unlucky splits like train=80% PASS, 20% FAIL

    TWO-STEP SPLITTING PROCESS:
    1. Split train from (dev+test) with ratio 15% vs 85%
    2. Split the 85% into dev and test with adjusted ratios

    Args:
        traces: List of all labeled traces (~150)
        train_ratio: Fraction for train set (default 0.15 = 15%)
        dev_ratio: Fraction for dev set (default 0.40 = 40%)
        test_ratio: Fraction for test set (default 0.45 = 45%)
        random_state: Random seed for reproducibility (42 = consistent splits)

    Returns:
        Tuple of (train_traces, dev_traces, test_traces)
    """

    # Validate that ratios sum to 1.0 (allow tiny floating point error)
    assert (
        abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    # Convert list of dicts to DataFrame for sklearn compatibility
    df = pd.DataFrame(traces)

    # STEP 1: First split - separate train from (dev + test)
    # Example: 15% train vs 85% temp (which will become dev+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(dev_ratio + test_ratio),  # 0.40 + 0.45 = 0.85 (85% goes to temp)
        stratify=df["label"],  # Keep PASS/FAIL ratio balanced in both splits
        random_state=random_state,  # Seed for reproducible splits
    )

    # STEP 2: Second split - separate dev from test (from the 85% temp data)
    # We need to calculate what fraction of temp_df should go to dev
    # If we want 40% dev and 45% test out of 100% total:
    #   - temp_df is 85% of total
    #   - dev should be 40/85 = 0.47 of temp_df
    #   - test should be 45/85 = 0.53 of temp_df
    dev_test_ratio = dev_ratio / (dev_ratio + test_ratio)  # 0.40 / 0.85 = 0.47
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_test_ratio),  # 1 - 0.47 = 0.53 (53% of temp goes to test)
        stratify=temp_df["label"],  # Maintain PASS/FAIL balance
        random_state=random_state,
    )

    # Convert DataFrames back to list of dictionaries
    train_traces = train_df.to_dict("records")
    dev_traces = dev_df.to_dict("records")
    test_traces = test_df.to_dict("records")

    return train_traces, dev_traces, test_traces


def save_split(traces: List[Dict[str, Any]], output_path: str, split_name: str) -> None:
    """Save a data split to CSV file."""
    df = pd.DataFrame(traces)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Saved {len(traces)} {split_name} traces to {output_path}")


def print_split_statistics(
    train_traces: List[Dict[str, Any]],
    dev_traces: List[Dict[str, Any]],
    test_traces: List[Dict[str, Any]],
) -> None:
    """Print statistics about the data splits."""

    def get_label_counts(traces: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {}
        for trace in traces:
            label = trace["label"]
            counts[label] = counts.get(label, 0) + 1
        return counts

    def get_restriction_counts(traces: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {}
        for trace in traces:
            restriction = trace["dietary_restriction"]
            counts[restriction] = counts.get(restriction, 0) + 1
        return counts

    total_traces = len(train_traces) + len(dev_traces) + len(test_traces)

    console.print("\n[bold]Data Split Statistics:")
    console.print(f"Total traces: {total_traces}")
    console.print(f"Train: {len(train_traces)} ({len(train_traces)/total_traces:.1%})")
    console.print(f"Dev: {len(dev_traces)} ({len(dev_traces)/total_traces:.1%})")
    console.print(f"Test: {len(test_traces)} ({len(test_traces)/total_traces:.1%})")

    # Label distribution
    console.print("\n[bold]Label Distribution:")
    for split_name, traces in [
        ("Train", train_traces),
        ("Dev", dev_traces),
        ("Test", test_traces),
    ]:
        label_counts = get_label_counts(traces)
        console.print(f"{split_name}:")
        for label, count in sorted(label_counts.items()):
            console.print(f"  {label}: {count} ({count/len(traces):.1%})")

    # Dietary restriction distribution (for train set)
    console.print("\n[bold]Dietary Restrictions in Train Set:")
    restriction_counts = get_restriction_counts(train_traces)
    for restriction, count in sorted(restriction_counts.items()):
        console.print(f"  {restriction}: {count}")


def validate_splits(
    train_traces: List[Dict[str, Any]],
    dev_traces: List[Dict[str, Any]],
    test_traces: List[Dict[str, Any]],
) -> bool:
    """Validate that the splits are reasonable.

    This is a CRITICAL quality control check that prevents silent failures.
    Without this validation, you could waste hours developing a judge on bad data.

    WHAT THIS FUNCTION CHECKS:
    1. Each split (train/dev/test) has BOTH labels (PASS and FAIL)
       - Why: Can't train or evaluate a binary classifier with only one class
       - Example bad case: Test set with only PASS examples can't measure FAIL detection

    2. Train set has at least 3 different dietary restrictions
       - Why: Train set is used for few-shot examples in judge prompt
       - Example bad case: Only "vegan" and "gluten-free" means judge won't learn about "keto", "paleo"
       - With ~23 train examples, we should have good dietary diversity

    Args:
        train_traces: Training split (~23 traces)
        dev_traces: Development split (~60 traces)
        test_traces: Test split (~67 traces)

    Returns:
        True if all validations pass, False if any validation fails
    """

    # CHECK #1: Verify each split has both PASS and FAIL labels
    # This ensures we can properly train/evaluate binary classification
    for split_name, traces in [
        ("Train", train_traces),
        ("Dev", dev_traces),
        ("Test", test_traces),
    ]:
        # Get unique labels in this split (should be {'PASS', 'FAIL'})
        labels = set(trace["label"] for trace in traces)

        # If we have fewer than 2 unique labels, something is wrong
        if len(labels) < 2:
            console.print(f"[red]Warning: {split_name} set only has labels: {labels}")
            console.print(
                "[red]This means you can't evaluate both PASS and FAIL cases!"
            )
            return False

    # CHECK #2: Verify train set has reasonable dietary diversity
    # Train set is used for few-shot examples, so needs variety
    train_restrictions = set(trace["dietary_restriction"] for trace in train_traces)

    # We expect at least 3 different dietary restrictions in ~23 train examples
    if len(train_restrictions) < 3:
        console.print(
            f"[red]Warning: Train set only has {len(train_restrictions)} dietary restrictions"
        )
        console.print(
            "[red]Few-shot examples need more diversity to teach the judge properly!"
        )
        return False

    # All checks passed!
    console.print("[green]Data splits validation passed!")
    return True


def main():
    """Main function to split labeled data.

    PIPELINE:
    1. Load labeled_traces.csv (~150 traces with PASS/FAIL labels)
    2. Split with stratification into train (15%), dev (40%), test (45%)
    3. Validate splits (check for both labels, dietary diversity)
    4. Save three CSV files: train_set.csv, dev_set.csv, test_set.csv
    5. Print detailed statistics
    """
    console.print("[bold blue]Data Splitting for LLM Judge Development")
    console.print("=" * 50)

    # Set up file paths
    # Example: if script is at hw3/scripts/3_split_data.py
    # Then: script_dir = hw3/scripts/, hw3_dir = hw3/, data_dir = hw3/data/
    script_dir = Path(__file__).parent  # Directory containing this script
    hw3_dir = script_dir.parent  # Parent directory (hw3/)
    data_dir = hw3_dir / "data"  # Data directory (hw3/data/)

    # Load labeled traces from previous step (2_label_data.py output)
    labeled_path = data_dir / "labeled_traces.csv"
    if not labeled_path.exists():
        console.print(f"[red]Error: {labeled_path} not found!")
        console.print("[yellow]Please run label_data.py first.")
        return

    # Load ~150 labeled traces into list of dictionaries
    traces = load_labeled_traces(str(labeled_path))
    console.print(f"[green]Loaded {len(traces)} labeled traces")

    # Perform stratified split into train/dev/test
    # Stratification maintains balanced PASS/FAIL ratios in all splits
    console.print("[yellow]Splitting data into train/dev/test sets...")
    train_traces, dev_traces, test_traces = stratified_split(
        traces,
        train_ratio=0.15,  # 15% (~23 traces) - Small, for few-shot examples
        dev_ratio=0.40,  # 40% (~60 traces) - Large, for iterative development
        test_ratio=0.45,  # 45% (~67 traces) - Largest, for final evaluation
    )

    # CRITICAL: Validate splits before saving
    # This prevents wasting time with bad data (e.g., test set with only PASS examples)
    if not validate_splits(train_traces, dev_traces, test_traces):
        console.print("[red]Data split validation failed!")
        console.print("[yellow]Splits are not usable - stopping execution!")
        return  # Don't save bad splits!

    # Save the three splits to separate CSV files
    train_path = data_dir / "train_set.csv"
    dev_path = data_dir / "dev_set.csv"
    test_path = data_dir / "test_set.csv"

    save_split(train_traces, str(train_path), "train")
    save_split(dev_traces, str(dev_path), "dev")
    save_split(test_traces, str(test_path), "test")

    # Print detailed statistics about the splits
    # Shows: sizes, percentages, label distributions, dietary restrictions
    print_split_statistics(train_traces, dev_traces, test_traces)

    # Success message with rationale
    console.print("\n[bold green]Data splitting completed successfully!")
    console.print("\n[bold]Split Rationale:")
    console.print("• Train (15%): Small set for few-shot examples in judge prompt")
    console.print("• Dev (40%): Large set for iterative judge development and tuning")
    console.print(
        "• Test (45%): Large set for final unbiased evaluation of judge performance"
    )


if __name__ == "__main__":
    main()
