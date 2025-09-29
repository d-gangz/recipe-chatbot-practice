"""
Generate recipes from synthetic queries using LLM and output as CSV with message arrays.

This script processes synthetic queries from hw2-1/synthetic_queries_for_analysis.csv,
generates recipe responses using the LLM, and outputs the results as a CSV containing
the original query data plus the complete conversation message arrays.

Prerequisites:
- Set your `OPENAI_API_KEY` environment variable for `gpt-4o-mini` access.
- Ensure the synthetic_queries_for_analysis.csv file exists in hw2-1 directory.

Output Format:
- ID: Original query ID from input CSV
- query: Original user query text
- dimension_tuple: JSON string of the dimension tuple from input
- message_array: JSON string containing [system_prompt, user_query, ai_response]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Add the project root to the Python path to import backend.utils
current_dir = Path(__file__).parent
project_root = current_dir.resolve().parent.parent
sys.path.insert(0, str(project_root))

# pylint: disable=import-error,wrong-import-position
from backend.utils import get_agent_response

load_dotenv()

# --- Configuration ---
INPUT_CSV_PATH = (
    Path(__file__).parent.parent / "hw2-1" / "synthetic_queries_for_analysis.csv"
)
OUTPUT_CSV_PATH = Path(__file__).parent / "recipe_responses.csv"
MAX_WORKERS = 5  # Number of parallel LLM calls
BATCH_SIZE = 25  # Process in batches to avoid overwhelming the API
TEST_LIMIT = 3  # Only process first 3 queries in test mode


def process_single_query(row_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single query and generate recipe response.

    Args:
        row_data: Dictionary containing the row data from input CSV

    Returns:
        Dictionary with processed data including message array, or None if failed
    """
    try:
        query_id = row_data["id"]
        query_text = row_data["query"]
        dimension_tuple = row_data["dimension_tuple_json"]

        # Create the conversation with user query
        user_message = {"role": "user", "content": query_text}

        # Get agent response which includes system prompt + user query + AI response
        complete_conversation = get_agent_response([user_message])

        # The get_agent_response function returns the full conversation including system prompt
        message_array = complete_conversation

        # Prepare the output row
        result = {
            "id": query_id,
            "query": query_text,
            "dimension_tuple": dimension_tuple,
            "message_array": json.dumps(message_array, ensure_ascii=False),
        }

        return result

    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Error processing query {row_data.get('id', 'unknown')}: {exc}")
        return None


def process_queries_parallel(
    df: pd.DataFrame,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_workers: int = MAX_WORKERS,
) -> List[Dict[str, Any]]:
    """
    Process queries in parallel using ThreadPoolExecutor.

    Args:
        df: DataFrame containing the input queries
        start_idx: Starting index for processing
        end_idx: Ending index for processing (None means process all remaining)

    Returns:
        List of processed results
    """
    if end_idx is None:
        end_idx = len(df)

    # Convert DataFrame rows to dictionaries for processing
    rows_to_process = df.iloc[start_idx:end_idx].to_dict("records")

    results = []

    print(f"Processing queries {start_idx + 1} to {end_idx} of {len(df)}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_query, row): row for row in rows_to_process
        }

        # Process completed tasks as they finish
        with tqdm(total=len(rows_to_process), desc="Generating Recipes") as pbar:
            for future in as_completed(future_to_row):
                row = future_to_row[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(
                        f"Query {row.get('id', 'unknown')} generated an exception: {exc}"
                    )
                    pbar.update(1)

    return results


def load_input_data() -> pd.DataFrame:
    """Load and validate the input CSV file."""
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)

    # Validate required columns
    required_columns = ["id", "query", "dimension_tuple_json"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in input CSV: {missing_columns}")

    print(f"Loaded {len(df)} queries from {INPUT_CSV_PATH}")
    return df


def save_results_to_csv(results: List[Dict[str, Any]], output_path: Path):
    """Save processed results to CSV file."""
    if not results:
        print("No results to save.")
        return

    # Create DataFrame from results
    df_output = pd.DataFrame(results)

    # Save to CSV
    df_output.to_csv(output_path, index=False)
    print(f"Saved {len(results)} recipe responses to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate recipes from synthetic queries using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (process first 3 queries)
  uv run python homeworks/hw2-2/generate_recipes.py --test
  
  # Production mode (process all queries)
  uv run python homeworks/hw2-2/generate_recipes.py
  
  # Custom test limit and batch size
  uv run python homeworks/hw2-2/generate_recipes.py --test --test-limit 10 --batch-size 50
        """,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode to process only a limited number of queries",
    )

    parser.add_argument(
        "--test-limit",
        type=int,
        default=TEST_LIMIT,
        help=f"Number of queries to process in test mode (default: {TEST_LIMIT})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of queries per batch (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV_PATH),
        help=f"Output CSV path (default: {OUTPUT_CSV_PATH})",
    )

    return parser.parse_args()


def main():
    """Main function to process queries and generate recipes."""
    args = parse_arguments()
    # Check for API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    start_time = time.time()

    # Parse and display configuration
    test_mode = args.test
    output_path = Path(args.output)

    print(f"🔧 Configuration:")
    print(f"   Mode: {'Test' if test_mode else 'Production'}")
    print(f"   Max Workers: {args.max_workers}")
    print(f"   Batch Size: {args.batch_size}")
    if test_mode:
        print(f"   Test Limit: {args.test_limit}")
    print(f"   Output: {output_path}")
    print()

    try:
        # Load input data
        print("Step 1: Loading input data...")
        df = load_input_data()

        # In test mode, limit to first few queries
        if test_mode:
            df = df.head(args.test_limit)
            print(f"🧪 TEST MODE: Processing only first {args.test_limit} queries")
        else:
            print(f"🚀 PRODUCTION MODE: Processing all {len(df)} queries")

        # Check if output file already exists and ask for confirmation
        if output_path.exists():
            response = input(
                f"Output file {output_path} already exists. Overwrite? (y/N): "
            )
            if response.lower() != "y":
                print("Operation cancelled.")
                return

        # Process all queries
        print("\nStep 2: Processing queries and generating recipes...")
        all_results = []

        # Process in batches to avoid overwhelming the API
        total_queries = len(df)
        for batch_start in range(0, total_queries, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_queries)

            batch_number = batch_start // args.batch_size + 1
            total_batches = (total_queries - 1) // args.batch_size + 1
            print(f"\nProcessing batch {batch_number}/{total_batches}")
            batch_results = process_queries_parallel(
                df, batch_start, batch_end, args.max_workers
            )
            all_results.extend(batch_results)

            # Small delay between batches to be respectful to the API
            if batch_end < total_queries:
                print("Waiting 2 seconds before next batch...")
                time.sleep(2)

        # Save results
        print("\nStep 3: Saving results to CSV...")
        save_results_to_csv(all_results, output_path)

        # Summary
        elapsed_time = time.time() - start_time
        success_rate = len(all_results) / total_queries * 100
        print(f"\nCompleted successfully in {elapsed_time:.2f} seconds.")
        print(
            f"Processed {len(all_results)}/{total_queries} queries "
            f"({success_rate:.1f}% success rate)"
        )

    except Exception as exc:
        print(f"Error in main execution: {exc}")
        raise


if __name__ == "__main__":
    main()
