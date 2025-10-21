#!/usr/bin/env python3
"""Generate Recipe Bot traces for dietary adherence evaluation.

This script sends dietary preference queries to the Recipe Bot and collects
the responses to create a dataset for LLM-as-Judge evaluation.

KEY CONCEPTS:
=============

1. Query vs Trace:
   - QUERY: A unique question (e.g., "Can you suggest a vegan pasta recipe?")
   - TRACE: A single execution/response for that query
   - We generate 40 TRACES per QUERY because LLMs are non-deterministic

2. Why Query ID AND Trace ID?
   - query_id: Identifies the question (e.g., 1, 2, 3...)
   - trace_id: Identifies specific execution (e.g., "1_5" = 5th execution of query 1)
   - Without trace IDs, we couldn't distinguish between different responses to the same query

3. Example Data Structure:
   Query 1: "I'm vegan but I really want to make something with honey - is there a good substitute? i am craving a yogurt breakfast"
   ├─ Trace 1_1: "Vegan Coconut & Berry Yogurt with agave syrup" ✓
   ├─ Trace 1_2: "Yogurt Parfait with maple syrup" ✓
   ├─ Trace 1_3: "Greek Yogurt Bowl" ❌ (contains dairy!)
   └─ ... (37 more traces)

   This lets us measure: "For vegan queries, bot succeeded 38/40 times (95%)"

4. Parallel Processing with ThreadPoolExecutor:
   - We have 60 queries × 40 traces each = 2,400 total tasks
   - Using 32 worker threads to run tasks in parallel
   - Uses Pattern 2b: Map Future -> task tuple to track which task produced which result
   - Processes results with as_completed() to show progress in real-time
"""

import sys
import os
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Group

# Add the backend to the path so we can import the Recipe Bot
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils import get_agent_response

MAX_WORKERS = 32

console = Console()


def load_dietary_queries(csv_path: str) -> List[Dict[str, Any]]:
    """Load dietary preference queries from CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def generate_trace(query: str, dietary_restriction: str) -> Dict[str, Any]:
    """Generate a single Recipe Bot trace for a dietary query."""
    try:
        # Create the conversation with just the user query
        messages = [{"role": "user", "content": query}]

        # Get the bot's response
        updated_messages = get_agent_response(messages)

        # Extract the assistant's response
        assistant_response = updated_messages[-1]["content"]

        return {
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": assistant_response,
            "success": True,
            "error": None,
        }
    except Exception as e:
        console.print(f"[red]Error generating trace for query: {query}")
        console.print(f"[red]Error: {str(e)}")
        return {
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": None,
            "success": False,
            "error": str(e),
        }


def generate_trace_with_id(args: tuple) -> Dict[str, Any]:
    """Wrapper function for parallel processing.

    This function receives a tuple and unpacks it to create unique trace IDs.

    Args:
        args: A 2-element tuple containing:
              - Element 0: query_data (dict with 'id', 'query', 'dietary_restriction')
              - Element 1: trace_num (int, the trace number for this query)

    Example tuple:
        (
            {"id": 1, "query": "I'm vegan but I really want to make something with honey...", "dietary_restriction": "vegan"},
            5  # This is the 5th trace for query 1
        )

    Returns:
        Dict containing the trace with a unique trace_id like "1_5"
    """
    # Tuple unpacking: Split the tuple into its two parts
    # This is equivalent to: query_data = args[0], trace_num = args[1]
    query_data, trace_num = args

    # Extract individual fields from the query_data dictionary
    query = query_data[
        "query"
    ]  # e.g., "I'm vegan but I really want to make something with honey..."
    dietary_restriction = query_data["dietary_restriction"]  # e.g., "vegan"

    # Generate the actual trace by calling the Recipe Bot
    trace = generate_trace(query, dietary_restriction)

    # Create unique trace ID by combining query ID with trace number
    # Example: 1 + "_" + 5 = "1_5"
    # This allows us to distinguish between different executions of the same query
    trace["trace_id"] = f"{query_data['id']}_{trace_num}"
    trace["query_id"] = query_data["id"]
    return trace


def generate_multiple_traces_per_query(
    queries: List[Dict[str, Any]],
    traces_per_query: int = 40,
    max_workers: int = MAX_WORKERS,
) -> List[Dict[str, Any]]:
    """Generate multiple traces for each query using parallel processing.

    WHY MULTIPLE TRACES PER QUERY?
    LLMs are non-deterministic! The same query can produce different responses each time.
    By generating 40 traces per query, we can:
    1. Measure consistency (how often does the bot follow dietary restrictions?)
    2. Find edge cases (catch the times when the bot fails)
    3. Get statistical significance for evaluating the LLM judge

    Example: For query 1 ("I'm vegan but I really want to make something with honey..."):
        - Trace 1_1: "Vegan Coconut & Berry Yogurt with agave syrup" ✓
        - Trace 1_2: "Yogurt Parfait with maple syrup" ✓
        - Trace 1_3: "Greek Yogurt Bowl" ❌ (contains dairy - not vegan!)
        - ... 37 more traces ...

    Args:
        queries: List of query dictionaries with 'id', 'query', 'dietary_restriction'
        traces_per_query: Number of times to execute each query (default: 40)
        max_workers: Number of parallel threads (default: 32)

    Returns:
        List of all generated trace dictionaries
    """

    # ============================================================
    # STEP 1: BUILD THE TASKS LIST
    # ============================================================
    # We use a two-step process instead of submitting directly because:
    # 1. Nested loops are cleaner when separated from submission
    # 2. We need len(tasks) upfront for progress tracking
    # 3. Makes the dict comprehension simpler later

    tasks = []  # Will hold 2,400 tuples (60 queries × 40 traces each)

    # Outer loop: iterate through each query
    for query_data in queries:  # 60 queries from dietary_queries.csv
        # Inner loop: create 40 traces for THIS specific query
        for i in range(traces_per_query):  # i goes from 0 to 39
            # Create a tuple pairing the query data with its trace number
            # Example tuple: ({"id": 1, "query": "I'm vegan but...", ...}, 1)
            tasks.append((query_data, i + 1))  # i+1 so trace numbers start at 1

    # Visual representation of what tasks looks like:
    # [
    #   (query_1_data, 1),  (query_1_data, 2),  ..., (query_1_data, 40),  # 40 traces for query 1
    #   (query_2_data, 1),  (query_2_data, 2),  ..., (query_2_data, 40),  # 40 traces for query 2
    #   ...
    #   (query_60_data, 1), (query_60_data, 2), ..., (query_60_data, 40), # 40 traces for query 60
    # ]
    # Total: 2,400 tuples

    all_traces = []  # Will collect results as they complete

    # ============================================================
    # STEP 2: SUBMIT ALL TASKS TO THREADPOOL
    # ============================================================
    # This uses Pattern 2b from the ThreadPool documentation:
    # - Map each Future back to its originating task
    # - Use as_completed() to process results as they finish (not in submission order)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all 2,400 tasks at once using a dict comprehension
        # This is equivalent to:
        #   future_to_task = {}
        #   for task in tasks:
        #       future = executor.submit(generate_trace_with_id, task)
        #       future_to_task[future] = task
        #
        # The dictionary maps: Future -> original task tuple
        # Example: {<Future_A>: (query_1_data, 1), <Future_B>: (query_1_data, 2), ...}
        future_to_task = {
            executor.submit(generate_trace_with_id, task): task for task in tasks
        }

        # ============================================================
        # STEP 3: PROCESS RESULTS AS THEY COMPLETE
        # ============================================================
        # as_completed() yields futures in the order they FINISH (not submission order)
        # This allows us to:
        # - Update progress bar in real-time
        # - Show sample traces as they're generated
        # - Collect results immediately without waiting for all tasks

        # Process completed tasks with progress tracking
        with console.status("[yellow]Generating traces in parallel...") as status:
            completed = 0
            total = len(tasks)  # 2,400 total tasks

            # Loop through futures as they complete (finish order, not submission order)
            for future in as_completed(future_to_task):
                # Get the result from this completed future
                trace = future.result()  # Blocks until this specific future is done
                all_traces.append(trace)
                completed += 1

                # Every 100 traces, display a sample for verification
                if completed % 100 == 0:
                    # Display the trace just generated for verification
                    panel_content = Text()
                    panel_content.append(
                        f"Trace ID: {trace['trace_id']}\n", style="bold magenta"
                    )
                    panel_content.append(
                        f"Query ID: {trace['query_id']}\n", style="bold cyan"
                    )
                    panel_content.append(
                        f"Dietary Restriction: {trace['dietary_restriction']}\n",
                        style="bold yellow",
                    )
                    panel_content.append(
                        f"Success: {trace['success']}\n",
                        style="bold green" if trace["success"] else "bold red",
                    )
                    panel_content.append("Query:\n", style="bold blue")
                    panel_content.append(f"{trace['query']}\n\n")

                    if trace["success"] and trace["response"]:
                        response_markdown = Markdown(trace["response"])
                        panel_group = Group(
                            panel_content,
                            Markdown("--- Response ---"),
                            response_markdown,
                        )
                    else:
                        error_text = Text(
                            f"Error: {trace.get('error', 'Unknown error')}",
                            style="bold red",
                        )
                        panel_group = Group(panel_content, error_text)

                    console.print(
                        Panel(
                            panel_group,
                            title="Sample Generated Trace",
                            border_style="cyan",
                        )
                    )

                # Update the progress status message
                status.update(
                    f"[yellow]Generated {completed}/{total} traces ({completed/total*100:.1f}%)"
                )

    # All tasks are now complete!
    console.print(f"[green]Completed parallel generation of {len(all_traces)} traces")
    return all_traces  # Returns ~2,400 traces (60 queries × 40 traces each)


def save_traces(traces: List[Dict[str, Any]], output_path: str) -> None:
    """Save traces to CSV file."""
    df = pd.DataFrame(traces)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Saved {len(traces)} traces to {output_path}")


def main():
    """Main function to generate Recipe Bot traces."""
    console.print("[bold blue]Recipe Bot Trace Generation")
    console.print("=" * 50)

    # Set up paths
    script_dir = Path(__file__).parent
    hw3_dir = script_dir.parent
    data_dir = hw3_dir / "data"

    # Load dietary queries
    queries_path = data_dir / "dietary_queries.csv"
    if not queries_path.exists():
        console.print(f"[red]Error: {queries_path} not found!")
        return

    queries = load_dietary_queries(str(queries_path))
    console.print(f"[green]Loaded {len(queries)} dietary queries")

    # Generate traces (40 traces per query)
    console.print("[yellow]Generating traces... This may take a while.")
    traces = generate_multiple_traces_per_query(queries, traces_per_query=40)

    # Filter successful traces
    successful_traces = [t for t in traces if t["success"]]
    failed_traces = [t for t in traces if not t["success"]]

    console.print(f"[green]Successfully generated {len(successful_traces)} traces")
    if failed_traces:
        console.print(f"[yellow]Failed to generate {len(failed_traces)} traces")

    # Save traces
    output_path = data_dir / "raw_traces.csv"
    save_traces(successful_traces, str(output_path))

    # Print summary statistics
    console.print("\n[bold]Summary Statistics:")
    console.print(f"Total traces generated: {len(successful_traces)}")

    # Count by dietary restriction
    restriction_counts = {}
    for trace in successful_traces:
        restriction = trace["dietary_restriction"]
        restriction_counts[restriction] = restriction_counts.get(restriction, 0) + 1

    console.print("\nTraces per dietary restriction:")
    for restriction, count in sorted(restriction_counts.items()):
        console.print(f"  {restriction}: {count}")


if __name__ == "__main__":
    main()
