"""
Bulk testing utility for recipe chatbot agent.

Reads a CSV file containing user queries and stores the results for manual evaluation.
"""

import sys
from pathlib import Path

# Adding PROJECT_ROOT = Path(__file__).resolve().parent.parent followed by sys.path.insert(0, str(PROJECT_ROOT)) tells Python to treat the project root as an importable location so scripts in `scripts/`` can load modules from sibling directories like `backend``; to reach higher-level directories, chain additional .parent calls (e.g., .parent.parent.parent).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Tools for configuring command-line arguments.
import argparse
import csv
import datetime as dt
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from backend.utils import get_agent_response, SYSTEM_PROMPT

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

DEFAULT_CSV: Path = Path("data/sample_queries.csv")
RESULTS_DIR: Path = Path("results")
# Create the results directory if it doesn't exist.
RESULTS_DIR.mkdir(exist_ok=True)

MAX_WORKERS = 32  # For ThreadPoolExecutor

# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------


# --- Sync function for ThreadPoolExecutor ---
# Handles one query. this function is ordinary (blocking) Python code, so ThreadPoolExecutor can run many copies of it in parallel threads—each thread runs one query independently.
def process_query_sync(query_id: str, query: str) -> Tuple[str, str, str]:
    """Processes a single query by calling the agent directly."""
    initial_messages: List[Dict[str, str]] = [{"role": "user", "content": query}]
    try:
        # get_agent_response now returns the full history
        updated_history = get_agent_response(initial_messages)
        # Extract the last assistant message for the result
        assistant_reply = ""
        if updated_history and updated_history[-1]["role"] == "assistant":
            assistant_reply = updated_history[-1]["content"]
        else:  # Should not happen with current logic but good to handle
            assistant_reply = "Error: No assistant reply found in history."
        return query_id, query, assistant_reply
    except Exception as e:
        return query_id, query, f"Error processing query: {str(e)}"


# Renamed and made sync
def run_bulk_test(csv_path: Path, num_workers: int = MAX_WORKERS) -> None:
    """Main entry point for bulk testing (synchronous version)."""

    # Open the CSV file and read the data into a list of dictionaries. Only keep rows with 'id' and 'query' columns.
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        # Expects columns 'id' and 'query'
        input_data: List[Dict[str, str]] = [
            row for row in reader if row.get("id") and row.get("query")
        ]

    if not input_data:
        raise ValueError(
            "No valid data (with 'id' and 'query') found in the provided CSV file."
        )

    console = Console()
    results_data: List[Tuple[str, str, str]] = (
        []
    )  # Will store the results (id, query, response)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map each scheduled future back to its originating CSV row. Executor.submit(func, arg1, arg2, ...) takes the callable first, then any positional (or keyword) arguments to pass when the worker executes it. Here, item["id"] and item["query"] come from the current CSV row (item), so each submitted task runs process_query_sync with that row’s id and query.
        future_to_data = {
            executor.submit(process_query_sync, item["id"], item["query"]): item
            for item in input_data
        }
        console.print(
            f"[bold blue]Submitting {len(input_data)} queries to the executor...[/bold blue]"
        )
        # Iterate futures as they finish, pulling their inputs and outputs in completion order.
        for i, future in enumerate(as_completed(future_to_data)):
            # Recover the input row for this completed future so we can report against it.
            item_data = future_to_data[future]
            item_id = item_data["id"]
            item_query = item_data["query"]
            try:
                # Block until this future finishes and unpack the agent response tuple.
                processed_id, original_query, response_text = future.result()
                results_data.append((processed_id, original_query, response_text))

                # Assemble styled text for the result header and original query using Rich.
                panel_content = Text()
                panel_content.append(f"ID: {processed_id}\n", style="bold magenta")
                panel_content.append("Query:\n", style="bold yellow")
                panel_content.append(f"{original_query}\n\n")

                # Wrap the response so Rich renders its Markdown formatting in the terminal.
                response_markdown = Markdown(response_text)

                # Group the different parts for the Panel
                panel_group = Group(
                    panel_content,  # Contains ID and Query
                    Markdown("--- Response ---"),  # A small separator for clarity
                    response_markdown,  # The Markdown rendered response
                )

                # Visualize the completed worker's output as a Rich panel in real time. Panel wraps formatted ID, query, separator and response into one Rich renderable. So console.print(Panel) will print the entire panel.
                console.print(
                    # Panel is a Rich component that wraps the panel_group and adds a title and border style.
                    Panel(
                        panel_group,  # Pass the group as the single renderable
                        title=f"Result {i+1}/{len(input_data)} - ID: {processed_id}",
                        border_style="cyan",
                    )
                )

            except Exception as exc:
                console.print(
                    Panel(
                        f"[bold red]Exception for ID {item_id}, Query:[/bold red]\n{item_query}\n\n[bold red]Error:[/bold red]\n{exc}",
                        title=f"Error in Result {i+1}/{len(input_data)} - ID: {item_id}",
                        border_style="red",
                    )
                )
                results_data.append(
                    (item_id, item_query, f"Exception during processing: {str(exc)}")
                )
        console.print("[bold blue]All queries processed.[/bold blue]")

    # After all futures finish and with With ThreadPoolExecutor(...) block exits, it stamps the time, builds the output path and opens the file for writing.
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{timestamp}.csv"

    # Opens the file for writing, writes the header row and then writes the results data as csv rows.
    with out_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "query", "response"])
        writer.writerows(results_data)

    console.print(
        f"[bold green]Saved {len(results_data)} results to {str(out_path)}[/bold green]"
    )


if __name__ == "__main__":
    # CLI usage notes:
    # - uv run python scripts/bulk_test.py
    # - uv run python scripts/bulk_test.py --csv data/custom_queries.csv
    # - uv run python scripts/bulk_test.py --csv data/vegan.csv --workers 8
    # These optional flags let you point at a different CSV and tune the worker count.
    parser = argparse.ArgumentParser(description="Bulk test the recipe chatbot")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to CSV file containing queries (column name: 'query').",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of worker threads (default: {MAX_WORKERS}).",
    )
    args = parser.parse_args()
    run_bulk_test(args.csv, args.workers)
