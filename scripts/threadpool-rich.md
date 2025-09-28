## ThreadPoolExecutor + Rich Console Cheatsheet

### Why ThreadPoolExecutor?
- Runs blocking functions in parallel threads without rewriting them as async code.
- In `bulk_test.py`, each CSV row is processed concurrently so slow agent calls do not block one another.
- Constructor accepts `max_workers`; pick a value that matches the level of parallelism your API and machine can handle.

### Submitting Work
```python
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_data = {
        executor.submit(process_query_sync, item["id"], item["query"]): item
        for item in input_data
    }
```
- `executor.submit(func, *args, **kwargs)` schedules `func` with the provided arguments and returns a `Future` handle immediately.
- Dict comprehension maps each `Future` back to the CSV row (`item`) so we can recover `id` and `query` later.
- Keys (futures) are yielded by `as_completed`, letting us process results in the order they finish, not the order submitted.

### Harvesting Results
```python
for i, future in enumerate(as_completed(future_to_data)):
    item_data = future_to_data[future]
    item_id = item_data["id"]
    item_query = item_data["query"]
    try:
        processed_id, original_query, response_text = future.result()
        results_data.append((processed_id, original_query, response_text))
```
- `future.result()` blocks until that worker is done, raising any exception it encountered.
- Storing `item_id`/`item_query` before the `try` lets the `except` block log the failing row even when the future raises.
- Append successes to a shared list so the caller can save or post-process all responses once the pool closes.

### Handling Errors
```python
    except Exception as exc:
        console.print(
            Panel.fit(
                f"Exception for ID {item_id} | Query: {item_query}\n{exc}",
                title=f"Error {i+1}/{len(input_data)}"
            ),
            style="bold red"
        )
        results_data.append((item_id, item_query, f"Exception during processing: {exc}"))
```
- Pull the original input from `future_to_data[future]` so logs reference the same row the worker saw.
- Capture exceptions to keep the batch running; the pool continues servicing other futures.

### Using Rich for Live Feedback
- `Console()` prints styled output without manual ANSI codes.
- `Panel`, `Text`, `Markdown`, and `Group` wrap result metadata, a separator, and the model response for readable terminal blocks.
- Color-coding (`bold magenta`, `bold yellow`, etc.) highlights IDs, queries, and statuses while the batch runs.
- Rich handles Markdown rendering for responses, so the assistant output stays readable even for multi-line formatted text.

### Practical Tips
- Validate the CSV upfront so the pool only runs with `id` + `query` rows.
- Limit worker count if the upstream API enforces rate limits.
- Keep the mappings lightweight—store only what you need (ID and query) to keep memory usage predictable.
- Collect results in `results_data` for later persistence (JSON/CSV) once the executor exits.
