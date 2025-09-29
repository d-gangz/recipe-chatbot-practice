## ThreadPoolExecutor Basics

`ThreadPoolExecutor` spawns a fixed worker pool so you can run blocking functions in parallel without writing manual thread management code. You submit callables, the executor queues them, schedules them on the worker threads, and hands you `Future` objects that represent the eventual results.

### Core Workflow

1. Create the executor (usually with a `with` block so threads are cleaned up automatically).
2. Submit work with `executor.submit(func, *args, **kwargs)` or `executor.map`.
3. Work with the returned `Future`s to retrieve results, handle errors, or track progress.

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_page(url: str) -> str:
    response = requests.get(url)
    return response.text

urls = ["https://example.com", "https://httpbin.org/get"]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(fetch_page, url) for url in urls]
    results = [future.result() for future in futures]
```

The last line is a list comprehension. It asks every finished `Future` for its value and puts the answers into a new list. Written without the shortcut it looks like this:

```python
results = []
for future in futures:
    results.append(future.result())
```

Both snippets do the same work; the comprehension is just a compact way to express the loop.

## Two Common Patterns for Tracking Futures

When you submit multiple tasks you need some way to keep track of the `Future`s. Two patterns show up often: storing the futures in a list versus mapping them back to their originating inputs.

### Pattern 1: List of Futures (Uniform Tasks)

Use a simple list when every task is interchangeable and you only need to gather the results.

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = []
    for url in urls:
        futures.append(executor.submit(fetch_page, url))

    pages = []
    for future in futures:
        pages.append(future.result())
```

`future.result()` is called in submission order. If the tasks truly do not depend on which input produced which result, a list keeps the code short and direct.

### Pattern 2: Future-to-Input Map (Tracking Origins or Progress)

When the caller needs to know which input produced each result—or when you want to react as soon as each task finishes—create a dictionary that maps the future back to the original input.

```python
from concurrent.futures import as_completed

with ThreadPoolExecutor(max_workers=5) as executor:
    # Map each submitted Future back to the URL that produced it.
    future_to_url = {}
    for url in urls:
        future = executor.submit(fetch_page, url)
        future_to_url[future] = url  # Remember which URL created this Future.

    for future in as_completed(future_to_url):
        url = future_to_url[future]  # Look up the original URL for this finished Future.
        try:
            page = future.result()
            print(f"Fetched {url} ({len(page)} bytes)")
        except Exception as exc:
            print(f"{url} failed: {exc}")
```

`as_completed` yields futures in the order they finish, so you can stream progress updates, update a progress bar, or record per-input metadata immediately.

In this pattern the dictionary stores the mapping explicitly: the loop submits each URL, captures the resulting `Future`, and saves the (future → url) pair. When a particular worker finishes, `future_to_url[future]` recovers the exact URL that spawned it. That lookup is the point where we translate "which Future just completed" back into "which input produced this result."

## Choosing Between the Patterns

| Requirement                                               | Use a List | Use a Map + `as_completed` |
| --------------------------------------------------------- | ---------- | -------------------------- |
| Just gather results at the end                            | ✅         |                            |
| Need to know which input produced each result             |            | ✅                         |
| Want to react immediately as tasks finish                 |            | ✅                         |
| Prefer submission order even if slower task finishes last | ✅         |                            |

In the HW2 scripts you saw both approaches: `generate_dimension_tuples` submits identical LLM calls and only needs the payloads, so it stores futures in a list and collects results afterwards. `generate_queries_parallel` must associate each result with its originating dimension tuple and update a progress bar as soon as work finishes, so it maps futures back to their tuple index and iterates with `as_completed`.

## Additional Tips

- Always exit the executor context (`with ThreadPoolExecutor(...)`) so threads shut down cleanly.
- Catch exceptions around `future.result()`; if the worker raised, the exception reappears when you request the result.
- For CPU-bound work prefer `ProcessPoolExecutor`; threads best suit IO-bound or network-bound tasks.

### Extra Example for Pattern 2: Tracking Rich Input Objects

You can map a `Future` back to any kind of metadata—not just a string. Here each task carries a small object storing both a name and a URL.

```python
from dataclasses import dataclass

@dataclass
class RequestInfo:
    name: str
    url: str

requests_to_fetch = [
    RequestInfo(name="Docs", url="https://example.com/docs"),
    RequestInfo(name="Status", url="https://httpbin.org/status/200"),
]

with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_request = {}
    for request in requests_to_fetch:
        future = executor.submit(fetch_page, request.url)  # Launch work for this request.
        future_to_request[future] = request  # Remember the entire RequestInfo object.

    for future in as_completed(future_to_request):
        request = future_to_request[future]  # Recover the same RequestInfo we stored earlier.
        try:
            page = future.result()
            print(f"{request.name} ({request.url}) returned {len(page)} bytes")
        except Exception as exc:
            print(f"{request.name} failed: {exc}")
```

The logic mirrors the earlier example—we simply keep richer metadata alongside each `Future` so the result handler still knows exactly which input produced the output.
