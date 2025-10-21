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

**Both Pattern 2 variants use the same ThreadPoolExecutor mechanics.** The only difference is **what value you store in the dictionary**: the input object itself vs. its index position.

#### Pattern 2a: Store the Input Object Directly

Store the actual input object in the dictionary. This gives you direct access to the input when a Future completes.

```python
from concurrent.futures import as_completed

urls = ["https://example.com", "https://httpbin.org/get", "https://github.com"]

with ThreadPoolExecutor(max_workers=5) as executor:
    # Map each submitted Future back to the URL that produced it.
    future_to_url = {}
    for url in urls:
        future = executor.submit(fetch_page, url)
        future_to_url[future] = url  # Store the URL string itself

    for future in as_completed(future_to_url):
        url = future_to_url[future]  # Get the URL string directly
        try:
            page = future.result()
            print(f"Fetched {url} ({len(page)} bytes)")
        except Exception as exc:
            print(f"{url} failed: {exc}")
```

**What's stored in the dictionary:**
```python
{
    <Future_A>: "https://example.com",      # The actual URL string
    <Future_B>: "https://httpbin.org/get",  # The actual URL string
    <Future_C>: "https://github.com",       # The actual URL string
}
```

`as_completed` yields futures in the order they finish, so you can stream progress updates, update a progress bar, or record per-input metadata immediately. The dictionary lookup `future_to_url[future]` translates "which Future just completed" back into "which input produced this result."

**When to use Pattern 2a:**
- You're building inputs dynamically (no pre-existing list)
- The input objects are small and simple (strings, numbers)
- You want direct, one-step access to the input
- You prefer explicit, clear code over dict comprehensions

#### Pattern 2b: Store the Index Position Instead

Store the **index** (position number) instead of the input object itself. This requires an extra lookup step but is more memory-efficient and works well with dict comprehensions.

```python
# Your inputs are already in a list
urls = ["https://example.com", "https://httpbin.org/get", "https://github.com"]

with ThreadPoolExecutor(max_workers=5) as executor:
    # Map each Future to its index in the original list
    future_to_index = {
        executor.submit(fetch_page, url): i  # Store index (0, 1, 2)
        for i, url in enumerate(urls)
    }

    for future in as_completed(future_to_index):
        index = future_to_index[future]  # Get the index number (0, 1, 2)
        url = urls[index]  # Look up the URL using the index
        try:
            page = future.result()
            print(f"Fetched {url} ({len(page)} bytes)")
        except Exception as exc:
            print(f"{url} failed: {exc}")
```

**What's stored in the dictionary:**
```python
{
    <Future_A>: 0,  # Just a number pointing to position in the list
    <Future_B>: 1,  # Just a number pointing to position in the list
    <Future_C>: 2,  # Just a number pointing to position in the list
}
```

**When to use Pattern 2b:**
- You already have a list of inputs (no need to duplicate them)
- The input objects are large or complex (more memory-efficient to store integers)
- You need to access neighboring items in the list (`urls[index + 1]`)
- You want to use dict comprehension for cleaner code

#### Side-by-Side Comparison

The **exact same task** implemented both ways:

```python
# Pattern 2a: Store the object directly
future_to_url = {}
for url in urls:
    future = executor.submit(fetch_page, url)
    future_to_url[future] = url  # Store "https://example.com"

# Later...
url = future_to_url[future]  # Get "https://example.com" directly
```

```python
# Pattern 2b: Store the index
future_to_index = {
    executor.submit(fetch_page, url): i  # Store 0, 1, 2
    for i, url in enumerate(urls)
}

# Later...
index = future_to_index[future]  # Get 0, 1, or 2
url = urls[index]  # Extra step: look up URL from list
```

**Key difference:** Pattern 2a retrieves the input in one step; Pattern 2b retrieves an index first, then uses it to get the input.

**Performance:** For most use cases, the difference is negligible. Choose based on readability and whether you already have a list.

## Choosing Between the Patterns

| Requirement                                               | Pattern 1 (List) | Pattern 2a (Map to Object) | Pattern 2b (Map to Index) |
| --------------------------------------------------------- | ---------------- | -------------------------- | ------------------------- |
| Just gather results at the end                            | ✅               |                            |                           |
| Need to know which input produced each result             |                  | ✅                         | ✅                        |
| Want to react immediately as tasks finish                 |                  | ✅                         | ✅                        |
| Prefer submission order even if slower task finishes last | ✅               |                            |                           |
| Building inputs dynamically (no pre-existing list)        |                  | ✅                         |                           |
| Already have inputs in a list                             | ✅               |                            | ✅                        |
| Input objects are large/complex                           |                  |                            | ✅                        |
| Want dict comprehension syntax                            |                  |                            | ✅                        |
| Need to access neighboring list items                     |                  |                            | ✅                        |

### Examples from HW2 Scripts

**`generate_dimension_tuples` uses Pattern 1:**
- Submits 5 identical LLM calls
- Only needs to collect the payloads
- Order doesn't matter
- Stores futures in a list and collects results afterwards

```python
futures = [executor.submit(call_llm, messages, DimensionTuplesList) for _ in range(5)]
responses = [future.result() for future in futures]
```

**`generate_queries_parallel` uses Pattern 2b:**
- Must associate each result with its originating dimension tuple
- Updates a progress bar as soon as each task finishes
- Maps futures back to their tuple index
- Uses `as_completed` to process results immediately

```python
future_to_tuple = {
    executor.submit(generate_queries_for_tuple, dim_tuple): i
    for i, dim_tuple in enumerate(dimension_tuples)
}

for future in as_completed(future_to_tuple):
    tuple_idx = future_to_tuple[future]  # Get index
    dimension_tuple = dimension_tuples[tuple_idx]  # Use index to retrieve tuple
    queries = future.result()
    # Attach queries back to their source dimension_tuple
```

**Why Pattern 2b was chosen here:**
1. ✅ `dimension_tuples` is already a list
2. ✅ Each `DimensionTuple` is a Pydantic model with 6 fields (relatively large)
3. ✅ Dict comprehension provides clean, compact syntax
4. ✅ More memory-efficient to store integers than duplicate object references

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
