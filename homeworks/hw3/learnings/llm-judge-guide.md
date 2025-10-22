<!--
Document Type: Guide
Purpose: Comprehensive implementation guide for building an LLM-as-Judge evaluation system from scratch
Context: Created for HW3 to help junior developers understand the complete process of implementing LLM-as-Judge
Key Topics: LLM evaluation, judge development, data labeling, train/dev/test splits, TPR/TNR, judgy package, bias correction
Target Use: Step-by-step reference for implementing production-grade LLM evaluation systems
-->

# LLM-as-Judge Implementation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Complete Process Overview](#complete-process-overview)
3. [Step 1: Generate Traces](#step-1-generate-traces)
4. [Step 2: Create Ground Truth Labels](#step-2-create-ground-truth-labels)
5. [Step 3: Split Your Data](#step-3-split-your-data)
6. [Step 4: Develop the Judge Prompt](#step-4-develop-the-judge-prompt)
7. [Step 5: Evaluate on Test Set](#step-5-evaluate-on-test-set)
8. [Step 6: Run Full Evaluation with Judgy](#step-6-run-full-evaluation-with-judgy)
9. [Data Shapes Reference](#data-shapes-reference)
10. [Key Concepts](#key-concepts)
11. [Best Practices](#best-practices)
12. [Common Pitfalls](#common-pitfalls)

---

## Introduction

### What is LLM-as-Judge?

**LLM-as-Judge** is an evaluation approach where you use a Language Model to automatically evaluate the quality of outputs from another LLM (or any system). Instead of manual human evaluation, you train a judge LLM to assess whether outputs meet specific criteria.

**Example Use Case**: Recipe Bot
- Your app generates recipes based on dietary restrictions (vegan, gluten-free, etc.)
- You need to verify that recipes actually follow those restrictions
- Manual checking of thousands of recipes is impractical
- Solution: Build an LLM judge to automatically evaluate dietary adherence

### Why Build an LLM Judge?

**Advantages:**
- **Scale**: Evaluate thousands of outputs automatically
- **Consistency**: Same evaluation criteria applied every time
- **Cost**: Cheaper than hiring human evaluators
- **Speed**: Real-time evaluation in production

**Limitations:**
- **Imperfect**: LLM judges make mistakes (false positives/negatives)
- **Bias**: Judges can be systematically too strict or too lenient
- **Requires Validation**: Must measure judge accuracy on labeled data

### The Solution: Statistical Bias Correction

Since LLM judges aren't perfect, we use the **judgy** package to:
1. Measure judge error patterns on labeled test data (TPR/TNR)
2. Mathematically correct biased predictions on unlabeled production data
3. Provide confidence intervals for the corrected estimates

---

## Complete Process Overview

Here's the complete 6-step pipeline for implementing LLM-as-Judge:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM-as-Judge Pipeline                            │
└─────────────────────────────────────────────────────────────────────┘

STEP 1: Generate Traces
┌────────────────────┐
│ Dietary Queries    │ ──→ Run through Recipe Bot (40 times each)
│ (60 queries)       │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Raw Traces         │ ← 2,400 recipe responses (60 × 40)
│ raw_traces.csv     │
└────────────────────┘

STEP 2: Create Ground Truth Labels
         │
         ▼
┌────────────────────┐
│ Label Traces       │ ← Use GPT-4o to label PASS/FAIL
│ (sample 200)       │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Labeled Traces     │ ← 150 labeled (75 PASS, 75 FAIL)
│ labeled_traces.csv │
└────────────────────┘

STEP 3: Split Data
         │
         ├──→ Train (15%) ──→ train_set.csv (~23 examples)
         ├──→ Dev (40%)   ──→ dev_set.csv (~60 examples)
         └──→ Test (45%)  ──→ test_set.csv (~67 examples)

STEP 4: Develop Judge (Iterate on Dev Set)
┌────────────────────┐
│ Train Set          │ ──→ Select few-shot examples (1 PASS, 3 FAIL)
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Judge Prompt       │ ← Base prompt + few-shot examples
│ judge_prompt.txt   │
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ Evaluate on Dev    │ ← Test judge on 50 dev samples
│ (Iterate & Refine) │
└────────────────────┘
         │
         ▼ (When satisfied)

STEP 5: Evaluate on Test Set (Run ONCE!)
┌────────────────────┐
│ Test Set           │ ── Evaluate ALL test traces
└────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Judge Performance              │
│ • TPR (True Positive Rate)     │ ← For judgy correction
│ • TNR (True Negative Rate)     │
│ judgy_test_data.json          │
└────────────────────────────────┘

STEP 6: Run Full Evaluation with Judgy
┌────────────────────┐
│ All Traces         │ ── Judge evaluates all 2,400 traces
│ (2,400 unlabeled)  │
└────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ Raw Predictions                │
│ (biased by judge errors)       │
└────────────────────────────────┘
         │
         ▼
┌────────────────────────────────┐
│ judgy Bias Correction          │ ← Uses TPR/TNR to correct
│ • Corrected success rate       │
│ • Confidence interval          │
└────────────────────────────────┘
```

---

## Step 1: Generate Traces

### Purpose
Generate multiple recipe responses for each query to capture LLM non-determinism.

### Why Multiple Traces Per Query?
LLMs are non-deterministic! The same query can produce different responses. By generating 40 traces per query, we:
- Measure consistency (how often does the bot follow dietary restrictions?)
- Find edge cases (catch the times when the bot fails)
- Get statistical significance for evaluating the LLM judge

### Code Example

```python
from backend.utils import get_agent_response

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
        return {
            "query": query,
            "dietary_restriction": dietary_restriction,
            "response": None,
            "success": False,
            "error": str(e),
        }
```

### Parallel Processing for Speed

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_multiple_traces_per_query(
    queries: List[Dict[str, Any]],
    traces_per_query: int = 40,
    max_workers: int = 32,
) -> List[Dict[str, Any]]:
    """Generate multiple traces for each query using parallel processing."""

    # Build task list: 60 queries × 40 traces each = 2,400 tasks
    tasks = []
    for query_data in queries:
        for i in range(traces_per_query):
            tasks.append((query_data, i + 1))

    all_traces = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_trace_with_id, task): task
            for task in tasks
        }

        # Process as they complete
        for future in as_completed(future_to_task):
            trace = future.result()
            all_traces.append(trace)

    return all_traces
```

### Input Data Format

**dietary_queries.csv**:
```csv
id,query,dietary_restriction
1,"I'm vegan but I really want to make something with honey - is there a good substitute?",vegan
2,"Need a quick gluten-free breakfast. I hate eggs though.",gluten-free
3,"Keto breakfast that I can meal prep for the week",keto
```

### Output Data Format

**raw_traces.csv** (~2,400 rows):
```csv
query,dietary_restriction,response,success,error,trace_id,query_id
"I'm vegan but I...",vegan,"Try this Vegan Coconut Yogurt...",True,,1_1,1
"I'm vegan but I...",vegan,"Here's a Maple Glazed Breakfast Bowl...",True,,1_2,1
"I'm vegan but I...",vegan,"Greek Yogurt Parfait...",True,,1_3,1
```

**Key Fields:**
- `trace_id`: Format `{query_id}_{trace_num}` (e.g., "1_5" = 5th execution of query 1)
- `query_id`: Identifies the original query
- `response`: The recipe generated by the bot

---

## Step 2: Create Ground Truth Labels

### Purpose
Create labeled examples with ground truth PASS/FAIL labels for training and evaluating the judge.

### Important Warning ⚠️
**Using an LLM to create ground truth is NOT ideal!** You're essentially using an LLM as a judge to create labels for training another LLM judge. This can introduce biases and circular dependencies.

**Best Practice**: Label data manually with domain experts. If using LLM labeling:
- Use a much more powerful model (e.g., GPT-4o) than your judge
- **Always** manually review a sample of LLM-created labels
- Look extensively at the data yourself

### Code Example

```python
import litellm

LABELING_PROMPT = """You are an expert nutritionist and dietary specialist.
Your task is to evaluate whether a recipe response properly adheres to the
specified dietary restriction.

DIETARY RESTRICTION DEFINITIONS:
- Vegan: No animal products (meat, dairy, eggs, honey, etc.)
- Vegetarian: No meat or fish, but dairy and eggs are allowed
- Gluten-free: No wheat, barley, rye, or other gluten-containing grains
- Dairy-free: No milk, cheese, butter, yogurt, or other dairy products
...

Query: {query}
Dietary Restriction: {dietary_restriction}
Recipe Response: {response}

Provide your analysis in the following JSON format:
{{
    "reasoning": "Detailed explanation citing specific ingredients",
    "label": "PASS" or "FAIL",
    "confidence": "HIGH", "MEDIUM", or "LOW"
}}"""

def get_labeling_response(
    query: str,
    dietary_restriction: str,
    response: str
) -> Optional[Dict[str, Any]]:
    """Get labeling response from GPT-4o."""
    try:
        prompt = LABELING_PROMPT.format(
            query=query,
            dietary_restriction=dietary_restriction,
            response=response
        )

        completion = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic for consistent labels
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse JSON (handle markdown code blocks)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text

        result = json.loads(json_text)
        return result

    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        return None
```

### Parallel Labeling

```python
def label_traces(
    traces: List[Dict[str, Any]],
    sample_size: int = 200,
    max_workers: int = 32
) -> List[Dict[str, Any]]:
    """Label traces in parallel using GPT-4o."""

    # Sample traces
    sampled_traces = random.sample(traces, sample_size)
    labeled_traces = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_trace = {
            executor.submit(label_single_trace, trace): trace
            for trace in sampled_traces
        }

        for future in as_completed(future_to_trace):
            labeled_trace = future.result()
            labeled_traces.append(labeled_trace)

    return labeled_traces
```

### Balancing the Dataset

```python
def balance_labels(
    labeled_traces: List[Dict[str, Any]],
    target_positive: int = 75,
    target_negative: int = 75,
) -> List[Dict[str, Any]]:
    """Balance to have equal PASS and FAIL examples."""

    # Separate by label
    pass_traces = [t for t in labeled_traces if t["label"] == "PASS"]
    fail_traces = [t for t in labeled_traces if t["label"] == "FAIL"]

    # Sample equal numbers
    selected_pass = random.sample(pass_traces, min(target_positive, len(pass_traces)))
    selected_fail = random.sample(fail_traces, min(target_negative, len(fail_traces)))

    # Combine and shuffle
    balanced_traces = selected_pass + selected_fail
    random.shuffle(balanced_traces)

    return balanced_traces
```

### Output Data Format

**labeled_traces.csv** (~150 rows):
```csv
query,dietary_restriction,response,success,error,trace_id,query_id,label,reasoning,confidence,labeled
"I'm vegan...",vegan,"Vegan Coconut Yogurt...",True,,1_1,1,PASS,"Recipe uses coconut yogurt and agave syrup, both vegan-friendly",HIGH,True
"I'm vegan...",vegan,"Greek Yogurt Parfait...",True,,1_3,1,FAIL,"Contains Greek yogurt which is a dairy product, not vegan",HIGH,True
```

**New Fields:**
- `label`: "PASS" or "FAIL"
- `reasoning`: Explanation for the label
- `confidence`: "HIGH", "MEDIUM", or "LOW"
- `labeled`: True if labeling succeeded

---

## Step 3: Split Your Data

### Purpose
Split labeled data into train/dev/test sets to avoid overfitting and get unbiased evaluation metrics.

### The Three Sets

| Set | Size | Purpose | Usage Frequency |
|-----|------|---------|----------------|
| **Train** | 15% (~23) | Create few-shot examples for judge prompt | Once when building prompt |
| **Dev** | 40% (~60) | Iteratively develop and refine judge | Many times during development |
| **Test** | 45% (~67) | Final unbiased evaluation of judge | **ONCE** when judge is finalized |

### Why You Need Three Sets

**Train Set**:
- Used to create few-shot examples in your judge prompt
- You can "look at" and "use" this data in your prompt
- Small because you only need a few good examples

**Dev Set**:
- Used to measure judge performance during development
- You iterate by testing changes and measuring on dev set
- **CANNOT** be used in your prompt or for RAG
- Large enough to get stable performance estimates

**Test Set**:
- Your ultimate protection against overfitting
- Provides unbiased estimate of real-world performance
- **Only use once** when judge is finalized
- Every time you look at test errors, you lose some objectivity

### Stratified Splitting

**Stratification** ensures each split maintains the same PASS/FAIL ratio as the original data.

**Without stratification**: Random bad luck could give you:
- Train: 80% PASS, 20% FAIL
- Dev: 40% PASS, 60% FAIL  ← Imbalanced!
- Test: 60% PASS, 40% FAIL

**With stratification**: All splits maintain original ratio:
- Train: 50% PASS, 50% FAIL
- Dev: 50% PASS, 50% FAIL
- Test: 50% PASS, 50% FAIL

### Code Example

```python
from sklearn.model_selection import train_test_split

def stratified_split(
    traces: List[Dict[str, Any]],
    train_ratio: float = 0.15,
    dev_ratio: float = 0.40,
    test_ratio: float = 0.45,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split traces into train/dev/test with stratification by label."""

    df = pd.DataFrame(traces)

    # STEP 1: Split train from (dev + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(dev_ratio + test_ratio),  # 0.85
        stratify=df["label"],  # Maintain PASS/FAIL balance
        random_state=random_state,
    )

    # STEP 2: Split dev from test
    dev_test_ratio = dev_ratio / (dev_ratio + test_ratio)  # 0.47
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - dev_test_ratio),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    return (
        train_df.to_dict("records"),
        dev_df.to_dict("records"),
        test_df.to_dict("records")
    )
```

### Validation Checks

```python
def validate_splits(
    train_traces: List[Dict[str, Any]],
    dev_traces: List[Dict[str, Any]],
    test_traces: List[Dict[str, Any]],
) -> bool:
    """Validate that splits are reasonable."""

    # CHECK 1: Each split has both PASS and FAIL labels
    for split_name, traces in [
        ("Train", train_traces),
        ("Dev", dev_traces),
        ("Test", test_traces),
    ]:
        labels = set(trace["label"] for trace in traces)
        if len(labels) < 2:
            console.print(f"[red]Warning: {split_name} only has {labels}")
            return False

    # CHECK 2: Train set has reasonable dietary diversity
    train_restrictions = set(
        trace["dietary_restriction"] for trace in train_traces
    )
    if len(train_restrictions) < 3:
        console.print(f"[red]Train set only has {len(train_restrictions)} restrictions")
        return False

    return True
```

### Output Files

**train_set.csv** (~23 rows): For few-shot examples
**dev_set.csv** (~60 rows): For iterative development
**test_set.csv** (~67 rows): For final evaluation

---

## Step 4: Develop the Judge Prompt

### Purpose
Create an effective judge prompt that can accurately evaluate whether recipes follow dietary restrictions.

### Judge Prompt Components

A good judge prompt contains:

1. **Role Definition**: Who is the judge?
2. **Evaluation Criteria**: Clear rules for PASS vs FAIL
3. **Dietary Definitions**: Specific definitions for each restriction
4. **Few-Shot Examples**: Examples showing correct evaluation
5. **Output Format**: JSON schema for structured responses

### Example Judge Prompt Structure

```python
def create_judge_prompt(few_shot_examples: List[Dict[str, Any]]) -> str:
    """Create the LLM judge prompt with few-shot examples."""

    # 1. Role and Criteria
    base_prompt = """You are an expert nutritionist and dietary specialist
evaluating whether recipe responses properly adhere to specified dietary restrictions.

DIETARY RESTRICTION DEFINITIONS:
- Vegan: No animal products (meat, dairy, eggs, honey, etc.)
- Vegetarian: No meat or fish, but dairy and eggs are allowed
- Gluten-free: No wheat, barley, rye, or other gluten-containing grains
...

EVALUATION CRITERIA:
- PASS: Recipe clearly adheres to dietary restriction
- FAIL: Recipe contains ingredients or methods that violate restriction

Here are some examples of how to evaluate dietary adherence:

"""

    # 2. Add few-shot examples
    for i, example in enumerate(few_shot_examples, 1):
        base_prompt += f"\nExample {i}:\n"
        base_prompt += f"Query: {example['query']}\n"
        base_prompt += f"Recipe Response: {example['response']}\n"
        base_prompt += f"Reasoning: {example['reasoning']}\n"
        base_prompt += f"Label: {example['label']}\n"

    # 3. Add evaluation template with placeholders
    base_prompt += """

Now evaluate the following recipe response:

Query: __QUERY__
Dietary Restriction: __DIETARY_RESTRICTION__
Recipe Response: __RESPONSE__

Provide your evaluation in the following JSON format:
{
    "reasoning": "Detailed explanation citing specific ingredients",
    "label": "PASS" or "FAIL"
}"""

    return base_prompt
```

### Selecting Few-Shot Examples

**Guidelines**:
- Include both PASS and FAIL examples
- Show edge cases and subtle violations
- Provide detailed reasoning in examples
- Asymmetric ratio (e.g., 1 PASS, 3 FAIL) to make judge more critical

```python
def select_few_shot_examples(
    train_traces: List[Dict[str, Any]],
    num_positive: int = 1,
    num_negative: int = 3,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select few-shot examples from train set."""

    # Separate by label
    train_pass = [t for t in train_traces if t["label"] == "PASS"]
    train_fail = [t for t in train_traces if t["label"] == "FAIL"]

    selected = []

    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Select examples
    if len(train_pass) >= num_positive:
        selected.extend(random.sample(train_pass, num_positive))
    if len(train_fail) >= num_negative:
        selected.extend(random.sample(train_fail, num_negative))

    return selected
```

### Evaluating a Single Trace

```python
def evaluate_single_trace(args: tuple) -> Dict[str, Any]:
    """Evaluate a single trace with the judge."""
    trace, judge_prompt = args

    # Extract data
    query = trace["query"]
    dietary_restriction = trace["dietary_restriction"]
    response = trace["response"]
    true_label = trace["label"]

    # Replace placeholders
    formatted_prompt = judge_prompt.replace("__QUERY__", query)
    formatted_prompt = formatted_prompt.replace(
        "__DIETARY_RESTRICTION__", dietary_restriction
    )
    formatted_prompt = formatted_prompt.replace("__RESPONSE__", response)

    # Call LLM judge
    completion = litellm.completion(
        model="gpt-4o-nano",  # Cheaper model for judge
        messages=[{"role": "user", "content": formatted_prompt}],
    )

    response_text = completion.choices[0].message.content.strip()

    # Parse JSON response
    result = json.loads(json_text)
    predicted_label = result.get("label", "UNKNOWN")

    return {
        "true_label": true_label,
        "predicted_label": predicted_label,
        "query": query,
        "dietary_restriction": dietary_restriction,
    }
```

### Iterative Development on Dev Set

```python
def evaluate_judge_on_dev(
    judge_prompt: str,
    dev_traces: List[Dict[str, Any]],
    sample_size: int = 50,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate judge on dev set."""

    # Sample dev traces
    sampled = random.sample(dev_traces, sample_size)

    # Evaluate in parallel
    tasks = [(trace, judge_prompt) for trace in sampled]
    predictions = []

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(evaluate_single_trace, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            predictions.append(future.result())

    # Calculate metrics
    tp = sum(1 for p in predictions
             if p["true_label"] == "PASS" and p["predicted_label"] == "PASS")
    fn = sum(1 for p in predictions
             if p["true_label"] == "PASS" and p["predicted_label"] == "FAIL")
    tn = sum(1 for p in predictions
             if p["true_label"] == "FAIL" and p["predicted_label"] == "FAIL")
    fp = sum(1 for p in predictions
             if p["true_label"] == "FAIL" and p["predicted_label"] == "PASS")

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return tpr, tnr, predictions
```

### Development Workflow

1. **Initial Prompt**: Create baseline with random few-shot examples
2. **Measure**: Run on dev set, calculate TPR/TNR
3. **Analyze Errors**: Look at predictions where judge was wrong
4. **Refine**: Update prompt based on error patterns
5. **Repeat**: Iterate steps 2-4 until satisfied
6. **Lock**: When happy, save final prompt for test evaluation

---

## Step 5: Evaluate on Test Set

### Purpose
Get **unbiased** estimates of judge performance (TPR/TNR) for use with judgy bias correction.

### Critical Rule: Run Only ONCE! ⚠️

The test set is your protection against overfitting. Every time you:
- Look at test set results
- Adjust your prompt based on test errors
- Re-evaluate on the test set

...you **leak information** and compromise the test set's objectivity.

**Best Practice**: Only evaluate on test set when you're completely satisfied with judge performance on dev set.

### Code Example

```python
def evaluate_judge_on_test(
    judge_prompt: str,
    test_traces: List[Dict[str, Any]],
    max_workers: int = 32,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Evaluate judge on ALL test traces (no sampling)."""

    console.print(f"[yellow]Evaluating on {len(test_traces)} test traces...")

    # Prepare all tasks (evaluate ALL test traces, not a sample)
    tasks = [(trace, judge_prompt) for trace in test_traces]
    predictions = []

    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_trace, task): task
            for task in tasks
        }

        for future in as_completed(futures):
            result = future.result()
            predictions.append(result)

    # Calculate confusion matrix
    tp = sum(1 for p in predictions
             if p["true_label"] == "PASS" and p["predicted_label"] == "PASS")
    fn = sum(1 for p in predictions
             if p["true_label"] == "PASS" and p["predicted_label"] == "FAIL")
    tn = sum(1 for p in predictions
             if p["true_label"] == "FAIL" and p["predicted_label"] == "FAIL")
    fp = sum(1 for p in predictions
             if p["true_label"] == "FAIL" and p["predicted_label"] == "PASS")

    # Calculate TPR and TNR
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return tpr, tnr, predictions
```

### Saving Results for Judgy

```python
def save_results_for_judgy(
    predictions: List[Dict[str, Any]],
    results_dir: Path
) -> None:
    """Save test results in judgy-compatible format."""

    # Convert PASS/FAIL to 1/0 format
    test_labels = [
        1 if p["true_label"] == "PASS" else 0
        for p in predictions
    ]
    test_preds = [
        1 if p["predicted_label"] == "PASS" else 0
        for p in predictions
    ]

    # Save for judgy
    judgy_data = {
        "test_labels": test_labels,  # Ground truth
        "test_preds": test_preds,    # Judge predictions
        "description": "Test set for judgy bias correction"
    }

    with open(results_dir / "judgy_test_data.json", "w") as f:
        json.dump(judgy_data, f, indent=2)
```

### What Gets Saved

**judgy_test_data.json**:
```json
{
  "test_labels": [1, 0, 1, 1, 0, 1, 0, ...],  // Ground truth (1=PASS, 0=FAIL)
  "test_preds": [1, 0, 0, 1, 0, 1, 0, ...],   // Judge predictions
  "description": "Test set for judgy bias correction"
}
```

This data enables judgy to:
1. Calculate confusion matrix (TP, FP, TN, FN)
2. Compute TPR = TP/(TP+FN) and TNR = TN/(TN+FP)
3. Use these rates to correct bias on unlabeled data

---

## Step 6: Run Full Evaluation with Judgy

### Purpose
Apply the finalized judge to all unlabeled production data and use judgy to correct for judge bias.

### The Bias Problem

Your judge evaluated the test set:
- **TPR = 1.0** (catches 100% of passes)
- **TNR = 0.75** (catches only 75% of fails)

When you run it on 2,400 unlabeled traces, it says **77.3% pass**.

Is that accurate? **NO!** The judge has a 25% false positive rate (misses 25% of violations).

### How Judgy Corrects Bias

Judgy uses the **Rogan-Gladen correction formula**:

```
corrected_rate = (observed_rate + TNR - 1) / (TPR + TNR - 1)
```

**Example**:
```python
observed_rate = 0.773  # Judge says 77.3% pass
TPR = 1.0              # From test set
TNR = 0.75             # From test set

corrected_rate = (0.773 + 0.75 - 1) / (1.0 + 0.75 - 1)
               = (1.523 - 1) / (1.75 - 1)
               = 0.523 / 0.75
               = 0.697  # True rate is ~69.7%!
```

The correction went DOWN because the judge is too lenient (low TNR).

### Code Example

```python
from judgy import estimate_success_rate
import numpy as np

def compute_metrics_with_judgy(
    test_labels: List[int],      # Ground truth from test set
    test_preds: List[int],       # Judge predictions on test set
    unlabeled_preds: List[int],  # Judge predictions on all traces
) -> Tuple[float, float, float, float]:
    """Compute corrected success rate with judgy."""

    # Use judgy to estimate true success rate
    theta_hat, lower_bound, upper_bound = estimate_success_rate(
        test_labels=test_labels,
        test_preds=test_preds,
        unlabeled_preds=unlabeled_preds
    )

    # Also compute raw observed rate
    raw_rate = np.mean(unlabeled_preds)

    return theta_hat, lower_bound, upper_bound, raw_rate
```

### Running Judge on All Traces

```python
def run_judge_on_traces(
    judge_prompt: str,
    traces: List[Dict[str, Any]],
    max_workers: int = 32,
) -> List[int]:
    """Run judge on all traces, return binary predictions."""

    tasks = [(trace, judge_prompt) for trace in traces]
    predictions = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_trace_binary, task): task
            for task in tasks
        }

        for future in as_completed(futures):
            result = future.result()  # 1 for PASS, 0 for FAIL
            predictions.append(result)

    return predictions

def evaluate_trace_binary(args: tuple) -> int:
    """Evaluate a trace and return 1 (PASS) or 0 (FAIL)."""
    trace, judge_prompt = args

    # Format prompt
    formatted = judge_prompt.replace("__QUERY__", trace["query"])
    formatted = formatted.replace("__DIETARY_RESTRICTION__",
                                  trace["dietary_restriction"])
    formatted = formatted.replace("__RESPONSE__", trace["response"])

    # Call judge
    completion = litellm.completion(
        model="gpt-4o-nano",
        messages=[{"role": "user", "content": formatted}],
    )

    # Parse response
    result = json.loads(response_text)
    label = result.get("label", "FAIL")

    return 1 if label == "PASS" else 0
```

### Complete Pipeline

```python
def main():
    """Run full evaluation with judgy correction."""

    # 1. Load judge prompt
    judge_prompt = load_judge_prompt("results/judge_prompt.txt")

    # 2. Load test set performance data
    with open("results/judgy_test_data.json") as f:
        data = json.load(f)
    test_labels = data["test_labels"]
    test_preds = data["test_preds"]

    # 3. Load all unlabeled traces
    all_traces = load_traces("data/raw_traces.csv")  # 2,400 traces

    # 4. Run judge on all traces
    predictions = run_judge_on_traces(judge_prompt, all_traces)

    # 5. Compute corrected metrics with judgy
    theta_hat, lower, upper, raw_rate = compute_metrics_with_judgy(
        test_labels, test_preds, predictions
    )

    # 6. Display results
    print(f"Raw Observed Rate: {raw_rate:.3f} ({raw_rate*100:.1f}%)")
    print(f"Corrected Rate: {theta_hat:.3f} ({theta_hat*100:.1f}%)")
    print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    print(f"Correction: {abs(raw_rate - theta_hat):.3f} ({abs(raw_rate - theta_hat)*100:.1f} pp)")
```

### Example Output

```
Raw Observed Success Rate: 0.773 (77.3%)
Corrected Success Rate: 0.697 (69.7%)
95% Confidence Interval: [0.586, 0.808]
                        [58.6%, 80.8%]
Correction Applied: 0.076 (7.6 percentage points)
```

---

## Data Shapes Reference

### For generate_traces.py

**Input**: `dietary_queries.csv`
```python
{
    "id": 1,
    "query": "I'm vegan but I really want to make something with honey...",
    "dietary_restriction": "vegan"
}
```

**Output**: `raw_traces.csv`
```python
{
    "query": "I'm vegan but...",
    "dietary_restriction": "vegan",
    "response": "Here's a Vegan Coconut Yogurt recipe...",
    "success": True,
    "error": None,
    "trace_id": "1_5",  # Format: {query_id}_{trace_num}
    "query_id": 1
}
```

### For label_data.py

**Input**: `raw_traces.csv` (from Step 1)

**Output**: `labeled_traces.csv`
```python
{
    "query": "I'm vegan but...",
    "dietary_restriction": "vegan",
    "response": "Greek Yogurt Parfait...",
    "label": "FAIL",  # NEW: PASS or FAIL
    "reasoning": "Contains Greek yogurt (dairy), not vegan",  # NEW
    "confidence": "HIGH",  # NEW: HIGH/MEDIUM/LOW
    "labeled": True,  # NEW: labeling success flag
    "trace_id": "1_3",
    "query_id": 1
}
```

### For split_data.py

**Input**: `labeled_traces.csv` (~150 labeled traces)

**Output**: Three CSV files with same schema:
- `train_set.csv` (~23 rows, 15%)
- `dev_set.csv` (~60 rows, 40%)
- `test_set.csv` (~67 rows, 45%)

Schema matches `labeled_traces.csv`.

### For develop_judge.py

**Input from train_set.csv** (for few-shot examples):
```python
{
    "query": "...",
    "dietary_restriction": "vegan",
    "response": "...",
    "label": "PASS",
    "reasoning": "...",
    # ... other fields
}
```

**Output**: `judge_prompt.txt`
```text
You are an expert nutritionist...

Example 1:
Query: ...
Recipe Response: ...
Reasoning: ...
Label: PASS

...

Now evaluate:
Query: __QUERY__
Dietary Restriction: __DIETARY_RESTRICTION__
Recipe Response: __RESPONSE__
```

**Dev Predictions**: `dev_predictions.json`
```python
[
    {
        "trace_id": "5_12",
        "true_label": "PASS",
        "predicted_label": "PASS",
        "query": "...",
        "dietary_restriction": "vegan",
        "success": True
    },
    # ... 50 predictions
]
```

### For evaluate_judge.py

**Input from test_set.csv**: Same schema as labeled_traces.csv

**Output**: `judgy_test_data.json`
```python
{
    "test_labels": [1, 0, 1, 1, 0, ...],  # 1=PASS, 0=FAIL (ground truth)
    "test_preds": [1, 0, 0, 1, 0, ...],   # Judge predictions
    "description": "Test set for judgy"
}
```

**Also saves**: `judge_performance.json`
```python
{
    "test_set_performance": {
        "true_positive_rate": 1.0,      # TPR
        "true_negative_rate": 0.75,     # TNR
        "balanced_accuracy": 0.875,
        "total_predictions": 67,
        "correct_predictions": 60,
        "accuracy": 0.896
    }
}
```

### For run_full_evaluation.py (judgy)

**Input**:
1. `judgy_test_data.json` (from Step 5):
```python
{
    "test_labels": [1, 0, 1, ...],  # Ground truth
    "test_preds": [1, 0, 0, ...]    # Judge predictions on test
}
```

2. All unlabeled traces from `raw_traces.csv`

**Process**: Judge evaluates all 2,400 traces → binary predictions

**Call judgy**:
```python
from judgy import estimate_success_rate

theta_hat, lower, upper = estimate_success_rate(
    test_labels=[1, 0, 1, 1, 0, ...],      # From judgy_test_data.json
    test_preds=[1, 0, 0, 1, 0, ...],       # From judgy_test_data.json
    unlabeled_preds=[1, 1, 0, 1, ...]      # Judge predictions on 2,400 traces
)
```

**Output**: `final_evaluation.json`
```python
{
    "final_evaluation": {
        "total_traces_evaluated": 2400,
        "raw_observed_success_rate": 0.773,
        "corrected_success_rate": 0.697,
        "confidence_interval_95": {
            "lower_bound": 0.586,
            "upper_bound": 0.808
        }
    }
}
```

---

## Key Concepts

### 1. TPR (True Positive Rate)

**Definition**: Of all items that should be labeled PASS, what percentage did the judge correctly identify?

**Formula**: `TPR = TP / (TP + FN)`

**Example**:
- 100 recipes should PASS
- Judge correctly identifies 95 → TP = 95
- Judge incorrectly rejects 5 → FN = 5
- **TPR = 95 / (95 + 5) = 0.95**

**Interpretation**:
- TPR = 1.0 → Perfect! Judge never misses a passing case
- TPR = 0.75 → Judge misses 25% of passing cases (too strict)

### 2. TNR (True Negative Rate)

**Definition**: Of all items that should be labeled FAIL, what percentage did the judge correctly identify?

**Formula**: `TNR = TN / (TN + FP)`

**Example**:
- 100 recipes should FAIL
- Judge correctly identifies 80 → TN = 80
- Judge incorrectly passes 20 → FP = 20
- **TNR = 80 / (80 + 20) = 0.80**

**Interpretation**:
- TNR = 1.0 → Perfect! Judge never misses a violation
- TNR = 0.75 → Judge misses 25% of violations (too lenient)

### 3. Confusion Matrix

```
                    Predicted
                 PASS      FAIL
        PASS │   TP    │   FN   │
Actual       │─────────│────────│
        FAIL │   FP    │   TN   │
```

- **TP (True Positive)**: Correctly identified PASS
- **TN (True Negative)**: Correctly identified FAIL
- **FP (False Positive)**: Said PASS, should be FAIL (missed violation!)
- **FN (False Negative)**: Said FAIL, should be PASS (too strict)

### 4. Rogan-Gladen Correction

Mathematical formula judgy uses to correct for judge bias:

```
θ̂ = (p_obs + TNR - 1) / (TPR + TNR - 1)
```

Where:
- `θ̂` = Corrected (true) pass rate
- `p_obs` = Observed pass rate from judge
- `TPR` = True Positive Rate from test set
- `TNR` = True Negative Rate from test set

**Intuition**: The observed rate is a biased mixture of:
- True passes that the judge correctly identified (weighted by TPR)
- True fails that the judge incorrectly passed (weighted by 1-TNR = FPR)

Rogan-Gladen "unmixes" this to estimate the true underlying rate.

### 5. Bootstrapping for Confidence Intervals

Judgy uses bootstrapping to quantify uncertainty:

1. **Resample** test set 20,000 times (with replacement)
2. For each resample, calculate TPR and TNR
3. For each resample, apply Rogan-Gladen correction
4. **95% CI** = [2.5th percentile, 97.5th percentile] of 20,000 estimates

This gives you a range: "We're 95% confident the true rate is between X% and Y%"

### 6. Stratification

**Purpose**: Ensure each data split maintains the same class balance.

**Without stratification**:
```
Original: 50% PASS, 50% FAIL

Random split could give:
- Train: 80% PASS, 20% FAIL  ← Imbalanced!
- Dev: 40% PASS, 60% FAIL
- Test: 55% PASS, 45% FAIL
```

**With stratification**:
```
Original: 50% PASS, 50% FAIL

Stratified split gives:
- Train: 50% PASS, 50% FAIL  ✓ Balanced
- Dev: 50% PASS, 50% FAIL    ✓ Balanced
- Test: 50% PASS, 50% FAIL   ✓ Balanced
```

---

## Best Practices

### 1. Data Labeling

✅ **DO**:
- Label data manually with domain experts when possible
- Look at every label, even if LLM-generated
- Review low-confidence labels carefully
- Check for labeling consistency across similar examples

❌ **DON'T**:
- Blindly trust LLM-generated labels
- Use the same model for labeling and judging
- Skip manual review of labeled data
- Ignore edge cases or ambiguous examples

### 2. Data Splitting

✅ **DO**:
- Use stratification to maintain class balance
- Set random seed for reproducibility
- Validate splits before using them
- Save split assignments for later reference

❌ **DON'T**:
- Use purely random splitting (risks imbalance)
- Change split assignments mid-development
- Look at test set during judge development
- Use test set examples in your judge prompt

### 3. Judge Development

✅ **DO**:
- Iterate freely on dev set
- Analyze dev set errors to improve prompt
- Use asymmetric few-shot ratios (more FAIL examples)
- Test multiple prompt variations
- Track TPR/TNR metrics during development

❌ **DON'T**:
- Stop at first "good enough" results
- Ignore systematic error patterns
- Use all PASS or all FAIL few-shot examples
- Optimize only for overall accuracy (track TPR and TNR separately!)

### 4. Test Set Evaluation

✅ **DO**:
- Run test evaluation only ONCE when judge is finalized
- Evaluate ALL test traces (no sampling)
- Save test results for judgy
- Document test set performance clearly

❌ **DON'T**:
- Look at test set during development
- Iterate based on test set errors
- Re-run test evaluation multiple times
- Use test set for prompt refinement

### 5. Production Evaluation

✅ **DO**:
- Use judgy to correct for judge bias
- Report both raw and corrected rates
- Include confidence intervals
- Document TPR/TNR assumptions

❌ **DON'T**:
- Report only raw observed rates
- Ignore judge error patterns
- Assume judge is unbiased
- Skip uncertainty quantification

---

## Common Pitfalls

### 1. Contaminating Test Set

**Problem**: Looking at test set results and adjusting prompt based on test errors.

**Why it's bad**: Each time you look at test set and adjust, you're "training" on the test set. The test set can no longer give unbiased estimates of real-world performance.

**Solution**:
- Only evaluate on test set when judge is finalized
- If you must look at test errors, create a NEW test set
- Turn old test set into training data and collect fresh test data

### 2. Imbalanced Splits

**Problem**: Random splitting creates train/dev/test sets with different PASS/FAIL ratios.

**Why it's bad**:
- Judge trained on 80% PASS data will be too lenient
- Dev set with 30% PASS can't measure performance well
- Test set with few FAIL examples can't measure TNR accurately

**Solution**:
- Use stratified splitting (sklearn's `train_test_split` with `stratify` parameter)
- Validate splits before using them
- Ensure each split has reasonable numbers of both classes

### 3. Insufficient Few-Shot Examples

**Problem**: Using only 1-2 few-shot examples, or only PASS examples.

**Why it's bad**:
- Judge doesn't learn what violations look like
- Performance degrades on edge cases
- May be biased toward one label

**Solution**:
- Use 3-5 few-shot examples minimum
- Include both PASS and FAIL examples
- Deliberately select examples showing subtle violations
- Consider asymmetric ratios (e.g., 1 PASS, 3 FAIL) to make judge more critical

### 4. Ignoring TPR/TNR Tradeoffs

**Problem**: Only optimizing for overall accuracy, ignoring TPR and TNR individually.

**Why it's bad**:
- Judge might be great at catching passes (high TPR) but terrible at catching violations (low TNR)
- Or vice versa
- Overall accuracy can be high even if one metric is terrible

**Example**:
```
Judge A: TPR=0.95, TNR=0.50, Accuracy=0.73  ← Misses 50% of violations!
Judge B: TPR=0.80, TNR=0.80, Accuracy=0.80  ← Better balanced
```

**Solution**:
- Always track TPR and TNR separately
- Consider which errors are more costly in your use case
- Balance performance on both metrics

### 5. Not Using Judgy

**Problem**: Reporting raw observed success rates without correction.

**Why it's bad**: Raw rates are biased by judge errors and can be misleading.

**Example**:
```
Judge says: 90% of recipes pass  ← Observed rate
Reality: 82% actually pass       ← True rate (after judgy correction)

Difference: 8 percentage points of bias!
```

**Solution**:
- Always use judgy for production evaluation
- Report both raw and corrected rates
- Include confidence intervals
- Explain the correction to stakeholders

### 6. Treating All Errors Equally

**Problem**: Not considering which type of error is more problematic.

**False Positive (FP)**: Judge says PASS, should be FAIL
- Dietary violation goes undetected
- Could harm users (allergies, health conditions)
- Damages user trust

**False Negative (FN)**: Judge says FAIL, should be PASS
- Valid recipe gets rejected
- Increased latency (retry needed)
- Minor user annoyance

**In this case, FP is much worse!**

**Solution**:
- Identify which error type is more costly
- Adjust judge to prioritize avoiding that error
- Document this decision clearly
- Use asymmetric few-shot examples (more FAIL examples if FP is worse)

### 7. Parallel Processing Errors

**Problem**: Not handling API errors gracefully in parallel processing.

**Why it's bad**: One API error can crash entire evaluation.

**Solution**:
```python
def evaluate_single_trace(args: tuple) -> Dict[str, Any]:
    try:
        # ... evaluation code ...
        return {
            "predicted_label": label,
            "success": True,
        }
    except Exception as e:
        # Don't crash - return error result
        return {
            "predicted_label": "ERROR",
            "success": False,
            "error": str(e),
        }
```

### 8. Not Validating Data Shapes

**Problem**: Assuming data is in the right format without checking.

**Why it's bad**: Silent failures, incorrect results, wasted time debugging.

**Solution**:
```python
# Always validate
def validate_traces(traces: List[Dict]) -> bool:
    required_fields = ["query", "dietary_restriction", "response"]

    for trace in traces:
        for field in required_fields:
            if field not in trace:
                raise ValueError(f"Missing field: {field}")

    return True
```

---

## Summary

Implementing an LLM-as-Judge system involves:

1. **Generate Traces**: Create diverse outputs to evaluate
2. **Label Data**: Create ground truth PASS/FAIL labels
3. **Split Data**: Train/dev/test for iterative development
4. **Develop Judge**: Create effective prompt with few-shot examples
5. **Test Evaluation**: Measure unbiased TPR/TNR on held-out data
6. **Production Evaluation**: Use judgy to correct bias on unlabeled data

**Key Takeaways**:
- LLM judges are powerful but imperfect
- Use stratified splits and separate train/dev/test sets
- Track TPR and TNR, not just accuracy
- Test set is sacred - use only once
- Always use judgy for bias correction on production data
- Consider cost of different error types

**Data Flow**:
```
Queries → Traces → Labels → Splits → Judge → Test Metrics → Production Eval → Corrected Results
```

This guide provides the foundation for building production-grade LLM evaluation systems that are rigorous, unbiased, and actionable.
