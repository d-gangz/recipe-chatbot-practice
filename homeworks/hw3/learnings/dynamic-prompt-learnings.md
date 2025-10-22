<!--
Document Type: Learning Notes
Purpose: Capture best practices for structuring LLM prompts with dynamic values
Context: Created while analyzing the LLM-as-Judge prompt construction in hw3
Key Topics: Prompt templating, placeholder patterns, string concatenation, when to use different approaches
Target Use: Reference guide for future prompt engineering work
-->

# Dynamic Prompt Construction: Patterns and Best Practices

**Context**: Learnings from analyzing `4_develop_judge.py` prompt construction patterns.

---

## The Core Problem

When building LLM prompts, you often need to:
1. Create a **reusable template** with static instructions
2. Add **dynamic content** that changes (examples, user input, retrieved data)
3. Insert **specific values** for each execution

**Key Question**: How do you structure prompts that have dynamic values?

---

## Pattern 1: String Concatenation + Placeholder Replacement

### What the hw3 Code Does

```python
def create_judge_prompt(few_shot_examples: List[Dict]) -> str:
    """Build a reusable prompt template."""

    # PHASE 1: Build template with concatenation
    # ==========================================

    # Step 1: Start with base instructions (static)
    base_prompt = """You are an expert evaluator...

INSTRUCTIONS:
- Criteria 1
- Criteria 2
- Criteria 3

Here are examples:

"""

    # Step 2: Add dynamic examples via loop (concatenation)
    for i, example in enumerate(few_shot_examples, 1):
        base_prompt += f"\nExample {i}:\n"
        base_prompt += f"Query: {example['query']}\n"
        base_prompt += f"Response: {example['response']}\n"
        base_prompt += f"Label: {example['label']}\n"

    # Step 3: Add evaluation template with placeholders (concatenation)
    base_prompt += """

Now evaluate:

Query: __QUERY__
Context: __CONTEXT__
Response: __RESPONSE__

Output JSON:
{"reasoning": "...", "label": "..."}
"""

    return base_prompt  # This is the TEMPLATE


# PHASE 2: Fill in specific values
# ==================================

def evaluate_item(item, template):
    """Use the template for a specific item."""

    # Replace placeholders with actual values
    formatted_prompt = template.replace("__QUERY__", item.query)
    formatted_prompt = formatted_prompt.replace("__CONTEXT__", item.context)
    formatted_prompt = formatted_prompt.replace("__RESPONSE__", item.response)

    # Send to LLM
    result = llm.complete(formatted_prompt)
    return result
```

### Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Create Template (Once)                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  base_prompt = "Instructions..."                            │
│                                                              │
│         ↓ (concatenate via +=)                              │
│                                                              │
│  base_prompt = "Instructions...                             │
│                 Example 1: ...                              │
│                 Example 2: ...                              │
│                 Example 3: ..."                             │
│                                                              │
│         ↓ (concatenate via +=)                              │
│                                                              │
│  base_prompt = "Instructions...                             │
│                 Example 1: ...                              │
│                 Example 2: ...                              │
│                 Example 3: ...                              │
│                 Now evaluate:                               │
│                 Query: __QUERY__                            │
│                 Response: __RESPONSE__"                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Fill Template (For Each Item)                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For item 1:                                                │
│    formatted = template.replace("__QUERY__", item1.query)  │
│    formatted = formatted.replace("__RESPONSE__", item1.res) │
│    → Send to LLM                                            │
│                                                              │
│  For item 2:                                                │
│    formatted = template.replace("__QUERY__", item2.query)  │
│    formatted = formatted.replace("__RESPONSE__", item2.res) │
│    → Send to LLM                                            │
│                                                              │
│  ... (repeat for N items)                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Why This Pattern Works

✅ **Separation of concerns**:
- Template creation = one-time setup
- Value insertion = per-execution

✅ **Reusability**:
- Build the template once
- Reuse for 100s or 1000s of evaluations

✅ **Saveable**:
- Template can be saved to a file
- Template can be manually edited
- Template can be loaded without needing actual data

✅ **Explicit placeholders**:
- `__QUERY__` is clearly a placeholder
- No confusion with actual content
- No conflicts with `{` `}` in user text

---

## Pattern 2: F-Strings (When Data is Available)

### When to Use

Use f-strings when you have **all the data at template creation time**.

```python
def create_one_time_prompt(query, context, response):
    """Create a prompt for immediate use."""

    prompt = f"""You are an expert evaluator.

Query: {query}
Context: {context}
Response: {response}

Evaluate this and provide reasoning.
"""

    return prompt

# Usage: create and use immediately
prompt = create_one_time_prompt("What is vegan?", "diet", "No animal products")
result = llm.complete(prompt)
```

### Pros and Cons

✅ **Pros**:
- Clean, readable syntax
- Python native (no extra dependencies)
- Direct variable substitution

❌ **Cons**:
- Can't separate template creation from data insertion
- Not reusable across multiple items
- Can't save template without data
- Risk of format conflicts if user text contains `{` `}`

---

## Pattern 3: str.format() or Template Strings

### When to Use

Use when you want **named placeholders** but data comes later.

```python
def create_template():
    """Create a template with named placeholders."""

    template = """You are an evaluator.

Query: {query}
Context: {context}
Response: {response}

Evaluate and provide reasoning.
"""

    return template

# Usage: create template, fill later
template = create_template()

# Fill for item 1
prompt1 = template.format(
    query="What is vegan?",
    context="diet",
    response="No animal products"
)

# Fill for item 2
prompt2 = template.format(
    query="What is gluten-free?",
    context="diet",
    response="No wheat or gluten"
)
```

### Pros and Cons

✅ **Pros**:
- Named placeholders (clearer than positional)
- Reusable template
- Clean syntax

❌ **Cons**:
- Still risk of format conflicts with `{` `}` in user text
- Less explicit than `__PLACEHOLDER__` pattern

---

## Pattern 4: Jinja2 Templates

### When to Use

Use Jinja2 for **complex templating logic** with conditionals, filters, and inheritance.

```python
from jinja2 import Template

def create_complex_template():
    """Create template with complex logic."""

    template = Template("""You are an evaluator.

{% if difficulty == 'beginner' %}
Use simple language in your evaluation.
{% else %}
Provide detailed technical analysis.
{% endif %}

Examples:
{% for example in examples %}
  {% if example.is_relevant %}
  Example {{ loop.index }}:
  Query: {{ example.query }}
  Response: {{ example.response }}
  Label: {{ example.label }}
  {% endif %}
{% endfor %}

Now evaluate:
Query: {{ query }}
Response: {{ response }}
""")

    return template

# Usage
template = create_complex_template()

prompt = template.render(
    difficulty='beginner',
    examples=[...],
    query="What is vegan?",
    response="No animal products"
)
```

### When Jinja2 is Better

✅ Use Jinja2 when you need:
- Conditional rendering (`{% if %}`)
- Complex loops with logic (`{% for %}` with filters)
- Template inheritance (DRY principles)
- Custom filters and functions
- Multiple template files

❌ Don't use Jinja2 when:
- Simple string concatenation works
- No conditional logic needed
- Avoiding extra dependencies
- Teaching fundamentals to beginners

---

## Decision Matrix: Which Pattern to Use?

| Situation | Best Pattern | Why |
|-----------|-------------|-----|
| Build template once, use many times | `__PLACEHOLDER__` + `.replace()` | Reusable, explicit, saveable |
| All data available immediately | F-strings | Clean, native Python |
| Need named placeholders, data comes later | `str.format()` | Named placeholders, reusable |
| Complex conditional logic | Jinja2 | Powerful templating engine |
| Template saved to file for manual editing | `__PLACEHOLDER__` + `.replace()` | Explicit, no syntax conflicts |
| Educational/learning context | String concatenation + placeholders | Simple, explicit, no magic |

---

## Real-World Example from hw3

### The Problem

In the LLM-as-Judge system:
1. Create a prompt template **once** (with base instructions + few-shot examples)
2. Evaluate **50 recipes** using the same template
3. Allow **manual refinement** of the saved prompt

### The Solution

```python
# In main() - create template ONCE
judge_prompt = create_judge_prompt(few_shot_examples)
# Result: Full template with "__QUERY__", "__DIETARY_RESTRICTION__", "__RESPONSE__"

# Save template for manual editing
save_judge_prompt(judge_prompt, "judge_prompt.txt")

# In evaluate_judge_on_dev() - use template 50 times
for recipe in dev_recipes[:50]:
    result = evaluate_single_trace((recipe, judge_prompt))
    # Each call replaces placeholders with recipe-specific data
```

### Why This Works Better Than Alternatives

**If they used f-strings:**
```python
# ❌ Would need to pass recipe data when creating template
judge_prompt = create_judge_prompt(few_shot_examples, recipe.query, recipe.response)
# Problem: Can only use for ONE recipe, need to rebuild for each recipe
```

**If they used Jinja2:**
```python
# ❌ Overkill for simple substitution
template = Template("Query: {{ query }}\nResponse: {{ response }}")
# Problem: Extra dependency, more complex, no real benefit here
```

---

## Key Takeaways

1. **Two-Phase Approach** is powerful:
   - Phase 1: Build reusable template
   - Phase 2: Fill in specific values

2. **Choose based on use case**:
   - One-time use? → F-strings
   - Reusable template? → Placeholders + replace()
   - Complex logic? → Jinja2

3. **Placeholder naming convention**:
   - Use `__ALL_CAPS__` for clarity
   - Makes placeholders obvious in saved files
   - Avoids conflicts with actual content

4. **String concatenation is fine**:
   - Don't over-engineer
   - Simple concatenation is readable
   - Works great for educational contexts

5. **Separation of concerns**:
   - Template structure (static + examples)
   - Template population (specific values)
   - Keeps code clean and maintainable

---

## References

- **hw3 Code**: `homeworks/hw3/scripts/4_develop_judge.py` (lines 200-270, 292-390)
- **Pattern**: Two-phase prompt construction with concatenation and placeholder replacement
- **Use Case**: LLM-as-Judge evaluation system with reusable prompt template
