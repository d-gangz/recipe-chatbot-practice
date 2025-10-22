<!--
Document Type: Learning Notes
Purpose: Detailed explanation of how the judgy package works for bias correction in LLM-as-Judge evaluation
Context: Created while understanding HW3 step 6 (run_full_evaluation.py) - how judgy corrects for judge bias
Key Topics: Rogan-Gladen correction, bootstrapping, confidence intervals, TPR/TNR, bias correction
Target Use: Reference guide for understanding judgy's two-step process (point estimate + uncertainty quantification)
-->

# How judgy Works: Complete Explanation

## Overview

The `judgy` package solves a critical problem in LLM evaluation: **How do you measure quality on unlabeled production data when your judge isn't perfect?**

judgy uses a two-step process:
1. **Rogan-Gladen correction** - Compute a bias-corrected point estimate
2. **Bootstrapping** - Quantify uncertainty with confidence intervals

---

## The Problem judgy Solves

You have an LLM judge that evaluates recipes:
- **Test set** (46 examples): You know the ground truth (PASS/FAIL)
- **Production data** (2400 examples): No ground truth labels!

Your judge isn't perfect:
- **TPR = 1.0** (catches 100% of passes)
- **TNR = 0.75** (catches only 75% of fails, misses 25%!)

When your judge says **77.3% of production recipes pass**, is that accurate?

**No!** The judge is too lenient (low TNR), so it overcounts passes by missing violations.

judgy corrects this bias to estimate the **true pass rate** (~69.7%).

---

## Step 1: Point Estimate (Rogan-Gladen Correction)

**IMPORTANT**: The point estimate is calculated **ONCE** using the **ORIGINAL data** (NOT from bootstrap iterations).

### The Formula

```
θ̂ = (p_obs + TNR - 1) / (TPR + TNR - 1)
```

Where:
- `p_obs` = observed pass rate from judge on **original** unlabeled data
- `TPR` = True Positive Rate from **original** test set
- `TNR` = True Negative Rate from **original** test set
- `θ̂` = corrected (true) pass rate **point estimate**

### Your Example

**Input** (from ORIGINAL data, no resampling):
```python
# From test set evaluation (5_evaluate_judge.py)
# Using ORIGINAL test set (46 examples)
TPR = 1.0    # Judge catches 100% of passes
TNR = 0.75   # Judge catches 75% of fails (misses 25%!)

# From unlabeled production data (6_run_full_evaluation.py)
# Using ORIGINAL unlabeled predictions (2400 examples)
p_obs = 0.773  # Judge says 77.3% pass
```

**Calculation** (performed ONCE on original data):
```python
θ̂ = (0.773 + 0.75 - 1) / (1.0 + 0.75 - 1)
  = (1.523 - 1) / (1.75 - 1)
  = 0.523 / 0.75
  = 0.697
```

**Result**: True pass rate ≈ **69.7%** (not 77.3%!)

**Key Point**: This 0.697 is your **point estimate** - the single best guess. It does NOT come from averaging 20,000 bootstrap iterations!

---

### Why the Correction Goes DOWN

Your judge has **TNR = 0.75**, which means:
- **FPR = 1 - TNR = 0.25** (25% false positive rate)
- Judge is **too lenient** - it lets violations slip through

**What really happened**:

```
Judge says: 1855 PASS, 545 FAIL (77.3% pass)

But with FPR = 0.25, some of those "passes" are actually fails:
- Estimated false positives ≈ 1855 × 0.25 ≈ 464 violations missed
- True passes ≈ 1855 - 464 ≈ 1391 ??? (too simplistic)

Better calculation using Rogan-Gladen:
- True passes ≈ 1673
- False positives ≈ 182 violations the judge missed
- Corrected pass rate = 1673/2400 = 69.7%
```

The judge **overestimated** by 7.6 percentage points (77.3% → 69.7%) because it's too lenient.

---

### The Intuition: Mixture Model

The observed pass rate is a **mixture** of true passes and false positives:

```
observed_passes = (TPR × true_passes) + (FPR × true_fails)

Where:
- TPR × true_passes = passes the judge correctly identified
- FPR × true_fails = fails the judge incorrectly marked as pass
```

Rogan-Gladen **solves** this equation for `true_passes`:

```python
# We know:
observed_passes / total = 0.773
TPR = 1.0
FPR = 0.25
true_passes + true_fails = total

# Rearrange and solve
# (Result is the Rogan-Gladen formula)
true_pass_rate = (observed_rate + TNR - 1) / (TPR + TNR - 1)
```

---

## Step 2: Confidence Intervals (Bootstrapping)

The Rogan-Gladen formula gives you **69.7%**, but how certain should you be?

- Is it precisely 69.7% ± 1%? (very confident)
- Or could it be anywhere from 55% to 77%? (very uncertain)

**Bootstrapping answers this question!**

### Critical Clarification: What Bootstrap Does vs. Doesn't Do

❌ **Bootstrap does NOT calculate the point estimate (θ̂ = 0.697)**
- The point estimate comes from Step 1 (Rogan-Gladen on original data)
- Calculated ONCE, not 20,000 times

✅ **Bootstrap ONLY calculates the confidence interval bounds**
- Lower bound: 0.546 (2.5th percentile of bootstrap distribution)
- Upper bound: 0.773 (97.5th percentile of bootstrap distribution)
- Calculated from 20,000 resampled iterations

**Analogy**:
- Point estimate = Your best measurement with the original ruler (0.697)
- Confidence interval = Range if you measured 20,000 times with slight variations ([0.546, 0.773])

---

### What is Bootstrapping?

Bootstrap is a statistical technique to estimate **uncertainty** by **resampling your data**.

**Core idea**: "If we repeated the experiment many times, what range of results would we get?"

Since we can't re-collect data, we **simulate** repeat experiments by resampling.

**What bootstrap produces**: A distribution of 20,000 possible θ̂* values, which we use to calculate uncertainty bounds (NOT to calculate θ̂ itself!)

---

### The Bootstrap Process in judgy

judgy runs **20,000 bootstrap iterations** (default). Each iteration:

#### Iteration i (of 20,000):

**1. Resample test set** (with replacement):
```python
# Original test set: 46 examples
original_test = [
    {"true": 1, "pred": 1},  # Example 1
    {"true": 0, "pred": 0},  # Example 2
    {"true": 1, "pred": 1},  # Example 3
    ...
    {"true": 0, "pred": 1},  # Example 46
]

# Bootstrap: randomly pick 46 examples (with replacement)
bootstrap_test = random.choices(original_test, k=46)
# Result might be:
# - Example 1 appears 3 times
# - Example 2 appears 0 times (not selected)
# - Example 3 appears 1 time
# - Example 5 appears 2 times
# ... etc (total = 46)
```

**2. Calculate TPR*, TNR* from bootstrap test set**:
```python
# Original: TPR = 1.0, TNR = 0.75
# Bootstrap iteration i might give:
TPR_i* = 0.94  # Different due to resampling!
TNR_i* = 0.80  # Different due to resampling!
```

**3. Resample unlabeled data** (with replacement):
```python
# Original unlabeled: 2400 predictions
original_unlabeled = [1, 1, 0, 1, 0, 1, ...]  # 1855 ones, 545 zeros

# Bootstrap: randomly pick 2400 (with replacement)
bootstrap_unlabeled = random.choices(original_unlabeled, k=2400)
```

**4. Calculate p_obs* from bootstrap unlabeled set**:
```python
# Original: p_obs = 0.773
# Bootstrap iteration i might give:
p_obs_i* = 0.780  # Slightly different!
```

**5. Apply Rogan-Gladen with bootstrap values**:
```python
θ̂_i* = (p_obs_i* + TNR_i* - 1) / (TPR_i* + TNR_i* - 1)
     = (0.780 + 0.80 - 1) / (0.94 + 0.80 - 1)
     = 0.580 / 0.74
     = 0.784
```

**6. Store this value**:
```python
bootstrap_thetas[i] = 0.784
```

---

### After 20,000 Iterations

You have a distribution of 20,000 corrected rates:

```python
bootstrap_thetas = [
    0.784,  # Iteration 1
    0.649,  # Iteration 2
    0.721,  # Iteration 3
    0.658,  # Iteration 4
    ...
    0.701,  # Iteration 20000
]
```

Each value represents: "What would the corrected rate be if we had a slightly different test set or unlabeled sample?"

---

### Calculate Confidence Interval

**Sort the 20,000 values**:
```python
sorted_thetas = sorted(bootstrap_thetas)
# [0.546, 0.551, 0.558, ..., 0.697, ..., 0.771, 0.773]
#   ↑                        ↑                        ↑
#  Min                    Median                    Max
```

**Extract percentiles**:
```python
import numpy as np

lower_bound = np.percentile(sorted_thetas, 2.5)   # 500th value
upper_bound = np.percentile(sorted_thetas, 97.5)  # 19,500th value

# Your results:
# lower_bound = 0.546  (54.6%)
# upper_bound = 0.773  (77.3%)
```

**Interpretation**: "We're 95% confident the true pass rate is between 54.6% and 77.3%"

---

### Why NOT Use Average for Point Estimate?

**Common misconception**: "Take the average (or median) of 20,000 bootstrap values as the point estimate"

```python
average = mean(bootstrap_thetas) ≈ 0.697
median = median(bootstrap_thetas) ≈ 0.697
```

#### Why We Don't Do This

**Reason 1: Bootstrap is centered around the original estimate anyway!**

The bootstrap distribution is designed to vary **around** the original Rogan-Gladen estimate:

```python
# Original calculation (Step 1)
theta_hat = 0.697  (from original data)

# Bootstrap distribution (Step 2)
mean(bootstrap_thetas) ≈ 0.697    # Very close to original!
median(bootstrap_thetas) ≈ 0.697  # Very close to original!
```

The bootstrap doesn't give you a "better" point estimate - it just shows variability around the original!

**Reason 2: We want the confidence interval, not another point estimate!**

Using percentiles gives us the **spread** (uncertainty), which is the whole point:

```python
# What we DON'T want (just the center):
average = 0.697  # Tells us nothing about uncertainty

# What we DO want (the range):
lower_bound = 2.5th percentile = 0.546
upper_bound = 97.5th percentile = 0.773
# Tells us: "95% of bootstrap samples fell between 0.546 and 0.773"
```

**Reason 3: Cleaner interpretation!**

- **Point estimate (0.697)**: "Our best guess using original data"
- **Confidence interval [0.546, 0.773]**: "Uncertainty range from bootstrap resampling"

This separation makes it clear what comes from the original data vs. what comes from bootstrap uncertainty quantification.

#### The Correct Approach

✅ **Point estimate**: Calculate ONCE from original data using Rogan-Gladen
✅ **Confidence interval**: Use 2.5th and 97.5th percentiles from 20,000 bootstrap iterations

❌ **Don't**: Average the bootstrap values to get the point estimate

---

## Visualizing the Bootstrap Distribution

```
20,000 Bootstrap Values (histogram):

Frequency
    │
800 │         ┌───┐
    │         │   │
600 │     ┌───┤   ├───┐
    │     │   │   │   │
400 │ ┌───┤   │   │   ├───┐
    │ │   │   │   │   │   │  ┌───┐
200 │ │   │   │   │   │   ├──┤   │
    └─┴───┴───┴───┴───┴───┴──┴───┴─── Corrected Rate (θ̂*)
      0.55 0.60 0.65 0.70 0.75 0.77
       ↑                         ↑
    Lower                     Upper
    Bound                     Bound
   (2.5%ile)                (97.5%ile)

Point Estimate (from original data): 0.697
95% Confidence Interval: [0.546, 0.773]
```

---

## What Does Each Bound Mean?

### Lower Bound (0.546)

**Meaning**: "In the most pessimistic scenarios (bottom 2.5% of bootstrap samples), the true pass rate could be as low as 54.6%"

**Why so low?**
- Some bootstrap test sets had worse TPR/TNR (judge looked less reliable)
- Some bootstrap unlabeled sets had fewer passes (unlucky sample)
- Combined effect: very low corrected rate

**Example pessimistic bootstrap**:
```python
TPR* = 0.94  # Judge missed some passes in this sample
TNR* = 0.70  # Judge missed many fails in this sample
p_obs* = 0.760  # Fewer passes in this unlabeled sample

θ̂* = (0.760 + 0.70 - 1) / (0.94 + 0.70 - 1)
   = 0.460 / 0.64
   = 0.719  # Still lower than median

# Even more pessimistic samples give 0.546
```

---

### Upper Bound (0.773)

**Meaning**: "In the most optimistic scenarios (top 2.5% of bootstrap samples), the true pass rate could be as high as 77.3%"

**Why does it equal the observed rate?**
- Some bootstrap test sets had perfect TPR/TNR (judge looked amazing!)
- Some bootstrap unlabeled sets had more passes (lucky sample)
- When judge is "perfect", no correction needed → corrected ≈ observed

**Example optimistic bootstrap**:
```python
TPR* = 1.0   # Judge caught everything in this sample
TNR* = 0.83  # Judge had high TNR in this sample
p_obs* = 0.780  # More passes in this unlabeled sample

θ̂* = (0.780 + 0.83 - 1) / (1.0 + 0.83 - 1)
   = 0.610 / 0.83
   = 0.735

# Even more optimistic samples reach 0.773 (observed rate)
```

When the upper bound equals the observed rate, it means: "If the judge were perfect, the true rate could be what we observed."

---

## Why Your Confidence Interval is WIDE

Your 95% CI: **[54.6%, 77.3%]** - that's a **22.7 percentage point spread!**

This is **very wide** and indicates **high uncertainty**. Why?

### Source 1: Small Test Set (46 examples)

With only 46 test examples:
- TPR estimate is unstable (only 34 PASS examples)
- TNR estimate is VERY unstable (only 12 FAIL examples!)
- Bootstrap resamples vary wildly

**Example**:
```
Original test set: 34 PASS, 12 FAIL
Bootstrap sample 1: 38 PASS, 8 FAIL (different mix!)
Bootstrap sample 2: 30 PASS, 16 FAIL (very different!)

→ TPR and TNR swing wildly
→ Wide confidence interval
```

**Fix**: Label more test data (aim for 100-200+ examples)

---

### Source 2: Imbalanced Test Set

Your test set breakdown (from earlier analysis):
- **34 PASS** (74%) - reasonable sample
- **12 FAIL** (26%) - very small sample!

With only 12 FAIL examples:
- TNR = 9/12 = 0.75 (based on just 9 correct)
- Adding or removing 1 example changes TNR by 8%!
- Bootstrap creates huge variability

**Example variation**:
```
Bootstrap 1: 10 FAIL examples → TNR = 7/10 = 0.70
Bootstrap 2: 14 FAIL examples → TNR = 11/14 = 0.79
Bootstrap 3: 11 FAIL examples → TNR = 9/11 = 0.82

→ TNR ranges from 0.70 to 0.82 (huge swing!)
→ Wide confidence interval
```

**Fix**: Ensure balanced test set (~50/50 PASS/FAIL)

---

### Source 3: Low Judge Performance (TNR = 0.75)

Your judge's TNR = 0.75 means it misses 25% of violations.

**Problem**: Low TNR makes the correction formula sensitive to small changes:

```python
# Original
θ̂ = (0.773 + 0.75 - 1) / (1.0 + 0.75 - 1)
  = 0.523 / 0.75
  = 0.697

# If TNR drops to 0.70 in a bootstrap sample:
θ̂* = (0.773 + 0.70 - 1) / (1.0 + 0.70 - 1)
   = 0.473 / 0.70
   = 0.676  (2 percentage points lower!)

# If TNR rises to 0.80 in a bootstrap sample:
θ̂* = (0.773 + 0.80 - 1) / (1.0 + 0.80 - 1)
   = 0.573 / 0.80
   = 0.716  (2 percentage points higher!)
```

Small changes in TNR → large changes in corrected rate → wide CI

**Fix**: Improve judge's TNR (better prompt, more examples, better model)

---

### Source 4: Finite Unlabeled Sample (2400 traces)

Even with 2400 traces, there's natural sampling variation:

```
Bootstrap 1: p_obs* = 0.770 (1848 pass)
Bootstrap 2: p_obs* = 0.776 (1862 pass)
Bootstrap 3: p_obs* = 0.768 (1843 pass)
```

This adds additional uncertainty (though less than test set issues).

**Fix**: Evaluate more production data (already pretty good at 2400)

---

## The Complete judgy Code Flow

From `6_run_full_evaluation.py`:

```python
from judgy import estimate_success_rate

# STEP 1: Load test set performance (from step 5)
test_labels = [1, 0, 1, 1, 0, ...]  # Ground truth (46 examples)
test_preds = [1, 0, 1, 0, 0, ...]   # Judge predictions (46 examples)

# STEP 2: Run judge on unlabeled production data
unlabeled_preds = [1, 1, 0, 1, ...]  # Judge on 2400 recipes

# STEP 3: Call judgy (does BOTH Rogan-Gladen + Bootstrap)
theta_hat, lower_bound, upper_bound = estimate_success_rate(
    test_labels=test_labels,        # Ground truth from test set
    test_preds=test_preds,          # Judge predictions on test set
    unlabeled_preds=unlabeled_preds, # Judge predictions on unlabeled data
    bootstrap_iterations=20000,      # Default: 20,000 iterations
    confidence_level=0.95            # Default: 95% confidence interval
)

# STEP 4: Results
print(f"Point estimate: {theta_hat:.3f}")        # 0.697
print(f"95% CI: [{lower_bound:.3f}, {upper_bound:.3f}]")  # [0.546, 0.773]
```

---

## What judgy Does Internally

```python
def estimate_success_rate(test_labels, test_preds, unlabeled_preds,
                         bootstrap_iterations=20000, confidence_level=0.95):
    """
    Estimate true success rate with bias correction and confidence intervals.
    """

    # === STEP 1: POINT ESTIMATE (Rogan-Gladen on original data) ===

    # Calculate TPR and TNR from test set
    tp = sum((true == 1) & (pred == 1) for true, pred in zip(test_labels, test_preds))
    fn = sum((true == 1) & (pred == 0) for true, pred in zip(test_labels, test_preds))
    tn = sum((true == 0) & (pred == 0) for true, pred in zip(test_labels, test_preds))
    fp = sum((true == 0) & (pred == 1) for true, pred in zip(test_labels, test_preds))

    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)

    # Calculate observed pass rate on unlabeled data
    p_obs = mean(unlabeled_preds)

    # Apply Rogan-Gladen correction
    theta_hat = (p_obs + TNR - 1) / (TPR + TNR - 1)

    # === STEP 2: CONFIDENCE INTERVAL (Bootstrap) ===

    bootstrap_thetas = []

    for i in range(bootstrap_iterations):
        # Resample test set (with replacement)
        test_indices = random.choices(range(len(test_labels)), k=len(test_labels))
        test_labels_boot = [test_labels[i] for i in test_indices]
        test_preds_boot = [test_preds[i] for i in test_indices]

        # Calculate TPR*, TNR* from bootstrap test set
        tp_boot = sum((true == 1) & (pred == 1) for true, pred in zip(test_labels_boot, test_preds_boot))
        fn_boot = sum((true == 1) & (pred == 0) for true, pred in zip(test_labels_boot, test_preds_boot))
        tn_boot = sum((true == 0) & (pred == 0) for true, pred in zip(test_labels_boot, test_preds_boot))
        fp_boot = sum((true == 0) & (pred == 1) for true, pred in zip(test_labels_boot, test_preds_boot))

        TPR_boot = tp_boot / (tp_boot + fn_boot) if (tp_boot + fn_boot) > 0 else 1.0
        TNR_boot = tn_boot / (tn_boot + fp_boot) if (tn_boot + fp_boot) > 0 else 1.0

        # Resample unlabeled data (with replacement)
        unlabeled_indices = random.choices(range(len(unlabeled_preds)), k=len(unlabeled_preds))
        unlabeled_preds_boot = [unlabeled_preds[i] for i in unlabeled_indices]

        # Calculate p_obs* from bootstrap unlabeled set
        p_obs_boot = mean(unlabeled_preds_boot)

        # Apply Rogan-Gladen with bootstrap values
        theta_boot = (p_obs_boot + TNR_boot - 1) / (TPR_boot + TNR_boot - 1)

        # Store bootstrap estimate
        bootstrap_thetas.append(theta_boot)

    # Calculate confidence interval from bootstrap distribution
    alpha = 1 - confidence_level  # 0.05 for 95% CI
    lower_bound = percentile(bootstrap_thetas, alpha/2 * 100)      # 2.5th percentile
    upper_bound = percentile(bootstrap_thetas, (1 - alpha/2) * 100) # 97.5th percentile

    return theta_hat, lower_bound, upper_bound
```

---

## Key Insights

### 1. Two Sources of Uncertainty

judgy bootstraps **BOTH**:
- **Test set** → Captures uncertainty in TPR/TNR estimates
- **Unlabeled data** → Captures sampling variation in production data

This gives you a realistic confidence interval that accounts for all uncertainty sources.

### 2. Point Estimate ≠ Bootstrap Average ⚠️ IMPORTANT!

**Common Misconception**: "The point estimate (0.697) comes from averaging 20,000 bootstrap values"

❌ **WRONG**:
```python
# Don't do this:
bootstrap_thetas = [θ̂_1*, θ̂_2*, ..., θ̂_20000*]
theta_hat = mean(bootstrap_thetas)  # ← NOT how judgy works!
```

✅ **CORRECT**:
```python
# Point estimate: calculated ONCE from ORIGINAL data
theta_hat = (p_obs + TNR - 1) / (TPR + TNR - 1)  # = 0.697

# Bootstrap: only for confidence interval
bootstrap_thetas = [θ̂_1*, θ̂_2*, ..., θ̂_20000*]
lower_bound = percentile(bootstrap_thetas, 2.5)   # = 0.546
upper_bound = percentile(bootstrap_thetas, 97.5)  # = 0.773

# Note: median(bootstrap_thetas) ≈ 0.697 (close to point estimate)
# But we use the ORIGINAL calculation, not the bootstrap median!
```

**Why this matters**:
- Point estimate uses **original data** (unbiased, clear interpretation)
- Bootstrap shows **uncertainty** around that estimate
- We don't "improve" the point estimate with bootstrap - we quantify its uncertainty!

### 3. Percentiles vs. Average

- **Percentiles** give you the spread (2.5th to 97.5th = 95% of values)
- **Average** only gives you the center (loses information about uncertainty)

### 4. Wide CI = Low Confidence

Your wide CI [54.6%, 77.3%] means:
- Small test set (46 examples)
- Imbalanced (only 12 FAIL examples)
- Low judge performance (TNR = 0.75)

**Solution**: Label more balanced test data, improve judge's TNR!

---

## Comparison: With vs. Without judgy

### Without judgy (Naive Approach)

```python
# Just use what the judge says
raw_pass_rate = 0.773  # 77.3%
```

**Problems**:
- ❌ Ignores judge bias (TNR = 0.75)
- ❌ Overcounts by ~7.6 percentage points
- ❌ No uncertainty quantification
- ❌ Could be serving violations to ~182 users!

### With judgy (Bias-Corrected Approach)

```python
# Correct for judge bias
corrected_pass_rate = 0.697  # 69.7%
confidence_interval = [0.546, 0.773]  # 95% CI
```

**Benefits**:
- ✅ Accounts for judge errors (TPR/TNR)
- ✅ More accurate estimate (69.7% vs 77.3%)
- ✅ Quantifies uncertainty (wide CI → need more data)
- ✅ Better business decisions (know the true quality)

---

## Practical Recommendations

### 1. Improve Your Test Set

**Current**: 46 examples (34 PASS, 12 FAIL)

**Target**: 100-200 examples (50/50 split)

**Why**:
- Narrower confidence intervals
- More stable TPR/TNR estimates
- More reliable corrections

### 2. Improve Your Judge's TNR

**Current**: TNR = 0.75 (misses 25% of violations)

**Target**: TNR ≥ 0.85 (miss <15% of violations)

**How**:
- Add more FAIL examples to few-shot prompt
- Make criteria more explicit about violations
- Use a better/larger model
- Add reasoning chains to judge prompt

### 3. Monitor Confidence Interval Width

**Rule of thumb**:
- CI width < 10 points → Good (high confidence)
- CI width 10-20 points → Acceptable (medium confidence)
- CI width > 20 points → Poor (low confidence) ← **You are here!**

**Your CI width**: 77.3% - 54.6% = 22.7 points (too wide!)

### 4. Understand Business Impact

**Corrected**: 69.7% pass (1673/2400 recipes)
**Observed**: 77.3% pass (1855/2400 recipes)
**Difference**: ~182 recipes (7.6%) are false positives

**Question**: Can your business tolerate serving dietary violations to 182 users? If not, improve TNR!

---

## Summary: The Complete Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JUDGY WORKFLOW                               │
└─────────────────────────────────────────────────────────────────┘

INPUTS:
├─ test_labels: [1, 0, 1, ...] (46 ground truth labels)
├─ test_preds: [1, 0, 1, ...]  (46 judge predictions)
└─ unlabeled_preds: [1, 1, 0, ...] (2400 judge predictions)

STEP 1: POINT ESTIMATE (Rogan-Gladen) ⚠️ ORIGINAL DATA ONLY
├─ Calculate from ORIGINAL data (NO resampling):
│  ├─ TPR = 1.0, TNR = 0.75 (from original test set)
│  ├─ p_obs = 0.773 (from original unlabeled)
│  └─ θ̂ = (0.773 + 0.75 - 1) / (1.0 + 0.75 - 1) = 0.697
│
└─ Point Estimate: 0.697 (69.7%) ✓ DONE, calculated ONCE

STEP 2: CONFIDENCE INTERVAL (Bootstrap 20,000 times) ⚠️ RESAMPLED DATA
├─ For each iteration i:
│  ├─ Resample test set (with replacement) → TPR_i*, TNR_i*
│  ├─ Resample unlabeled (with replacement) → p_obs_i*
│  ├─ Calculate θ̂_i* using Rogan-Gladen formula
│  └─ Store θ̂_i* (one of 20,000 values)
│
├─ After 20,000 iterations, we have:
│  └─ bootstrap_thetas = [θ̂_1*, θ̂_2*, ..., θ̂_20000*]
│
├─ Sort 20,000 values: [0.546, ..., 0.697, ..., 0.773]
│  (Note: median ≈ 0.697, close to point estimate!)
│
└─ Extract percentiles:
   ├─ 2.5th percentile → 0.546 (lower bound)
   └─ 97.5th percentile → 0.773 (upper bound)

OUTPUTS:
├─ Corrected success rate: 0.697 (from Step 1, original data)
└─ 95% Confidence Interval: [0.546, 0.773] (from Step 2, bootstrap)

INTERPRETATION:
├─ True pass rate: ~69.7% (not 77.3%!)
├─ Judge overcounted by 7.6 points (too lenient)
├─ 95% confident true rate is between 54.6% and 77.3%
└─ Wide CI → Need more test data and better judge!

KEY INSIGHT:
┌──────────────────────────────────────────────────────────────┐
│ ⚠️ CRITICAL: Point Estimate vs. Confidence Interval          │
│                                                               │
│ Point Estimate (0.697):                                       │
│   - Calculated ONCE from ORIGINAL data                       │
│   - NOT from averaging 20,000 bootstrap values               │
│   - Best single guess of true pass rate                      │
│                                                               │
│ Confidence Interval [0.546, 0.773]:                          │
│   - Calculated from 20,000 BOOTSTRAP iterations              │
│   - Uses percentiles (2.5th and 97.5th), NOT average        │
│   - Tells you the RANGE of uncertainty                       │
│                                                               │
│ Bootstrap median ≈ 0.697 (close to point estimate)          │
│ But we use ORIGINAL estimate, not bootstrap average!         │
└──────────────────────────────────────────────────────────────┘
```

---

## Further Reading

- **judgy GitHub**: https://github.com/ai-evals-course/judgy
- **Rogan-Gladen Paper** (1978): "Estimating prevalence from results of a screening test"
- **Bootstrap Method**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- **TPR/TNR Deep Dive**: See `tpr-tnr.md` in this folder

---

**Key Takeaway**: judgy transforms your imperfect judge's biased predictions into an unbiased estimate with quantified uncertainty. The wider your confidence interval, the more you need to improve your test set and judge! 🎯
