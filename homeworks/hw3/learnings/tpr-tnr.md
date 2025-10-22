<!--
Document Type: Learning Notes
Purpose: Understand how TPR/TNR metrics enable judgy bias correction on unlabeled data
Context: Created while learning HW3 LLM-as-Judge evaluation workflow
Key Topics: True Positive Rate, True Negative Rate, bias correction, confusion matrix, judgy package
Target Use: Reference guide for understanding judge performance metrics and correction math
-->

# Understanding TPR/TNR and Judgy Bias Correction

## The Core Problem

Your LLM judge isn't perfect. When you apply it to unlabeled production data and it says "90% of recipes pass", that's **biased** by the judge's error patterns. The real pass rate might be higher or lower!

**judgy's solution**: Use the judge's known error rates (TPR/TNR from test set) to mathematically correct the observed rate and estimate the **true** pass rate.

---

## What Are TPR and TNR?

### TPR (True Positive Rate) = Sensitivity = Recall for PASS class

**Question it answers**: "Of all recipes that *should* PASS, what % did the judge correctly identify?"

```
TPR = TP / (TP + FN)
    = True Positives / All Actual Positives
    = Correctly identified PASSes / All recipes that should PASS
```

**Example**: If 100 recipes should pass dietary restrictions:
- Judge correctly identifies 75 → **TPR = 0.75 (75%)**
- Judge incorrectly rejects 25 → FN = 25

---

### TNR (True Negative Rate) = Specificity = Recall for FAIL class

**Question it answers**: "Of all recipes that *should* FAIL, what % did the judge correctly identify?"

```
TNR = TN / (TN + FP)
    = True Negatives / All Actual Negatives
    = Correctly identified FAILs / All recipes that should FAIL
```

**Example**: If 100 recipes should fail (violate dietary restrictions):
- Judge correctly identifies 80 → **TNR = 0.80 (80%)**
- Judge incorrectly passes 20 → FP = 20

---

## How Judgy Corrects Bias

### The Math Behind It

When you run your judge on unlabeled data, you observe some pass rate. But this is biased! Here's how judgy corrects it:

```python
# What you observe on unlabeled data
observed_pass_rate = 0.90  # Judge says 90% pass

# Judge's error patterns from test set
TPR = 0.75  # Catches 75% of actual passes
TNR = 0.80  # Catches 80% of actual fails

# judgy's correction formula (simplified):
# The judge's prediction is a "weighted mixture" of true PASSes and FAILs
# We solve for the true pass rate that would produce the observed rate

corrected_pass_rate = judgy.correct(
    observed_rate=0.90,
    tpr=0.75,
    tnr=0.80
)
# Result: corrected_pass_rate ≈ 0.875 (87.5%)
```

**Intuition**: The judge has a 25% miss rate on PASSes and 20% miss rate on FAILs. When it says 90% pass, some of those are correct (true PASSes it caught) and some are mistakes (FAILs it missed). judgy mathematically untangles this!

---

## Scenario Analysis: Different TPR/TNR Combinations

Let's analyze what different TPR/TNR combinations mean and how they affect correction.

**Setup**: Your judge evaluates 1000 unlabeled recipes and says **850 PASS, 150 FAIL** (85% observed pass rate).

---

### 🟢 Scenario 1: High TPR (0.95) + High TNR (0.90)

**What this means**: Your judge is excellent! It catches 95% of passes and 90% of fails.

```
Test Set Results:
- TPR = 0.95 (misses only 5% of passes)
- TNR = 0.90 (misses only 10% of fails)

Observed on Unlabeled Data:
- 850 recipes judged as PASS
- 150 recipes judged as FAIL
- Observed pass rate: 85%

Judgy Correction:
- The judge is highly accurate, so observed ≈ true rate
- Corrected pass rate: ~84-86%
- Confidence interval: [82%, 88%] (narrow!)
```

**Interpretation**:
- ✅ Trust your judge! The correction is minimal
- ✅ Narrow confidence interval means high certainty
- ✅ Your true pass rate is very close to what you observed

**Example Impact**: If judge says 85% pass, the real rate is probably 84-86% (very close!).

---

### 🟡 Scenario 2: High TPR (0.95) + Low TNR (0.60)

**What this means**: Judge is **too lenient** - it catches passes well but misses many fails!

```
Test Set Results:
- TPR = 0.95 (catches 95% of passes - excellent!)
- TNR = 0.60 (catches only 60% of fails - misses 40%!)

Observed on Unlabeled Data:
- 850 recipes judged as PASS
- 150 recipes judged as FAIL
- Observed pass rate: 85%

Judgy Correction:
- The judge lets too many fails slip through as passes
- Many of those 850 "passes" are actually fails the judge missed
- Corrected pass rate: ~68-72%
- Confidence interval: [62%, 78%] (wider)
```

**Interpretation**:
- ⚠️ Your judge is dangerously optimistic!
- ⚠️ When it says 85% pass, the real rate is only ~70%
- ⚠️ You're serving dietary violations to ~15% of users!

**Real-World Example**:
Imagine a vegan user asking for recipes. Judge says 850/1000 recipes are vegan-compliant. But TNR=0.60 means the judge missed 40% of non-vegan recipes (cheese, eggs, etc.).

After correction: Only ~700 recipes are truly vegan. The other ~150 "passes" are false positives (recipes with animal products that slipped through).

**Business Impact**: High churn risk! Users with allergies/restrictions get wrong recipes.

---

### 🔴 Scenario 3: Low TPR (0.65) + High TNR (0.90)

**What this means**: Judge is **too strict** - it catches fails well but rejects many valid recipes!

```
Test Set Results:
- TPR = 0.65 (catches only 65% of passes - misses 35%!)
- TNR = 0.90 (catches 90% of fails - excellent!)

Observed on Unlabeled Data:
- 850 recipes judged as PASS
- 150 recipes judged as FAIL
- Observed pass rate: 85%

Judgy Correction:
- The judge rejects too many valid recipes
- Many actual passes were incorrectly marked as fail
- Corrected pass rate: ~92-95%
- Confidence interval: [88%, 98%] (wider)
```

**Interpretation**:
- 😤 Your judge is overly cautious and pessimistic
- 😤 When it says 85% pass, the real rate is ~93%
- 😤 You're incorrectly rejecting ~8% of good recipes

**Real-World Example**:
Judge says only 850/1000 recipes comply with dietary restrictions. But TPR=0.65 means it incorrectly rejected 35% of valid recipes.

After correction: Actually ~930 recipes are compliant! The judge wrongly failed ~80 perfectly good recipes.

**Business Impact**: Poor user experience. Users ask for vegan recipes, judge says "no good options" when there actually are. Frustration and churn.

---

### 🔵 Scenario 4: Low TPR (0.70) + Low TNR (0.65)

**What this means**: Judge is **unreliable** - it makes lots of errors in both directions!

```
Test Set Results:
- TPR = 0.70 (misses 30% of passes)
- TNR = 0.65 (misses 35% of fails)

Observed on Unlabeled Data:
- 850 recipes judged as PASS
- 150 recipes judged as FAIL
- Observed pass rate: 85%

Judgy Correction:
- The judge makes many errors both ways
- Hard to trust any individual prediction
- Corrected pass rate: ~78-82%
- Confidence interval: [70%, 90%] (VERY wide!)
```

**Interpretation**:
- ❌ Your judge is unreliable and noisy
- ❌ Wide confidence interval = high uncertainty
- ❌ You can't trust aggregate stats OR individual predictions
- ❌ Time to improve your judge prompt!

**What to do**: Go back to step 4 (develop_judge.py) and:
1. Add more/better few-shot examples
2. Make criteria more explicit
3. Add more diverse training examples
4. Consider using a better/larger model

---

## Visual Summary: TPR/TNR Combinations

```
┌─────────────────────────────────────────────────────────────────┐
│                    TPR (Catch PASS rate)                        │
│              Low (≤0.70)              High (≥0.85)              │
├──────────────────────────┬──────────────────────────────────────┤
│ High TNR                 │ 🔴 TOO STRICT             │ 🟢 EXCELLENT      │
│ (≥0.85)                  │ False rejects many good   │ Trust your judge! │
│ Catch FAIL rate          │ Correction: UP            │ Minimal correction│
│                          │ Impact: Poor UX           │                   │
├──────────────────────────┼──────────────────────────┼───────────────────┤
│ Low TNR                  │ 🔵 UNRELIABLE             │ 🟡 TOO LENIENT    │
│ (≤0.70)                  │ Errors in both directions │ Misses violations │
│ Catch FAIL rate          │ Correction: Unpredictable │ Correction: DOWN  │
│                          │ Impact: Can't trust       │ Impact: Safety!   │
└──────────────────────────┴──────────────────────────┴───────────────────┘
```

---

## Detailed Example: Correction Math

Let's walk through how judgy actually computes the correction.

### Test Set Evaluation (Step 5)

```
Test set: 100 recipes with ground truth labels
- 60 should PASS (vegan recipes that are compliant)
- 40 should FAIL (vegan recipes with animal products)

Judge predictions:
- Of 60 PASSes: Judge correctly identifies 45 → TPR = 45/60 = 0.75
                Judge incorrectly rejects 15 → FN = 15

- Of 40 FAILs: Judge correctly identifies 30 → TNR = 30/40 = 0.75
               Judge incorrectly passes 10 → FP = 10

Confusion Matrix:
                Actual PASS    Actual FAIL
Judge PASS          45            10          = 55 total "pass" predictions
Judge FAIL          15            30          = 45 total "fail" predictions
               ─────────────────────────────
                    60            40          = 100 total
```

### Production Evaluation (Step 6)

```
Unlabeled production data: 1000 recipes (no ground truth!)

Judge predictions:
- 700 recipes marked as PASS
- 300 recipes marked as FAIL
- Observed pass rate: 70%

But wait! We know the judge has error patterns:
- TPR = 0.75 means it misses 25% of actual passes
- TNR = 0.75 means it misses 25% of actual fails
```

### Judgy's Correction Logic

The judge's observed rate is a **mixture**:

```python
# The 700 "PASS" predictions come from two sources:
# 1. True passes that the judge caught (TPR × true_pass_count)
# 2. False positives (fails that slipped through as passes)

observed_passes = (TPR × true_passes) + (FP_rate × true_fails)
                = (TPR × true_passes) + ((1 - TNR) × true_fails)

# We know:
observed_passes = 700
TPR = 0.75
TNR = 0.75
true_passes + true_fails = 1000

# Solve for true_passes:
700 = (0.75 × true_passes) + (0.25 × (1000 - true_passes))
700 = 0.75 × true_passes + 250 - 0.25 × true_passes
700 = 0.50 × true_passes + 250
450 = 0.50 × true_passes
true_passes = 900

# Corrected pass rate = 900/1000 = 90%!
```

**What happened**:
- Observed: 70% pass rate
- Corrected: 90% pass rate
- **Correction: +20 percentage points!**

The judge is pessimistic (low TPR) so it rejects many valid recipes. judgy corrects this bias upward.

---

## When Correction Goes UP vs DOWN

### Correction Goes UP (Corrected > Observed)

This happens when: **Judge is too strict (low TPR)**

```
Low TPR → Judge rejects many valid recipes (false negatives)
       → Observed pass rate is artificially low
       → True pass rate is higher
       → judgy corrects UPWARD
```

**Example**: TPR=0.70, TNR=0.90, Observed=75% → Corrected=85%

---

### Correction Goes DOWN (Corrected < Observed)

This happens when: **Judge is too lenient (low TNR)**

```
Low TNR → Judge accepts many bad recipes (false positives)
       → Observed pass rate is artificially high
       → True pass rate is lower
       → judgy corrects DOWNWARD
```

**Example**: TPR=0.90, TNR=0.60, Observed=85% → Corrected=70%

---

### Correction Is Minimal (Corrected ≈ Observed)

This happens when: **Judge is accurate (high TPR and high TNR)**

```
High TPR + High TNR → Judge makes few errors
                    → Observed rate ≈ true rate
                    → judgy correction is minimal
```

**Example**: TPR=0.95, TNR=0.95, Observed=85% → Corrected=85%

---

## Confidence Intervals: How Sure Are We?

judgy also provides confidence intervals around the corrected rate:

```python
# Example output from judgy
{
    "observed_pass_rate": 0.85,
    "corrected_pass_rate": 0.92,
    "confidence_interval_95": {
        "lower_bound": 0.88,
        "upper_bound": 0.96
    }
}
```

### What This Means

**95% confidence interval [88%, 96%]**: If we repeated this experiment many times, 95% of the time the true pass rate would fall in this range.

### Factors That Affect Confidence Interval Width

1. **Sample Size (Test Set)**:
   - Small test set (20 traces) → Wide interval [70%, 95%]
   - Large test set (200 traces) → Narrow interval [87%, 93%]
   - **Fix**: Label more test data!

2. **Judge Accuracy (TPR/TNR)**:
   - Low TPR/TNR (0.60, 0.65) → Wide interval (high uncertainty)
   - High TPR/TNR (0.95, 0.95) → Narrow interval (high certainty)
   - **Fix**: Improve your judge prompt!

3. **Production Sample Size**:
   - Evaluating 100 traces → Wider interval
   - Evaluating 10,000 traces → Narrower interval
   - **Fix**: Evaluate more production data!

---

## Practical Guidelines (Research-Backed)

### The Only Hard Rule: Better Than Random

According to the `judgy` package documentation and research:

❌ **Unacceptable (Worse than random chance)**: **TPR + TNR ≤ 1.0**
- If your judge's TPR + TNR sum to 1.0 or less, it performs **worse than random guessing**
- Example: TPR=0.60, TNR=0.40 → Sum=1.0 (random chance)
- Example: TPR=0.55, TNR=0.40 → Sum=0.95 (worse than flipping a coin!)
- **Action**: Your judge is fundamentally broken. Start over.

### Research Findings on LLM Judges

From recent LLM-as-Judge research (2024-2025):

**Common Pattern**: LLM judges typically show:
- ✅ **High TPR** (>0.80): Good at identifying correct/compliant outputs
- ❌ **Low TNR** (<0.25): Poor at catching violations ("agreeableness bias")

This "agreeableness bias" means LLMs are often too lenient - they miss violations and false positives.

**Key Research Quote**:
> "Their high variance and low True Negative Rate (TNR) makes them unreliable to be used as a sole judge for benchmarking."

### Practical Thresholds (Context-Dependent!)

There is **no universal "good enough" threshold** - it depends entirely on your use case. Here's a framework:

#### Minimum for Rogan-Gladen Correction

From epidemiological research on the Rogan-Gladen method:
- **Minimum studied**: TPR ≥ 0.60 AND TNR ≥ 0.60
- Below this, corrections become mathematically unstable (produce negative values or >100%)
- **Recommendation**: Treat 0.60 as absolute floor, but aim much higher

#### Risk-Based Thresholds

**High-risk scenarios** (allergies, medical, legal compliance):
- **Required**: TPR ≥ 0.90 AND TNR ≥ 0.90
- False positives (low TNR) can cause harm (serving allergens)
- Cannot tolerate missed violations
- **Example**: Nut-free diet for severe allergy

**Medium-risk scenarios** (dietary preferences, UX quality):
- **Target**: TPR ≥ 0.75 AND TNR ≥ 0.75
- Balance between safety and usability
- Some corrections acceptable if confidence intervals reasonable
- **Example**: Vegan preference (not allergy)

**Low-risk scenarios** (recommendations, non-critical):
- **Acceptable**: TPR ≥ 0.60 AND TNR ≥ 0.60 (but aim higher!)
- Large corrections needed but tolerable
- Focus on aggregate statistics, not individual predictions
- **Example**: Recipe style preferences

### Validation Best Practices

Research recommends treating your LLM judge like any classifier:

1. **Measure on labeled test data**:
   - Compute TPR, TNR, precision, recall, F1
   - Calculate inter-rater agreement with human judges
   - Check that TPR + TNR > 1.0 (better than random)

2. **Consider the trade-offs**:
   - Is TPR or TNR more important for your use case?
   - Can you tolerate 20% correction? 40%? Know your limits.
   - Look at confidence interval width - wide = unreliable

3. **Don't use as sole judge if**:
   - TNR < 0.50 (misses >50% of violations)
   - TPR + TNR < 1.2 (barely better than random)
   - Confidence intervals span >30 percentage points

4. **Consider cost-benefit**:
   - What's the cost of false positives? (serving violations)
   - What's the cost of false negatives? (rejecting good recipes)
   - Which error is more expensive in your product?

### Warning Signs Your Judge Needs Work

🚨 **Critical Issues** (fix immediately):
- TPR + TNR ≤ 1.0 (worse than random)
- TNR < 0.25 (typical agreeableness bias - too lenient)
- Correction shifts observed rate by >40 percentage points
- Confidence interval width >50 percentage points

⚠️ **Moderate Issues** (improve if possible):
- Either metric < 0.60 (Rogan-Gladen floor)
- Large imbalance (TPR=0.95 but TNR=0.40)
- Correction shifts observed rate by 20-40 points
- Confidence interval width >30 points

✅ **Acceptable Performance** (depends on risk tolerance):
- TPR + TNR > 1.4 (solidly better than random)
- Both metrics ≥ 0.70 (for medium-risk scenarios)
- Correction shifts observed rate by <20 points
- Confidence interval width <20 points

### Which Error Is Worse?

This depends on your product!

**Low TNR is worse if**: Safety/compliance matters
- Food allergies (serving dairy to lactose-intolerant users)
- Medical restrictions (serving pork to Muslim users)
- Legal compliance (serving alcohol to minors)
- **Cost**: Lawsuits, health issues, churn

**Low TPR is worse if**: User experience matters
- Recipe discovery (rejecting good recipes frustrates users)
- Conversion (users leave if "no results found")
- Engagement (users stop asking if always rejected)
- **Cost**: Churn, poor reviews, lost revenue

**Most common**: Low TNR is worse (false positives = safety issues)

---

## Summary: The judgy Workflow

```
Step 5: 5_evaluate_judge.py
├─ Input: test_set.csv (ground truth labels)
├─ Output: judgy_test_data.json
│   └─ Contains: test_labels, test_preds arrays
│   └─ Used to compute: TPR, TNR
│
└─ Purpose: Measure judge's error patterns on held-out data

Step 6: run_full_evaluation.py
├─ Input: unlabeled production data (no ground truth!)
├─ Input: judgy_test_data.json (TPR/TNR from step 5)
├─ Process:
│   1. Judge evaluates unlabeled data → observed_pass_rate
│   2. judgy.correct(observed_rate, TPR, TNR) → corrected_pass_rate
│   3. judgy.confidence_interval() → uncertainty bounds
│
└─ Output: Bias-corrected estimate of true pass rate
```

---

## Key Takeaways (Updated with Research)

1. **TPR/TNR capture your judge's error patterns** from the test set
2. **judgy uses these patterns to correct bias** on unlabeled production data using the Rogan-Gladen method
3. **Hard requirement**: TPR + TNR > 1.0 (better than random chance) - from judgy docs
4. **Mathematical stability floor**: TPR, TNR ≥ 0.60 - from Rogan-Gladen research
5. **Common LLM failure mode**: High TPR (>0.80) but Low TNR (<0.25) - "agreeableness bias"
6. **Correction direction depends on which error dominates**:
   - Low TPR (too strict) → Correction goes UP
   - Low TNR (too lenient) → Correction goes DOWN (common for LLMs!)
7. **Confidence intervals tell you how certain to be** about the correction
8. **Acceptable thresholds are context-dependent**:
   - High-risk (allergies): TPR, TNR ≥ 0.90
   - Medium-risk (preferences): TPR, TNR ≥ 0.75
   - Low-risk (recommendations): TPR, TNR ≥ 0.60 (but aim higher!)
9. **Validation is critical**: Treat your judge like a classifier - measure TPR, TNR, F1, inter-rater agreement on labeled test data

Without judgy, you'd be flying blind on production data. With it, you can measure quality even when you don't have ground truth labels! 🎯

---

## References and Research Basis

This document is based on:

### Primary Sources

#### 1. judgy Package Documentation
- **GitHub Repository**: [ai-evals-course/judgy](https://github.com/ai-evals-course/judgy)
- **Key finding**: "This library assumes that your LLM judge performs better than random chance (TPR + TNR > 1)"
- **Method**: Uses Rogan-Gladen correction for prevalence estimation

#### 2. Rogan-Gladen Method Research

**Original Paper**:
- Rogan, W. J., & Gladen, B. (1978). "Estimating prevalence from results of a screening test"
- *American Journal of Epidemiology*, 107(1), 71-76
- [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/623091/)

**Applied Research**:
- Blaker, H. (2000). "Confidence limits for prevalence of disease adjusted for estimated sensitivity and specificity"
- *Preventive Veterinary Medicine*, 113(1), 13-22
- [ScienceDirect Link](https://www.sciencedirect.com/science/article/abs/pii/S0167587713002936)
- [PubMed Link](https://pubmed.ncbi.nlm.nih.gov/24416798/)

**Validation Study**:
- Lang, S., et al. (2020). "Comparison of Bayesian and frequentist methods for prevalence estimation under misclassification"
- *BMC Public Health*, 20(1), 1135
- [Full Text (Open Access)](https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-020-09177-4)
- **Key finding**: Sensitivity/Specificity ≥ 0.60 used in validation studies; below this produces unstable corrections

#### 3. LLM-as-Judge Research (2024-2025)

**Agreeableness Bias Study**:
- "Beyond Consensus: Mitigating the Agreeableness Bias in LLM Judge Evaluations" (2025)
- [arXiv:2510.11822](https://arxiv.org/html/2510.11822)
- **Key finding**: LLM judges typically show high TPR (>0.80) but low TNR (<0.25) - "agreeableness bias"
- **Quote**: "Their high variance and low True Negative Rate (TNR) makes them unreliable to be used as a sole judge for benchmarking"

**Comprehensive Survey**:
- "LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods"
- [arXiv:2412.05579](https://arxiv.org/html/2412.05579v2)
- **Recommendation**: Validate with labeled data using TPR, TNR, precision, recall, F1, inter-rater agreement

**Industry Best Practices**:
- MLflow: [LLM as Judge](https://mlflow.org/blog/llm-as-judge)
- Evidently AI: [LLM-as-a-judge: A Complete Guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- Hugging Face: [Using LLM-as-a-judge for Automated Evaluation](https://huggingface.co/learn/cookbook/en/llm_judge)

#### 4. Statistical Methods for Classification

**Prevalence Estimation**:
- Lang, S., et al. (2020). "Comparison of Bayesian and frequentist methods for prevalence estimation under misclassification"
- *BMC Public Health*, 20(1), 1135
- [Full Text](https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-020-09177-4)
- **Key finding**: Advanced regression frameworks reduce estimation error to ~1.2%

**Decision Criteria**:
- "A decision criterion for the application of the Rogan-Gladen estimator in prevalence studies"
- [ResearchGate](https://www.researchgate.net/publication/259008854_A_decision_criterion_for_the_application_of_the_Rogan-Gladen_estimator_in_prevalence_studies)

---

### What's Evidence-Based vs. Heuristic

#### Evidence-Based (from research)

✅ **TPR + TNR > 1.0** (better than random)
- Source: judgy package requirement
- Rationale: If your judge doesn't exceed random chance, it's useless

✅ **TPR, TNR ≥ 0.60** (Rogan-Gladen stability floor)
- Source: Lang et al. (2020) validation study
- Rationale: Below this, mathematical corrections become unstable (negative values, >100%)

✅ **Common LLM pattern: High TPR (>0.80), Low TNR (<0.25)**
- Source: "Beyond Consensus" (arXiv:2510.11822, 2025)
- Rationale: LLMs exhibit "agreeableness bias" - too lenient on violations

✅ **Validation approach: Treat as classifier, measure multiple metrics**
- Source: LLM-as-Judge survey (arXiv:2412.05579) + industry best practices
- Rationale: Research consensus on proper evaluation methodology

#### Context-Dependent Heuristics

⚠️ **Risk-based thresholds** (0.70, 0.75, 0.90)
- Source: Practical guidelines based on cost-benefit analysis
- Rationale: Not derived from research, but from domain considerations (allergies vs. preferences)

⚠️ **Correction magnitude thresholds** (20%, 40%)
- Source: Practical experience, not research-derived
- Rationale: Rule of thumb for when corrections become too large to trust

⚠️ **Confidence interval width guidelines** (20, 30, 50 points)
- Source: Practical heuristics
- Rationale: Based on what's actionable in business contexts, not statistical theory

#### Important Caveat

The "acceptable" threshold depends entirely on your specific use case:
1. **Risk tolerance**: Allergies (high-risk) vs. preferences (medium-risk) vs. recommendations (low-risk)
2. **Cost asymmetry**: Which is more expensive - false positives (serving violations) or false negatives (rejecting good recipes)?
3. **Business requirements**: Safety, UX quality, legal compliance
4. **Statistical confidence**: Width of confidence intervals you can tolerate

---

### Recommended Reading

#### Course Materials
- **Isaac Moran's AI Evals Course**: [GitHub Repository](https://github.com/ai-evals-course)
- **Hamel Husain's Blog**: [hamel.dev](https://hamel.dev/) - Practical LLM evaluation guides

#### Technical Implementation
- **judgy Source Code**: [GitHub](https://github.com/ai-evals-course/judgy) - See Rogan-Gladen implementation
- **LangWatch Blog**: [Essential LLM Evaluation Metrics](https://langwatch.ai/blog/essential-llm-evaluation-metrics-for-ai-quality-control)

#### Research Papers
- **LLM Judge Survey**: [arXiv:2412.05579](https://arxiv.org/html/2412.05579v2) - Comprehensive overview
- **Agreeableness Bias**: [arXiv:2510.11822](https://arxiv.org/html/2510.11822) - Critical finding on LLM failure modes
- **Statistical Methods**: [BMC Public Health](https://bmcpublichealth.biomedcentral.com/articles/10.1186/s12889-020-09177-4) - Rogan-Gladen validation

#### Industry Guides
- **Confident AI**: [LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- **MLflow**: [LLM as Judge Blog Post](https://mlflow.org/blog/llm-as-judge)
- **Hugging Face**: [LLM Judge Cookbook](https://huggingface.co/learn/cookbook/en/llm_judge)
