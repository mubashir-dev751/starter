# Prioritizing Content Refresh Decisions with Machine Learning

## Abstract
We investigate whether machine learning can outperform a heuristic baseline for prioritizing content refresh candidates in a large, multi‑client search dataset. Using 30,000 anonymized content performance records, we build a ranking model that predicts the probability that a page is declining in organic traffic. Our grouped‑split evaluation shows that a Random Forest model achieves a ROC‑AUC of 0.601 and a Precision@50 of 0.640, compared to 0.490 and 0.480 for the baseline heuristic. The results indicate that ML‑based ranking can provide directional decision‑support to content teams, though the model’s performance is limited by overlapping time windows in the features and label. We recommend the model be used only to surface pages for human review, never for automated content deletion or redirects.

## 1. Introduction / Problem Statement
Content teams face a common challenge: a large backlog of pages that may need updating, but limited editorial capacity to review them all. Static rules often flag every page that shows any traffic decline, producing an unmanageable queue where high‑value opportunities are buried under noise. The goal of this work is to build a *prioritized queue* that helps human editors decide **which declining pages to review first**.

We frame this as a ranking problem: given a set of observable signals (impressions, average position, staleness, engagement), can we assign a score to each page that correlates with the likelihood that it is currently experiencing a traffic decline? A machine learning model can capture non‑linear interactions and client‑specific patterns that would be impractical to encode in a fixed rule.

Our primary question is: **Can a Random Forest model trained on historical content performance data rank true declining pages higher than a simple heuristic baseline, as measured by Precision@50?** The answer to this question provides decision‑support for human editors, not a replacement for their judgment.

## 2. Data
We use the `content_refresh_anonymized.csv` dataset provided as part of the FlyRank ML Internship. The dataset contains **30,000 rows**, each representing one content item’s 90‑day performance window across 32 anonymized clients. The data includes both Search Console (GSC) and Google Analytics 4 (GA4) metrics, but for this analysis we rely primarily on the following fields:

- `impressions_90d`, `clicks_90d`, `ctr` – search performance
- `avg_position` – average position in search results (0 indicates missing data)
- `days_since_last_update` – staleness
- `word_count` – content length (with some missing values)
- `engagement_rate`, `scroll_rate`, `ai_traffic_pct` – user engagement signals
- `content_type` – categorical content type
- `trend_direction` – ground truth label for decline (used only for evaluation)

We exclude `trend_pct` and `trend_direction` from the feature set to avoid target leakage; they are used solely to define the evaluation label and measure model performance.

**Data exclusions:** No rows were removed from the dataset. Missing values (`avg_position = 0`, missing `word_count`) were handled explicitly during preprocessing with median imputation and missing‑value flags.

## 3. Methodology
### 3.1 Label Definition
The target variable `is_declining_label` is derived from `trend_direction`: we assign `1` if the value is `'down'`, otherwise `0`. This is an observed outcome based on the dataset’s internal trend classification, not a rule we defined ourselves. We acknowledge that this label may be noisy and is not a perfect proxy for “needs refresh,” but it serves as a reasonable signal for evaluation.

### 3.2 Feature Engineering
We engineer the following features:
- `has_no_position_data` – binary flag for `avg_position == 0`
- `clean_avg_position` – copy of `avg_position` with 0 replaced by `NaN` (to be imputed)
- `has_missing_word_count` – binary flag for missing `word_count`
- `clean_word_count` – `word_count` with missing values filled by median

The final feature set includes: `impressions_90d`, `clicks_90d`, `ctr`, `clean_avg_position`, `has_no_position_data`, `days_since_last_update`, `clean_word_count`, `has_missing_word_count`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, and `content_type` (one‑hot encoded).

### 3.3 Validation Design
We use a **grouped split by `client_id`** to ensure that no client appears in both training and test sets. This prevents the model from learning client‑specific quirks that would not generalize to unseen clients. We use an 80/20 split with `GroupShuffleSplit` (random seed 42).

We compare three approaches:
1. **Heuristic Baseline** – a rule that combines staleness (`days_since_last_update >= 180`), striking distance (`avg_position` between 11 and 30), and log impressions.
2. **Logistic Regression** – linear model with standard scaling and one‑hot encoding.
3. **Random Forest** – 100 trees, max depth 8, using the same preprocessing pipeline.

All models are evaluated on the same held‑out test set using ROC‑AUC, Precision@20, Precision@50, and Precision@100.

### 3.4 Leakage Checks
We performed a thorough leakage audit:
- **Target leakage:** `trend_direction` and `trend_pct` are excluded from features.
- **Time‑window overlap:** The features `impressions_90d` and `clicks_90d` span a 90‑day window that may overlap with the 30‑day window used to compute the decline label. This overlap artificially inflates model performance because the model can partially “see” the outcome during training. We explicitly flag this as a limitation; in a production setting, features must be strictly time‑boxed to precede the label window.
- **Missing data:** `avg_position = 0` is treated as missing, not as a rank of zero.

## 4. Results
### 4.1 Model Comparison on Grouped Split
| Method | ROC‑AUC | Precision@20 | Precision@50 | Precision@100 |
|--------|---------|--------------|--------------|---------------|
| Test set base rate | 0.500 | 0.511 | 0.511 | 0.511 |
| Heuristic baseline | 0.490 | 0.450 | 0.480 | 0.430 |
| Logistic Regression | 0.563 | 0.600 | 0.620 | 0.570 |
| Random Forest (max_depth=8) | **0.601** | 0.450 | **0.640** | **0.640** |

The Random Forest model outperforms the heuristic baseline and logistic regression in ROC‑AUC and Precision@50/100. However, the improvement is modest, and the model’s ROC‑AUC of 0.601 indicates limited discriminative power.

![Model comparison chart](figures/model_comparison.png)  
*Figure 1: Precision@k comparison across methods on the grouped test split.*

### 4.2 Feature Importance
Permutation importance on the test set reveals that `impressions_90d` is the most influential feature, followed by `clicks_90d`, `scroll_rate`, and `clean_avg_position`. This suggests that high‑traffic pages that are slipping in engagement are most likely to be flagged by the model.

![Feature importance](figures/feature_importance.png)  
*Figure 2: Permutation importance (decrease in ROC‑AUC) for top features.*

### 4.3 Error Analysis
**False positives** (predicted decline, actual stable) tend to be old pages with weak CTR on page 2. The model over‑penalizes staleness and high position even when traffic is stable. **False negatives** (predicted stable, actual decline) are often pages with recent updates and good rankings but a sharp drop – possibly due to a content change that hurt relevance. The model’s reliance on aggregate features misses these edge cases.

## 5. Limitations & Honest Framing
This work must be interpreted carefully:

- **No causal claims:** The model identifies associations, not causation. We cannot say that refreshing a high‑scoring page will reverse its decline.
- **Time‑window leakage:** The 90‑day features overlap with the 30‑day label window, inflating performance metrics. In a production deployment, we would compute features strictly from periods before the label window.
- **Weak overall signal:** The ROC‑AUC of 0.601 is only slightly above random chance. The model provides a *ranking* improvement over the baseline but is not a high‑confidence classifier.
- **Label noise:** The `trend_direction` label may be derived from data that itself contains noise (e.g., seasonal effects, algorithm changes). The model cannot distinguish between a page that is truly stale and one that is temporarily down.
- **Generalization:** The model is trained on 32 clients and may not transfer to clients with different content types or search patterns.

We therefore position this model as **directional decision‑support** for content teams: it surfaces pages that *share characteristics* with historically declining pages, but a human must always verify the context and make the final call.

## 6. Ranked Recommendations (Action Playbook)
Based on the model output and our understanding of its limits, we recommend the following actions, in order of priority:

1. **Integrate the ML queue into the content review workflow.** Use the model’s probability scores to rank pages for human review, focusing on the top 20% of the queue each week.
2. **Use reason codes to guide reviewers.** For each flagged page, automatically assign a reason code based on the dominant feature (e.g., “Stale Content”, “Low Engagement”, “Missing Position Data”). This helps the editor quickly understand *why* the page was flagged.
3. **Implement a mandatory human audit before any action.** The model must never trigger automated content deletion, redirects, or unpublishing. A human must confirm that the decline is real and that a refresh is the appropriate response.
4. **Fix the time‑window leakage in the feature pipeline.** To obtain honest performance metrics and avoid over‑optimistic expectations, recompute features using only data that precedes the label window. This will likely reduce the apparent AUC but give a more realistic view of model usefulness.
5. **Monitor model drift and data quality.** Track the distribution of `has_no_position_data` and the model’s score distribution. A sudden increase in missing data or a shift in score distribution indicates that the input pipeline has changed and the model needs retraining.
6. **Do not use model scores for individual performance evaluation.** The scores reflect page characteristics, not the quality of the writers or SEOs.

## 7. Reproducibility
All code and notebooks used in this work are available in the repository (see `work/notebooks/`). The main notebooks are:

- `ML-02` – research question and data exploration
- `ML-03` – ML task framing
- `ML-04` – data contract and leakage demonstration
- `ML-07` – baseline action score and top‑20 review
- `ML-08` – capstone modeling (split, train, compare)
- `ML-09` – validation and claim audit
- `ML-10` – content action playbook and exports

To reproduce the analysis:

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order. The final outputs (ranked queue, figures) are saved to `work/outputs/` and `work/figures/`.

The dataset `content_refresh_anonymized.csv` is available in the repository or via the provided raw URL in the notebooks.

## 8. Acknowledgments & Data Credit
Built on the [FlyRank ML Internship dataset](https://flyrank.ai). This dataset was provided as part of the FlyRank ML Internship program. We thank the program organizers for making this data available for learning purposes.
