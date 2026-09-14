# Prioritizing Content Refresh Decisions with Machine Learning

## Abstract

This study investigates whether machine learning can outperform a heuristic baseline for prioritizing content refresh candidates in a large multi-client search dataset. The work addresses a common content operations challenge: identifying which pages should be reviewed first when editorial resources are limited. Using 30,000 anonymized content-performance records, we develop a ranking model that estimates the likelihood that a page is experiencing organic traffic decline. A grouped validation strategy that holds out entire clients shows that a Random Forest model achieves a ROC-AUC of **0.601** and a Precision@50 of **0.640**, outperforming a heuristic baseline that achieves **0.490** and **0.480** respectively. The results suggest that ML-based ranking can provide useful decision-support for content review workflows, although overlapping feature and label windows limit the strength of the conclusions and require careful interpretation.

---

# 1. Introduction / Problem Statement

Content teams often face a practical prioritization problem: a large backlog of pages may benefit from review, but editorial capacity is limited. Traditional rule-based approaches frequently flag every page showing signs of decline, producing large queues that make it difficult to identify the most valuable opportunities.

The objective of this project is to build a prioritized review queue that helps editors determine which pages warrant investigation first. Rather than replacing human decision-making, the goal is to improve allocation of editorial attention.

We frame the problem as a ranking task. Given observable search and engagement signals such as impressions, clicks, ranking position, content freshness, and user engagement metrics, can a model assign higher scores to pages that exhibit characteristics associated with decline?

### Research Question

> Can a machine learning model rank true declining pages higher than a simple heuristic baseline, thereby improving content-refresh prioritization?

A positive answer would provide evidence that ML-based prioritization may improve operational efficiency compared to static rules.

---

# 2. Data

This analysis uses the `content_refresh_anonymized.csv` dataset provided as part of the FlyRank ML Internship program.

The dataset contains approximately **30,000 content records** spanning **32 anonymized clients**. Each record represents a content asset summarized across a ninety-day observation window.

### Variables Used

- `impressions_90d`
- `clicks_90d`
- `ctr`
- `avg_position`
- `days_since_last_update`
- `word_count`
- `engagement_rate`
- `scroll_rate`
- `ai_traffic_pct`
- `content_type`

### Variables Used Only for Label Construction and Evaluation

- `trend_direction`
- `trend_pct`

These variables were excluded from model features to prevent target leakage.

## Data Quality and Exclusions

No rows were removed from the dataset.

Missing values were handled as follows:

- `avg_position = 0` was interpreted as missing rather than a valid search position.
- Missing `word_count` values were imputed using the median.
- Missing-value indicator flags were created where appropriate.

No client names, domains, URLs, credentials, or raw search queries were used in the analysis.

---

# 3. Methodology

## 3.1 Label Definition

The target variable, `is_declining_label`, is derived from the provided `trend_direction` field.

- Declining (`down`) = 1
- All other values = 0

This label represents an observed outcome defined in the dataset and should not be interpreted as a direct measure of refresh necessity. A page experiencing decline may require a refresh, a technical fix, competitive analysis, or no action at all.

The label is therefore treated as a directional signal suitable for ranking evaluation.

---

## 3.2 Feature Engineering

The following engineered features were created.

### Position Signals

- `has_no_position_data`
- `clean_avg_position`

### Content Signals

- `clean_word_count`
- `has_missing_word_count`

### Search Performance Signals

- `impressions_90d`
- `clicks_90d`
- `ctr`

### Engagement Signals

- `engagement_rate`
- `scroll_rate`
- `ai_traffic_pct`

### Freshness Signals

- `days_since_last_update`

### Content Classification

- One-hot encoded `content_type`

The feature set intentionally excludes variables directly used to define the evaluation label.

---

## 3.3 Validation Design

A key objective of this study is estimating how well the model generalizes to unseen clients.

To achieve this, an 80/20 grouped split was created using `GroupShuffleSplit`, with `client_id` as the grouping variable and a fixed random seed of **42**.

This design ensures that records from the same client cannot appear in both training and testing partitions.

### Models Compared

#### 1. Heuristic Baseline

A rule-based score combining:

- Content staleness (`days_since_last_update >= 180`)
- Striking-distance rankings (`avg_position` between 11 and 30)
- Log-transformed impressions

#### 2. Logistic Regression

A linear baseline using standardized numeric features and one-hot encoded categorical variables.

#### 3. Random Forest

A Random Forest classifier consisting of:

- 100 trees
- Maximum depth = 8
- Identical preprocessing pipeline

### Evaluation Metrics

- ROC-AUC
- Precision@20
- Precision@50
- Precision@100

---

## 3.4 Leakage Assessment

Potential sources of leakage were reviewed before model evaluation.

### Target Leakage

The following variables were excluded from model features:

- `trend_direction`
- `trend_pct`

### Temporal Overlap

A more subtle issue exists because the ninety-day feature windows may overlap with the period used to derive decline labels.

As a result, the model may partially observe information correlated with the outcome, potentially inflating performance metrics.

Accordingly, this project should be interpreted primarily as a ranking and prioritization exercise rather than a strict future forecasting system.

A production-grade implementation would require feature windows that terminate before the label window begins.

### Missing Data Handling

Missing ranking positions were treated as unavailable data rather than valid zero-rank positions.

---

# 4. Results

## 4.1 Model Performance

| Method | ROC-AUC | Precision@20 | Precision@50 | Precision@100 |
|----------|----------|----------|----------|----------|
| Base Rate | 0.500 | 0.511 | 0.511 | 0.511 |
| Heuristic Baseline | 0.490 | 0.450 | 0.480 | 0.430 |
| Logistic Regression | 0.563 | 0.600 | 0.620 | 0.570 |
| Random Forest | **0.601** | 0.450 | **0.640** | **0.640** |

The Random Forest model achieves the strongest overall ranking performance and exceeds both the heuristic baseline and Logistic Regression on ROC-AUC, Precision@50, and Precision@100.

Although the gains are encouraging, the overall performance remains moderate and should not be interpreted as evidence of a highly predictive system.

> **Figure 1:** Precision@k comparison across models.

*Insert chart here.*

---

## 4.2 Feature Importance

Permutation importance analysis identifies the following variables as the strongest contributors:

1. `impressions_90d`
2. `clicks_90d`
3. `scroll_rate`
4. `clean_avg_position`

These findings suggest that traffic scale, engagement patterns, and ranking visibility contribute most strongly to the model's prioritization behavior.

> **Figure 2:** Permutation importance of top predictive features.

*Insert feature importance chart here.*

---

## 4.3 Error Analysis

### False Positives

False positives often consist of:

- Older content
- Weak CTR
- Lower ranking positions
- Stable traffic despite appearing vulnerable

The model appears to overweight signals related to content
