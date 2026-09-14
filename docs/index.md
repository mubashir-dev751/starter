Prioritizing Content Refresh Decisions with Machine Learning
Abstract

This study investigates whether machine learning can outperform a heuristic baseline for prioritizing content refresh candidates in a large multi-client search dataset. The work addresses a common content operations challenge: identifying which pages should be reviewed first when editorial resources are limited. Using 30,000 anonymized content-performance records, we develop a ranking model that estimates the likelihood that a page is experiencing organic traffic decline. A grouped validation strategy that holds out entire clients shows that a Random Forest model achieves a ROC-AUC of 0.601 and a Precision@50 of 0.640, outperforming a heuristic baseline that achieves 0.490 and 0.480 respectively. The results suggest that ML-based ranking can provide useful decision-support for content review workflows, although overlapping feature and label windows limit the strength of the conclusions and require careful interpretation.

1. Introduction / Problem Statement

Content teams often face a practical prioritization problem: a large backlog of pages may benefit from review, but editorial capacity is limited. Traditional rule-based approaches frequently flag every page showing signs of decline, producing large queues that make it difficult to identify the most valuable opportunities.

The objective of this project is to build a prioritized review queue that helps editors determine which pages warrant investigation first. Rather than replacing human decision-making, the goal is to improve allocation of editorial attention.

We frame the problem as a ranking task. Given observable search and engagement signals such as impressions, clicks, ranking position, content freshness, and user engagement metrics, can a model assign higher scores to pages that exhibit characteristics associated with decline?

The primary research question is:

Can a machine learning model rank true declining pages higher than a simple heuristic baseline, thereby improving content-refresh prioritization?

A positive answer would provide evidence that ML-based prioritization may improve operational efficiency compared to static rules.

2. Data

This analysis uses the content_refresh_anonymized.csv dataset provided as part of the FlyRank ML Internship program.

The dataset contains approximately 30,000 content records spanning 32 anonymized clients. Each record represents a content asset summarized across a ninety-day observation window.

The analysis uses the following variables:

impressions_90d
clicks_90d
ctr
avg_position
days_since_last_update
word_count
engagement_rate
scroll_rate
ai_traffic_pct
content_type

The dataset also contains:

trend_direction
trend_pct

These variables are excluded from model features and used only for evaluation and label construction.

Data Quality and Exclusions

No rows were removed from the dataset.

Missing values were handled as follows:

avg_position = 0 was interpreted as missing rather than a valid search position.
Missing word_count values were imputed using the median.
Missing-value indicator flags were created where appropriate.

No client names, domains, URLs, or raw search queries were used in the analysis.

3. Methodology
3.1 Label Definition

The target variable, is_declining_label, is derived from the provided trend_direction field.

Declining (down) = 1
All other values = 0

This label represents an observed outcome defined in the dataset and should not be interpreted as a direct measure of refresh necessity. A page experiencing decline may require a refresh, a technical fix, competitive analysis, or no action at all.

The label is therefore treated as a directional signal suitable for ranking evaluation.

3.2 Feature Engineering

The following engineered features were created:

Position Signals
has_no_position_data
clean_avg_position
Content Signals
clean_word_count
has_missing_word_count
Search Performance Signals
impressions_90d
clicks_90d
ctr
Engagement Signals
engagement_rate
scroll_rate
ai_traffic_pct
Freshness Signals
days_since_last_update
Content Classification
One-hot encoded content_type

The feature set intentionally excludes variables directly used to define the evaluation label.

3.3 Validation Design

A key objective of this study is estimating how well the model generalizes to unseen clients.

To achieve this, an 80/20 grouped split was created using GroupShuffleSplit, with client_id as the grouping variable and a fixed random seed of 42.

This design ensures that records from the same client cannot appear in both training and testing partitions.

Three approaches are compared:

Heuristic Baseline

A rule-based score combining:

Content staleness (days_since_last_update >= 180)
Striking-distance rankings (avg_position between 11 and 30)
Log-transformed impressions
Logistic Regression

A linear baseline using standardized numeric features and one-hot encoded categorical variables.

Random Forest

A Random Forest classifier consisting of:

100 trees
Maximum depth of 8
Identical preprocessing pipeline

Models were evaluated using:

ROC-AUC
Precision@20
Precision@50
Precision@100
3.4 Leakage Assessment

Potential sources of leakage were reviewed before model evaluation.

Target Leakage

The variables:

trend_direction
trend_pct

were excluded from all model features.

Temporal Overlap

A more subtle issue exists because the ninety-day feature windows may overlap with the period used to derive decline labels.

As a result, the model may partially observe information correlated with the outcome, potentially inflating performance metrics.

Accordingly, this project should be interpreted primarily as a ranking and prioritization exercise rather than a strict future forecasting system.

A production-grade implementation would require feature windows that terminate before the label window begins.

Missing Data Handling

Missing ranking positions were treated as unavailable data rather than valid zero-rank positions.

4. Results
4.1 Model Performance
Method	ROC-AUC	Precision@20	Precision@50	Precision@100Base Rate	0.500	0.511	0.511	0.511
Heuristic Baseline	0.490	0.450	0.480	0.430
Logistic Regression	0.563	0.600	0.620	0.570
Random Forest	0.601	0.450	0.640	0.640

The Random Forest model achieves the strongest overall ranking performance and exceeds both the heuristic baseline and Logistic Regression on ROC-AUC, Precision@50, and Precision@100.

Although the gains are statistically encouraging, the overall performance remains moderate and should not be interpreted as evidence of a highly predictive system.

Figure 1: Precision@k comparison across models.

4.2 Feature Importance

Permutation importance analysis identifies the following variables as the strongest contributors:

impressions_90d
clicks_90d
scroll_rate
clean_avg_position

These findings suggest that traffic scale, engagement patterns, and ranking visibility contribute most strongly to the model's prioritization behavior.

Figure 2: Permutation importance of top predictive features.

4.3 Error Analysis
False Positives

False positives often consist of:

Older content
Weak CTR
Lower ranking positions
Stable traffic despite appearing vulnerable

The model appears to over-weight signals related to content age and visibility.

False Negatives

False negatives frequently include:

Recently updated pages
Strong rankings
Abrupt traffic loss

These cases may reflect external factors not represented in the available feature set.

5. Limitations and Honest Framing

Several limitations should be considered when interpreting the results.

Observational Analysis

The study identifies associations rather than causal relationships.

The results do not demonstrate that refreshing a page will improve performance.

Temporal Leakage Risk

Partial overlap between feature windows and outcome windows likely inflates measured performance.

Future studies should enforce strict chronological separation.

Limited Predictive Strength

A ROC-AUC of 0.601 indicates modest discrimination.

The model is more useful for prioritization than prediction.

Label Noise

The decline label may reflect seasonality, market conditions, competitive activity, tracking variation, or algorithm changes.

Not all observed declines represent content-quality problems.

External Validity

The dataset includes only 32 anonymized clients.

Performance may differ for other industries, websites, or search environments.

Appropriate Use

The system should be used exclusively for decision support.

It should not be used to automate:

Content deletion
URL removals
Redirects
Editorial performance evaluation

Human review remains essential.

6. Ranked Recommendations

Based on the model outputs and observed limitations, the following action framework is recommended.

Priority 1: Review High-Score Candidates

Focus editorial review on pages appearing within the highest-scoring segment of the ranking queue.

These pages represent the strongest concentration of historically declining characteristics.

Priority 2: Add Explainable Reason Codes

Provide reviewers with feature-based explanation labels such as:

Stale Content
Declining Engagement
Weak Visibility
Missing Position Data

This improves adoption and reduces reviewer effort.

Priority 3: Establish a Tiered Review Framework
Priority Tier	Characteristics	Recommended ActionP1	High score, high traffic	Immediate review
P2	Medium score, stale content	Scheduled refresh review
P3	Moderate score, stable traffic	Monitor
P4	Low score, low business impact	Defer
Priority 4: Remove Temporal Overlap in Future Versions

Future iterations should construct all features from periods that occur strictly before outcome measurement windows.

This will produce more trustworthy estimates of real-world performance.

Priority 5: Monitor Model Drift

Track:

Missing-data frequency
Score distributions
Feature distributions
Precision@k over time

Significant shifts may indicate retraining is required.

7. Reproducibility

All notebooks, scripts, and outputs are maintained in the project repository.

Key notebooks include:

ML-02 Data Exploration
ML-07 Baseline Opportunity Scoring
ML-08 Model Development
ML-09 Validation and Claim Audit
ML-10 Recommendation Framework

To reproduce the project:

Clone the repository.
Install dependencies using:
Shell
1
pip install -r requirements.txt
Show more lines
Execute notebooks in numerical order.
Review generated outputs in work/outputs/.

Random seeds were fixed at 42 wherever applicable to improve reproducibility.

8. Acknowledgments and Data Credit

Built on the FlyRank ML Internship Dataset.

Data source: https://flyrank.ai/

This project was completed as part of the FlyRank ML Internship program using anonymized search-performance and engagement data. All analysis, interpretations, recommendations, and limitations presented in this paper are the author's own.
