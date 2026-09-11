# CURRENT STATUS (read first)

- **Stage 1 (Planning) is COMPLETE and APPROVED.** The approved design is in `PLAN.md`.
- Do NOT redo Stage 1. Sections 19 and 21 below describe the original process; the first response they describe has already happened.
- Where `PLAN.md` is more specific than this file (dataset, features, project structure, metrics, outputs), follow `PLAN.md`.
- Proceed stage by stage and stop at each checkpoint I give you.

---

You are helping me build my submission for the **GitHub Community SRM (GCSRM) Recruitment 2026 – Technical Track: AI & Machine Learning (Year 2)**.

I have chosen **OPTION A: Stock Price Movement Predictor**.

The official assignment PDF has two options, and I have deliberately selected Option A because it is faster and more practical to complete within the deadline. **Do not implement Option B.**

## 1. OFFICIAL ASSIGNMENT REQUIREMENTS

### Option A: Stock Price Movement Predictor

**Track:** Time-Series Machine Learning

**Required tech stack:**

* Python
* pandas
* scikit-learn
* matplotlib
* Technical indicators may be implemented directly in pandas or using the `ta` library.
* Do NOT use unmaintained packages such as `pandas-ta`.

### Goal

Engineer temporal features without data leakage, establish rigorous naive baselines, conduct forward time-series splits, and honestly analyze market classification performance.

### Core task

Predict the **next-day direction** (`Up` or `Down`) using daily historical **OHLCV** data for any liquid equity or index.

OHLCV means:

* Open
* High
* Low
* Close
* Volume

### Required technical indicators

Calculate at least **3 technical indicators**, either directly in pandas or using the `ta` library.

### Critical requirement: NO TEMPORAL DATA LEAKAGE

The project must construct the target so that today's features are used to predict tomorrow's direction.

We need to explicitly verify this.

The notebook must print/show feature rows alongside target labels so that it is clear that there is **zero forward leakage**.

Do NOT accidentally use future prices, future indicators, future rolling statistics, or information from the test period when constructing training features.

### Required baselines

We must compare against BOTH:

1. **Persistence baseline**

   * Predict tomorrow's direction based on today's direction.
   * Essentially: tomorrow matches today.

2. **Majority Class baseline**

   * Always predict the dominant class in the training data.

Do not omit either baseline.

### Required feature comparison

We need two model versions:

1. **Raw price/volume features**

   * Use raw OHLCV-related features.

2. **Engineered technical-indicator features**

   * Add the engineered technical indicators.

The purpose is to compare whether technical feature engineering actually improves directional prediction.

### Required time-based evaluation

The data MUST be split chronologically.

Training data must always come from earlier dates than testing data.

Do NOT randomly shuffle the dataset.

If scaling is used:

* Fit the scaler ONLY on the training partition.
* Transform the test partition using the already-fitted training scaler.

Never fit preprocessing on the complete dataset before splitting.

### Required visualization

Plot **predicted vs actual directional movement across the test window**.

### Required deliverables

The final repository must contain:

1. **Jupyter Notebook**
2. **Four-way comparison table**

   * Persistence
   * Majority Class
   * Raw Features
   * Engineered Features
3. **Class balance report**
4. **Prediction plot**
5. **README.md**
6. Reproducible code
7. Public GitHub repository

The assignment also explicitly says:

* Avoid temporal data leakage.
* Verify generalization under cross-session real-world constraints.
* Submit public GitHub repository links containing reproducible code and documentation.

The submission deadline is:

**23:59 PM, 12 September 2026**

---

# 2. OUR DECISION

We discussed both options.

Option B is a real-time computer vision project involving:

* MediaPipe
* webcam data collection
* 4 gesture classes
* independent test session
* invariant geometric features
* latency/FPS benchmarking
* live deployment
* demo video

We decided **Option A is faster to implement and safer for the deadline.**

Therefore:

**DO NOT switch to Option B.**

---

# 3. PROJECT PHILOSOPHY

I am a CSE student and I know Python and basic ML, but I want this project to be technically legitimate rather than just something that produces a high accuracy number.

The project should demonstrate that I understand:

* Time-series prediction
* Feature engineering
* Classification
* Baseline models
* Temporal leakage
* Chronological train/test splitting
* Model evaluation
* Honest interpretation of results

Do NOT optimize for artificially high accuracy.

If the model performs poorly, that is acceptable. We should explain why.

The assignment specifically values an honest analysis of market classification performance.

---

# 4. KEEP THE PROJECT SIMPLE

Do NOT over-engineer this.

We do NOT need:

* Deep learning
* LSTM
* Transformers
* Reinforcement learning
* Trading bots
* Portfolio optimization
* Streamlit
* Flask/FastAPI
* Authentication
* Databases
* Cloud deployment
* Complex frontend
* Real-time trading
* Cryptocurrency systems

This is an academic **time-series ML experiment**.

A clean, reproducible implementation is more important than unnecessary complexity.

---

# 5. PROPOSED PROJECT STRUCTURE

(Superseded by the structure in `PLAN.md`. Original proposal kept for reference.)

A reasonable starting structure is:

```
stock-price-predictor/
│
├── data/
│
├── notebooks/
│   └── stock_prediction.ipynb
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── evaluation.py
│
├── outputs/
│   ├── comparison_table.csv
│   ├── class_balance.png
│   └── predictions.png
│
├── requirements.txt
│
└── README.md
```

You may adjust this structure if there is a genuinely good reason, but keep it simple.

The notebook should be understandable on its own.

---

# 6. DATASET

(Decided in `PLAN.md`: SPY, 2005-01-01 to 2025-12-31, via yfinance, saved to `data/spy_daily.csv`.)

Use daily historical OHLCV data for a **liquid equity or index**.

Before selecting the final dataset, explain:

* Which equity/index you selected
* Why it qualifies as liquid
* Where the data comes from
* The date range
* What each column represents

Prefer a dataset/source that can be reproduced easily.

Do not use a tiny toy dataset.

Do not fabricate data.

---

# 7. TARGET CONSTRUCTION

The target should represent the NEXT trading day's direction.

Conceptually:

future_close = close shifted by -1

target = 1 if future_close > current_close else 0

However, implement this carefully and verify it.

The final row will not have a known next-day target and therefore must be handled appropriately.

The notebook should explicitly demonstrate several rows showing:

Date | Current Close | Next-Day Close | Target

This is important because the assignment explicitly asks us to verify zero forward leakage.

---

# 8. FEATURE ENGINEERING

(Decided in `PLAN.md`: 5 raw single-day features + 5 indicators, frozen.)

Start with raw OHLCV-related information.

Then create at least 3 technical indicators.

Potential indicators can include:

* SMA
* EMA
* RSI
* MACD
* Bollinger Bands
* Momentum
* Rolling volatility

But do not blindly add dozens of indicators.

Choose approximately **3–5 useful indicators** and explain why they were selected.

Every feature must only use information available on or before that date.

Rolling calculations must be causal.

Be extremely careful with:

* `.shift()`
* rolling windows
* future values
* target construction

---

# 9. MODEL

(Decided in `PLAN.md`: Logistic Regression in a StandardScaler pipeline, same settings for both feature sets.)

Use a simple scikit-learn classification model.

A reasonable first choice is:

**Logistic Regression**

Another possible model is:

**Random Forest**

We can decide after reviewing the dataset and feature behavior.

Do not introduce complicated models unless they materially improve the experiment.

The key experiment is:

Persistence vs Majority vs Raw Features vs Engineered Features.

---

# 10. EXPERIMENT DESIGN

We need a clear experiment.

### Experiment 1 — Persistence

Predict that tomorrow's direction will be the same as today's.

### Experiment 2 — Majority Class

Always predict the majority class.

### Experiment 3 — Raw Features

Train a classifier using raw price/volume features.

### Experiment 4 — Engineered Features

Train the same or comparable classifier using raw features + technical indicators.

Then produce a table similar to:

| Approach            | Accuracy |
| ------------------- | -------: |
| Persistence         |      XX% |
| Majority Class      |      XX% |
| Raw Features        |      XX% |
| Engineered Features |      XX% |

Do not hardcode numbers.

Everything must be calculated from the actual test results.

---

# 11. TRAIN/TEST SPLIT

Use a strict chronological split.

Example:

Earlier 80% → training
Later 20% → testing

But choose the exact split based on the dataset and explain it.

NO random train_test_split with shuffle=True.

The test period must represent a genuinely later period.

If using StandardScaler:

1. fit scaler on X_train
2. transform X_train
3. transform X_test

Never:

scaler.fit_transform(X_all)

before splitting.

---

# 12. CLASS BALANCE

We need a class balance report showing:

* Number of Up samples
* Number of Down samples
* Percent Up
* Percent Down

Ideally show this for the relevant dataset partitions as appropriate.

The report should help explain why the Majority Class baseline may achieve a seemingly decent accuracy.

---

# 13. VISUALIZATION

Create the required prediction visualization.

The main plot should show actual vs predicted direction over the test window.

Make it readable.

Avoid misleading visualizations.

The plot should have:

* Date on x-axis
* Actual direction
* Predicted direction
* Clear legend
* Clear title
* Proper labels

Save the figure to `outputs/`.

---

# 14. NOTEBOOK

The notebook should tell a coherent story.

Suggested sections:

1. Project Title
2. Objective
3. Dataset
4. Data Loading
5. Data Cleaning
6. Exploratory Data Analysis
7. Target Construction
8. Explicit Leakage Verification
9. Class Balance
10. Raw Feature Construction
11. Technical Indicator Engineering
12. Chronological Train/Test Split
13. Persistence Baseline
14. Majority Class Baseline
15. Raw Feature Model
16. Engineered Feature Model
17. Results Comparison
18. Prediction Visualization
19. Analysis
20. Limitations
21. Conclusion

Do not create filler sections.

---

# 15. README

README.md should explain:

* Project title
* Problem statement
* Objective
* Dataset/source
* Methodology
* Target definition
* Features
* Technical indicators
* Baselines
* Model
* Train/test methodology
* Leakage prevention
* Results
* How to run
* Requirements
* Project structure
* Limitations
* Conclusion

Make it look like a serious GitHub ML project.

---

# 16. REPRODUCIBILITY

Create a pinned `requirements.txt`.

At minimum, likely dependencies include:

* pandas
* numpy
* scikit-learn
* matplotlib
* jupyter
* yfinance or whatever legitimate data source we ultimately choose
* ta, ONLY if we decide to use it

Pin versions so another person can reproduce the project.

Do not add packages that aren't actually used.

---

# 17. IMPORTANT REVIEW RULES

Throughout development, constantly check:

### Leakage checklist

Ask:

1. Does any feature use tomorrow's information?
2. Does any rolling calculation accidentally include future data?
3. Is the target shifted correctly?
4. Is the scaler fitted only on training data?
5. Are model decisions based only on training data?
6. Is the test period strictly later than the training period?
7. Did any preprocessing happen before splitting that should have happened after splitting?

If there is any leakage, fix it before proceeding.

---

# 18. DO NOT FAKE RESULTS

Never manufacture accuracy values.

Never claim the model is profitable.

Never claim the model can predict the stock market reliably.

If engineered features make performance worse, report that.

If the model barely beats the baselines, report that.

The point is to perform an honest experiment.

---

# 19. DEVELOPMENT PROCESS

(Stage 1 is already complete — see the status note at the top.)

We are going to build this in stages.

### Stage 1 — Planning (DONE)

* Understand the requirements above.
* Inspect the current project directory.
* Propose the final architecture.
* Recommend a dataset/equity/index.
* Recommend the technical indicators.
* Recommend the initial classifier.
* Explain the experimental design.
* Identify potential leakage risks.

### Stage 2 — Implementation

* Create the project structure.
* Create requirements.txt.
* Implement data loading.
* Implement feature engineering.
* Implement target construction.
* Implement baselines.
* Implement chronological evaluation.
* Implement models.
* Implement visualizations.
* Create notebook.
* Create README.

### Stage 3 — Validation

After implementation:

* Run the project.
* Check for errors.
* Verify the target construction manually.
* Verify chronological split.
* Verify no preprocessing leakage.
* Verify all four approaches appear in the comparison table.
* Verify required plots are generated.
* Verify README is complete.
* Verify requirements are reproducible.

### Stage 4 — Final Audit

Perform an explicit audit against EVERY requirement in the original assignment.

Create a checklist:

Requirement | Implemented? | Evidence/File

Do not declare the project finished until every requirement is satisfied.

---

# 20. COMMUNICATION STYLE

I want you to act as an engineering collaborator, not blindly agree with me.

If I suggest something technically weak:

* Tell me.
* Explain why.
* Recommend a better approach.

If something violates the assignment:

* Stop me.
* Point it out.

If something introduces data leakage:

* Treat it as a serious bug.

Prioritize correctness and reproducibility over flashy results.

---

# 21. FIRST RESPONSE (ALREADY COMPLETED)

The original first-response instructions (no code; present the plan and wait for approval) have been fulfilled. The approved plan is in `PLAN.md`. Proceed with Stage 2 as directed in my prompts.
