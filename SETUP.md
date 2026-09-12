# Setup Guide

Simple steps to get this project running on your machine.

## 1. Get the code

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

(If you already have the folder locally, just open a terminal there and skip this step.)

## 2. Create a virtual environment

```bash
python -m venv .venv
```

## 3. Activate it

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

You'll know it worked because your terminal prompt now starts with `(.venv)`.

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Open and run the notebook

```bash
jupyter notebook notebooks/stock_prediction.ipynb
```

This opens Jupyter in your browser. Click **Kernel → Restart & Run All** to run the whole thing top to bottom.

(The dataset is already included at `data/spy_daily.csv`, so this just works — no download needed.)

## 6. See the results without running anything

Everything the notebook produces is already saved in the `outputs/` folder:

- `comparison_table.csv` — accuracy of all 4 approaches
- `walk_forward_results.csv` — generalization check across time
- `class_balance.csv` / `class_balance.png` — how many Up vs Down days
- `predictions.png` / `predictions_zoom.png` — predicted vs actual direction charts

Just open these files directly if you don't want to run the notebook.

## That's it

- Full write-up of what the project does and what it found: **`README.md`**
- The original approved design: **`PLAN.md`**

## (Optional) Re-download the data

The dataset is already committed, so you don't need this. Only run it if you want a fresh pull from Yahoo Finance:

```bash
python -m src.data
```

This overwrites `data/spy_daily.csv` with newly downloaded data.
