# NMF Behavioral Pattern Decomposition — Churn Risk Prototype

## What This Prototype Validates

This is a **scientific validation prototype** — not a production system. It answers four specific questions about whether Non-Negative Matrix Factorization (NMF) can detect customer churn risk from real unlabeled behavioral data:

| Question | What It Tests |
|----------|--------------|
| **Q1. Pattern Quality** | Does NMF discover meaningful, interpretable behavioral patterns? |
| **Q2. K Stability** | Does the K selection logic produce a stable number of patterns? |
| **Q3. Outcome Prediction** | Do customer blend shifts predict confirmed churn events? |
| **Q4. Pattern Alignment** | Do patterns hold consistent across retraining windows? |

The prototype reads from your existing `ml_db` database and writes results only to the `output/` folder. It never modifies any production tables.

## Setup

### 1. Prerequisites

- Python 3.9+
- MySQL Server running with the `ml_db` database
- The following tables must exist: `ml_feature_store`, `ml_feedback`, `tenant_feature_config`, `feature_registry`

### 2. Install Dependencies

```bash
cd churn_pattern_prototype
pip install -r requirements.txt
```

### 3. Configure Database Credentials

Create or edit the `.env` file in the `churn_pattern_prototype/` directory:

```env
ML_DB_HOST=localhost
ML_DB_USER=root
ML_DB_PASSWORD=your_password_here
ML_DB_DATABASE=ml_db
ML_DB_PORT=3306
```

> **Security Note:** Never commit the `.env` file to version control.

## How to Run

### Single Tenant

```bash
python run_prototype.py --tenant sciqusams
```

### Multi-Tenant (All Active Tenants)

```bash
python run_prototype.py --tenant all
```

### With Temporal Split (for Q4 Alignment Validation)

```bash
python run_prototype.py --tenant sciqusams --split-date 2026-06-01
```

### Interactive Mode (No Arguments)

```bash
python run_prototype.py
```

This will list available tenants and let you choose interactively.

## Output Files

Results are saved to the `output/` directory:

| File | Contents |
|------|----------|
| `patterns_{tenant}.csv` | H matrix — each row is a pattern, columns are feature loadings |
| `blends_{tenant}.csv` | Customer blend vectors — how much of each pattern each customer expresses |
| `drift_scores_{tenant}.csv` | Customer drift scores, risk levels, and pattern movement direction |
| `validation_report_{tenant}.json` | All four validation results with detailed metrics |
| `run_metadata_{tenant}.json` | Run timestamp, K selected, feature list, data quality report |

## Interpreting Results

### PASS / MARGINAL / FAIL

Each of the four validation questions produces a verdict:

| Verdict | Meaning |
|---------|---------|
| **PASS** | The check meets its threshold. This aspect of NMF works well on this data. |
| **MARGINAL** | Partially passes. Works but with noted limitations. Review before proceeding. |
| **FAIL** | Does not meet the threshold. This aspect of NMF is problematic on this data. |
| **INSUFFICIENT_DATA** | Not enough data to evaluate (e.g., no feedback labels for Q3). |

### Overall Recommendation

- **All PASS**: NMF is validated. Proceed to production design.
- **3 PASS + 1 MARGINAL**: Viable with noted limitation. Review the marginal area.
- **Any FAIL**: Fundamental issue exists. Review the failing check before proceeding.

### Key Metrics to Watch

| Metric | Good | Concerning |
|--------|------|-----------|
| Reconstruction explained (Q1) | > 50% | < 30% |
| Inter-pattern distance (Q1) | > 0.3 | < 0.2 |
| K stability similarity (Q2) | > 0.8 | < 0.6 |
| Alignment quality (Q4) | > 0.75 | < 0.5 |

## What to Do Next Based on Results

### If Validated (All/Most PASS)
1. Design the production NMF pipeline using these validated parameters
2. Implement the pattern engine as a scheduled scoring service
3. Build drift monitoring into the existing churn prediction system
4. Add pattern labels to the customer dashboard

### If Marginal
1. Investigate the marginal check — is it a data volume issue?
2. Try adding more features or adjusting the feature set
3. Re-run with more data if available

### If Failed
1. Check if the feature matrix has enough variance (too many zeros?)
2. Verify that features are relevant to churn behavior
3. Consider whether the tenant has enough customers for NMF
4. Review the specific failure — pattern instability may suggest the data
   doesn't have clear behavioral clusters

## Architecture

```
run_prototype.py          ← Entry point (single or multi-tenant)
    ├── config.py          ← DB connection, settings, feature categories
    ├── data_loader.py     ← Load data from ml_db tables
    ├── pattern_engine.py  ← NMF, K selection, labeling, alignment
    ├── drift_detector.py  ← Blend vectors, drift scores, trajectories
    ├── validator.py       ← Four validation questions
    └── results.py         ← Display and save results
```

This prototype is completely standalone — it does not import from the main `app/` project.

## Multi-Tenant Architecture

When run with `--tenant all`, the prototype:
1. Queries all active tenants from `tenant_feature_config`
2. Runs the full pipeline independently for each tenant
3. Produces per-tenant output files in `output/`
4. Prints a cross-tenant comparison table at the end

Each tenant uses its own feature configuration, so patterns discovered for `tenant_engagement` (focused on meeting/activity features) will differ from `tenant_revenue` (focused on revenue/subscription features).
