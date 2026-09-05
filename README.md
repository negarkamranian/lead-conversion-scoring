# Lead conversion scoring

Ranks abandoned-quote leads by purchase probability so telesales can contact the most promising leads first. 

## Run

```bash
cp .env.example .env
docker compose build --no-cache app
docker compose up -d db
make pipeline
```

Individual stages: `make ingest`, `make quality`, `make train`, `make evaluate`, `make score`, and `make monitor`. Use `make mlflow-ui` to compare runs at <http://localhost:5000>.

Docker persists reports, charts, and MLflow state in named volumes. Local runs write to `artifacts/`, `charts/`, and `mlruns/`. Settings come from `.env`; `TOP_FRACTION=0.1` means contacting the top 10%.


## Run as dev
clear MLFlow history via:
```bash
docker compose down
docker volume rm lead-conversion-scoring_mlflow_data
```

local development:

```bash
uv sync --locked
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pytest -q -m 'not integration'
```

## Design decisions and tradeoffs

- **Prediction point:** score a lead after quote abandonment and before contact or purchase. The target, Lead ID, and absolute creation timestamp are excluded from the model; hour and weekday are derived from the timestamp. Quote, payment, incoming-call, and margin fields require point-in-time verification before production use.
- **Data contract:** parse timestamps and numeric fields strictly, validate binary values and plausible ranges, reject missing Lead IDs, and keep the earliest row for each Lead ID (50,180 raw rows → 50,000 unique leads). Strict validation catches upstream changes early, but a production pipeline may need an explicit quarantine path for bad rows.
- **Split:** split chronologically at the 70th and 85th percentile timestamps into approximately 70% train, 15% validation, and 15% test, keeping equal timestamps in the same partition. This better represents training on the past and scoring the future than a random split. A single time window and the lack of customer IDs limit the evidence for stability and customer-level leakage.
- **Preprocessing:** fit and persist all transforms inside each model pipeline. Linear models, Extra Trees, and XGBoost use median/most-frequent imputation and one-hot encoding; the other scikit-learn trees use ordinal encoding. CatBoost receives categorical strings directly and preserves numeric missingness for native handling.
- **Models:** compare nine fixed candidates: a dummy baseline, two logistic variants, four scikit-learn tree ensembles, CatBoost, and XGBoost. CatBoost and XGBoost use the previously selected depth 4 with 300 boosting rounds and regularization. Fixed settings keep the comparison reproducible and affordable, but the limited search does not guarantee optimal tuning.
- **Feature sensitivity:** train an additional logistic model without insurance company, payment type, price, discount, recent incoming call, and expected margin. Its validation metrics show how performance changes if these potentially unavailable or post-outcome fields must be removed; it is diagnostic and is not part of model selection.
- **Selection and test isolation:** select by validation AP, preferring logistic regression when its AP and log loss are both within 0.01 of the validation leader. The later test partition is evaluated only after selection. No resampling, class weighting, probability calibration, or test-driven tuning is applied.
- **Scoring and lineage:** derive an operating threshold from the validation scores for threshold-based evaluation, but production scoring always calls exactly the top `ceil(rows × TOP_FRACTION)` leads. Scores are ordered by probability with Lead ID as the deterministic tie-breaker. CSV ingestion is idempotent, and PostgreSQL stores immutable scoring batches with model, source-data, and timestamp lineage.

## Metrics

- **Average precision (AP):** the primary model-selection metric. Purchases are rare, so AP focuses on how well the model ranks the positive class without being dominated by non-purchasers.
- **Precision, recall, and lift at capacity:** measure the leads found within the top 1%, 5%, 10%, and 20% of scores. They directly match the telesales constraint: precision is the expected conversion rate of contacted leads, recall is the share of all purchasers reached, and lift compares that precision with contacting leads at random.
- **ROC-AUC:** measures overall ranking quality across every possible threshold. It is useful as a broad discrimination check, but AP and capacity metrics are closer to the imbalanced business objective.
- **Log loss and Brier score:** assess the quality of the predicted probabilities and penalize overconfident errors. This matters when scores are used for planning or expected-value decisions, rather than ranking alone.
- **Precision, recall, F1, and confusion matrix:** describe behavior at the selected operating threshold. F1 summarizes the precision-recall balance, while the confusion matrix keeps the underlying error counts visible.
- **Bootstrap intervals:** give 95% uncertainty ranges for test AP and ROC-AUC, showing how much the reported ranking performance may vary across samples.

## Results

Rerun on 2026-09-05, seed 42, locked dependencies. The earlier 13-configuration comparison used one train/validation split; current training retains nine candidates. Best depth by AP shown below; see the [full comparison](artifacts/validation_model_comparison.json) for all results.

| Candidate | Validation AP ↑ | ROC-AUC ↑ | Log loss ↓ | Top-10% lift ↑ |
|---|---:|---:|---:|---:|
| Logistic regression | **0.1934** | **0.7159** | **0.2781** | **2.63×** |
| CatBoost (depth 4) | 0.1904 | 0.7136 | 0.2787 | 2.52× |
| Spline logistic regression | 0.1902 | 0.7136 | 0.2786 | 2.51× |
| XGBoost (depth 4) | 0.1900 | 0.7080 | 0.2798 | 2.45× |
| Histogram gradient boosting | 0.1828 | 0.7054 | 0.2808 | 2.46× |
| Gradient boosting | 0.1807 | 0.7075 | 0.2807 | 2.55× |
| Random forest | 0.1738 | 0.7003 | 0.2817 | 2.30× |
| Extra Trees | 0.1727 | 0.6898 | 0.2844 | 2.36× |
| Dummy prior | 0.0899 | 0.5000 | 0.3024 | 0.95× |

**Logistic regression remains selected.** AP measures precision–recall ranking for rare purchases; ROC-AUC measures discrimination; log loss/Brier measure probability error. Lift measures conversion concentration relative to the population.

Only the selected model was evaluated on the later test window (7,501 leads, 558 purchases):

| Test metric | Value |
|---|---:|
| AP / ROC-AUC | 0.1531 / 0.6938 |
| Log loss / Brier score | 0.2512 / 0.0674 |
| Top-10% precision / recall | 17.58% / 23.66% |
| Top-10% lift | 2.36× |

The top 751 leads contain 132 purchases: **2.36×** the population purchase rate. Test AP is lower, alongside lower purchase prevalence (7.44% versus validation's 8.99%). See [test metrics](artifacts/test_metrics.json) for confidence intervals and [metadata](artifacts/model_metadata.json) for split/feature sensitivity details.

Training/evaluation ran directly from CSV. **29 unit tests and Ruff checks pass.** PostgreSQL integration was not rerun; 16 pre-existing type errors remain in unchanged modules.

This synthetic dataset measures purchase propensity, not incremental sales from calling. Next steps: verified feature timestamps, rolling temporal validation, and a contact/no-contact experiment.

## View scored leads in PostgreSQL!

After `make pipeline` or `make score` finishes, see the 100 highest-priority leads from the
latest successful scoring batch with:

```bash
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
SELECT lead_id, purchase_probability, priority_rank, priority_tier,
       scored_at, model_version
FROM lead_scores
WHERE scoring_batch_id = (
    SELECT scoring_batch_id
    FROM scoring_batches
    WHERE status = '\''succeeded'\''
    ORDER BY completed_at DESC, scoring_batch_id DESC
    LIMIT 1
)
ORDER BY priority_rank
LIMIT 100;
"'
```

Rows with `priority_tier = 'call'` are the leads selected for contact under the configured
`TOP_FRACTION`; `purchase_probability` is the model score, and a lower `priority_rank` means
higher priority. Remove `LIMIT 100` to display every scored lead.

To explore the database interactively, run:

```bash
docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Useful psql commands are `\dt` to list tables and `\q` to exit. Scoring history is stored in
`scoring_batches`, while the per-lead results are stored in `lead_scores`.
