# Pre-emptive fraud mitigation dashboard.
#
# Layout:
#   Tier 1  src/data_pipeline.py   loading, feature engineering, splitting
#   Tier 2  src/models.py          the three models
#           src/evaluation.py      metrics
#   Tier 3  app.py, explanations.py, actions.py
#
# Run from the terminal: python -m streamlit run app.py

import os

import numpy as np
import pandas as pd
import streamlit as st

from src import (actions, data_pipeline as dp, evaluation, explanations,
                 ingest, models, persistence)

# Reads from a hosted URL when NIBSS_DATA_URL is set (used for deployment,
# since the full dataset is too large for a GitHub repository), otherwise
# falls back to the local copy used for development.
def _data_path():
    try:
        if "NIBSS_DATA_URL" in st.secrets:
            return st.secrets["NIBSS_DATA_URL"]
    except Exception:
        pass
    return os.environ.get("NIBSS_DATA_URL", "data/nibss_fraud_dataset.csv")


DATA_PATH = _data_path()

st.set_page_config(page_title="Pre-emptive Fraud Mitigation Prototype",
                   layout="wide")

st.title("AI-Driven Pre-emptive Cybersecurity Prototype")
st.caption("Fraud mitigation for small-scale Nigerian fintechs. "
           "Research prototype, NIBSS-aligned synthetic data.")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Simulation Controls")
sample_size = st.sidebar.select_slider(
    "Dataset sample size", options=[50_000, 100_000, 200_000, 500_000, 1_000_000],
    value=200_000,
    help="Random sample keeping the natural 0.3% fraud rate. "
         "Larger samples give steadier metrics but train more slowly.")
contamination = st.sidebar.slider(
    "Isolation Forest contamination", 0.001, 0.02, 0.003, 0.001,
    format="%.3f",
    help="Expected share of anomalies. Set to 0.003 to match the actual "
         "fraud rate in the data.")
rf_threshold = st.sidebar.slider(
    "Random Forest decision threshold", 0.1, 0.9, 0.5, 0.05,
    help="Probability above which the Random Forest flags a transaction.")
flag_rate = st.sidebar.slider(
    "Deviation Scorer flag rate (%)", 0.3, 3.0, 1.0, 0.1,
    help="Share of traffic flagged for intervention.") / 100.0
prior_set = st.sidebar.selectbox(
    "Channel risk priors", list(models.PRIOR_SETS.keys()), index=0,
    help="Sensitivity setting. 'uniform' applies no channel weighting. "
         "The alternatives encode two conflicting published rankings; "
         "results are reported on 'uniform'.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models**\n\n"
    "*Isolation Forest*: first unsupervised attempt. Kept as a negative "
    "result, since fraud here is not an outlier.\n\n"
    "*Behavioural Deviation Scorer*: the label-free model. Scores upward "
    "deviation of transaction value from the baseline.\n\n"
    "*Random Forest*: supervised benchmark. Shows what labelled history "
    "would add.")
st.sidebar.markdown("---")
if st.sidebar.button("Clear saved models",
                     help="Delete saved model files and retrain on the "
                          "next run."):
    removed = persistence.clear_store()
    st.cache_resource.clear()
    st.sidebar.success(f"Cleared {removed} saved model bundle(s).")


# ---------------------------------------------------------------- cached pipeline
@st.cache_data(show_spinner="Loading dataset...")
def load_and_split(path: str, n: int):
    df = dp.load_dataset(path, sample_size=n)
    df_train, df_test = dp.split(df)
    return df, df_train, df_test


@st.cache_resource(show_spinner="Preparing models...")
def get_bundle(n: int, cont: float, frate: float, priors: str) -> dict:
    """Load fitted models from disk if available, otherwise train and
    persist them. Mirrors deployment practice: training is an offline
    cost paid once, not per transaction."""
    key = persistence.cache_key(n=n, contamination=cont, flag_rate=frate,
                                priors=priors, version=3)
    bundle = persistence.load_bundle(key)
    if bundle is not None:
        bundle["from_cache"] = True
        return bundle

    _, df_train, _ = load_and_split(DATA_PATH, n)
    X_train, y_train = dp.build_features(df_train)
    bundle = {
        "iso": models.train_isolation_forest(X_train, contamination=cont),
        "rf": models.train_random_forest(X_train, y_train),
        "bds": models.BehaviouralDeviationScorer(
            flag_rate=frate, priors=priors).fit(df_train),
        "stats": dp.legitimate_feature_stats(df_train),
        "feature_cols": X_train.columns.tolist(),
        "trained_on": n,
    }
    persistence.save_bundle(key, bundle)
    bundle["from_cache"] = False
    return bundle


try:
    df, df_train, df_test = load_and_split(DATA_PATH, sample_size)
except FileNotFoundError:
    st.error(f"Dataset not found at `{DATA_PATH}`. It should sit in the data/ folder.")
    st.stop()

bundle = get_bundle(sample_size, contamination, flag_rate, prior_set)
iso, rf, bds = bundle["iso"], bundle["rf"], bundle["bds"]
legit_stats, feature_cols = bundle["stats"], bundle["feature_cols"]

# Score the held-out test partition (the only honest place to report metrics).
X_test, y_test = dp.build_features(df_test)
X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)
iso_risk, iso_flag = models.score_isolation_forest(iso, X_test)
rf_prob, rf_flag = models.score_random_forest(rf, X_test, threshold=rf_threshold)
bds_risk, bds_flag = bds.score(df_test)

tab_context, tab_pipeline, tab_eval, tab_feed, tab_new = st.tabs([
    "Sector Context", "Data & Pipeline",
    "Model Evaluation", "Live Threat Feed", "Test on New Data"])

# ---------------------------------------------------------------- Tab 1: sector context
with tab_context:
    st.subheader("Nigerian Fintech Fraud Landscape, 2021-2025")
    st.markdown(
        "Published NIBSS and CBN figures. Incident volume fell while loss "
        "values stayed volatile, which is the pattern this prototype targets.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fraud Incident Rate (FIR)**: reported incidents")
        st.bar_chart(pd.DataFrame(
            {"Incidents": [123_918, 67_518]}, index=["2021", "2025"]))
        st.caption("−45.5% over the period (NIBSS, 2025; 2026)")
    with c2:
        st.markdown("**Fraud Loss Value (FLV)**: ₦ billions")
        st.bar_chart(pd.DataFrame(
            {"₦bn lost": [52.26, 25.85]}, index=["2024", "2025"]))
        st.caption("−51% year-on-year (NIBSS, 2026)")
    st.warning(
        "**Read the 51% fall with care.** NIBSS reports that the 2024 total "
        "was largely driven by a single fraud incident of ₦31.1 billion "
        "involving one entity, roughly 60% of that year's losses. Excluding "
        "it, the 2024 base is nearer ₦21bn, so 2025 may be a modest **rise** "
        "in ordinary fraud rather than a halving.")
    st.info(
        "The CBN reports a 45% fraud surge with about 70% of losses traced to "
        "digital channels, attributed **particularly to unregulated virtual "
        "asset platforms** (Cardoso, EFCC lecture, July 2025) rather than to "
        "licensed neo-banks alone. NIBSS (2026) ranks e-commerce and internet "
        "banking as the most affected channels, then POS, mobile and web.")

# ---------------------------------------------------------------- Tab 2: pipeline
with tab_pipeline:
    st.subheader("Tier 1: Ingestion and Feature Engineering")
    fraud_rate = df["is_fraud"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions loaded", f"{len(df):,}")
    c2.metric("Confirmed fraud cases", f"{int(df['is_fraud'].sum()):,}")
    c3.metric("Fraud rate", f"{fraud_rate:.3%}")
    c4.metric("Model features", len(feature_cols))
    st.markdown("**Raw data sample**")
    st.dataframe(df[dp.DISPLAY_COLS].head(10), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fraud volume by channel**")
        st.bar_chart(df[df.is_fraud == 1]["channel"].value_counts())
    with c2:
        st.markdown("**Fraud technique distribution**")
        st.bar_chart(df[df.is_fraud == 1]["fraud_technique"].value_counts())
    with st.expander("Feature engineering decisions (Chapter 3 documentation)"):
        st.markdown(
            "- **Excluded columns:** " + ", ".join(
                f"`{k}` ({v})" for k, v in dp.EXCLUDED.items()) + "\n"
            "- **One-hot encoding** for categoricals, not label encoding\n"
            "- **Class imbalance kept** in both sampling and the 70/30 "
            "stratified split\n"
            "- `is_fraud` and `fraud_technique` reach only the Random Forest, "
            "which uses labels by design")

# ---------------------------------------------------------------- Tab 3: evaluation
with tab_eval:
    st.subheader("Tier 2: Performance on Held-out Test Data "
                 f"({len(df_test):,} transactions)")

    iso_cls = evaluation.classification_metrics(y_test, iso_flag, iso_risk)
    bds_cls = evaluation.classification_metrics(y_test, bds_flag, bds_risk)
    rf_cls = evaluation.classification_metrics(y_test, rf_flag, rf_prob)
    amounts = df_test["amount"].values
    iso_flv = evaluation.flv_metrics(y_test, iso_flag, amounts)
    bds_flv = evaluation.flv_metrics(y_test, bds_flag, amounts)
    rf_flv = evaluation.flv_metrics(y_test, rf_flag, amounts)

    st.markdown("**1 · Classification quality** (vs ground-truth `is_fraud`)")

    def _col(cls):
        return {
            "Precision": f"{cls['precision']:.3f}",
            "Recall": f"{cls['recall']:.3f}",
            "F1": f"{cls['f1']:.3f}",
            "ROC-AUC": f"{cls['roc_auc']:.3f}",
            "True positives": str(cls["true_positives"]),
            "False positives": str(cls["false_positives"]),
            "Missed fraud (FN)": str(cls["false_negatives"]),
        }

    st.dataframe(pd.DataFrame({
        "Isolation Forest (initial candidate)": _col(iso_cls),
        "Deviation Scorer (label-free primary)": _col(bds_cls),
        "Random Forest (supervised ceiling)": _col(rf_cls),
    }), use_container_width=True)
    st.caption(
        "The Isolation Forest's near-random AUC is a negative result. Fraud "
        "here sits at elevated value inside the normal range rather than in "
        "the outlier tails, so outlier detection cannot find it. The "
        "Deviation Scorer is the label-free replacement; the Random Forest "
        "shows what labelled history would add.")

    st.markdown("**2 · Financial impact: FLV protection vs friction**")
    c1, c2, c3 = st.columns(3)
    for col, name, flv in ((c1, "Isolation Forest", iso_flv),
                           (c2, "Deviation Scorer", bds_flv),
                           (c3, "Random Forest", rf_flv)):
        with col:
            st.markdown(f"*{name}*")
            st.metric("FLV Protection Ratio",
                      f"{flv['protection_ratio']:.1%}",
                      help="Share of fraudulent value intercepted")
            st.metric("Fraud value intercepted", f"₦{flv['protected_value']:,.0f}")
            st.metric("Fraud value missed", f"₦{flv['missed_value']:,.0f}")
            st.metric("Legitimate value wrongly held",
                      f"₦{flv['blocked_legit_value']:,.0f}",
                      delta=f"{flv['friction_ratio']:.2%} of legitimate flow",
                      delta_color="inverse",
                      help="Customer friction caused by false positives")

    st.markdown("**3 · Intercept latency** "
                "(per-transaction scoring time, decision step only)")
    if st.button("Measure latency (200 single-transaction scores per model)"):
        with st.spinner("Scoring transactions one at a time..."):
            iso_lat = evaluation.intercept_latency(
                lambda r: iso.decision_function(r), X_test)
            rf_lat = evaluation.intercept_latency(
                lambda r: rf.predict_proba(r), X_test)
            bds_lat = evaluation.intercept_latency(
                lambda r: bds.score(r), df_test.reset_index(drop=True))
        c1, c2, c3 = st.columns(3)
        c1.metric("Isolation Forest, mean / p95",
                  f"{iso_lat['mean_ms']:.1f} ms / {iso_lat['p95_ms']:.1f} ms")
        c2.metric("Deviation Scorer, mean / p95",
                  f"{bds_lat['mean_ms']:.2f} ms / {bds_lat['p95_ms']:.2f} ms")
        c3.metric("Random Forest, mean / p95",
                  f"{rf_lat['mean_ms']:.1f} ms / {rf_lat['p95_ms']:.1f} ms")
        st.caption("Inside real-time clearing windows.")

    st.info(
        "**How to read this.** The first unsupervised model failed because "
        "its assumption that fraud is an outlier does not hold here. The "
        "label-free scorer replaced it, and the supervised benchmark sets "
        "the ceiling. Moderate scores are expected on data with this much "
        "class imbalance.")

# ---------------------------------------------------------------- Tab 4: live feed
with tab_feed:
    st.subheader("Tier 3: Live Threat Feed")
    st.markdown(
        "Transactions flagged by the **Behavioural Deviation Scorer** on the "
        "test partition, ranked by risk. Each alert carries a plain-language "
        "explanation and a *simulated* intervention.")

    feed = df_test.copy().reset_index(drop=True)
    feed["risk_score"] = bds_risk
    feed["flagged"] = bds_flag
    flagged = feed[feed.flagged == 1].sort_values("risk_score", ascending=False)

    n_show = st.slider("Alerts to display", 5, 50, 15, 5)
    if flagged.empty:
        st.success("No transactions flagged at current settings.")
    else:
        ranks = flagged["risk_score"].rank(pct=True)
        shown = flagged.head(n_show)
        st.caption(f"{len(flagged):,} transactions flagged "
                   f"({len(flagged) / len(feed):.2%} of test traffic)")
        for idx, row in shown.iterrows():
            action = actions.assign_action(float(ranks.loc[idx]))
            truth = "confirmed fraud" if row.is_fraud == 1 else "legitimate (false positive)"
            with st.expander(
                    f"{action.icon} [{action.tier}] ₦{row.amount:,.0f} · "
                    f"{row.channel} · {row.bank} · {row.timestamp} · {truth}"):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.markdown("**Why this was flagged:**")
                    for reason in explanations.explain_transaction(row, legit_stats):
                        st.markdown(f"- {reason}")
                    st.markdown(f"**Anomaly risk score:** `{row.risk_score:.4f}`")
                with c2:
                    st.markdown(f"**Simulated action: {action.name}**")
                    st.markdown(action.description)
                    held = ("held for review"
                            if action.tier in ("CRITICAL", "HIGH")
                            else "sent for additional verification")
                    st.markdown(
                        f"*Customer notice:* Unusual activity was detected on "
                        f"your account. This {row.channel} transaction of "
                        f"₦{row.amount:,.0f} has been {held}. "
                        f"No funds have left your account.")

    st.caption("Ground-truth labels are shown so alert quality can be "
               "checked. A live system would not have them.")

# ---------------------------------------------------------------- Tab 5: new data
with tab_new:
    st.subheader("Score a Transaction File the Models Have Never Seen")
    st.markdown(
        "The models were fitted on historical data only. Here they score a "
        "file supplied at runtime, the way a deployed engine would score "
        "incoming traffic. Nothing is retrained.")

    with st.expander("Where to get a valid test file"):
        st.markdown(
            "Run this once in the project folder to create a file drawn "
            "**only from rows excluded from training**:")
        st.code("python scripts/generate_holdout.py --rows 5000\n"
                "python scripts/generate_holdout.py --rows 5000 --no-labels  "
                "# blind test", language="bash")
        st.markdown(
            f"Any CSV works if it carries the same schema "
            f"({len(ingest.REQUIRED)} required fields). Include an "
            "`is_fraud` column to also score accuracy; without it the file "
            "is treated as live traffic and only alerts are produced.")

    uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}")
            st.stop()

        try:
            report = ingest.validate(new_df)
        except ingest.SchemaError as exc:
            st.error(f"**Schema check failed.** {exc}")
            st.info("The file must carry the same fields as the dataset. "
                    "Generate a valid example with "
                    "`python scripts/generate_holdout.py`.")
            st.stop()

        st.success(f"Schema valid. {report['n_rows']:,} transactions accepted.")
        if report["unknown_categories"]:
            st.warning("Unseen category values (zero-filled, but they "
                       "indicate the data differs from training): "
                       + "; ".join(f"`{k}`: {', '.join(map(str, v))}"
                                   for k, v in report["unknown_categories"].items()))
        if report["null_counts"]:
            st.info(f"Missing values were zero-filled in "
                    f"{len(report['null_counts'])} column(s).")

        with st.spinner("Scoring with the fitted models..."):
            scored = ingest.score_file(new_df, bundle, rf_threshold=rf_threshold)

        n_flag = int(scored["deviation_flag"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions scored", f"{len(scored):,}")
        c2.metric("Flagged for intervention", f"{n_flag:,}",
                  delta=f"{n_flag / max(len(scored), 1):.2%} of traffic")
        c3.metric("Value under intervention",
                  f"₦{scored.loc[scored.deviation_flag == 1, 'amount'].sum():,.0f}")
        c4.metric("Models retrained?", "No",
                  help="The fitted engine is applied as-is. This is scoring "
                       "on unseen data, not refitting.")

        if report["has_labels"]:
            st.markdown("### Detection performance on this unseen file")
            y_new = scored["is_fraud"].astype(int)
            amt_new = scored["amount"].values
            rows = {}
            for label, flag, risk in (
                    ("Deviation Scorer (label-free)", scored["deviation_flag"],
                     scored["deviation_risk"]),
                    ("Random Forest (supervised)", scored["rf_flag"],
                     scored["rf_fraud_probability"])):
                cls = evaluation.classification_metrics(y_new, flag, risk)
                flv = evaluation.flv_metrics(y_new, flag, amt_new)
                rows[label] = {
                    "Precision": f"{cls['precision']:.3f}",
                    "Recall": f"{cls['recall']:.3f}",
                    "ROC-AUC": f"{cls['roc_auc']:.3f}",
                    "Fraud caught": str(cls["true_positives"]),
                    "Fraud missed": str(cls["false_negatives"]),
                    "FLV protection": f"{flv['protection_ratio']:.1%}",
                    "Legit value held": f"₦{flv['blocked_legit_value']:,.0f}",
                }
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(
                f"This file contains {int(y_new.sum())} fraudulent "
                f"transactions the models never saw. Metrics close to the "
                "Model Evaluation tab mean the models generalise.")
        else:
            st.info("No `is_fraud` column, so the file is treated as live "
                    "traffic. Alerts are produced below, but accuracy cannot "
                    "be scored without ground truth.")

        st.markdown("### Pre-emptive interventions triggered")
        flagged_new = scored[scored.deviation_flag == 1].sort_values(
            "deviation_risk", ascending=False)
        if flagged_new.empty:
            st.success("No transactions in this file crossed the "
                       "intervention threshold.")
        else:
            tier_counts = flagged_new["action_tier"].value_counts()
            st.bar_chart(tier_counts)
            for idx, row in flagged_new.head(10).iterrows():
                truth = ""
                if report["has_labels"]:
                    truth = (" · confirmed fraud" if row.is_fraud == 1
                             else " · legitimate (false positive)")
                with st.expander(
                        f"[{row.action_tier}] ₦{row.amount:,.0f} · "
                        f"{row.channel} · {row.get('bank', 'n/a')}{truth}"):
                    st.markdown("**Why this was flagged:**")
                    for reason in explanations.explain_transaction(row, legit_stats):
                        st.markdown(f"- {reason}")
                    st.markdown(
                        f"**Deviation risk:** `{row.deviation_risk:.3f}` · "
                        f"**RF fraud probability:** "
                        f"`{row.rf_fraud_probability:.3f}`")

        st.download_button(
            "Download scored results (CSV)",
            data=scored.to_csv(index=False).encode(),
            file_name="scored_transactions.csv", mime="text/csv",
            help="Full file with risk scores, flags and assigned action "
                 "tiers.")
