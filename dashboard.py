from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "artifacts" / "runs"

st.set_page_config(
    page_title="Federated Learning | MLOps",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #102a43, #176b87);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        font-size: 2.15rem;
        margin-bottom: 8px;
    }
    .hero p {
        color: #d9edf7;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .section-note {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .client-card {
        background: #effaf4;
        border: 1px solid #b7e4c7;
        border-radius: 14px;
        padding: 20px 12px;
        text-align: center;
        min-height: 150px;
    }
    .client-card h3 {
        color: #166534;
        margin: 8px 0;
    }
    .client-card p {
        color: #365c46;
        font-size: 0.9rem;
        margin: 4px 0;
    }
    .server-card {
        background: #fff7e6;
        border: 2px solid #f0c36d;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
    }
    .server-card h3 {
        color: #92400e;
        margin-bottom: 8px;
    }
    .server-card p {
        color: #854d0e;
        margin: 4px 0;
    }
    .global-card {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }
    .global-card h3 {
        color: #3730a3;
        margin: 5px 0;
    }
    .global-card p {
        color: #4338ca;
        margin: 3px 0;
    }
    .flow-arrow {
        text-align: center;
        color: #64748b;
        font-size: 1.8rem;
        line-height: 1.5;
        padding: 5px;
    }
    .privacy-note {
        background: #f0fdfa;
        border-left: 4px solid #0d9488;
        border-radius: 5px;
        padding: 13px 16px;
        color: #115e59;
        margin: 12px 0 20px 0;
    }
    .stage-card {
        background: #f8fafc;
        border: 1px solid #dbe4ee;
        border-radius: 12px;
        padding: 16px;
        min-height: 130px;
    }
    .stage-card h4 {
        color: #1e3a5f;
        margin-bottom: 8px;
    }
    .stage-card p {
        color: #475569;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🧬 Federated Learning with MLOps</h1>
    <p>
        Privacy-aware collaborative model training, FedAvg aggregation,
        model evaluation and experiment tracking.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("### The problem we are addressing")

p1, p2 = st.columns(2)

with p1:
    st.markdown("#### ⚠️ Centralized learning challenge")
    st.write(
        "Organizations may be unable or unwilling to pool their raw "
        "data because of privacy, governance or data-sharing constraints."
    )

with p2:
    st.markdown("#### ✅ Our proposed approach")
    st.write(
        "Train models on separate client data partitions and combine "
        "their model weights through Federated Averaging (FedAvg)."
    )

st.info(
    "This is a simulation using a bundled breast-cancer dataset. "
    "The clients are not real hospitals and do not communicate over a network."
)


# --------------------------------------------------
# LOAD EXPERIMENT
# --------------------------------------------------

if not RUNS_DIR.exists():
    st.warning("No saved runs found. Run `python main.py train` first.")
    st.stop()

run_dirs = sorted(
    [p for p in RUNS_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.name,
    reverse=True,
)

if not run_dirs:
    st.warning("No saved experiments found.")
    st.stop()

selected_run = st.selectbox(
    "Choose an experiment",
    [p.name for p in run_dirs],
)

run_dir = RUNS_DIR / selected_run
metrics_path = run_dir / "metrics.csv"
config_path = run_dir / "config.json"

if not metrics_path.exists():
    st.error("metrics.csv is missing from this run.")
    st.stop()

metrics = pd.read_csv(metrics_path)

if metrics.empty:
    st.error("The selected run contains no metrics.")
    st.stop()

config = {}
if config_path.exists():
    config = json.loads(config_path.read_text(encoding="utf-8"))

latest = metrics.iloc[-1]


# --------------------------------------------------
# PERFORMANCE CARDS
# --------------------------------------------------

st.divider()
st.header("📊 Model performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Final accuracy", f"{latest['accuracy']:.2%}")
m2.metric("Final loss", f"{latest['loss']:.4f}")
m3.metric("Training rounds", str(len(metrics)))
m4.metric(
    "Simulated clients",
    str(config.get("num_clients", "N/A")),
)


# --------------------------------------------------
# CLEAN FEDERATED ARCHITECTURE
# --------------------------------------------------

st.divider()
st.header("🗺️ How our solution works")

st.markdown(
    '<p class="section-note">Raw training records stay in each local '
    'partition. Clients train locally; the server combines model weights.</p>',
    unsafe_allow_html=True,
)

# Dataset source
st.markdown("""
<div style="
    max-width:540px;
    margin:0 auto;
    text-align:center;
    background:#eff6ff;
    border:1px solid #bfdbfe;
    border-radius:14px;
    padding:17px;">
    <div style="font-size:1.7rem;">🗄️</div>
    <h3 style="color:#1d4ed8;margin:5px 0;">Training Dataset</h3>
    <p style="color:#475569;margin:0;">
        Breast-cancer data split into client partitions
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="flow-arrow">↓ &nbsp; Data partitioning</div>',
    unsafe_allow_html=True,
)

# Client cards
num_clients = config.get("num_clients", 5)
num_clients = max(2, min(int(num_clients), 6))

client_columns = st.columns(num_clients)

for i, column in enumerate(client_columns):
    with column:
        st.markdown(f"""
        <div class="client-card">
            <div style="font-size:1.8rem;">💻</div>
            <h3>Client {i + 1}</h3>
            <p>Private local partition</p>
            <p>Local model training</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    '<div class="flow-arrow">↓ &nbsp; Local model weights</div>',
    unsafe_allow_html=True,
)

# Aggregation server
st.markdown("""
<div class="server-card">
    <div style="font-size:1.8rem;">🔄</div>
    <h3>Central FedAvg Server</h3>
    <p>Combines client model weights using sample-count weighting</p>
    <p><b>Raw client records are not sent to the aggregation step.</b></p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="flow-arrow">↓ &nbsp; Aggregated model</div>',
    unsafe_allow_html=True,
)

# Global model
st.markdown("""
<div class="global-card">
    <div style="font-size:1.7rem;">🧠</div>
    <h3>Global Model</h3>
    <p>Updated after each federated round</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="flow-arrow">↓ &nbsp; Evaluate and track</div>',
    unsafe_allow_html=True,
)

# MLOps stages
stage_cols = st.columns(3)

stages = [
    (
        "📈 Evaluation",
        "Measure loss, accuracy, precision, recall and F1-score.",
    ),
    (
        "🧪 Experiment tracking",
        "Save training configuration and round-by-round metrics.",
    ),
    (
        "📦 Model registry",
        "Register the trained model as a versioned artifact.",
    ),
]

for column, (title, description) in zip(stage_cols, stages):
    with column:
        st.markdown(f"""
        <div class="stage-card">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="privacy-note">
    <b>Privacy clarification:</b> This prototype supports experimental
    gradient clipping and Gaussian noise for DP-FedAvg, with a theoretical
    privacy-accounting estimate. An end-to-end differential privacy guarantee
    has not been established, and secure aggregation is not implemented.
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TRAINING CHARTS
# --------------------------------------------------

st.divider()
st.header("📈 Training progress")

left, right = st.columns(2)

with left:
    loss_fig = px.line(
        metrics,
        x="round",
        y="loss",
        markers=True,
        title="Loss by federated round",
        labels={"round": "Round", "loss": "Loss"},
    )
    loss_fig.update_layout(template="plotly_white")
    st.plotly_chart(loss_fig, use_container_width=True)

with right:
    acc_fig = px.line(
        metrics,
        x="round",
        y="accuracy",
        markers=True,
        title="Accuracy by federated round",
        labels={"round": "Round", "accuracy": "Accuracy"},
    )
    acc_fig.update_yaxes(tickformat=".0%")
    acc_fig.update_layout(template="plotly_white")
    st.plotly_chart(acc_fig, use_container_width=True)


# --------------------------------------------------
# CLASSIFICATION METRICS
# --------------------------------------------------

st.divider()
st.header("🎯 Classification metrics")

score_names = [
    name for name in ["precision", "recall", "f1_score"]
    if name in metrics.columns
]

if score_names:
    score_data = pd.DataFrame({
        "Metric": score_names,
        "Score": [float(latest[name]) for name in score_names],
    })

    score_fig = px.bar(
        score_data,
        x="Metric",
        y="Score",
        color="Metric",
        text=score_data["Score"].map(lambda x: f"{x:.2%}"),
        title="Final-round scores",
    )
    score_fig.update_yaxes(range=[0, 1], tickformat=".0%")
    score_fig.update_layout(template="plotly_white", showlegend=False)
    st.plotly_chart(score_fig, use_container_width=True)
else:
    st.warning("Classification metrics are unavailable for this run.")


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.divider()
st.header("🧩 Confusion matrix")

required_cm_columns = [
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
]

if all(column in metrics.columns for column in required_cm_columns):
    tp = int(latest["true_positive"])
    tn = int(latest["true_negative"])
    fp = int(latest["false_positive"])
    fn = int(latest["false_negative"])

    confusion_matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual malignant (0)", "Actual benign (1)"],
        columns=["Predicted malignant (0)", "Predicted benign (1)"],
    )

    confusion_fig = px.imshow(
        confusion_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "Predicted class",
            "y": "Actual class",
            "color": "Samples",
        },
        title="Final-round prediction results",
    )
    confusion_fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(
        confusion_fig,
        use_container_width=True,
        key="confusion_matrix_chart",
    )
    st.caption(
        f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}. "
        "Label 0 = malignant; label 1 = benign."
    )
else:
    st.warning(
        "Confusion matrix values are unavailable in this experiment. "
        "Check that metrics.csv contains true_positive, true_negative, "
        "false_positive and false_negative."
    )


# Centralized vs Federated Model Comparison
st.divider()
st.header("Centralized vs Federated Learning")

# Communication Tracking Dashboard
st.divider()
st.header("Federated Communication Analysis")

communication_file = (
    ROOT / "artifacts" / "runs" / selected_run / "communication.csv"
)

if communication_file.exists():
    communication = pd.read_csv(communication_file)

    total_upload = int(communication["upload_bytes"].sum())
    total_download = int(communication["download_bytes"].sum())
    total_communication = int(communication["total_bytes"].sum())

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Upload",
        f"{total_upload:,} bytes",
    )
    col2.metric(
        "Total Download",
        f"{total_download:,} bytes",
    )
    col3.metric(
        "Total Communication",
        f"{total_communication:,} bytes",
    )

    st.subheader("Communication Per Round")

    communication_chart = px.bar(
        communication,
        x="round",
        y=["upload_bytes", "download_bytes"],
        barmode="group",
        labels={
            "round": "Training Round",
            "value": "Payload Size (bytes)",
            "variable": "Direction",
        },
        title="Upload vs Download per Round",
    )

    st.plotly_chart(
        communication_chart,
        use_container_width=True,
    )

    communication["cumulative_bytes"] = (
        communication["total_bytes"].cumsum()
    )

    st.subheader("Cumulative Communication")

    cumulative_chart = px.line(
        communication,
        x="round",
        y="cumulative_bytes",
        markers=True,
        labels={
            "round": "Training Round",
            "cumulative_bytes": "Cumulative Payload (bytes)",
        },
        title="Total Communication Across Training Rounds",
    )

    st.plotly_chart(
        cumulative_chart,
        use_container_width=True,
    )

    st.caption(
        "Communication values are estimated model-payload sizes "
        "for the simulated federated workflow, not measured "
        "network traffic."
    )

else:
    st.info(
        "No communication log exists for this experiment run. "
        "Select a run that contains communication.csv, or run "
        "python main.py to generate a new one."
    )


comparison_file = (
    ROOT / "artifacts" / "comparison" / "comparison.csv"
)

if comparison_file.exists():
    comparison = pd.read_csv(comparison_file)

    st.subheader("Performance Comparison")

    metric = st.selectbox(
        "Select a metric",
        ["accuracy", "precision", "recall", "f1_score", "loss"],
    )

    chart = px.bar(
        comparison,
        x="model",
        y=metric,
        color="model",
        text_auto=".4f",
        title=f"Centralized vs Federated: {metric.replace('_', ' ').title()}",
        labels={
            "model": "Training Approach",
            metric: metric.replace("_", " ").title(),
        },
    )

    chart.update_layout(
        showlegend=False,
        xaxis_title="Training Approach",
        yaxis_title=metric.replace("_", " ").title(),
    )

    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Exact Results")
    st.dataframe(comparison, use_container_width=True)

else:
    st.info(
        "Comparison results not found. Run "
        "`python compare_models.py` first."
    )






# --------------------------------------------------
# FEDERATED EXPERIMENT COMPARISON AND PRIVACY ANALYSIS
# --------------------------------------------------

st.divider()
st.header("Federated Experiment Comparison")

experiment_comparison_file = (
    ROOT / "artifacts" / "comparison" / "experiment_comparison.csv"
)

if experiment_comparison_file.exists():
    experiment_comparison = pd.read_csv(experiment_comparison_file)

    if not experiment_comparison.empty:
        st.subheader("Filter and Compare Experiments")

        available_modes = (
            experiment_comparison["mode"].dropna().unique().tolist()
            if "mode" in experiment_comparison.columns else []
        )
        selected_modes = st.multiselect(
            "Training mode",
            options=available_modes,
            default=available_modes,
            key="experiment_mode_filter",
        )
        filtered_comparison = experiment_comparison[
            experiment_comparison["mode"].isin(selected_modes)
        ].copy() if "mode" in experiment_comparison.columns else experiment_comparison.copy()

        if "noise_multiplier" in filtered_comparison.columns:
            noise_values = sorted(
                pd.to_numeric(
                    filtered_comparison.loc[
                        filtered_comparison["mode"].eq("DP-FedAvg"),
                        "noise_multiplier",
                    ], errors="coerce"
                ).dropna().unique().tolist()
            )
            selected_noise = st.multiselect(
                "Noise multiplier (DP runs)",
                options=noise_values,
                default=noise_values,
                key="experiment_noise_filter",
            )
            dp_mask = filtered_comparison["mode"].eq("DP-FedAvg")
            noise_numeric = pd.to_numeric(
                filtered_comparison["noise_multiplier"], errors="coerce"
            )
            filtered_comparison = filtered_comparison[
                (~dp_mask) | noise_numeric.isin(selected_noise)
            ].copy()

        st.caption(
            f"Showing {len(filtered_comparison)} of "
            f"{len(experiment_comparison)} saved experiments."
        )
        st.dataframe(
            filtered_comparison,
            use_container_width=True,
            hide_index=True,
        )

        accuracy_data = filtered_comparison.copy()
        accuracy_data["accuracy"] = pd.to_numeric(
            accuracy_data["accuracy"], errors="coerce"
        )
        accuracy_data = accuracy_data.dropna(subset=["accuracy"])

        if not accuracy_data.empty:
            accuracy_data["noise_multiplier"] = (
                accuracy_data["noise_multiplier"]
                if "noise_multiplier" in accuracy_data.columns
                else ""
            )

            def experiment_label(row):
                noise = row.get("noise_multiplier", "")
                noise_text = (
                    f" (noise={noise})"
                    if pd.notna(noise) and str(noise).strip()
                    else ""
                )
                return f"{row['mode']}{noise_text} | {row['run_id']}"

            accuracy_data["experiment"] = accuracy_data.apply(
                experiment_label, axis=1
            )

            accuracy_fig = px.bar(
                accuracy_data,
                x="experiment",
                y="accuracy",
                color="mode",
                title="Final Accuracy by Experiment",
                labels={
                    "experiment": "Experiment run",
                    "accuracy": "Final test accuracy",
                    "mode": "Training mode",
                },
                hover_data=[
                    col for col in ["loss", "epsilon", "delta"]
                    if col in accuracy_data.columns
                ],
            )
            accuracy_fig.update_yaxes(tickformat=".0%", range=[0, 1])
            accuracy_fig.update_layout(
                template="plotly_white",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(accuracy_fig, use_container_width=True)

        # Only runs with an actual saved privacy report have epsilon values.
        privacy_data = filtered_comparison.copy()
        if "epsilon" in privacy_data.columns:
            privacy_data["epsilon"] = pd.to_numeric(
                privacy_data["epsilon"], errors="coerce"
            )
            privacy_data = privacy_data.dropna(subset=["epsilon"])
        else:
            privacy_data = pd.DataFrame()

        st.subheader("Privacy Budget Comparison")
        if not privacy_data.empty:
            noise_values = (
                privacy_data["noise_multiplier"].astype(str)
                if "noise_multiplier" in privacy_data.columns
                else pd.Series(["unknown"] * len(privacy_data), index=privacy_data.index)
            )
            privacy_data["experiment"] = (
                privacy_data["mode"].astype(str)
                + " (noise=" + noise_values + ") | "
                + privacy_data["run_id"].astype(str)
            )

            epsilon_fig = px.bar(
                privacy_data,
                x="experiment",
                y="epsilon",
                title="Estimated Epsilon by DP Experiment",
                labels={
                    "experiment": "Experiment run",
                    "epsilon": "Estimated epsilon (lower is generally better)",
                },
                hover_data=[
                    col for col in ["delta", "accuracy", "noise_multiplier"]
                    if col in privacy_data.columns
                ],
            )
            epsilon_fig.update_layout(
                template="plotly_white",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(epsilon_fig, use_container_width=True)
            st.caption(
                "Epsilon values are theoretical estimates under the accountant's "
                "assumptions; they are not proof of end-to-end privacy."
            )
        else:
            st.info(
                "No privacy reports are available in the comparison file. "
                "Run `python compare_experiments.py` after creating DP runs."
            )
    else:
        st.info("The experiment comparison file is empty.")
else:
    st.info(
        "Experiment comparison not found. Run `python compare_experiments.py` first."
    )


# --------------------------------------------------
# PRIVACY REPORT FOR THE SELECTED RUN
# --------------------------------------------------

st.divider()
st.header("Differential Privacy Report")
privacy_file = run_dir / "privacy.json"

if privacy_file.exists():
    try:
        privacy_report = json.loads(
            privacy_file.read_text(encoding="utf-8")
        )

        p1, p2, p3 = st.columns(3)
        epsilon_value = privacy_report.get("epsilon")
        delta_value = privacy_report.get("delta")
        noise_value = privacy_report.get("noise_multiplier")

        p1.metric(
            "Estimated epsilon",
            f"{float(epsilon_value):.4f}" if epsilon_value is not None else "N/A",
        )
        p2.metric(
            "Delta",
            f"{float(delta_value):.2e}" if delta_value is not None else "N/A",
        )
        p3.metric(
            "Noise multiplier",
            str(noise_value) if noise_value is not None else "N/A",
        )

        st.warning(
            "This is a theoretical privacy estimate, not an end-to-end "
            "differential privacy guarantee. Review the assumptions and "
            "limitations below."
        )
        st.write(
            "**Accounting method:**",
            privacy_report.get("accounting_method", "Not specified"),
        )
        st.write(
            "**Adjacency:**",
            privacy_report.get("adjacency", "Not specified"),
        )
        st.write(
            "**Accounted steps per client:**",
            privacy_report.get("accounted_steps_per_client", "N/A"),
        )

        limitations = privacy_report.get("limitations", [])
        if limitations:
            with st.expander("Privacy assumptions and limitations"):
                for limitation in limitations:
                    st.markdown(f"- {limitation}")

        with st.expander("View complete privacy report"):
            st.json(privacy_report)

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        st.error(f"Could not read the privacy report: {error}")
else:
    st.info(
        "The selected run has no privacy.json file. Select a DP-FedAvg run "
        "to view its privacy report."
    )


# --------------------------------------------------
# MLOPS DETAILS
# --------------------------------------------------

st.divider()
st.header("⚙️ MLOps experiment details")

st.write(f"**Experiment:** `{selected_run}`")

with st.expander("View experiment configuration"):
    st.json(config)

with st.expander("View detailed training results"):
    st.dataframe(metrics, use_container_width=True)

st.success(f"Loaded experiment {selected_run}")
