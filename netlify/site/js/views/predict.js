/**
 * Predict view: run the production model over the bundled dataset or an upload.
 *
 * The threshold slider matters clinically -- moving it trades recall against
 * precision, and the confusion matrix updates so you can see that trade.
 */

import { api } from "../api.js";
import { confusionMatrix } from "../charts.js";
import {
  badge, card, el, errorToast, int, metricTile, num, pct, table, toast, when,
} from "../ui.js";

export async function renderPredict(root) {
  const state = {
    target: "production",
    threshold: 0.5,
    source: "bundled",
    limit: 569,
    file: null,
    result: null,
  };

  let versions = [];
  try {
    versions = await api.listModels({ name: "wdbc-logreg" });
  } catch { /* the model picker degrades to the production default */ }

  root.append(el("div", { class: "page-head" },
    el("div", {},
      el("h1", {}, "Predict"),
      el("p", { class: "sub" },
        "Serve the registered model. The scaler travels with the version, so raw " +
        "feature columns are enough."),
    ),
    el("div", { class: "page-actions" },
      el("a", { class: "btn", href: "#/registry" }, "Registry")),
  ));

  /* ---------------------------------------------------------- form */
  const versionSelect = el("select", {
    onchange: (event) => { state.target = event.target.value; },
  },
    el("option", { value: "production" }, "production (what serving reads)"),
    el("option", { value: "staging" }, "staging (latest candidate)"),
    ...versions.map((v) => el("option", { value: `version:${v.id}` },
      `pin ${v.name}:${v.version} (${v.stage}) — ${pct(v.test_accuracy)} test`)),
  );

  // "version:12" pins one exact version; anything else is a stage name, which
  // the server resolves to the newest version in that stage. Selecting a version
  // by *stage* used to silently score with whatever was newest instead.
  function resolveTarget() {
    if (state.target.startsWith("version:")) {
      return { model_version_id: Number(state.target.slice("version:".length)) };
    }
    return { stage: state.target };
  }

  const thresholdValue = el("span", { class: "val" }, "0.50");
  const thresholdInput = el("input", {
    type: "range", min: 0.05, max: 0.95, step: 0.01, value: state.threshold,
    oninput: (event) => {
      state.threshold = Number(event.target.value);
      thresholdValue.textContent = state.threshold.toFixed(2);
    },
  });

  const limitSelect = el("select", {
    onchange: (event) => { state.limit = Number(event.target.value); },
  },
    el("option", { value: "50" }, "first 50 rows"),
    el("option", { value: "200" }, "first 200 rows"),
    el("option", { value: "569", selected: true }, "all 569 rows"),
  );

  const fileInput = el("input", {
    type: "file", accept: ".csv,text/csv",
    onchange: (event) => { state.file = event.target.files?.[0] || null; },
  });

  const uploadRow = el("div", { class: "field", style: { display: "none" } },
    el("label", {}, "CSV file"), fileInput,
    el("div", { class: "help" }, "header must contain the 30 feature columns; a target column is optional"),
  );

  const sourceSelect = el("select", {
    onchange: (event) => {
      state.source = event.target.value;
      uploadRow.style.display = state.source === "upload" ? "" : "none";
      bundledRow.style.display = state.source === "bundled" ? "" : "none";
    },
  },
    el("option", { value: "bundled" }, "bundled dataset (breast_cancer.csv)"),
    el("option", { value: "upload" }, "upload a CSV"),
  );

  const bundledRow = el("div", { class: "field" }, el("label", {}, "Rows"), limitSelect);

  const runButton = el("button", { class: "btn btn-primary btn-block" }, "Run prediction");

  const form = el("form", { class: "form-grid", onsubmit: (event) => event.preventDefault() },
    el("div", { class: "field" }, el("label", {}, "Model version"), versionSelect,
      el("div", { class: "help" }, "only versions in a stage can serve")),
    el("div", { class: "field" }, el("label", {}, "Decision threshold", thresholdValue), thresholdInput,
      el("div", { class: "help" }, "p_benign ≥ threshold → predicted benign")),
    el("div", { class: "field" }, el("label", {}, "Input"), sourceSelect),
    bundledRow,
    uploadRow,
    runButton,
  );

  const resultsHost = el("div", { class: "stack", style: { gap: "16px" } },
    el("div", { class: "card card-pad" },
      el("div", { class: "empty" },
        el("strong", {}, "No prediction yet"),
        "Pick a model and run it — results appear here, with a downloadable CSV.")));

  root.append(el("div", { class: "grid grid-side mt-20" },
    card("Scoring", el("div", { class: "card-body" }, form), { hint: "POST /api/predict" }),
    resultsHost,
  ));

  runButton.addEventListener("click", async (event) => {
    event.preventDefault();
    const btn = event.currentTarget;
    btn.disabled = true;
    btn.textContent = "Scoring…";
    try {
      let result;
      const target = resolveTarget();
      if (state.source === "upload") {
        if (!state.file) throw new Error("choose a CSV file first");
        result = await api.predictUpload({
          file: state.file,
          threshold: state.threshold,
          ...target,
        });
      } else {
        result = await api.predict({
          ...target,
          threshold: state.threshold,
          use_bundled_dataset: true,
          limit: state.limit,
        });
      }
      state.result = result;
      toast(`${result.n_rows} rows scored with ${result.model_label}`, "ok");
      paintResult(resultsHost, result);
      await paintHistory(historyHost);
    } catch (err) {
      errorToast(err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run prediction";
    }
  });

  /* ---------------------------------------------------------- history */
  const historyHost = el("div", { class: "mt-16" });
  root.append(historyHost);
  await paintHistory(historyHost);
}

function paintResult(host, result) {
  const summary = result.summary || {};
  const tiles = el("div", { class: "metric-grid" },
    metricTile("rows scored", int(result.n_rows)),
    metricTile("predicted benign", int(summary.predicted_benign)),
    metricTile("predicted malignant", int(summary.predicted_malignant)),
    metricTile("threshold", num(result.threshold, 2)),
    summary.accuracy != null ? metricTile("accuracy vs target", pct(summary.accuracy)) : null,
  );

  const children = [
    el("div", { class: "card-head" },
      el("div", {},
        el("div", { class: "row" },
          badge(result.model_label, "running"),
          el("span", { class: "small dim" }, result.source_label)),
        el("div", { class: "small dim mt-8" }, `created ${when(result.created_at)}`),
      ),
      result.download_url
        ? el("a", { class: "btn btn-sm btn-primary", href: result.download_url, download: "" }, "Download CSV")
        : el("span", { class: "small dim" }, "in-browser demo — no CSV"),
    ),
    el("div", { class: "card-body" }, tiles),
  ];

  if (summary.confusion) {
    children.push(el("div", { class: "card-body", style: { borderTop: "1px solid var(--line)" } },
      el("h2", { style: { marginBottom: "12px" } }, "Confusion matrix"),
      confusionMatrix(summary.confusion),
      el("div", { class: "small dim mt-12" }, summary.evaluated_against),
    ));
  }

  if (result.preview?.length) {
    const columns = Object.keys(result.preview[0]).map((key) => ({
      label: key,
      render: (row) => (key === "prediction_label"
        ? badge(row[key], row[key] === "benign" ? "production" : "failed")
        : key === "p_benign" ? num(row[key], 3) : String(row[key] ?? "—")),
      num: key === "p_benign",
    }));
    children.push(el("div", { style: { borderTop: "1px solid var(--line)" } },
      table(columns, result.preview, { empty: "No preview rows." })));
    children.push(el("div", { class: "card-body small dim" },
      `Showing the first ${result.preview.length} of ${result.n_rows} rows. The full file is in the download.`));
  }

  host.replaceChildren(el("section", { class: "card" }, ...children));
}

async function paintHistory(host) {
  let predictions = [];
  try {
    predictions = await api.listPredictions(8);
  } catch {
    host.replaceChildren();
    return;
  }
  if (!predictions.length) {
    host.replaceChildren();
    return;
  }
  host.replaceChildren(card("Recent prediction jobs",
    table([
      { label: "model", render: (p) => el("span", { class: "mono small" }, p.model_label) },
      { label: "source", render: (p) => p.source_label },
      { label: "rows", render: (p) => int(p.n_rows), num: true },
      { label: "threshold", render: (p) => num(p.threshold, 2), num: true },
      { label: "accuracy", render: (p) => (p.summary?.accuracy != null ? pct(p.summary.accuracy) : el("span", { class: "dim" }, "—")), num: true },
      { label: "when", render: (p) => el("span", { class: "dim nowrap" }, when(p.created_at)) },
      { label: "", render: (p) => el("div", { class: "row-actions" },
          el("a", { class: "btn btn-sm", href: p.download_url, download: "" }, "CSV")) },
    ], predictions.map((p) => ({ ...p }))),
    { hint: "every scoring job is recorded" },
  ));
}
