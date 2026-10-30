/**
 * The run detail drawer: one run, opened from anywhere.
 *
 * Kept in its own module because both the dashboard and the runs table need it,
 * and because it is where the MLOps actions live (evaluate -> register), so it
 * needs to be able to refresh whatever view opened it.
 */

import { api } from "../api.js";
import { classBar, legend, lineChart } from "../charts.js";
import {
  badge, clock, el, errorToast, int, metricTile, num, pct, progressBar,
  stageBadge, statusBadge, table, toast, when, clear,
} from "../ui.js";

let scrim = null;
let drawer = null;
let onChange = null;
let currentId = null;
let activeTab = "overview";

function ensureShell() {
  if (drawer) return;
  scrim = el("div", { class: "scrim", onclick: () => closeDrawer() });
  drawer = el("aside", { class: "drawer", role: "dialog", "aria-modal": "true" });
  document.body.append(scrim, drawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
}

export function closeDrawer() {
  if (!drawer) return;
  drawer.classList.remove("open");
  scrim.classList.remove("open");
  currentId = null;
}

export async function openRunDrawer(runId, { onChange: cb = null } = {}) {
  ensureShell();
  onChange = cb;
  currentId = runId;
  activeTab = "overview";
  drawer.classList.add("open");
  scrim.classList.add("open");
  await paint();
}

async function paint() {
  if (!currentId) return;
  clear(drawer);
  drawer.append(el("div", { class: "drawer-body" }, el("div", { class: "empty" }, "Loading run…")));

  let run;
  try {
    run = await api.getRun(currentId);
  } catch (err) {
    errorToast(err);
    clear(drawer);
    drawer.append(el("div", { class: "drawer-body" }, el("div", { class: "empty" }, err.message)));
    return;
  }

  const head = el("div", { class: "drawer-head" },
    el("div", {},
      el("div", { class: "row", style: { gap: "8px" } },
        statusBadge(run.status),
        run.privacy_enabled ? badge("differential privacy", "dp") : null,
        badge(run.partition || "—", "neutral"),
      ),
      el("h1", { style: { fontSize: "17px", marginTop: "8px" } }, run.name || run.id),
      el("div", { class: "dim small mono", style: { marginTop: "2px" } }, run.id),
    ),
    el("button", { class: "btn btn-ghost btn-sm", onclick: closeDrawer, "aria-label": "Close" }, "Close"),
  );

  const tabs = el("div", { class: "tabs" });
  for (const [key, label] of [["overview", "Overview"], ["metrics", "Round metrics"], ["clients", "Clients"], ["config", "Config"]]) {
    tabs.append(el("button", {
      class: `tab ${activeTab === key ? "active" : ""}`,
      onclick: () => { activeTab = key; paint(); },
    }, label));
  }

  const body = el("div", { class: "drawer-body" }, ...renderTab(run));
  clear(drawer); // drop the "Loading run…" placeholder
  drawer.append(head, tabs, body);
}

function renderTab(run) {
  if (activeTab === "metrics") return [metricsTable(run)];
  if (activeTab === "clients") return [clientsTable(run)];
  if (activeTab === "config") return [configBlock(run)];
  return [overview(run)];
}

function overview(run) {
  const nodes = [];

  if (run.error) {
    nodes.push(el("div", { class: "note warn", style: { marginBottom: "16px" } }, `Run failed: ${run.error}`));
  }

  const history = run.history || [];
  const progress = run.status === "running" || run.status === "queued";
  if (progress) {
    const rounds = run.config?.federated?.rounds || 0;
    nodes.push(el("div", { class: "card card-pad", style: { marginBottom: "16px" } },
      el("div", { class: "spread" },
        el("h2", {}, "Training in progress"),
        el("span", { class: "small dim pulse" }, `${history.length}/${rounds} rounds`),
      ),
      el("div", { class: "mt-12" }, progressBar(rounds ? history.length / rounds : 0)),
    ));
  }

  const tiles = el("div", { class: "metric-grid" },
    metricTile("val accuracy", pct(run.val_accuracy)),
    metricTile("val F1", num(run.val_f1)),
    metricTile("val ROC-AUC", num(run.val_roc_auc)),
    metricTile("best round", run.best_round ?? "—"),
    run.is_scored ? metricTile("test accuracy", pct(run.test_accuracy)) : metricTile("test accuracy", "not scored"),
    metricTile("duration", run.duration_s != null ? `${run.duration_s.toFixed(2)}s` : "—"),
  );
  nodes.push(tiles);

  if (history.length) {
    const chart = lineChart({
      height: 220,
      yFormat: (v) => v.toFixed(3),
      series: [
        { name: "val accuracy", color: "#4f46e5", points: history.map((h) => ({ x: h.round, y: h.val_accuracy })) },
        { name: "val ROC-AUC", color: "#0d9488", points: history.map((h) => ({ x: h.round, y: h.val_roc_auc })) },
        { name: "val F1", color: "#f59e0b", points: history.map((h) => ({ x: h.round, y: h.val_f1 })) },
      ],
    });
    nodes.push(el("div", { class: "card mt-16" },
      el("div", { class: "card-head" },
        el("h2", {}, "Validation curve"),
        legend([
          { name: "accuracy", color: "#4f46e5" },
          { name: "ROC-AUC", color: "#0d9488" },
          { name: "F1", color: "#f59e0b" },
        ]),
      ),
      el("div", { class: "card-body" }, chart),
    ));

    nodes.push(el("div", { class: "card mt-16" },
      el("div", { class: "card-head" }, el("h2", {}, "Training loss"), el("span", { class: "hint" }, "mean client loss per round")),
      el("div", { class: "card-body" },
        lineChart({
          height: 170,
          showArea: true,
          yFormat: (v) => v.toFixed(2),
          series: [{ name: "train loss", color: "#64748b", points: history.map((h) => ({ x: h.round, y: h.train_loss })) }],
        })),
    ));
  }

  if (run.is_scored) {
    nodes.push(el("div", { class: "card mt-16" },
      el("div", { class: "card-head" }, el("h2", {}, "Test split (held out)")),
      el("div", { class: "card-body" },
        el("div", { class: "metric-grid" },
          metricTile("accuracy", pct(run.test_metrics.accuracy)),
          metricTile("precision", num(run.test_metrics.precision)),
          metricTile("recall", num(run.test_metrics.recall)),
          metricTile("F1", num(run.test_metrics.f1)),
          metricTile("ROC-AUC", num(run.test_metrics.roc_auc)),
          metricTile("log loss", num(run.test_metrics.log_loss)),
        )),
    ));
  }

  if (run.privacy_enabled) {
    nodes.push(el("div", { class: "note mt-16" },
      `Differential privacy: ε = ${num(run.epsilon, 2)} at δ = ${run.config?.privacy?.delta ?? "1e-5"}, ` +
      `σ = ${num(run.noise_multiplier, 3)}, clip norm C = ${num(run.clip_norm, 2)}. ` +
      "Epsilon is a loose Gaussian-composition bound, not RDP — see docs/PRIVACY.md.",
    ));
  }

  nodes.push(actions(run));
  return el("div", {}, ...nodes);
}

function actions(run) {
  const finished = run.status === "finished";
  const registerButton = el("button", {
    class: "btn btn-primary",
    disabled: !finished || !run.is_scored,
    onclick: async (event) => {
      const btn = event.currentTarget;
      btn.disabled = true;
      btn.textContent = "Registering…";
      try {
        const mv = await api.registerRun(run.id, { name: "wdbc-logreg", stage: "staging", description: run.name || "" });
        toast(`Registered ${mv.name}:${mv.version} (${mv.stage})`, "ok");
        onChange?.();
        paint();
      } catch (err) {
        errorToast(err);
        btn.disabled = false;
        btn.textContent = "Register as staging";
      }
    },
  }, "Register as staging");

  const evaluateButton = el("button", {
    class: "btn",
    disabled: !finished,
    onclick: async (event) => {
      const btn = event.currentTarget;
      btn.disabled = true;
      btn.textContent = "Evaluating…";
      try {
        const result = await api.evaluateRun(run.id);
        toast(`Test accuracy ${pct(result.test.accuracy)}`, "ok");
        onChange?.();
        paint();
      } catch (err) {
        errorToast(err);
        btn.disabled = false;
        btn.textContent = "Evaluate on test split";
      }
    },
  }, run.is_scored ? "Re-evaluate" : "Evaluate on test split");

  return el("div", { class: "card card-pad mt-16" },
    el("div", { class: "spread" },
      el("div", {},
        el("h2", {}, "Lifecycle"),
        el("div", { class: "small dim mt-8" },
          run.is_scored
            ? "Scored on the test split — eligible for the registry."
            : "A run must be scored before it can enter the registry."),
      ),
      el("div", { class: "row" }, evaluateButton, registerButton),
    ),
    run.registered_versions?.length
      ? el("div", { class: "small mt-12" }, "Registered as: ",
          ...run.registered_versions.map((v) => badge(v, "staging", null)))
      : null,
  );
}

function metricsTable(run) {
  const rows = (run.history || []).map((h) => ({ ...h }));
  return el("div", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", {}, "Per-round metrics"),
      el("span", { class: "hint" }, `${rows.length} rows written to the database`),
    ),
    table([
      { label: "round", key: "round", num: true },
      { label: "clients", key: "n_clients", num: true },
      { label: "train loss", render: (r) => num(r.train_loss), num: true },
      { label: "val acc", render: (r) => pct(r.val_accuracy), num: true },
      { label: "val F1", render: (r) => num(r.val_f1), num: true },
      { label: "val AUC", render: (r) => num(r.val_roc_auc), num: true },
      { label: "|Δ| norm", render: (r) => num(r.mean_update_norm), num: true },
      { label: "clipped", render: (r) => pct(r.clipped_fraction, 0), num: true },
      { label: "ε", render: (r) => (r.epsilon ? num(r.epsilon, 2) : "—"), num: true },
    ], rows),
  );
}

function clientsTable(run) {
  const clients = run.clients || [];
  if (!clients.length) return el("div", { class: "empty" }, "No client summary stored for this run.");
  return el("div", { class: "card" },
    el("div", { class: "card-head" },
      el("h2", {}, "Simulated clients"),
      el("span", { class: "hint" }, "green = benign, red = malignant"),
    ),
    el("div", { class: "card-body stack", style: { gap: "10px" } },
      clients.map((c) => el("div", { class: "client-row" },
        el("span", { class: "mono" }, c.client),
        classBar(c.positive_rate),
        el("span", { class: "dim", style: { textAlign: "right" } }, `${c.samples} rows · ${pct(c.positive_rate, 0)}`),
      )),
    ),
  );
}

function configBlock(run) {
  return el("div", { class: "card card-pad" },
    el("h2", {}, "Configuration"),
    el("p", { class: "small dim mt-8" }, "Exactly what was submitted — enough to reproduce the run."),
    el("pre", { class: "json mt-12" }, JSON.stringify(run.config, null, 2)),
  );
}
