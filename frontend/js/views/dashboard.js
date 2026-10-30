/**
 * Dashboard: is the pipeline healthy, what is in production, what did we last run.
 */

import { api } from "../api.js";
import { classBar, legend, lineChart } from "../charts.js";
import {
  badge, card, el, errorToast, int, kpi, num, pct, stageBadge, statusBadge,
  table, when,
} from "../ui.js";
import { openRunDrawer } from "./runDrawer.js";

export async function renderDashboard(root) {
  const [stats, runs, versions] = await Promise.all([
    api.stats(),
    api.listRuns({ limit: 8 }),
    api.listModels().catch(() => []),
  ]);

  const head = el("div", { class: "page-head" },
    el("div", {},
      el("h1", {}, "Dashboard"),
      el("p", { class: "sub" },
        "Federated training across simulated clients, with runs, a model registry and serving — " +
        "on the bundled Wisconsin breast-cancer dataset."),
    ),
    el("div", { class: "page-actions" },
      el("a", { class: "btn", href: "#/runs" }, "All runs"),
      el("a", { class: "btn btn-primary", href: "#/train" }, "New experiment"),
    ),
  );

  const production = stats.production_model;
  const kpis = el("div", { class: "grid grid-4 mt-20" },
    kpi("Runs tracked", int(stats.total_runs),
      `${stats.finished_runs} finished · ${stats.failed_runs} failed`),
    kpi("Best test accuracy", stats.best_test_accuracy != null ? pct(stats.best_test_accuracy) : "—",
      stats.best_run_id ? `from ${stats.best_run_id}` : "score a run to fill this",
      stats.best_test_accuracy != null ? "good" : ""),
    kpi("Production model", production ? `${production.name}:${production.version}` : "none",
      production ? `test ${pct(production.test_accuracy)} · ${production.stage}` : "register and promote a version"),
    kpi("Rounds trained", int(stats.total_rounds),
      `${stats.model_versions} model versions · ${stats.predictions} prediction job${stats.predictions === 1 ? "" : "s"}`),
  );

  const root_children = [head, kpis];

  if (!runs.length) {
    root_children.push(el("div", { class: "card card-pad mt-20" },
      el("div", { class: "empty" },
        el("strong", {}, "No experiments yet"),
        "Start a federated run — it takes about a second on this dataset.",
        el("div", { class: "mt-16" },
          el("a", { class: "btn btn-primary", href: "#/train" }, "Train a model")),
      )));
    root.append(...root_children);
    return;
  }

  // best run curve
  const bestRun = stats.best_run_id ? await api.getRun(stats.best_run_id).catch(() => null) : null;
  const curveCard = card("Validation curve of the best run",
    el("div", { class: "card-body" },
      bestRun && bestRun.history.length
        ? lineChart({
            height: 230,
            yFormat: (v) => v.toFixed(3),
            series: [
              { name: "val accuracy", color: "#4f46e5", points: bestRun.history.map((h) => ({ x: h.round, y: h.val_accuracy })) },
              { name: "val ROC-AUC", color: "#0d9488", points: bestRun.history.map((h) => ({ x: h.round, y: h.val_roc_auc })) },
            ],
          })
        : el("div", { class: "empty" }, "No history stored."),
    ),
    {
      hint: bestRun ? `${bestRun.name} · ${pct(bestRun.val_accuracy)} val accuracy` : undefined,
      actions: bestRun
        ? [el("button", { class: "btn btn-sm", onclick: () => openRunDrawer(bestRun.id) }, "Open run")]
        : null,
    },
  );

  const registryRows = (versions || []).slice(0, 4).map((v) => ({ ...v }));
  const registryCard = card("Model registry",
    versions.length
      ? el("div", {}, table([
          { label: "version", render: (v) => el("strong", {}, `${v.name}:${v.version}`) },
          { label: "stage", render: (v) => stageBadge(v.stage) },
          { label: "test acc", render: (v) => pct(v.test_accuracy), num: true },
        ], registryRows))
      : el("div", { class: "empty" },
          el("strong", {}, "Registry is empty"),
          "Score a run, then register it."),
    {
      hint: "staging → production",
      actions: [el("a", { class: "btn btn-sm", href: "#/registry" }, "Manage")],
    },
  );

  root_children.push(el("div", { class: "grid grid-side mt-16" }, curveCard, registryCard));

  // dataset card
  let balance = null;
  try {
    const preview = await api.datasetPreview(1);
    balance = preview;
  } catch { /* the dataset card is decoration; a failure here is not fatal */ }

  if (balance) {
    const total = balance.class_balance.benign + balance.class_balance.malignant;
    root_children.push(el("div", { class: "card card-pad mt-16" },
      el("div", { class: "spread" },
        el("div", {},
          el("h2", {}, "Bundled dataset"),
          el("div", { class: "small dim mt-8" },
            `Wisconsin Diagnostic Breast Cancer · ${int(balance.n_rows)} rows · ${balance.n_features} features · binary target`),
        ),
        el("div", { class: "row" },
          badge(`${balance.class_balance.benign} benign`, "production"),
          badge(`${balance.class_balance.malignant} malignant`, "failed"),
        ),
      ),
      el("div", { class: "mt-12" }, classBar(balance.class_balance.benign / total, 10)),
    ));
  }

  // recent runs
  root_children.push(card("Recent runs",
    table([
      { label: "run", render: (r) => el("div", {},
          el("div", { style: { fontWeight: "550" } }, r.name || r.id),
          el("div", { class: "dim mono", style: { fontSize: "11.5px" } }, r.id)) },
      { label: "status", render: (r) => statusBadge(r.status) },
      { label: "setup", render: (r) => `${r.n_clients ?? "—"} clients · ${r.partition ?? "—"}` },
      { label: "privacy", render: (r) => (r.privacy_enabled ? badge(`ε ${num(r.epsilon, 1)}`, "dp") : el("span", { class: "dim" }, "off")) },
      { label: "val acc", render: (r) => pct(r.val_accuracy), num: true },
      { label: "test acc", render: (r) => (r.is_scored ? pct(r.test_accuracy) : el("span", { class: "dim" }, "not scored")), num: true },
      { label: "created", render: (r) => el("span", { class: "dim" }, when(r.created_at)) },
    ], runs, { onRowClick: (r) => openRunDrawer(r.id, { onChange: () => renderDashboard(root) }) }),
    { hint: "click a row for the full run", actions: [el("a", { class: "btn btn-sm", href: "#/runs" }, "View all")] },
  ));

  root.append(...root_children);
}
