/**
 * Runs table: the experiment log, with the lifecycle actions inline.
 */

import { api } from "../api.js";
import {
  badge, card, el, errorToast, num, pct, statusBadge, table, toast, when,
} from "../ui.js";
import { openRunDrawer } from "./runDrawer.js";

const FILTERS = [
  { key: "", label: "All" },
  { key: "finished", label: "Finished" },
  { key: "running", label: "Running" },
  { key: "failed", label: "Failed" },
];

export async function renderRuns(root) {
  let filter = "";
  let scoredOnly = false;

  root.append(el("div", { class: "page-head" },
    el("div", {},
      el("h1", {}, "Runs"),
      el("p", { class: "sub" }, "Every training run, its configuration, and where it is in the model lifecycle."),
    ),
    el("div", { class: "page-actions" },
      el("button", { class: "btn", onclick: () => load() }, "Refresh"),
      el("a", { class: "btn btn-primary", href: "#/train" }, "New run")),
  ));

  const chips = el("div", { class: "chip-row" },
    ...FILTERS.map((f) => el("button", {
      class: `chip ${filter === f.key ? "active" : ""}`,
      onclick: (event) => {
        filter = f.key;
        chips.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
        event.currentTarget.classList.add("active");
        load();
      },
    }, f.label)),
    el("button", {
      class: "chip",
      onclick: (event) => {
        scoredOnly = !scoredOnly;
        event.currentTarget.classList.toggle("active", scoredOnly);
        load();
      },
    }, "scored only"),
  );

  const tableHost = el("div", { class: "mt-16" });
  root.append(el("div", { class: "mt-20" }, chips), tableHost);

  async function load() {
    tableHost.replaceChildren(el("div", { class: "empty" }, "Loading…"));
    let runs;
    try {
      runs = await api.listRuns({ status: filter || undefined, scored_only: scoredOnly, limit: 200 });
    } catch (err) {
      errorToast(err);
      tableHost.replaceChildren(el("div", { class: "empty" }, err.message));
      return;
    }

    const node = card("", table([
      { label: "run", render: (r) => el("div", {},
          el("div", { style: { fontWeight: "550" } }, r.name || r.id),
          el("div", { class: "dim mono", style: { fontSize: "11.5px" } }, r.id)) },
      { label: "status", render: (r) => statusBadge(r.status) },
      { label: "setup", render: (r) => el("span", { class: "nowrap" },
          `${r.n_clients ?? "—"} clients · ${r.partition ?? "—"} · ${r.rounds_run ?? 0} rounds`) },
      { label: "privacy", render: (r) => (r.privacy_enabled ? badge(`ε ${num(r.epsilon, 1)}`, "dp") : el("span", { class: "dim" }, "off")) },
      { label: "val acc", render: (r) => pct(r.val_accuracy), num: true },
      { label: "test acc", render: (r) => (r.is_scored ? pct(r.test_accuracy) : el("span", { class: "dim" }, "—")), num: true },
      { label: "test F1", render: (r) => (r.is_scored ? num(r.test_f1) : el("span", { class: "dim" }, "—")), num: true },
      { label: "created", render: (r) => el("span", { class: "dim nowrap" }, when(r.created_at)) },
      { label: "", render: (r) => el("div", { class: "row-actions" },
          el("button", {
            class: "btn btn-sm",
            disabled: r.status !== "finished",
            onclick: async (event) => {
              event.stopPropagation();
              const btn = event.currentTarget;
              btn.disabled = true;
              btn.textContent = "…";
              try {
                const result = await api.evaluateRun(r.id);
                toast(`${r.id}: test accuracy ${pct(result.test.accuracy)}`, "ok");
                load();
              } catch (err) {
                errorToast(err);
                btn.disabled = false;
                btn.textContent = "Score";
              }
            },
          }, r.is_scored ? "Re-score" : "Score"),
          el("button", {
            class: "btn btn-sm",
            disabled: r.status !== "finished" || !r.is_scored,
            title: r.is_scored ? "Register as a staging version" : "Score the run first",
            onclick: async (event) => {
              event.stopPropagation();
              const btn = event.currentTarget;
              btn.disabled = true;
              try {
                const mv = await api.registerRun(r.id, { name: "wdbc-logreg", stage: "staging", description: r.name || "" });
                toast(`Registered ${mv.name}:${mv.version}`, "ok");
                load();
              } catch (err) {
                errorToast(err);
                btn.disabled = false;
              }
            },
          }, "Register"),
          el("button", {
            class: "btn btn-sm btn-danger",
            title: "Delete this run",
            onclick: async (event) => {
              event.stopPropagation();
              if (!window.confirm(`Delete run ${r.id}? This removes its metrics and model.`)) return;
              try {
                await api.deleteRun(r.id);
                toast("Run deleted");
                load();
              } catch (err) {
                errorToast(err);
              }
            },
          }, "Delete"),
        ) },
    ], runs, {
      onRowClick: (r) => openRunDrawer(r.id, { onChange: load }),
      empty: "No runs match this filter.",
    }), { hint: `${runs.length} run(s)` });

    tableHost.replaceChildren(node);
  }

  await load();
}
