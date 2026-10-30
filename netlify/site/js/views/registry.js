/**
 * Registry view: versions, stages, and the promotion gate.
 *
 * A version is immutable. Promotion only moves a stage pointer, which is why
 * "roll back" is just promoting the previous version again.
 */

import { api } from "../api.js";
import {
  badge, card, el, errorToast, num, pct, stageBadge, table, toast, when,
} from "../ui.js";
import { openRunDrawer } from "./runDrawer.js";

export async function renderRegistry(root) {
  const gateInput = el("input", {
    type: "number", min: 0, max: 1, step: 0.01, value: 0.9,
    style: { width: "110px" },
  });
  const gate = () => Number(gateInput.value);

  root.append(el("div", { class: "page-head" },
    el("div", {},
      el("h1", {}, "Model registry"),
      el("p", { class: "sub" },
        "Immutable, numbered versions of the model. Staging is for candidates; " +
        "production is what serving reads."),
    ),
    el("div", { class: "page-actions" },
      el("button", { class: "btn", onclick: () => load() }, "Refresh"),
      el("a", { class: "btn btn-primary", href: "#/runs" }, "Register a run")),
  ));

  const banner = el("div", { class: "mt-20" });
  const gateCard = card("Promotion gate",
    el("div", { class: "card-body" },
      el("div", { class: "row", style: { gap: "12px", flexWrap: "wrap" } },
        el("div", { class: "field", style: { maxWidth: "150px" } },
          el("label", {}, "Minimum test accuracy"), gateInput),
        el("div", { class: "small muted", style: { maxWidth: "460px" } },
          "Promoting to production is refused if the version's recorded test accuracy is " +
          "below this value. The server enforces it — the browser is just the messenger."),
      ),
    ),
  );
  const versionsHost = el("div", { class: "mt-16" });
  root.append(banner, el("div", { class: "mt-16" }, gateCard), versionsHost);

  async function load() {
    banner.replaceChildren();
    versionsHost.replaceChildren(el("div", { class: "empty" }, "Loading…"));

    let versions = [];
    try {
      versions = await api.listModels({ name: "wdbc-logreg" });
    } catch (err) {
      errorToast(err);
      versionsHost.replaceChildren(el("div", { class: "empty" }, err.message));
      return;
    }

    const production = versions.find((v) => v.stage === "production");
    if (production) {
      banner.append(el("div", { class: "card card-pad" },
        el("div", { class: "spread" },
          el("div", {},
            el("div", { class: "row" },
              stageBadge("production"),
              el("h2", {}, `${production.name}:${production.version}`)),
            el("div", { class: "small dim mt-8" },
              `Serving from ${production.version} · test accuracy ${pct(production.test_accuracy)} · ` +
              `registered ${when(production.created_at)}`),
            production.description ? el("div", { class: "small mt-8" }, production.description) : null,
          ),
          el("div", { class: "row" },
            production.run_id
              ? el("button", { class: "btn btn-sm", onclick: () => openRunDrawer(production.run_id, { onChange: load }) }, "Source run")
              : null,
            el("a", { class: "btn btn-sm btn-primary", href: "#/predict" }, "Predict with it")),
        ),
      ));
    } else {
      banner.append(el("div", { class: "card card-pad" },
        el("div", { class: "empty" },
          el("strong", {}, "Nothing in production"),
          "Score a run, register it as staging, then promote it here."),
      ));
    }

    if (!versions.length) {
      versionsHost.replaceChildren(el("div", { class: "card card-pad" },
        el("div", { class: "empty" },
          el("strong", {}, "No versions registered"),
          "A run must be scored on the test split before it can be registered.")));
      return;
    }

    const rows = versions.map((v) => ({ ...v }));
    versionsHost.replaceChildren(card("Versions",
      table([
        { label: "version", render: (v) => el("div", {},
            el("strong", {}, `${v.name}:${v.version}`),
            v.description ? el("div", { class: "dim small" }, v.description) : null) },
        { label: "stage", render: (v) => stageBadge(v.stage) },
        { label: "test acc", render: (v) => pct(v.test_accuracy), num: true },
        { label: "test F1", render: (v) => num(v.test_f1), num: true },
        { label: "privacy", render: (v) => (v.epsilon ? badge(`ε ${num(v.epsilon, 1)}`, "dp") : el("span", { class: "dim" }, "off")) },
        { label: "source run", render: (v) => (v.run_id
            ? el("button", {
                class: "btn btn-ghost btn-sm mono",
                onclick: () => openRunDrawer(v.run_id, { onChange: load }),
              }, v.run_id)
            : el("span", { class: "dim" }, "—")) },
        { label: "registered", render: (v) => el("span", { class: "dim nowrap" }, when(v.created_at)) },
        { label: "", render: (v) => el("div", { class: "row-actions" },
            v.stage !== "production"
              ? el("button", {
                  class: "btn btn-sm btn-primary",
                  onclick: async (event) => {
                    const btn = event.currentTarget;
                    btn.disabled = true;
                    btn.textContent = "…";
                    try {
                      await api.promote(v.id, { stage: "production", min_accuracy: gate() });
                      toast(`${v.version} promoted to production`, "ok");
                      load();
                    } catch (err) {
                      errorToast(err);
                      btn.disabled = false;
                      btn.textContent = "Promote";
                    }
                  },
                }, "Promote")
              : null,
            v.stage !== "staging"
              ? el("button", {
                  class: "btn btn-sm",
                  onclick: async () => {
                    try {
                      await api.promote(v.id, { stage: "staging" });
                      toast(`${v.version} moved to staging`, "ok");
                      load();
                    } catch (err) { errorToast(err); }
                  },
                }, "To staging")
              : null,
            el("button", {
              class: "btn btn-sm btn-danger",
              disabled: v.stage === "production",
              title: v.stage === "production" ? "Promote another version first" : "Delete this version",
              onclick: async () => {
                if (!window.confirm(`Delete ${v.name}:${v.version}?`)) return;
                try {
                  await api.deleteVersion(v.id);
                  toast("Version deleted");
                  load();
                } catch (err) { errorToast(err); }
              },
            }, "Delete"),
          ) },
      ], rows, { empty: "No versions registered." }),
      { hint: `${rows.length} version(s) · promoting archives whoever held the stage` },
    ));
  }

  await load();
}
