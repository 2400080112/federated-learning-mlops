/**
 * Train view: the experiment form, plus a live panel that polls the run it just
 * started. The form is the interesting part of the demo -- it is where the
 * federated and privacy knobs become visible.
 */

import { api } from "../api.js";
import { legend, lineChart } from "../charts.js";
import {
  badge, card, clear, el, errorToast, num, pct, progressBar, statusBadge,
  table, toast,
} from "../ui.js";
import { openRunDrawer } from "./runDrawer.js";

let poller = null;

export async function renderTrain(root) {
  stopPolling();

  let defaults;
  try {
    defaults = await api.defaults();
  } catch (err) {
    errorToast(err);
    defaults = { n_clients: 8, rounds: 25, local_epochs: 3, batch_size: 16, learning_rate: 0.15, l2: 0.001, partition: "iid", dirichlet_alpha: 0.5, seed: 7, privacy_enabled: false, dp_clip: 1.0, dp_epsilon: null };
  }

  root.append(el("div", { class: "page-head" },
    el("div", {},
      el("h1", {}, "Train"),
      el("p", { class: "sub" },
        "Each round: send the global model to a subset of clients, train locally, " +
        "average the updates (FedAvg), validate. No client row ever leaves its client."),
    ),
    el("div", { class: "page-actions" },
      el("a", { class: "btn", href: "#/runs" }, "Run history")),
  ));

  const form = buildForm(defaults);
  const liveHost = el("div", { class: "stack", style: { gap: "16px" } },
    el("div", { class: "card card-pad" },
      el("h2", {}, "Live run"),
      el("p", { class: "small dim mt-8" },
        "Start a run and its progress appears here: one database row per round, " +
        "polled from /api/runs/{id}."),
      el("div", { class: "mt-16 dim small" }, "Nothing running yet."),
    ),
  );

  root.append(el("div", { class: "grid grid-side mt-20" },
    card("Experiment configuration", el("div", { class: "card-body" }, form.node),
      { hint: "every field maps to a validated API field" }),
    liveHost,
  ));

  if (api.isDemo) {
    const submitBtn = form.node.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Training needs the live API";
    }
    form.node.append(el("div", { class: "note warn" },
      "This build is a static snapshot of a seeded run, so the form is read-only. " +
      "Run the FastAPI backend locally, or point the site at a live API — see docs/DEPLOY_NETLIFY.md."));
  }

  form.onSubmit(async (payload, button) => {
    button.disabled = true;
    button.textContent = "Starting…";
    try {
      const run = await api.createRun(payload);
      toast(`Run ${run.id} queued`, "ok");
      watchRun(run.id, liveHost);
      window.location.hash = "#/train";
    } catch (err) {
      errorToast(err);
    } finally {
      button.disabled = false;
      button.textContent = "Start federated run";
    }
  });

  return stopPolling;
}

function stopPolling() {
  if (poller) {
    clearInterval(poller);
    poller = null;
  }
}

/* ------------------------------------------------------------------ form */
function buildForm(defaults) {
  const state = { ...defaults, clients_per_round: defaults.n_clients };

  const numberField = (key, label, { min, max, step = 1, help = "", format = null } = {}) => {
    const input = el("input", {
      type: "number", min, max, step, value: state[key],
      oninput: (event) => { state[key] = Number(event.target.value); },
    });
    return el("div", { class: "field" },
      el("label", {}, label),
      input,
      help ? el("div", { class: "help" }, help) : null,
    );
  };

  const rangeField = (key, label, { min, max, step, help = "", format = (v) => v }) => {
    const valueTag = el("span", { class: "val" }, format(state[key]));
    const input = el("input", {
      type: "range", min, max, step, value: state[key],
      oninput: (event) => {
        state[key] = Number(event.target.value);
        valueTag.textContent = format(state[key]);
        if (key === "n_clients") syncClientsPerRound();
      },
    });
    return el("div", { class: "field" },
      el("label", {}, label, valueTag),
      input,
      help ? el("div", { class: "help" }, help) : null,
    );
  };

  const nameInput = el("input", {
    type: "text", placeholder: "e.g. 8 clients · iid · 25 rounds",
    oninput: (event) => { state.name = event.target.value; },
  });

  const cprValue = el("span", { class: "val" }, String(state.clients_per_round));
  const cprInput = el("input", {
    type: "range", min: 1, max: state.n_clients, step: 1, value: state.clients_per_round,
    oninput: (event) => {
      state.clients_per_round = Number(event.target.value);
      cprValue.textContent = String(state.clients_per_round);
    },
  });

  function syncClientsPerRound() {
    cprInput.max = String(state.n_clients);
    if (state.clients_per_round > state.n_clients) {
      state.clients_per_round = state.n_clients;
      cprInput.value = String(state.n_clients);
    }
    cprValue.textContent = String(state.clients_per_round);
  }

  const partitionSelect = el("select", {
    onchange: (event) => {
      state.partition = event.target.value;
      alphaField.style.display = state.partition === "dirichlet" ? "" : "none";
    },
  },
    el("option", { value: "iid" }, "iid — evenly mixed clients"),
    el("option", { value: "dirichlet" }, "dirichlet — label-skewed clients"),
  );

  const alphaField = rangeField("dirichlet_alpha", "Dirichlet α (lower = more skew)", {
    min: 0.05, max: 2, step: 0.05, format: (v) => Number(v).toFixed(2),
    help: "α = 0.1 gives clients that hold almost one class only",
  });
  alphaField.style.display = state.partition === "dirichlet" ? "" : "none";

  // privacy
  const privacyToggle = el("input", {
    type: "checkbox",
    onchange: (event) => {
      state.privacy_enabled = event.target.checked;
      privacyBody.style.display = state.privacy_enabled ? "" : "none";
    },
  });
  const epsilonInput = el("input", {
    type: "number", min: 0.1, max: 100, step: 0.1, value: 8,
    oninput: (event) => { state.dp_epsilon = Number(event.target.value); },
  });
  const clipInput = el("input", {
    type: "number", min: 0.1, max: 20, step: 0.1, value: state.dp_clip ?? 1,
    oninput: (event) => { state.dp_clip = Number(event.target.value); },
  });

  const privacyBody = el("div", { class: "form-grid", style: { display: "none", marginTop: "12px" } },
    el("div", { class: "field" },
      el("label", {}, "ε budget (privacy loss)"),
      epsilonInput,
      el("div", { class: "help" }, "σ is derived from ε and the round count, so more rounds cost more privacy"),
    ),
    el("div", { class: "field" },
      el("label", {}, "clip norm C"),
      clipInput,
      el("div", { class: "help" }, "client updates are clipped to this L2 norm before noise is added"),
    ),
    el("div", { class: "note" },
      "Noise is added to the aggregated update, calibrated to max(wᵢ)·C. The reported ε is a " +
      "loose Gaussian-composition bound — see docs/PRIVACY.md."),
  );

  const submit = el("button", { class: "btn btn-primary btn-block", type: "submit" }, "Start federated run");

  const node = el("form", { class: "form-grid", onsubmit: (event) => { event.preventDefault(); } },
    el("div", { class: "field" },
      el("label", {}, "Run name"), nameInput,
      el("div", { class: "help" }, "optional — leave blank and the server names it")),

    el("fieldset", { class: "form-grid" },
      el("legend", {}, "Federation"),
      el("div", { class: "field-row" },
        rangeField("n_clients", "Clients", { min: 2, max: 32, step: 1 }),
        el("div", { class: "field" }, el("label", {}, "Clients per round", cprValue), cprInput),
      ),
      el("div", { class: "field" },
        el("label", {}, "Data partition"),
        partitionSelect,
        el("div", { class: "help" }, "iid mixes rows evenly; dirichlet makes clients look like real silos"),
      ),
      alphaField,
      rangeField("rounds", "Rounds", { min: 1, max: 200, step: 1 }),
    ),

    el("fieldset", { class: "form-grid" },
      el("legend", {}, "Local optimisation"),
      el("div", { class: "field-row-3" },
        numberField("local_epochs", "Local epochs", { min: 1, max: 20, step: 1 }),
        numberField("batch_size", "Batch size", { min: 4, max: 256, step: 4 }),
        numberField("seed", "Seed", { min: 0, max: 99999, step: 1 }),
      ),
      el("div", { class: "field-row" },
        rangeField("learning_rate", "Learning rate", {
          min: 0.01, max: 0.6, step: 0.01, format: (v) => Number(v).toFixed(2),
        }),
        rangeField("l2", "L2 penalty", {
          min: 0, max: 0.01, step: 0.0005, format: (v) => Number(v).toFixed(4),
        }),
      ),
    ),

    el("fieldset", { class: "form-grid" },
      el("legend", {}, "Privacy"),
      el("label", { class: "check" }, privacyToggle, "Differential privacy on client updates"),
      privacyBody,
    ),

    submit,
  );

  return {
    node,
    onSubmit: (handler) => {
      submit.addEventListener("click", (event) => {
        event.preventDefault();
        const payload = {
          name: state.name || "",
          n_clients: state.n_clients,
          clients_per_round: state.clients_per_round,
          rounds: state.rounds,
          local_epochs: state.local_epochs,
          batch_size: state.batch_size,
          learning_rate: state.learning_rate,
          l2: state.l2,
          partition: state.partition,
          dirichlet_alpha: state.dirichlet_alpha,
          seed: state.seed,
          privacy_enabled: Boolean(state.privacy_enabled),
          dp_clip: state.dp_clip,
          dp_epsilon: state.privacy_enabled ? (state.dp_epsilon ?? 8) : null,
          dp_sigma: null,
        };
        handler(payload, submit);
      });
    },
  };
}

/* ------------------------------------------------------------------ live */
function watchRun(runId, host) {
  stopPolling();
  let run = null;

  const paint = () => {
    if (!run) return;
    clear(host);
    host.append(liveCard(run, host));
  };

  const tick = async () => {
    try {
      run = await api.getRun(runId);
    } catch (err) {
      errorToast(err);
      stopPolling();
      return;
    }
    paint();
    if (run.status === "finished" || run.status === "failed") {
      stopPolling();
      if (run.status === "finished") toast(`Run finished — val accuracy ${pct(run.val_accuracy)}`, "ok");
    }
  };

  tick();
  poller = setInterval(tick, 700);
}

function liveCard(run, host) {
  const rounds = run.config?.federated?.rounds || 0;
  const done = run.history?.length || 0;
  const running = run.status === "queued" || run.status === "running";

  const header = el("div", { class: "card-head" },
    el("div", {},
      el("div", { class: "row" }, statusBadge(run.status),
        run.privacy_enabled ? badge(`ε ${num(run.epsilon, 2)}`, "dp") : null),
      el("div", { class: "small dim mono mt-8" }, run.id),
    ),
    el("button", {
      class: "btn btn-sm",
      // refresh the live panel, not the whole page: this callback used to repaint
      // into #main and wipe the form
      onclick: () => openRunDrawer(run.id, { onChange: () => watchRun(run.id, host) }),
    }, "Open"),
  );

  const body = el("div", { class: "card-body" },
    running
      ? el("div", {},
          el("div", { class: "spread small dim" },
            el("span", { class: "pulse" }, `training… ${done}/${rounds} rounds`),
            el("span", { class: "mono" }, done && rounds ? `${Math.round((done / rounds) * 100)}%` : ""),
          ),
          el("div", { class: "mt-8" }, progressBar(rounds ? done / rounds : 0)),
        )
      : el("div", { class: "metric-grid" },
          el("div", { class: "metric" }, el("div", { class: "k" }, "val accuracy"), el("div", { class: "v" }, pct(run.val_accuracy))),
          el("div", { class: "metric" }, el("div", { class: "k" }, "val ROC-AUC"), el("div", { class: "v" }, num(run.val_roc_auc))),
          el("div", { class: "metric" }, el("div", { class: "k" }, "best round"), el("div", { class: "v" }, String(run.best_round ?? "—"))),
          el("div", { class: "metric" }, el("div", { class: "k" }, "duration"), el("div", { class: "v" }, run.duration_s ? `${run.duration_s.toFixed(2)}s` : "—")),
        ),
  );

  const children = [header, body];

  if (run.error) {
    children.push(el("div", { class: "card-body" }, el("div", { class: "note warn" }, run.error)));
  }

  if (run.history?.length) {
    children.push(el("div", { class: "card-body", style: { borderTop: "1px solid var(--line)" } },
      lineChart({
        height: 190,
        yFormat: (v) => v.toFixed(3),
        series: [
          { name: "val accuracy", color: "#4f46e5", points: run.history.map((h) => ({ x: h.round, y: h.val_accuracy })) },
          { name: "val ROC-AUC", color: "#0d9488", points: run.history.map((h) => ({ x: h.round, y: h.val_roc_auc })) },
        ],
      }),
      legend([
        { name: "accuracy", color: "#4f46e5" },
        { name: "ROC-AUC", color: "#0d9488" },
      ]),
    ));
  }

  if (run.history?.length) {
    children.push(table([
      { label: "round", key: "round", num: true },
      { label: "loss", render: (r) => num(r.train_loss), num: true },
      { label: "val acc", render: (r) => pct(r.val_accuracy), num: true },
      { label: "val AUC", render: (r) => num(r.val_roc_auc), num: true },
      { label: "ε", render: (r) => (r.epsilon ? num(r.epsilon, 2) : "—"), num: true },
    ], run.history.slice(-6).reverse()));
  }

  if (run.status === "finished") {
    children.push(el("div", { class: "card-body" },
      el("div", { class: "note" },
        "Next: score this run on the held-out test split, then register it. " +
        "Both actions live in the run drawer."),
    ));
  }

  return el("section", { class: "card" }, ...children);
}
