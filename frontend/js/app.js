/**
 * App shell: hash router, sidebar wiring, health indicator.
 *
 * Each view module exports a render function that fills a root element and may
 * return a cleanup function (the train view uses it to stop polling when you
 * navigate away).
 */

import { api } from "./api.js";
import { $, el, errorToast, icon } from "./ui.js";
import { renderDashboard } from "./views/dashboard.js";
import { renderPredict } from "./views/predict.js";
import { renderRegistry } from "./views/registry.js";
import { renderRuns } from "./views/runs.js";
import { renderTrain } from "./views/train.js";

const ROUTES = {
  dashboard: { render: renderDashboard, title: "Dashboard" },
  train: { render: renderTrain, title: "Train" },
  runs: { render: renderRuns, title: "Runs" },
  registry: { render: renderRegistry, title: "Registry" },
  predict: { render: renderPredict, title: "Predict" },
};

let cleanup = null;
let currentRoute = null;

function hydrateIcons() {
  document.querySelectorAll("[data-icon]").forEach((node) => {
    const name = node.dataset.icon;
    const svg = icon(name);
    node.replaceWith(svg);
  });
}

function setActive(route) {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.route === route);
  });
}

async function navigate() {
  const hash = window.location.hash.replace(/^#\/?/, "") || "dashboard";
  const [route, ...rest] = hash.split("/");
  const entry = ROUTES[route] || ROUTES.dashboard;

  if (typeof cleanup === "function") {
    try { cleanup(); } catch { /* a failing cleanup must not block navigation */ }
  }
  cleanup = null;

  currentRoute = route;
  setActive(ROUTES[route] ? route : "dashboard");
  document.title = `FedMLOps · ${entry.title}`;

  const root = $("#main");
  root.replaceChildren();
  try {
    const result = await entry.render(root, { params: rest, navigate });
    if (typeof result === "function") cleanup = result;
  } catch (err) {
    errorToast(err);
    root.replaceChildren(el("div", { class: "empty" },
      el("strong", {}, "This view could not load"),
      err.message || String(err),
      el("div", { class: "mt-16" },
        el("button", { class: "btn", onclick: () => navigate() }, "Retry")),
    ));
  }
}

async function checkHealth() {
  const dot = $("#health-dot");
  const text = $("#health-text");
  if (api.isDemo) {
    const health = await api.health().catch(() => ({ version: "?" }));
    dot.className = "dot ok";
    text.textContent = `static snapshot · v${health.version}`;
    return;
  }
  try {
    const health = await api.health();
    dot.className = "dot ok";
    text.textContent = `API ok · v${health.version}`;
  } catch {
    dot.className = "dot bad";
    text.textContent = "API unreachable";
  }
}

hydrateIcons();

// The static build replays a snapshot; say so instead of letting writes fail
// mysteriously when someone clicks "Register".
if (api.isDemo) {
  const foot = document.querySelector(".sidebar-foot");
  // /api/docs does not exist on a static host
  foot?.querySelector('a[href="/api/docs"]')?.closest("div")?.remove();
  if (foot) {
    foot.insertBefore(
      el("div", { class: "demo-flag" }, "Read-only demo · static build"),
      foot.firstChild,
    );
  }
}

window.addEventListener("hashchange", navigate);
if (!window.location.hash) window.location.hash = "#/dashboard";
navigate();
checkHealth();
if (!api.isDemo) setInterval(checkHealth, 30000);

export { currentRoute };
