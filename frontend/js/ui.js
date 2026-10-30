/**
 * Tiny DOM helpers. No framework: `el()` builds nodes, everything else is
 * formatting. Keeping this small is why the whole frontend has no build step.
 */

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** Static SVG icons (inline so there is no icon-font dependency). */
export const icon = (name, size = 16) => {
  const paths = {
    dashboard: "M3 3h7v7H3zM14 3h7v4h-7zM14 11h7v10h-7zM3 14h7v7H3z",
    train: "M4 18V9m5 9V5m5 13v-7m5 7V8",
    runs: "M4 6h16M4 12h16M4 18h10",
    registry: "M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 17l9 5 9-5",
    predict: "M12 2l3 7h7l-5.5 4.5L18 21l-6-4-6 4 1.5-7.5L2 9h7l3-7z",
    play: "M6 4l14 8-14 8z",
    check: "M20 6L9 17l-5-5",
    plus: "M12 5v14M5 12h14",
    close: "M6 6l12 12M18 6L6 18",
    refresh: "M21 12a9 9 0 1 1-3-6.7M21 3v6h-6",
    trash: "M3 6h18M8 6V4h8v2m-9 0l1 14h8l1-14",
    download: "M12 3v12m0 0l4-4m-4 4l-4-4M4 21h16",
    upload: "M12 21V9m0 0l4 4m-4-4L8 13M4 3h16",
    shield: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z",
    info: "M12 8h.01M11 12h1v5h1",
  };
  return el("span", {
    class: "nav-icon",
    html: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
      stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || ""}"/></svg>`,
    style: { display: "inline-flex" },
  });
};

/* ------------------------------------------------------------ formatting */
export const pct = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(digits);

export const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-IN"));

export function when(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return d.toLocaleDateString(undefined, { day: "2-digit", month: "short" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

export const clock = (iso) =>
  iso ? new Date(iso).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "—";

/* ------------------------------------------------------------ components */
export function badge(text, kind = "neutral", extra = null) {
  return el("span", { class: `badge badge-${kind}` }, text, extra);
}

export function statusBadge(status) {
  const kind = { finished: "finished", running: "running", queued: "queued", failed: "failed" }[status] || "neutral";
  return badge(status, kind);
}

export function stageBadge(stage) {
  return badge(stage, stage);
}

export function kpi(label, value, foot, footClass = "") {
  return el("div", { class: "card kpi" },
    el("div", { class: "kpi-label" }, label),
    el("div", { class: "kpi-value" }, value),
    foot ? el("div", { class: `kpi-foot ${footClass}` }, foot) : null,
  );
}

export function card(title, body, { hint = null, actions = null, className = "" } = {}) {
  const head = title
    ? el("div", { class: "card-head" },
        el("div", {},
          el("h2", {}, title),
          hint ? el("div", { class: "hint" }, hint) : null,
        ),
        actions ? el("div", { class: "row" }, actions) : null,
      )
    : null;
  return el("section", { class: `card ${className}` }, head, body);
}

export function table(columns, rows, { onRowClick = null, empty = "Nothing here yet." } = {}) {
  if (!rows.length) {
    return el("div", { class: "empty" }, el("strong", {}, empty), "Run an experiment to fill this in.");
  }
  const thead = el("thead", {}, el("tr", {}, columns.map((c) =>
    el("th", { class: c.num ? "num" : "" }, c.label))));
  const tbody = el("tbody", {}, rows.map((row) =>
    el("tr", {
      class: onRowClick ? "clickable" : "",
      onclick: onRowClick ? () => onRowClick(row) : null,
    }, columns.map((c) => {
      const value = c.render ? c.render(row) : row[c.key];
      return el("td", { class: [c.num ? "num" : "", c.mono ? "mono" : ""].filter(Boolean).join(" ") },
        value === null || value === undefined ? "—" : value);
    }))));
  return el("div", { class: "table-wrap" }, el("table", {}, thead, tbody));
}

export function metricTile(key, value) {
  return el("div", { class: "metric" },
    el("div", { class: "k" }, key),
    el("div", { class: "v" }, value));
}

export function progressBar(fraction) {
  return el("div", { class: "progress" },
    el("div", { style: { width: `${Math.max(0, Math.min(1, fraction)) * 100}%` } }));
}

/* ------------------------------------------------------------ toasts */
let toastHost = null;

export function toast(message, kind = "") {
  if (!toastHost) {
    toastHost = el("div", { class: "toasts" });
    document.body.append(toastHost);
  }
  const node = el("div", { class: `toast ${kind}` }, el("div", {}, message));
  toastHost.append(node);
  setTimeout(() => {
    node.style.transition = "opacity .2s";
    node.style.opacity = "0";
    setTimeout(() => node.remove(), 220);
  }, kind === "err" ? 6000 : 3200);
}

export function errorToast(err) {
  toast(err?.message || String(err), "err");
}
