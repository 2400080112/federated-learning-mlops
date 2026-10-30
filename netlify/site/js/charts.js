/**
 * Charts, hand-rolled in SVG.
 *
 * A dependency-free line chart is ~120 lines, which is cheaper than shipping a
 * charting library and keeps the page a single static bundle. Values are drawn
 * in a fixed 720-unit viewBox and scaled by CSS, so the chart is crisp at any
 * container width.
 */

import { el } from "./ui.js";

const W = 720;
const PAD = { top: 14, right: 16, bottom: 26, left: 46 };

const svgEl = (tag, attrs = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

/**
 * @param {object} opts
 * @param {Array<{name:string,color:string,points:Array<{x:number,y:number}>}>} opts.series
 * @param {number} [opts.height]
 * @param {[number,number]} [opts.yDomain]
 * @param {(v:number)=>string} [opts.yFormat]
 * @param {string} [opts.xLabel]
 */
export function lineChart(opts) {
  const {
    series = [],
    height = 240,
    yDomain = null,
    yFormat = (v) => v.toFixed(2),
    xLabel = "",
    showArea = false,
  } = opts;

  const host = el("div", { class: "chart-host" });
  const tip = el("div", { class: "chart-tip" });
  const H = height;
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const points = series.flatMap((s) => s.points);
  if (!points.length) {
    host.append(el("div", { class: "empty" }, "No data yet."));
    return host;
  }

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs) === xMin ? xMin + 1 : Math.max(...xs);
  let [yMin, yMax] = yDomain || [Math.min(...ys), Math.max(...ys)];
  if (yMax === yMin) { yMax = yMin + 1; yMin = yMin - 1; }

  const sx = (x) => PAD.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const sy = (y) => PAD.top + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${W} ${H}`,
    width: "100%",
    height: "auto",
    role: "img",
    "aria-label": series.map((s) => s.name).join(", "),
  });

  // horizontal gridlines + y labels
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / ticks;
    const y = sy(value);
    svg.append(svgEl("line", {
      x1: PAD.left, x2: W - PAD.right, y1: y, y2: y,
      stroke: "#eef0f3", "stroke-width": 1,
    }));
    const label = svgEl("text", {
      x: PAD.left - 8, y: y + 4, "text-anchor": "end",
      "font-size": 10.5, fill: "#98a2b3", "font-family": "ui-monospace, monospace",
    });
    label.textContent = yFormat(value);
    svg.append(label);
  }

  // x axis labels: first, middle, last round
  const xTicks = [xMin, Math.round((xMin + xMax) / 2), xMax];
  for (const value of [...new Set(xTicks)]) {
    const label = svgEl("text", {
      x: sx(value), y: H - 8, "text-anchor": "middle",
      "font-size": 10.5, fill: "#98a2b3", "font-family": "ui-monospace, monospace",
    });
    label.textContent = String(value);
    svg.append(label);
  }

  if (xLabel) {
    const label = svgEl("text", {
      x: PAD.left, y: PAD.top - 2, "font-size": 10.5, fill: "#98a2b3",
      "text-transform": "uppercase", "letter-spacing": "0.05em",
    });
    label.textContent = xLabel;
    svg.append(label);
  }

  // series
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points
      .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
      .join(" ");

    if (showArea && series.length === 1) {
      const area = `${d} L${sx(s.points.at(-1).x)},${sy(yMin)} L${sx(s.points[0].x)},${sy(yMin)} Z`;
      svg.append(svgEl("path", { d: area, fill: s.color, opacity: 0.08 }));
    }
    svg.append(svgEl("path", {
      d, fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }));
  }

  // hover layer
  const guide = svgEl("line", {
    x1: 0, x2: 0, y1: PAD.top, y2: PAD.top + plotH,
    stroke: "#98a2b3", "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0,
  });
  svg.append(guide);

  const markers = series.map((s) => {
    const dot = svgEl("circle", { r: 3.5, fill: "#fff", stroke: s.color, "stroke-width": 2, opacity: 0 });
    svg.append(dot);
    return dot;
  });

  const overlay = svgEl("rect", {
    x: PAD.left, y: PAD.top, width: plotW, height: plotH, fill: "transparent",
  });
  svg.append(overlay);

  const first = series[0].points;
  overlay.addEventListener("mousemove", (event) => {
    const box = svg.getBoundingClientRect();
    const relX = ((event.clientX - box.left) / box.width) * W;
    const fraction = (relX - PAD.left) / plotW;
    const xValue = xMin + fraction * (xMax - xMin);

    let nearest = first[0];
    for (const p of first) {
      if (Math.abs(p.x - xValue) < Math.abs(nearest.x - xValue)) nearest = p;
    }

    guide.setAttribute("x1", sx(nearest.x));
    guide.setAttribute("x2", sx(nearest.x));
    guide.setAttribute("opacity", 0.9);

    const rows = series.map((s, i) => {
      const point = s.points.find((p) => p.x === nearest.x);
      if (!point) { markers[i].setAttribute("opacity", 0); return null; }
      markers[i].setAttribute("cx", sx(point.x));
      markers[i].setAttribute("cy", sy(point.y));
      markers[i].setAttribute("opacity", 1);
      return `<div class="t-row"><span><i class="sw" style="background:${s.color}"></i>${s.name}</span><b>${yFormat(point.y)}</b></div>`;
    }).filter(Boolean).join("");

    tip.innerHTML = `<div style="opacity:.7;margin-bottom:3px">round ${nearest.x}</div>${rows}`;
    tip.classList.add("show");

    // place the tooltip above the highest series point at this round
    const hovered = series
      .map((s) => s.points.find((p) => p.x === nearest.x)?.y)
      .filter((v) => v !== undefined);
    const topY = hovered.length ? Math.max(...hovered) : yMax;
    const hostBox = host.getBoundingClientRect();
    tip.style.left = `${(sx(nearest.x) / W) * hostBox.width}px`;
    tip.style.top = `${(sy(topY) / H) * hostBox.height - 12}px`;
  });

  overlay.addEventListener("mouseleave", () => {
    tip.classList.remove("show");
    guide.setAttribute("opacity", 0);
    markers.forEach((m) => m.setAttribute("opacity", 0));
  });

  host.append(svg, tip);
  return host;
}

export function legend(items) {
  return el("div", { class: "legend" }, items.map((item) =>
    el("span", { class: "item" },
      el("i", { style: { background: item.color } }), item.name)));
}

/** Confusion matrix block for the predict view. */
export function confusionMatrix(cm) {
  const { tp = 0, tn = 0, fp = 0, fn = 0 } = cm || {};
  const total = tp + tn + fp + fn || 1;
  const cell = (kind, value, label) =>
    el("div", { class: `cell ${kind}` }, String(value), el("small", {}, `${label} · ${((value / total) * 100).toFixed(0)}%`));

  return el("div", { class: "cm" },
    el("div", {}), el("div", { class: "h" }, "pred benign"), el("div", { class: "h" }, "pred malignant"),
    el("div", { class: "axis" }, "benign"), cell("tp", tp, "true pos"), cell("fn", fn, "false neg"),
    el("div", { class: "axis" }, "malignant"), cell("fp", fp, "false pos"), cell("tn", tn, "true neg"),
  );
}

/** Horizontal stacked bar showing a client's class balance. */
export function classBar(positiveRate, height = 8) {
  const pos = Math.max(0, Math.min(1, positiveRate));
  return el("div", { class: "client-bar", style: { height: `${height}px` } },
    el("div", { class: "pos", style: { width: `${pos * 100}%` } }),
    el("div", { class: "neg", style: { width: `${(1 - pos) * 100}%` } }),
  );
}
