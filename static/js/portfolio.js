(function () {
  "use strict";

  const els = {
    tickers: document.getElementById("tickers-input"),
    start: document.getElementById("start-date"),
    end: document.getElementById("end-date"),
    runBtn: document.getElementById("run-btn"),
    status: document.getElementById("status-badge"),
    errorBanner: document.getElementById("error-banner"),
    performanceChart: document.getElementById("performance-chart"),
    correlationChart: document.getElementById("correlation-chart"),
    riskTable: document.getElementById("risk-table"),
  };

  const LINE_COLORS = ["#2DD4BF", "#F5A623", "#F0546B", "#7F77DD", "#378ADD", "#639922", "#D4537E", "#888780"];

  function setStatus(text, cls) {
    els.status.textContent = text;
    els.status.className = "sp-status-badge" + (cls ? " " + cls : "");
  }

  function showError(message) {
    els.errorBanner.textContent = message;
    els.errorBanner.classList.remove("d-none");
  }

  function clearError() {
    els.errorBanner.classList.add("d-none");
  }

  function defaultDates() {
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 5);
    return [start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)];
  }

  function renderPerformance(data) {
    const traces = data.tickers.map((t, i) => ({
      x: data.dates,
      y: data.normalized_performance[t],
      type: "scatter",
      mode: "lines",
      name: t,
      line: { color: LINE_COLORS[i % LINE_COLORS.length], width: 2 },
    }));
    Plotly.newPlot(
      els.performanceChart,
      traces,
      {
        margin: { l: 50, r: 20, t: 10, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#8792A8", family: "IBM Plex Mono", size: 11 },
        xaxis: { gridcolor: "#232B45" },
        yaxis: { gridcolor: "#232B45", title: "Index (base 100)" },
        legend: { orientation: "h", y: -0.2 },
      },
      { responsive: true, displaylogo: false }
    );
  }

  function renderCorrelation(corr) {
    Plotly.newPlot(
      els.correlationChart,
      [
        {
          z: corr.matrix, x: corr.labels, y: corr.labels, type: "heatmap",
          colorscale: [[0, "#F0546B"], [0.5, "#121729"], [1, "#2DD4BF"]],
          zmin: -1, zmax: 1,
        },
      ],
      {
        margin: { l: 60, r: 20, t: 10, b: 50 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#8792A8", size: 11 },
      },
      { responsive: true, displaylogo: false }
    );
  }

  function renderRiskTable(rows, benchmark) {
    const cols = ["ticker", "total_return_pct", "annual_return_pct", "annual_volatility_pct", "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "beta"];
    const labels = { ticker: "Ticker", total_return_pct: "Total Ret %", annual_return_pct: "Ann. Ret %", annual_volatility_pct: "Ann. Vol %", sharpe_ratio: "Sharpe", sortino_ratio: "Sortino", max_drawdown_pct: "Max DD %", beta: "Beta" };

    let html = '<table class="sp-table"><thead><tr>';
    cols.forEach((c) => (html += `<th>${labels[c]}</th>`));
    html += "</tr></thead><tbody>";
    rows.forEach((r) => {
      html += `<tr class="${r.ticker === benchmark ? "sp-best-row" : ""}">`;
      cols.forEach((c) => {
        const v = r[c];
        html += `<td>${v === null || v === undefined ? "—" : v}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    els.riskTable.innerHTML = html + `<div class="sp-empty" style="text-align:left;padding-top:10px;">Highlighted row is the benchmark (first ticker) used for beta.</div>`;
  }

  async function run() {
    clearError();
    setStatus("Comparing…", "busy");
    els.runBtn.disabled = true;
    try {
      const tickers = els.tickers.value.split(",").map((t) => t.trim()).filter(Boolean);
      const res = await fetch("/api/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers, start: els.start.value, end: els.end.value }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");

      renderPerformance(data);
      renderCorrelation(data.correlation);
      renderRiskTable(data.risk_metrics, data.benchmark);
      setStatus("Done", "ok");
    } catch (err) {
      showError(err.message);
      setStatus("Error", "error");
    } finally {
      els.runBtn.disabled = false;
    }
  }

  function quickAdd(ticker) {
    const current = els.tickers.value.split(",").map((t) => t.trim()).filter(Boolean);
    if (!current.includes(ticker)) current.push(ticker);
    els.tickers.value = current.join(", ");
  }

  document.addEventListener("DOMContentLoaded", () => {
    const [start, end] = defaultDates();
    els.start.value = start;
    els.end.value = end;
    els.runBtn.addEventListener("click", run);
    document.querySelectorAll(".sp-quick-add").forEach((btn) => {
      btn.addEventListener("click", () => quickAdd(btn.dataset.ticker));
    });
  });
})();
