(function () {
  "use strict";

  const els = {
    ticker: document.getElementById("ticker-input"),
    start: document.getElementById("start-date"),
    end: document.getElementById("end-date"),
    target: document.getElementById("target-select"),
    model: document.getElementById("model-select"),
    positionMode: document.getElementById("position-mode"),
    feeBps: document.getElementById("fee-bps"),
    runBtn: document.getElementById("run-btn"),
    status: document.getElementById("status-badge"),
    errorBanner: document.getElementById("error-banner"),
    equityChart: document.getElementById("equity-chart"),
    metricsPanel: document.getElementById("metrics-panel"),
  };

  const METRIC_LABELS = {
    total_return_pct: "Total Return %",
    annual_return_pct: "Annual Return %",
    sharpe_ratio: "Sharpe Ratio",
    sortino_ratio: "Sortino Ratio",
    max_drawdown_pct: "Max Drawdown %",
    win_rate_pct: "Win Rate %",
    profit_factor: "Profit Factor",
    number_of_trades: "Number of Trades",
  };

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
    start.setFullYear(start.getFullYear() - 10);
    return [start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)];
  }

  function renderEquityCurve(bt) {
    Plotly.newPlot(
      els.equityChart,
      [
        {
          x: bt.dates, y: bt.strategy_equity_curve, type: "scatter", mode: "lines",
          name: "Strategy", line: { color: "#2DD4BF", width: 2 },
        },
        {
          x: bt.dates, y: bt.benchmark_equity_curve, type: "scatter", mode: "lines",
          name: "Buy & Hold", line: { color: "#8792A8", width: 1.5, dash: "dot" },
        },
      ],
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

  function renderMetrics(metrics, benchmarkReturn) {
    const rows = Object.entries(metrics)
      .map(
        ([k, v]) => `
        <div class="sp-metric-item">
          <div class="sp-metric-value">${v === null ? "—" : v}</div>
          <div class="sp-metric-label">${METRIC_LABELS[k] || k}</div>
        </div>`
      )
      .join("");

    els.metricsPanel.innerHTML = `
      <div class="sp-metric-grid">${rows}</div>
      <div class="sp-empty" style="text-align:left;padding-top:14px;">
        Buy-and-hold total return over the same out-of-sample period: ${benchmarkReturn}%
      </div>`;
  }

  async function run() {
    clearError();
    setStatus("Backtesting…", "busy");
    els.runBtn.disabled = true;
    try {
      const body = {
        ticker: els.ticker.value.trim(),
        start: els.start.value,
        end: els.end.value,
        target: els.target.value,
        model: els.model.value,
        feature_selection_method: "random_forest_importance",
        allow_short: els.positionMode.value === "long_short",
        fee_bps: parseFloat(els.feeBps.value || "0"),
      };

      const res = await fetch("/api/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");

      renderEquityCurve(data.backtest);
      renderMetrics(data.backtest.metrics, data.backtest.benchmark_total_return_pct);
      setStatus("Done", "ok");
    } catch (err) {
      showError(err.message);
      setStatus("Error", "error");
    } finally {
      els.runBtn.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const [start, end] = defaultDates();
    els.start.value = start;
    els.end.value = end;
    els.runBtn.addEventListener("click", run);
  });
})();
