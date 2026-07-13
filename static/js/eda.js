(function () {
  "use strict";

  const els = {
    ticker: document.getElementById("ticker-input"),
    start: document.getElementById("start-date"),
    end: document.getElementById("end-date"),
    runBtn: document.getElementById("run-btn"),
    status: document.getElementById("status-badge"),
    errorBanner: document.getElementById("error-banner"),
  };

  const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#8792A8", family: "IBM Plex Mono", size: 11 },
    margin: { l: 50, r: 20, t: 10, b: 40 },
    xaxis: { gridcolor: "#232B45" },
    yaxis: { gridcolor: "#232B45" },
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

  function plot(id, traces, layoutExtra) {
    Plotly.newPlot(id, traces, Object.assign({}, PLOTLY_LAYOUT_BASE, layoutExtra || {}), {
      responsive: true,
      displaylogo: false,
    });
  }

  async function run() {
    clearError();
    setStatus("Loading…", "busy");
    els.runBtn.disabled = true;
    try {
      const payload = {
        ticker: els.ticker.value.trim(),
        start: els.start.value,
        end: els.end.value,
      };

      const [dataRes, edaRes] = await Promise.all([
        fetch("/api/data", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }).then((r) => r.json()),
        fetch("/api/eda", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }).then((r) => r.json()),
      ]);
      if (dataRes.error) throw new Error(dataRes.error);
      if (edaRes.error) throw new Error(edaRes.error);

      const ps = dataRes.price_series;
      plot(
        "candlestick-chart",
        [
          {
            x: ps.dates, open: ps.open, high: ps.high, low: ps.low, close: ps.close,
            type: "candlestick",
            increasing: { line: { color: "#2DD4BF" } },
            decreasing: { line: { color: "#F0546B" } },
            xaxis: "x", yaxis: "y",
          },
          { x: ps.dates, y: ps.volume, type: "bar", yaxis: "y2", marker: { color: "#232B45" }, opacity: 0.6 },
        ],
        {
          xaxis: { rangeslider: { visible: false }, gridcolor: "#232B45" },
          yaxis: { domain: [0.3, 1], gridcolor: "#232B45" },
          yaxis2: { domain: [0, 0.2], gridcolor: "#232B45" },
          showlegend: false,
        }
      );

      const r = edaRes.returns;
      plot("returns-chart", [{ x: r.dates, y: r.daily_returns, type: "scatter", mode: "lines", line: { color: "#2DD4BF", width: 1 } }]);
      plot("hist-chart", [{ x: r.histogram.bin_edges, y: r.histogram.counts, type: "bar", marker: { color: "#F5A623" } }]);
      plot("box-chart", [{ y: r.boxplot_values, type: "box", marker: { color: "#2DD4BF" }, boxpoints: "outliers" }]);
      plot("vol-chart", [{ x: r.dates, y: r.rolling_volatility, type: "scatter", mode: "lines", line: { color: "#F0546B", width: 1.5 } }]);
      plot("cumret-chart", [{ x: r.dates, y: r.cumulative_returns, type: "scatter", mode: "lines", fill: "tozeroy", line: { color: "#2DD4BF" } }]);

      const c = edaRes.correlation;
      plot(
        "corr-chart",
        [
          {
            z: c.matrix, x: c.labels, y: c.labels, type: "heatmap",
            colorscale: [[0, "#F0546B"], [0.5, "#121729"], [1, "#2DD4BF"]],
            zmin: -1, zmax: 1,
          },
        ],
        { margin: { l: 70, r: 20, t: 10, b: 60 } }
      );

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
