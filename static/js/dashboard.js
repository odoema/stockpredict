(function () {
  "use strict";

  const els = {
    ticker: document.getElementById("ticker-input"),
    start: document.getElementById("start-date"),
    end: document.getElementById("end-date"),
    target: document.getElementById("target-select"),
    featureMethod: document.getElementById("feature-select-method"),
    model: document.getElementById("model-select"),
    indicatorGroups: document.getElementById("indicator-groups"),
    runBtn: document.getElementById("run-btn"),
    compareBtn: document.getElementById("compare-btn"),
    resetBtn: document.getElementById("reset-btn"),
    status: document.getElementById("status-badge"),
    errorBanner: document.getElementById("error-banner"),
    priceChart: document.getElementById("price-chart"),
    predictionPanel: document.getElementById("prediction-panel"),
    metricsPanel: document.getElementById("metrics-panel"),
    importanceChart: document.getElementById("importance-chart"),
    confusionChart: document.getElementById("confusion-chart"),
    compareCard: document.getElementById("compare-card"),
    compareTable: document.getElementById("compare-table"),
  };

  const DEFAULT_INDICATORS = ["sma", "ema", "macd", "rsi", "atr", "bollinger", "obv"];

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
    els.errorBanner.textContent = "";
  }

  function defaultDates() {
    const end = new Date();
    const start = new Date();
    start.setFullYear(start.getFullYear() - 10);
    return [start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)];
  }

  function getSelectedIndicators() {
    return Array.from(els.indicatorGroups.querySelectorAll("input[type=checkbox]:checked")).map(
      (cb) => cb.value
    );
  }

  function currentRequestBody(extra) {
    return Object.assign(
      {
        ticker: els.ticker.value.trim(),
        start: els.start.value,
        end: els.end.value,
        target: els.target.value,
        feature_selection_method: els.featureMethod.value,
        indicators: getSelectedIndicators(),
      },
      extra || {}
    );
  }

  async function postJSON(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
  }

  async function loadIndicatorGroups() {
    const res = await fetch("/api/indicators");
    const groups = await res.json();
    els.indicatorGroups.innerHTML = "";
    Object.entries(groups).forEach(([groupName, items]) => {
      const wrap = document.createElement("div");
      const title = document.createElement("div");
      title.className = "sp-indicator-group-title";
      title.textContent = groupName;
      wrap.appendChild(title);
      items.forEach((item) => {
        const label = document.createElement("label");
        label.className = "sp-indicator-check";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = item.key;
        cb.checked = DEFAULT_INDICATORS.includes(item.key);
        label.appendChild(cb);
        label.appendChild(document.createTextNode(item.label));
        wrap.appendChild(label);
      });
      els.indicatorGroups.appendChild(wrap);
    });
  }

  function renderPriceChart(priceSeries) {
    const trace = {
      x: priceSeries.dates,
      open: priceSeries.open,
      high: priceSeries.high,
      low: priceSeries.low,
      close: priceSeries.close,
      type: "candlestick",
      increasing: { line: { color: "#2DD4BF" } },
      decreasing: { line: { color: "#F0546B" } },
    };
    Plotly.newPlot(
      els.priceChart,
      [trace],
      {
        margin: { l: 40, r: 20, t: 10, b: 30 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#8792A8", family: "IBM Plex Mono" },
        xaxis: { rangeslider: { visible: false }, gridcolor: "#232B45" },
        yaxis: { gridcolor: "#232B45" },
      },
      { responsive: true, displaylogo: false }
    );
  }

  function renderPrediction(prediction, targetType) {
    const signalClass =
      prediction.signal === "BUY" ? "sp-signal-buy" : prediction.signal === "SELL" ? "sp-signal-sell" : "sp-signal-hold";

    let statsHtml = "";
    if (targetType === "classification") {
      statsHtml = `
        <div class="sp-pred-stats">
          <div class="sp-pred-stat"><div class="sp-pred-stat-value">${(prediction.probability_up * 100).toFixed(1)}%</div><div class="sp-pred-stat-label">Prob. Up</div></div>
          <div class="sp-pred-stat"><div class="sp-pred-stat-value">${(prediction.probability_down * 100).toFixed(1)}%</div><div class="sp-pred-stat-label">Prob. Down</div></div>
          <div class="sp-pred-stat"><div class="sp-pred-stat-value">${(prediction.confidence * 100).toFixed(1)}%</div><div class="sp-pred-stat-label">Confidence</div></div>
        </div>`;
    } else {
      statsHtml = `
        <div class="sp-pred-stats">
          <div class="sp-pred-stat"><div class="sp-pred-stat-value">${prediction.raw_prediction.toFixed(4)}</div><div class="sp-pred-stat-label">Predicted Value</div></div>
        </div>`;
    }

    els.predictionPanel.innerHTML = `
      <div class="sp-signal-badge ${signalClass}">${prediction.signal}</div>
      ${statsHtml}
    `;
  }

  function renderMetrics(evaluation) {
    const rows = Object.entries(evaluation.metrics)
      .filter(([, v]) => v !== null && v !== undefined)
      .map(
        ([k, v]) => `
        <div class="sp-metric-item">
          <div class="sp-metric-value">${typeof v === "number" ? v.toFixed(4) : v}</div>
          <div class="sp-metric-label">${k.replace(/_/g, " ")}</div>
        </div>`
      )
      .join("");

    els.metricsPanel.innerHTML = `
      <div class="sp-metric-grid">${rows}</div>
      <div class="sp-empty" style="text-align:left;padding-top:14px;">
        ${evaluation.model_label} &middot; trained on ${evaluation.n_train} rows, tested on ${evaluation.n_test} rows
        &middot; train ${evaluation.train_time_seconds}s / predict ${evaluation.prediction_time_seconds}s
      </div>`;
  }

  function renderImportance(featureImportance) {
    if (!featureImportance) {
      els.importanceChart.innerHTML = '<div class="sp-empty">This model does not expose feature importances.</div>';
      return;
    }
    const entries = Object.entries(featureImportance).slice(0, 15).reverse();
    Plotly.newPlot(
      els.importanceChart,
      [
        {
          x: entries.map((e) => e[1]),
          y: entries.map((e) => e[0]),
          type: "bar",
          orientation: "h",
          marker: { color: "#2DD4BF" },
        },
      ],
      {
        margin: { l: 120, r: 20, t: 10, b: 30 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#8792A8", size: 11 },
        xaxis: { gridcolor: "#232B45" },
      },
      { responsive: true, displaylogo: false }
    );
  }

  function renderConfusionOrFit(evalResult) {
    if (evalResult.confusion_matrix) {
      Plotly.newPlot(
        els.confusionChart,
        [
          {
            z: evalResult.confusion_matrix,
            x: ["Pred Down", "Pred Up"],
            y: ["Actual Down", "Actual Up"],
            type: "heatmap",
            colorscale: [
              [0, "#121729"],
              [1, "#2DD4BF"],
            ],
            showscale: false,
          },
        ],
        {
          margin: { l: 90, r: 20, t: 10, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#8792A8" },
        },
        { responsive: true, displaylogo: false }
      );
    } else {
      els.confusionChart.innerHTML =
        '<div class="sp-empty">Regression target selected — showing metrics instead of a confusion matrix.</div>';
    }
  }

  async function runAnalysis() {
    clearError();
    setStatus("Running…", "busy");
    els.runBtn.disabled = true;
    try {
      const body = currentRequestBody({ model: els.model.value });
      const [dataRes, trainRes] = await Promise.all([
        postJSON("/api/data", body),
        postJSON("/api/train", body),
      ]);

      renderPriceChart(dataRes.price_series);
      renderMetrics(trainRes.evaluation);
      renderImportance(trainRes.evaluation.feature_importance);
      renderConfusionOrFit(trainRes.evaluation);
      renderPrediction(trainRes.prediction, trainRes.evaluation.target_type);

      setStatus("Done", "ok");
    } catch (err) {
      showError(err.message);
      setStatus("Error", "error");
    } finally {
      els.runBtn.disabled = false;
    }
  }

  async function runCompare() {
    clearError();
    setStatus("Comparing models…", "busy");
    els.compareBtn.disabled = true;
    try {
      const body = currentRequestBody();
      const res = await postJSON("/api/train/compare", body);
      const rows = res.comparison;
      const cols = Object.keys(rows[0]).filter((c) => c !== "model");
      const bestModel = rows[0].model;

      let html = '<table class="sp-table"><thead><tr><th>Model</th>';
      cols.forEach((c) => (html += `<th>${c.replace(/_/g, " ")}</th>`));
      html += "</tr></thead><tbody>";
      rows.forEach((r) => {
        html += `<tr class="${r.model === bestModel ? "sp-best-row" : ""}"><td>${r.model}</td>`;
        cols.forEach((c) => {
          const v = r[c];
          html += `<td>${typeof v === "number" ? v.toFixed(4) : v ?? "—"}</td>`;
        });
        html += "</tr>";
      });
      html += "</tbody></table>";

      els.compareTable.innerHTML = html;
      els.compareCard.style.display = "block";
      setStatus("Done", "ok");
    } catch (err) {
      showError(err.message);
      setStatus("Error", "error");
    } finally {
      els.compareBtn.disabled = false;
    }
  }

  function resetForm() {
    const [start, end] = defaultDates();
    els.ticker.value = els.ticker.dataset.default || els.ticker.value;
    els.start.value = start;
    els.end.value = end;
    els.target.value = "next_day_direction";
    els.featureMethod.value = "random_forest_importance";
    document.querySelectorAll("#indicator-groups input[type=checkbox]").forEach((cb) => {
      cb.checked = DEFAULT_INDICATORS.includes(cb.value);
    });
    clearError();
    setStatus("Idle");
  }

  function exportData(format) {
    const body = currentRequestBody();
    body.format = format;
    fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((res) => {
        if (!res.ok) return res.json().then((d) => Promise.reject(new Error(d.error)));
        return res.blob();
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${els.ticker.value.trim().toUpperCase()}_export.${format}`;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch((err) => showError(err.message));
  }

  document.addEventListener("DOMContentLoaded", () => {
    const [start, end] = defaultDates();
    els.start.value = start;
    els.end.value = end;

    loadIndicatorGroups();

    els.runBtn.addEventListener("click", runAnalysis);
    els.compareBtn.addEventListener("click", runCompare);
    els.resetBtn.addEventListener("click", resetForm);
    document.querySelectorAll("[data-export]").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        exportData(e.target.dataset.export);
      });
    });
  });
})();
