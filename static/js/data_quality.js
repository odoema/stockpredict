(function () {
  "use strict";

  const els = {
    ticker: document.getElementById("ticker-input"),
    start: document.getElementById("start-date"),
    end: document.getElementById("end-date"),
    runBtn: document.getElementById("run-btn"),
    status: document.getElementById("status-badge"),
    errorBanner: document.getElementById("error-banner"),
    overview: document.getElementById("overview-panel"),
    missing: document.getElementById("missing-panel"),
    statsTable: document.getElementById("stats-table"),
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

  function renderOverview(report) {
    els.overview.innerHTML = `
      <div class="sp-metric-grid">
        <div class="sp-metric-item"><div class="sp-metric-value">${report.rows}</div><div class="sp-metric-label">Rows</div></div>
        <div class="sp-metric-item"><div class="sp-metric-value">${report.columns}</div><div class="sp-metric-label">Columns</div></div>
        <div class="sp-metric-item"><div class="sp-metric-value">${report.duplicate_rows}</div><div class="sp-metric-label">Duplicates</div></div>
        <div class="sp-metric-item"><div class="sp-metric-value">${report.total_missing}</div><div class="sp-metric-label">Missing values</div></div>
      </div>
      <div class="sp-empty" style="text-align:left;padding-top:14px;">
        Coverage: ${report.date_start} &rarr; ${report.date_end}
      </div>`;
  }

  function renderMissing(report) {
    const rows = Object.entries(report.missing_values)
      .map(([col, count]) => `<tr><td>${col}</td><td>${report.dtypes[col]}</td><td>${count}</td></tr>`)
      .join("");
    els.missing.innerHTML = `
      <table class="sp-table">
        <thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  function renderStats(report) {
    const cols = Object.keys(report.summary_stats);
    const statNames = Object.keys(report.summary_stats[cols[0]]);
    let html = '<table class="sp-table"><thead><tr><th>Statistic</th>';
    cols.forEach((c) => (html += `<th>${c}</th>`));
    html += "</tr></thead><tbody>";
    statNames.forEach((stat) => {
      html += `<tr><td>${stat}</td>`;
      cols.forEach((c) => {
        const v = report.summary_stats[c][stat];
        html += `<td>${v === null ? "—" : v.toFixed(4)}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody></table>";
    els.statsTable.innerHTML = html;
  }

  async function run() {
    clearError();
    setStatus("Checking…", "busy");
    els.runBtn.disabled = true;
    try {
      const res = await fetch("/api/data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ticker: els.ticker.value.trim(),
          start: els.start.value,
          end: els.end.value,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");

      renderOverview(data.quality_report);
      renderMissing(data.quality_report);
      renderStats(data.quality_report);
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
