/* Fraud Risk Console — static, no build step. Data: data/console.json (written by
   fraud_classifier/scripts/export_console_data.py). Decisions live in this browser only. */
(() => {
  'use strict';
  const $main = document.getElementById('main');
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt$ = n => '$' + Number(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const num = (v, d = 3) => v == null ? '—' : Number(v).toFixed(d);

  const DESC = {
    TransactionAmt: 'Amount (USD)', ProductCD: 'Product code', card1: 'Card ID (masked)', card6: 'Card type',
    P_emaildomain: 'Purchaser email domain', R_emaildomain: 'Recipient email domain', dist1: 'Address distance (masked)',
    amt_log: 'Log amount', amt_zscore: "Amount vs card's earlier amounts (σ)", card1_txn_count: 'Txns on this card (v0.1, all-time)',
    card1_txn_count_7d: 'Txns on this card, last 7 days', card1_prior_amt_mean: "Card's earlier average amount",
    secs_since_prev_card_txn: 'Seconds since previous card txn', email_mismatch: 'Purchaser ≠ recipient email',
    P_email_anon: 'Purchaser email missing/anonymous', is_large_txn: 'Amount over $500', C_sum: 'Sum of masked counts',
    is_mobile: 'Mobile device', dt_hour: 'Relative hour of day'
  };
  const describe = f => DESC[f] || (/^C\d+$/.test(f) ? 'Masked count' : /^D\d+$/.test(f) ? 'Masked time delta (days)'
    : /^M\d+$/.test(f) ? 'Masked match flag' : /^V\d+$/.test(f) ? 'Masked Vesta feature' : /^id_\d+$/.test(f) ? 'Masked identity feature' : 'Encoded feature');
  const ACTIONS = { LOW: 'No alert; approve.', MEDIUM: 'Manual review within 4 hours.',
    HIGH: 'Escalate to a senior analyst; consider a temporary hold.', CRITICAL: 'Block and notify the compliance officer.' };

  const store = {
    get(k, d) { try { const v = localStorage.getItem('frc.' + k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem('frc.' + k, JSON.stringify(v)); } catch { /* storage unavailable */ } }
  };
  const state = { data: null, filter: 'ALL', status: store.get('status', {}), audit: store.get('audit', []), lastCase: null };

  let tierOf;
  function setTiering(d) {
    const t = d.threshold;
    tierOf = t != null
      ? p => p < t ? 'LOW' : p < 0.5 ? 'MEDIUM' : p < 0.8 ? 'HIGH' : 'CRITICAL'
      : p => p >= 0.75 ? 'CRITICAL' : p >= 0.5 ? 'HIGH' : p >= 0.3 ? 'MEDIUM' : 'LOW';
  }
  const statusOf = id => state.status[id] || 'Open';
  const scoreLabel = () => state.data.probabilities_calibrated ? 'Fraud probability' : 'Model score';

  function narrative(c) {
    const t = tierOf(c.prob), risk = c.drivers.filter(d => d[2] > 0), prot = c.drivers.filter(d => d[2] < 0);
    const what = state.data.probabilities_calibrated ? `Fraud probability ${(c.prob * 100).toFixed(1)}%` : `Model score ${c.prob.toFixed(2)}`;
    let s = `${what} (${t}).`;
    if (risk.length) s += ` Largest contributions toward fraud: ${risk.slice(0, 2).map(d => `${d[0]} = ${d[1]} (${describe(d[0])})`).join(' and ')}.`;
    if (prot.length) s += ` Partly offset by ${prot[0][0]} = ${prot[0][1]}.`;
    return s + ' Recommended: ' + ACTIONS[t];
  }

  // ---------- views ----------
  function viewQueue() {
    const d = state.data, cases = d.cases;
    const open = cases.filter(c => statusOf(c.id) === 'Open').length;
    const crit = cases.filter(c => tierOf(c.prob) === 'CRITICAL').length;
    const op = d.operating_point;
    const kpis = [
      ['Open alerts', open, `of ${cases.length} ${d.cases_are_samples ? 'sample ' : ''}cases`],
      ['Critical', crit, d.threshold != null ? 'probability ≥ 0.80' : 'score ≥ 0.75'],
      ['Decided here', state.audit.length, 'saved in this browser'],
      op ? ['Alerts / day at threshold', op.alerts_per_day, `precision ${num(op.precision)} · recall ${num(op.recall)}`]
         : ['Challenger AUC-PR', num(d.metrics.challenger_xgboost.auc_pr), d.model_version + ' holdout']
    ];
    const rows = cases.filter(c => state.filter === 'ALL' || tierOf(c.prob) === state.filter);
    return `
      <div class="grid-kpi">${kpis.map(([l, v, s]) => `<div class="card kpi"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><div class="small muted">${esc(s)}</div></div>`).join('')}</div>
      <div class="row between" style="margin-bottom:12px">
        <h1>Alert queue</h1>
        <div class="filters" role="group" aria-label="Filter by risk tier">
          ${['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(f => `<button data-filter="${f}" aria-pressed="${state.filter === f}">${f === 'ALL' ? 'All' : f[0] + f.slice(1).toLowerCase()}</button>`).join('')}
        </div>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Transaction</th><th>${esc(scoreLabel())}</th><th>Tier</th><th>Product</th><th>Card</th><th>Email domain</th><th>Amount</th><th>Status</th><th>Age</th></tr></thead>
        <tbody>${rows.map(c => { const t = tierOf(c.prob); return `
          <tr data-open="${esc(c.id)}">
            <td><a href="#/case/${encodeURIComponent(c.id)}">TXN-${esc(c.id)}</a></td>
            <td class="mono"><strong>${c.prob.toFixed(2)}</strong></td>
            <td><span class="badge t-${t}">${t}</span></td>
            <td>${esc(c.product)}</td><td>${esc(c.card)}</td><td>${esc(c.email)}</td>
            <td class="mono">${fmt$(c.amount)}</td>
            <td>${esc(statusOf(c.id))}</td>
            <td class="muted">${c.age_min != null ? esc(c.age_min) + ' min' : '—'}</td>
          </tr>`; }).join('') || '<tr><td colspan="9" class="muted">No alerts in this tier.</td></tr>'}</tbody>
      </table></div>`;
  }

  function viewCase(id) {
    const d = state.data, c = d.cases.find(x => x.id === id) || d.cases.find(x => x.id === state.lastCase) || d.cases[0];
    state.lastCase = c.id;
    const t = tierOf(c.prob), s = statusOf(c.id), max = Math.max(...c.drivers.map(x => Math.abs(x[2])), 1e-9);
    const facts = [['Amount', fmt$(c.amount)], ['Product code', c.product], ['Card', c.card], ['Purchaser email', c.email], ['Device', c.device],
      ['Actual label', c.actual || 'unknown'], ['Model', 'XGBoost ' + d.model_version]];
    return `
      <div class="row" style="margin-bottom:16px">
        <a class="btn ghost" href="#/queue">← Queue</a>
        <h1 class="mono">TXN-${esc(c.id)}</h1>
        <span class="badge t-${t}">${t}</span>
        <span class="badge ${s === 'Open' ? 'b-info' : 'b-ok'}">${esc(s)}</span>
      </div>
      <div class="split">
        <div class="main-col">
          <section class="card">
            <div class="row between"><h2 style="margin:0">Why the model flagged this</h2><span class="small muted">SHAP contribution (log-odds)</span></div>
            <div class="bars" style="margin-top:16px">${c.drivers.map(([f, v, sv]) => `
              <div class="bar-row">
                <div><div class="mono"><strong>${esc(f)}</strong> = ${esc(v)}</div><div class="small muted">${esc(describe(f))}</div></div>
                <div class="track" role="img" aria-label="${esc(f)} ${sv > 0 ? 'raises' : 'lowers'} risk by ${Math.abs(sv).toFixed(2)}"><div class="mid"></div><div class="fill ${sv > 0 ? 'risk' : 'safe'}" style="width:${(Math.abs(sv) / max * 50).toFixed(1)}%"></div></div>
                <div class="mono" style="text-align:right;color:var(${sv > 0 ? '--risk-ink' : '--safe-ink'})">${sv > 0 ? '+' : ''}${sv.toFixed(2)}</div>
              </div>`).join('')}</div>
            <div class="legend"><span><span class="swatch" style="background:var(--risk)"></span>Raises risk</span><span><span class="swatch" style="background:var(--safe)"></span>Lowers risk</span></div>
          </section>
          <section class="card">
            <div class="row between"><h2 style="margin:0">Triage narrative</h2><span class="small muted mono">source: ${esc(c.narrative_source || 'template')}</span></div>
            <p style="margin:12px 0 0;font-size:15px;line-height:1.6">${esc(c.narrative || narrative(c))}</p>
            <p class="small muted" style="margin:10px 0 0">Every number in the narrative comes from the model inputs. LLM output that states any other number is rejected by the pipeline.</p>
          </section>
          <section class="card">
            <h2>Decision</h2>
            <label for="note">Reason (saved to the audit log)</label>
            <input id="note" type="text" placeholder="e.g. Customer confirmed by phone" autocomplete="off">
            <div class="row" style="margin-top:12px">
              <button class="btn" data-act="Approved">Approve</button>
              <button class="btn" data-act="Escalated">Escalate</button>
              <button class="btn danger" data-act="Blocked">Block card</button>
              <button class="btn primary" data-act="Confirmed fraud">Confirm fraud (label)</button>
            </div>
            <p class="small muted" style="margin:10px 0 0">Demo: decisions are stored in this browser only. In production they become training labels.</p>
          </section>
        </div>
        <aside class="side-col">
          <section class="card">
            <div class="small muted" style="text-transform:uppercase;letter-spacing:.6px">${esc(scoreLabel())}</div>
            <div class="mono" style="font-size:40px;font-weight:600">${c.prob.toFixed(2)}</div>
            <div class="meter"><div style="width:${Math.round(c.prob * 100)}%"></div></div>
            <p class="small muted" style="margin:10px 0 0">${d.probabilities_calibrated
              ? `Calibrated probability. Alert threshold ${num(d.threshold, 4)}.`
              : 'Uncalibrated score, not a probability. Tier cut-offs 0.30 / 0.50 / 0.75.'}</p>
          </section>
          <section class="card"><h2>Transaction</h2>${facts.map(([k, v]) => `<div class="kv"><span>${esc(k)}</span><span class="mono">${esc(v)}</span></div>`).join('')}</section>
        </aside>
      </div>`;
  }

  function viewHealth() {
    const d = state.data, dt = d.metrics.champion_decision_tree, xg = d.metrics.challenger_xgboost, op = d.operating_point;
    const M = [['KS', 'ks', 0.40], ['Gini', 'gini', 0.60], ['AUC-ROC', 'auc_roc', null], ['AUC-PR', 'auc_pr', null], ['Brier ↓', 'brier', null]]
      .filter(([, k]) => dt[k] != null || xg[k] != null);
    const mini = (v, col) => v == null ? '' : `<div class="mini"><div style="width:${Math.min(v * 100, 100)}%;background:var(${col})"></div></div>`;
    const top = d.shap[0] ? d.shap[0].value : 1;
    const psiBadge = v => v < 0.10 ? '<span class="badge b-ok">Stable</span>' : v < 0.25 ? '<span class="badge t-MEDIUM">Monitor</span>' : '<span class="badge t-CRITICAL">Review</span>';
    return `
      <h1>Model health</h1>
      <p class="muted" style="margin:6px 0 20px">Model ${esc(d.model_version)} · evaluated on ${esc(d.evaluation)}${d.generated_at ? ' · exported ' + esc(d.generated_at.slice(0, 10)) : ''}</p>
      <div class="split">
        <section class="card main-col" style="display:block">
          <h2>Champion vs challenger</h2>
          <div class="metric-row head"><span>Metric</span><span>Decision tree</span><span>XGBoost</span><span>Rule of thumb</span></div>
          ${M.map(([n, k, b]) => `<div class="metric-row"><strong>${n}</strong>
            <div><div class="mono">${num(dt[k], k === 'brier' ? 4 : 3)}</div>${k === 'brier' ? '' : mini(dt[k], '--model-2')}</div>
            <div><div class="mono"><strong>${num(xg[k], k === 'brier' ? 4 : 3)}</strong></div>${k === 'brier' ? '' : mini(xg[k], '--model')}</div>
            <span class="small ${b == null ? 'muted' : ''}">${b == null ? '—' : '&gt; ' + b.toFixed(2)}</span></div>`).join('')}
          <p class="small muted" style="margin:12px 0 0">KS &gt; 0.40 and Gini &gt; 0.60 are industry rules of thumb; SR 11-7 sets no numeric thresholds.</p>
          ${op ? `<h2 style="margin-top:20px">Operating point (out-of-time)</h2>
            ${[['Threshold (probability)', num(d.threshold, 4)], ['Alerts per day', op.alerts_per_day], ['Precision', num(op.precision)], ['Recall', num(op.recall)], ['Fraud dollars caught', op.fraud_amount_caught_pct + '%']]
              .map(([k, v]) => `<div class="kv"><span>${k}</span><span class="mono">${esc(v)}</span></div>`).join('')}` : ''}
        </section>
        <div class="side-col">
          ${d.psi.score.map(p => `<div class="card kpi"><div class="label">Score PSI · ${esc(p.label)}</div><div class="row" style="margin-top:4px"><span class="value">${num(p.value, 4)}</span>${psiBadge(p.value)}</div></div>`).join('')}
          ${d.model_version === 'v0.1' ? '<div class="note">These are v0.1 numbers: the holdout was also used for early stopping, so they are optimistic, and PSI covers the score only. Re-run the v0.2 pipeline and export to refresh.</div>' : ''}
        </div>
      </div>
      ${d.psi.features.length ? `<section class="card" style="margin-top:20px"><h2>Feature drift (top SHAP features)</h2>
        ${d.psi.features.slice(0, 12).map(f => `<div class="kv"><span class="mono">${esc(f.feature)}</span><span>${num(f.psi, 4)} ${psiBadge(f.psi)}</span></div>`).join('')}</section>` : ''}
      <section class="card" style="margin-top:20px">
        <h2 style="margin-bottom:4px">Global drivers</h2><p class="small muted" style="margin:0 0 14px">Mean |SHAP|, top ${d.shap.length}</p>
        <div class="bars">${d.shap.map(s => `<div class="bar-row" style="grid-template-columns:minmax(0,280px) minmax(0,1fr) 56px">
          <div><span class="mono"><strong>${esc(s.feature)}</strong></span> <span class="muted">· ${esc(describe(s.feature))}</span></div>
          <div class="hbar"><div style="width:${(s.value / top * 100).toFixed(1)}%"></div></div>
          <span class="mono" style="text-align:right">${num(s.value)}</span></div>`).join('')}</div>
      </section>`;
  }

  function viewGovernance() {
    const d = state.data, v01 = d.model_version === 'v0.1';
    const card = [['Purpose', 'Score card-not-present e-commerce transactions for fraud and route alerts to analysts'],
      ['Data', 'IEEE-CIS (Vesta): 590,540 transactions, 3.5% fraud'], ['Champion', 'Decision tree, depth 8'],
      ['Challenger', 'XGBoost, early-stopped on AUC-PR'], ['Split', v01 ? 'Time-ordered 80 / 20' : 'Time-ordered fit 70 / valid 15 / out-of-time 15'],
      ['Calibration', d.probabilities_calibrated ? 'Isotonic, fitted on the valid period' : 'None (raw scores)'],
      ['Explainability', 'TreeSHAP per transaction; grounded narratives'], ['Owner', '[MODEL OWNER]'], ['Next review', '[DATE]']];
    const F = (t, fixed) => [t, fixed ? 'Fixed in v0.2' : v01 ? 'Fix pending (PR #1)' : 'Fixed in v0.2', fixed || !v01];
    const findings = [F('Card features computed across the time split (leakage)'), F('Holdout reused for early stopping'),
      F('Scores uncalibrated; tiers on raw score'), F('No operating threshold from alert capacity'), F('PSI on score only, fixed-width bins'),
      F('Transaction data sent to rotating free LLMs'), ['Production service, tests and CI', 'Planned', false]];
    return `
      <h1 style="margin-bottom:20px">Governance</h1>
      <div class="split">
        <section class="card main-col" style="display:block"><h2>Model card</h2>${card.map(([k, v]) => `<div class="kv"><span>${k}</span><span>${esc(v)}</span></div>`).join('')}</section>
        <section class="card side-col" style="display:block;flex-basis:380px"><h2>Validation findings</h2>
          ${findings.map(([t, s, ok]) => `<div class="kv"><span style="color:var(--ink)">${esc(t)}</span><span class="badge ${ok ? 'b-ok' : 'b-open'}">${esc(s)}</span></div>`).join('')}</section>
      </div>
      <section class="card" style="margin-top:20px">
        <div class="row between"><h2 style="margin:0">Audit log</h2>${state.audit.length ? '<button class="btn" data-clear>Clear</button>' : ''}</div>
        <div style="margin-top:12px">${state.audit.length ? state.audit.map(a => `<div class="audit-row"><span class="mono muted">${esc(a.time)}</span><a class="mono" href="#/case/${encodeURIComponent(a.id)}">TXN-${esc(a.id)}</a><strong>${esc(a.action)}</strong><span>${esc(a.note)}</span></div>`).join('')
          : '<p class="muted" style="margin:0">No decisions yet. Open a case from the queue and record one.</p>'}</div>
      </section>`;
  }

  // ---------- routing ----------
  function render() {
    const [, view = 'queue', arg] = (location.hash || '#/queue').split('/');
    const v = ['queue', 'case', 'health', 'governance'].includes(view) ? view : 'queue';
    document.querySelectorAll('.tabs a').forEach(a => {
      if (a.dataset.view === v) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    $main.innerHTML = v === 'queue' ? viewQueue() : v === 'case' ? viewCase(arg && decodeURIComponent(arg)) : v === 'health' ? viewHealth() : viewGovernance();
    document.title = { queue: 'Alert queue', case: 'Case', health: 'Model health', governance: 'Governance' }[v] + ' · Fraud Risk Console';
  }

  $main.addEventListener('click', e => {
    const f = e.target.closest('[data-filter]');
    if (f) { state.filter = f.dataset.filter; return render(); }
    const act = e.target.closest('[data-act]');
    if (act) {
      const note = (document.getElementById('note').value || '').trim() || '—';
      state.status[state.lastCase] = act.dataset.act;
      state.audit.unshift({ time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), id: state.lastCase, action: act.dataset.act, note });
      store.set('status', state.status); store.set('audit', state.audit);
      return render();
    }
    if (e.target.closest('[data-clear]')) { state.status = {}; state.audit = []; store.set('status', {}); store.set('audit', []); return render(); }
    const row = e.target.closest('tr[data-open]');
    if (row && !e.target.closest('a')) location.hash = '#/case/' + encodeURIComponent(row.dataset.open);
  });
  window.addEventListener('hashchange', () => { render(); $main.focus({ preventScroll: true }); window.scrollTo(0, 0); });

  fetch('data/console.json', { cache: 'no-cache' }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }).then(d => {
    d.cases.forEach(c => { c.id = String(c.id); });
    state.data = d; setTiering(d);
    document.getElementById('meta').innerHTML =
      `<span class="chip">${d.cases_are_samples ? 'Sample cases' : 'Out-of-time cases'}</span><span class="chip">Model ${esc(d.model_version)}</span>`;
    if (d.cases_are_samples || d.model_version === 'v0.1') {
      const b = document.getElementById('banner');
      b.textContent = 'Demo data: the cases below are illustrative samples; model metrics are the published v0.1 results. Run the pipeline and export to show real out-of-time alerts.';
      b.hidden = false;
    }
    render();
  }).catch(err => { $main.innerHTML = `<div class="note">Could not load data/console.json (${esc(err.message)}).</div>`; });
})();
