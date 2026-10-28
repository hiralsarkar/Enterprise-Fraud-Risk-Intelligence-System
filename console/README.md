# Fraud Risk Console

A static analyst console for the fraud pipeline: alert queue, per-case SHAP review with triage narrative and decisions, model health (champion vs challenger, PSI, global drivers) and governance (model card, findings, audit log).

- **No build step.** Plain HTML, CSS and JavaScript. Data comes from `data/console.json`.
- **Refresh the data** after running the pipeline: `python fraud_classifier/scripts/export_console_data.py` (writes `data/console.json` from `fraud_classifier/outputs/`).
- **Run locally:** `python -m http.server -d console 8000`, then open http://localhost:8000.
- **Deploy on Vercel:** import the repo, set **Root Directory** to `console`, framework preset **Other**, no build command. `vercel.json` adds security headers.

Analyst decisions are kept in the browser's local storage; this is a demo, not a system of record.
