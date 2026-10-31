"""Write console/data/console.json for the analyst console from the pipeline outputs.

Run after notebooks 1-3:  python fraud_classifier/scripts/export_console_data.py
Optional: --out PATH to write somewhere else.
"""
import argparse, json, pathlib
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'fraud_classifier' / 'outputs'


def load(name):
    with open(OUT_DIR / name) as f:
        return json.load(f)


def main(out_path):
    metrics, op, psi = load('metrics.json'), load('operating_point.json'), load('psi_results.json')
    shap, cases = load('top_features.json'), load('triage_cases.json')
    split = metrics['split']['oot']

    def card(c):
        parts = [c.get('card4'), c.get('card6')]
        return ' '.join(p for p in parts if p) or '—'

    data = {
        'model_version': 'v0.2',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'evaluation': f"out-of-time period: {split['rows']:,} transactions over {split['days']} days",
        'cases_are_samples': False,
        'probabilities_calibrated': True,
        'threshold': op['threshold_probability'],
        'metrics': {k: metrics[k]['oot'] for k in ('champion_decision_tree', 'challenger_xgboost')},
        'operating_point': op['oot'],
        'psi': {
            'score': [{'label': 'Valid → out-of-time', 'value': psi['score']['psi_valid_to_oot']},
                      {'label': 'Valid → Kaggle test', 'value': psi['score']['psi_valid_to_kaggle_test']}],
            'features': [{'feature': f['feature'], 'psi': max(f['psi_oot'], f['psi_kaggle_test'])} for f in psi['features']],
        },
        'shap': [{'feature': s['feature'], 'value': round(s['mean_abs_shap'], 4)} for s in shap[:10]],
        'cases': [{
            'id': str(c['transaction_id']), 'prob': c['fraud_probability'], 'amount': c['amount'],
            'product': c.get('ProductCD') or '—', 'card': card(c), 'email': c.get('P_emaildomain') or '—',
            'device': c.get('DeviceType') or '—', 'age_min': None, 'actual': c['actual'],
            'narrative': c['narrative'], 'narrative_source': c['source'], 'drivers': c['drivers'],
        } for c in cases],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1) + '\n')
    print(f"Wrote {out_path} ({len(data['cases'])} cases)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=pathlib.Path, default=ROOT / 'console' / 'data' / 'console.json')
    main(ap.parse_args().out)
