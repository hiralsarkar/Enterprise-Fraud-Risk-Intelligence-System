"""Export the notebooks/*.py scripts (percent format: '# %%' code cells, '# %% [markdown]' text cells)
to .ipynb so the scripts stay the single source of truth.  Usage: python scripts/py_to_ipynb.py"""
import json, pathlib, re

def convert(py_path):
    cells, kind, buf = [], None, []
    def flush():
        text = '\n'.join(buf).strip('\n')
        if kind and text:
            if kind == 'markdown':
                text = '\n'.join(re.sub(r'^# ?', '', l) for l in text.splitlines())
                cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})
            else:
                cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                              'source': text.splitlines(True)})
    for line in pathlib.Path(py_path).read_text().splitlines():
        if line.startswith('# %%'):
            flush(); buf = []
            kind = 'markdown' if '[markdown]' in line else 'code'
        else:
            buf.append(line)
    flush()
    nb = {'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}}}
    out = pathlib.Path(py_path).with_suffix('.ipynb')
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    return out, len(cells)

if __name__ == '__main__':
    for p in sorted((pathlib.Path(__file__).parent.parent / 'notebooks').glob('0*.py')):
        print(*convert(p))
