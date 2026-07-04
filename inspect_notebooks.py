import json
from pathlib import Path

root = Path('.')
for path in sorted(root.glob('*.ipynb')):
    print('FILE', path)
    try:
        nb = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print('ERR', exc)
        continue
    for i, cell in enumerate(nb.get('cells', []), 1):
        src = ''.join(cell.get('source', []))
        if src.strip():
            print(f'--- cell {i} [{cell["cell_type"]}] ---')
            print(src[:2800])
            print()
    print('====')
