import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('MAAGAP_Objective1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
print('Total cells:', len(nb['cells']))
print()
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    preview = src[:90].replace(chr(10), ' ')
    ctype = cell['cell_type']
    print(f'[{i:02d}] {ctype:8s} | {preview}')
