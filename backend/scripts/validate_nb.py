import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('MAAGAP_Objective1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
for i, cell in enumerate(nb['cells']):
    if 'cell_type' not in cell: errors.append('Cell ' + str(i) + ': missing cell_type')
    if 'source' not in cell:    errors.append('Cell ' + str(i) + ': missing source')
    if 'id' not in cell:        errors.append('Cell ' + str(i) + ': missing id')

ids = [c['id'] for c in nb['cells']]
if len(ids) != len(set(ids)): errors.append('Duplicate cell IDs')

obj3 = [i for i, c in enumerate(nb['cells']) if 'logic_consistency_check' in ''.join(c['source'])]
obj4 = [i for i, c in enumerate(nb['cells']) if 'monte_carlo_robustness'   in ''.join(c['source'])]
ft   = [i for i, c in enumerate(nb['cells']) if 'load_fund_transfer_con'   in ''.join(c['source'])]
spl  = [i for i, c in enumerate(nb['cells']) if 'Train/Val overlap'        in ''.join(c['source'])]

ver = str(nb['nbformat']) + '.' + str(nb['nbformat_minor'])
print('Notebook format:', ver)
print('Total cells:', len(nb['cells']))
print('Obj3 cells (logic_consistency_check):', obj3)
print('Obj4 cells (monte_carlo_robustness):', obj4)
print('FT cells (load_fund_transfer_con):', ft)
print('Split check cells:', spl)
print()
if errors:
    for e in errors: print('ERROR:', e)
else:
    print('All validation checks PASSED')
