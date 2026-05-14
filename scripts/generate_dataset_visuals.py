import pandas as pd
import plotly.graph_objects as go
import os

os.makedirs('outputs', exist_ok=True)

# 1. Dataset Shape Summary Diagram
shapes_data = [
    ['Real PPDO Monitoring (2026)', '800 records', 'Extracted: budget distributions, infra/non-infra ratio'],
    ['Real Fund Transfer (2013-2026)', '21,083 records', 'Extracted: municipality distribution, funding source, 95% liquidation rate'],
    ['Synthetic Projects (Static)', '3,000 rows × 30 columns', 'Features: budget, agency capacity, contractor reliability, typhoon exposure, etc.'],
    ['Synthetic Quarterly (Temporal)', '3,000 projects × 4 quarters × 9 features', 'Features: planned vs actual progress, slippage, expenditure ratio, etc.'],
    ['Train / Val / Test Split', '2,100 / 450 / 450 projects', '70% / 15% / 15% stratified split']
]

fig1 = go.Figure(data=[go.Table(
    columnorder=[1,2,3],
    columnwidth=[80, 80, 150],
    header=dict(
        values=['<b>Dataset Component</b>', '<b>Shape / Size</b>', '<b>Description / Usage</b>'],
        line_color='darkslategray',
        fill_color='#2c3e50',
        align='left',
        font=dict(color='white', size=16),
        height=40
    ),
    cells=dict(
        values=[
            [row[0] for row in shapes_data],
            [row[1] for row in shapes_data],
            [row[2] for row in shapes_data]
        ],
        line_color='darkslategray',
        fill_color=['#ecf0f1', '#ecf0f1', '#ecf0f1'],
        align='left',
        font=dict(color='black', size=14),
        height=40
    )
)])

fig1.update_layout(title=dict(text='MAAGAP Dataset Dimensions & Usage', x=0.5), margin=dict(l=20, r=20, t=50, b=20), width=1000, height=350)
fig1.write_image('outputs/dataset_shapes_summary.png', scale=2)

# 2. Sample data snapshot (Static Features)
df_proj = pd.read_csv('data/processed/synthetic_projects.csv')
cols_to_show = ['project_id', 'implementing_agency', 'project_type', 'approved_budget', 'contractor_reliability', 'delay_probability', 'risk_category']
df_sample = df_proj[cols_to_show].head(6).copy()
df_sample['approved_budget'] = df_sample['approved_budget'].apply(lambda x: f'PHP {x:,.2f}')
df_sample['contractor_reliability'] = df_sample['contractor_reliability'].apply(lambda x: f'{x:.2f}')
df_sample['delay_probability'] = df_sample['delay_probability'].apply(lambda x: f'{x:.1%}')

fig2 = go.Figure(data=[go.Table(
    header=dict(
        values=[f'<b>{c.replace("_", " ").title()}</b>' for c in cols_to_show],
        fill_color='#3498db',
        align='center',
        font=dict(color='white', size=14),
        height=35
    ),
    cells=dict(
        values=[df_sample[c] for c in cols_to_show],
        fill_color='#f7f9f9',
        align='center',
        font=dict(color='#2c3e50', size=12),
        height=30
    )
)])

fig2.update_layout(title=dict(text='Sample Generated Project Data (Static Snapshot)', x=0.5), margin=dict(l=20, r=20, t=50, b=20), width=1100, height=300)
fig2.write_image('outputs/dataset_sample_static.png', scale=2)

# 3. Sample data snapshot (Temporal Features)
df_qtr = pd.read_csv('data/processed/synthetic_quarterly.csv')
cols_qtr = ['project_id', 'quarter', 'planned_progress_pct', 'actual_progress_pct', 'slippage_pct', 'rainfall_mm', 'typhoon_days']
df_qtr_sample = df_qtr[cols_qtr].head(8).copy()
for col in ['planned_progress_pct', 'actual_progress_pct', 'slippage_pct']:
    df_qtr_sample[col] = df_qtr_sample[col].apply(lambda x: f'{x:.1f}%')

fig3 = go.Figure(data=[go.Table(
    header=dict(
        values=[f'<b>{c.replace("_", " ").title()}</b>' for c in cols_qtr],
        fill_color='#e74c3c',
        align='center',
        font=dict(color='white', size=14),
        height=35
    ),
    cells=dict(
        values=[df_qtr_sample[c] for c in cols_qtr],
        fill_color='#f7f9f9',
        align='center',
        font=dict(color='#2c3e50', size=12),
        height=30
    )
)])

fig3.update_layout(title=dict(text='Sample Quarterly Monitoring Data (Temporal Snapshot)', x=0.5), margin=dict(l=20, r=20, t=50, b=20), width=1100, height=350)
fig3.write_image('outputs/dataset_sample_temporal.png', scale=2)

print('Generated dataset shapes and tables in outputs/')
