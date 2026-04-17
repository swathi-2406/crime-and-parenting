"""
Visualization script — generates all figures for the report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from data_sources import build_master_dataframe

REGION_COLORS = {
    'East Asia':         '#1D9E75',
    'Western Europe':    '#378ADD',
    'Latin America':     '#D85A30',
    'Sub-Saharan Africa':'#BA7517',
    'North America':     '#7F77DD',
    'MENA':              '#D4537E',
    'South/SE Asia':     '#5DCAA5',
    'Eastern Europe':    '#888780',
    'Oceania':           '#97C459',
}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def apply_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'figure.dpi': 150,
    })


def fig1_scatter_homicide_vs_idv(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Homicide Rate vs. Cultural Dimensions', fontsize=14, fontweight='bold', y=1.01)

    for ax, (xcol, xlabel) in zip(axes, [
        ('idv', 'Individualism score (Hofstede IDV)\n0 = fully collectivist, 100 = fully individualist'),
        ('wvs_obedience', 'WVS Obedience as child quality (%)'),
    ]):
        for region, grp in df.groupby('region'):
            ax.scatter(grp[xcol], grp['log_homicide'],
                       color=REGION_COLORS.get(region, '#888'),
                       s=55, alpha=0.85, label=region, zorder=3)
            for _, row in grp.iterrows():
                ax.annotate(row.name, (row[xcol], row['log_homicide']),
                            fontsize=6.5, ha='left', va='bottom', color='#444', xytext=(3, 2),
                            textcoords='offset points')

        # regression line
        m = smf.ols(f'log_homicide ~ {xcol}', data=df).fit()
        x_range = np.linspace(df[xcol].min(), df[xcol].max(), 100)
        y_pred = m.params['Intercept'] + m.params[xcol] * x_range
        r, p = stats.pearsonr(df[xcol], df['log_homicide'])
        ax.plot(x_range, y_pred, color='#333', lw=1.5, ls='--', alpha=0.7)
        ax.text(0.05, 0.95, f'r = {r:.2f}  (p = {p:.3f})',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Log homicide rate')

    handles = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in REGION_COLORS if r in df['region'].values]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08),
               fontsize=9, frameon=True)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_scatter_idv_obedience.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig2_regional_bars(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Regional Averages — Parenting & Crime Indicators', fontsize=13, fontweight='bold')

    reg_order = df.groupby('region')['homicide_rate'].mean().sort_values(ascending=False).index
    colors = [REGION_COLORS.get(r, '#888') for r in reg_order]

    metrics = [
        ('homicide_rate', 'Avg homicide rate (per 100k)\nUNODC 2020–2022'),
        ('wvs_obedience', 'Avg WVS obedience score (%)\nWorld Values Survey Wave 7'),
        ('idv', 'Avg individualism score (IDV)\nHofstede Cultural Dimensions'),
    ]

    for ax, (col, title) in zip(axes, metrics):
        vals = df.groupby('region')[col].mean().loc[reg_order]
        bars = ax.barh(range(len(reg_order)), vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(reg_order)))
        ax.set_yticklabels(reg_order, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        for i, v in enumerate(vals):
            ax.text(v + 0.3, i, f'{v:.1f}', va='center', fontsize=8.5)
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_regional_bars.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig3_correlation_heatmap(df):
    cols = ['homicide_rate', 'log_homicide', 'idv', 'collectivism',
            'wvs_obedience', 'wvs_independence', 'wvs_trust',
            'pdi', 'gini', 'log_gdp', 'unemployment']
    labels = ['Homicide rate', 'Log homicide', 'IDV (indiv.)', 'Collectivism',
              'WVS obedience', 'WVS independence', 'WVS trust',
              'Power distance', 'Gini coeff.', 'Log GDP', 'Unemployment']

    corr_matrix = df[cols].corr()
    p_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i != j:
                _, p = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
                p_matrix.loc[c1, c2] = p

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, ax=ax, annot=True, fmt='.2f',
                annot_kws={'size': 8}, xticklabels=labels, yticklabels=labels)

    # mark significant cells
    for i in range(len(cols)):
        for j in range(i):
            if p_matrix.iloc[i, j] < 0.05:
                ax.text(j + 0.85, i + 0.15, '*', fontsize=10, color='black', fontweight='bold')

    ax.set_title('Correlation Matrix — All Variables\n(* = p < 0.05)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=40, ha='right')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_correlation_heatmap.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig4_model_coefficients(df):
    import statsmodels.formula.api as smf
    m3 = smf.ols('log_homicide ~ idv + wvs_obedience + wvs_trust + log_gdp + gini + unemployment + urbanization + pdi', data=df).fit()

    coef_names = {
        'idv': 'Individualism (IDV)',
        'wvs_obedience': 'WVS Obedience',
        'wvs_trust': 'WVS Social Trust',
        'log_gdp': 'Log GDP per capita',
        'gini': 'Gini Coefficient',
        'unemployment': 'Unemployment %',
        'urbanization': 'Urbanization %',
        'pdi': 'Power Distance (PDI)',
    }

    params = m3.params.drop('Intercept')
    conf = m3.conf_int().drop('Intercept')
    pvals = m3.pvalues.drop('Intercept')

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = range(len(params))
    colors = ['#D85A30' if v > 0 else '#1D9E75' for v in params.values]
    alphas = [1.0 if p < 0.05 else 0.45 for p in pvals.values]

    for i, (name, val) in enumerate(params.items()):
        lo, hi = conf.loc[name]
        color = colors[i]
        alpha = alphas[i]
        ax.barh(i, val, color=color, alpha=alpha, height=0.6, zorder=3)
        ax.plot([lo, hi], [i, i], color='#333', lw=2, zorder=4)
        ax.plot([lo, lo], [i-0.15, i+0.15], color='#333', lw=2, zorder=4)
        ax.plot([hi, hi], [i-0.15, i+0.15], color='#333', lw=2, zorder=4)
        sig = '***' if pvals[name] < 0.001 else ('**' if pvals[name] < 0.01 else ('*' if pvals[name] < 0.05 else ''))
        ax.text(hi + 0.01, i, f' β={val:.3f} {sig}', va='center', fontsize=9)

    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([coef_names.get(n, n) for n in params.index])
    ax.set_xlabel('Coefficient (outcome = log homicide rate)')
    ax.set_title(f'OLS Regression Coefficients — Full Model (M3)\nAdj. R² = {m3.rsquared_adj:.3f}, n = {int(m3.nobs)}',
                 fontsize=12, fontweight='bold')

    red_patch = mpatches.Patch(color='#D85A30', label='Positive effect (↑ crime)')
    green_patch = mpatches.Patch(color='#1D9E75', label='Negative effect (↓ crime)')
    ax.legend(handles=[red_patch, green_patch], fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_regression_coefficients.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig5_trust_gini_homicide(df):
    """3-panel plot showing the trust-inequality-crime nexus."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('The Trust–Inequality–Crime Nexus', fontsize=13, fontweight='bold')

    pairs = [
        ('gini', 'log_homicide', 'Gini coefficient', 'Log homicide rate', 'Inequality → Crime'),
        ('wvs_trust', 'log_homicide', 'Social trust (% trusting others)', 'Log homicide rate', 'Trust → Crime'),
        ('gini', 'wvs_trust', 'Gini coefficient', 'Social trust (%)', 'Inequality → Trust'),
    ]

    for ax, (x, y, xl, yl, title) in zip(axes, pairs):
        for region, grp in df.groupby('region'):
            ax.scatter(grp[x], grp[y], color=REGION_COLORS.get(region, '#888'),
                       s=50, alpha=0.8, zorder=3)
        r, p = stats.pearsonr(df[x], df[y])
        xr = np.linspace(df[x].min(), df[x].max(), 100)
        m, b = np.polyfit(df[x], df[y], 1)
        ax.plot(xr, m * xr + b, color='#333', lw=1.5, ls='--', alpha=0.7)
        ax.text(0.05, 0.93, f'r = {r:.2f}  p = {p:.3f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

    handles = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in REGION_COLORS if r in df['region'].values]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_trust_gini_nexus.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def fig6_radar_regional(df):
    """Radar / spider chart comparing parenting profiles by region."""
    from matplotlib.patches import FancyArrowPatch

    regions = list(df['region'].unique())
    metrics = ['wvs_obedience', 'collectivism', 'wvs_hard_work', 'pdi', 'uai']
    metric_labels = ['Obedience\n(WVS)', 'Collectivism\n(Hofstede)', 'Hard work\n(WVS)',
                     'Power dist.\n(Hofstede)', 'Uncert. avoid.\n(Hofstede)']

    df_norm = df.copy()
    for col in metrics:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

    reg_means = df_norm.groupby('region')[metrics].mean()

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 100)

    for region, row in reg_means.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        color = REGION_COLORS.get(region, '#888')
        ax.plot(angles, values, color=color, lw=2, label=region)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_title('Parenting Culture Profiles by Region\n(All metrics normalized 0–100)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_radar_regional.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def generate_all():
    apply_style()
    df = build_master_dataframe()
    print(f"Generating figures for {len(df)} countries...")
    fig1_scatter_homicide_vs_idv(df)
    fig2_regional_bars(df)
    fig3_correlation_heatmap(df)
    fig4_model_coefficients(df)
    fig5_trust_gini_homicide(df)
    fig6_radar_regional(df)
    print("\nAll figures saved to outputs/figures/")


if __name__ == '__main__':
    generate_all()
