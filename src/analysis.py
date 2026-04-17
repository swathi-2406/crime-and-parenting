"""
Statistical analysis: Parenting Culture vs. Crime Rates
Author: Generated analysis
Methods: OLS regression, Pearson/Spearman correlation, residual analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_sources import build_master_dataframe


def correlation_analysis(df):
    """Pearson and Spearman correlations between parenting indicators and crime."""
    parenting_vars = ['idv', 'collectivism', 'wvs_obedience', 'wvs_independence',
                      'wvs_hard_work', 'wvs_trust', 'pdi', 'uai']
    control_vars   = ['gini', 'log_gdp', 'unemployment', 'urbanization']
    outcome_vars   = ['homicide_rate', 'log_homicide']

    results = []
    for pvar in parenting_vars + control_vars:
        for ovar in outcome_vars:
            valid = df[[pvar, ovar]].dropna()
            r_p, p_p = stats.pearsonr(valid[pvar], valid[ovar])
            r_s, p_s = stats.spearmanr(valid[pvar], valid[ovar])
            results.append({
                'predictor': pvar,
                'outcome': ovar,
                'pearson_r': round(r_p, 3),
                'pearson_p': round(p_p, 4),
                'spearman_r': round(r_s, 3),
                'spearman_p': round(p_s, 4),
                'n': len(valid),
                'significant_p05': p_p < 0.05,
            })
    return pd.DataFrame(results)


def ols_models(df):
    """
    Four nested OLS models (outcome = log_homicide):
      M1: Parenting only (IDV + WVS obedience)
      M2: Parenting + economic controls
      M3: Full model
      M4: Full model + interaction (IDV * Gini)
    """
    models = {}

    # M1 - parenting only
    m1 = smf.ols('log_homicide ~ idv + wvs_obedience + wvs_trust', data=df).fit()
    models['M1_parenting_only'] = m1

    # M2 - add economic controls
    m2 = smf.ols('log_homicide ~ idv + wvs_obedience + wvs_trust + log_gdp + gini', data=df).fit()
    models['M2_parenting_economic'] = m2

    # M3 - full model
    m3 = smf.ols('log_homicide ~ idv + wvs_obedience + wvs_trust + log_gdp + gini + unemployment + urbanization + pdi', data=df).fit()
    models['M3_full'] = m3

    # M4 - interaction term
    m4 = smf.ols('log_homicide ~ idv * gini + wvs_obedience + wvs_trust + log_gdp + unemployment', data=df).fit()
    models['M4_interaction'] = m4

    return models


def model_comparison_table(models):
    """Create a clean comparison table across models."""
    rows = []
    for name, m in models.items():
        rows.append({
            'Model': name,
            'N': int(m.nobs),
            'R²': round(m.rsquared, 3),
            'Adj. R²': round(m.rsquared_adj, 3),
            'AIC': round(m.aic, 1),
            'BIC': round(m.bic, 1),
            'F-stat': round(m.fvalue, 2),
            'F p-value': round(m.f_pvalue, 4),
        })
    return pd.DataFrame(rows)


def vif_check(df):
    """Variance Inflation Factor to check multicollinearity in M3."""
    cols = ['idv', 'wvs_obedience', 'wvs_trust', 'log_gdp', 'gini',
            'unemployment', 'urbanization', 'pdi']
    X = df[cols].dropna()
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame({
        'variable': cols,
        'VIF': [round(variance_inflation_factor(X_const.values, i+1), 2) for i in range(len(cols))]
    })
    return vif_data


def regional_summary(df):
    """Aggregate statistics by region."""
    numeric_cols = ['homicide_rate', 'idv', 'collectivism', 'wvs_obedience',
                    'wvs_trust', 'gini', 'gdp_per_capita', 'pdi']
    summary = df.groupby('region')[numeric_cols].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    return summary.round(2)


def outlier_analysis(df, model):
    """Identify influential outliers using Cook's distance."""
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]
    threshold = 4 / len(df)
    outliers = df.copy()
    outliers['cooks_d'] = cooks
    outliers['is_outlier'] = cooks > threshold
    return outliers[['region', 'homicide_rate', 'idv', 'wvs_obedience', 'gini', 'cooks_d', 'is_outlier']].sort_values('cooks_d', ascending=False)


def run_full_analysis():
    df = build_master_dataframe()

    print("=" * 60)
    print("PARENTING CULTURE VS CRIME RATES — ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nDataset: {len(df)} countries, {df['region'].nunique()} regions\n")

    print("--- CORRELATION ANALYSIS ---")
    corr = correlation_analysis(df)
    print(corr[corr['outcome'] == 'log_homicide'].to_string(index=False))

    print("\n--- OLS REGRESSION MODELS ---")
    models = ols_models(df)
    print(model_comparison_table(models).to_string(index=False))

    print("\n--- FULL MODEL SUMMARY (M3) ---")
    print(models['M3_full'].summary())

    print("\n--- VIF CHECK (multicollinearity) ---")
    print(vif_check(df).to_string(index=False))

    print("\n--- REGIONAL SUMMARY ---")
    print(regional_summary(df))

    print("\n--- OUTLIER ANALYSIS (Cook's D) ---")
    outliers = outlier_analysis(df, models['M3_full'])
    print(outliers.head(10).to_string())

    return df, models, corr


if __name__ == '__main__':
    df, models, corr = run_full_analysis()
