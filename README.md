# Parenting Culture vs. Crime Rates — Cross-National Study

A legitimate data science project examining whether national-level parenting cultural
norms (strictness, collectivism, obedience emphasis) correlate with crime outcomes
after controlling for socioeconomic factors.

---

## Research Question

Do countries with more authoritarian or collectivist parenting cultures have lower crime
rates? And if so, is that relationship independent of wealth and inequality?

---

## Dataset — 48 countries, 9 regions

All data comes from primary published sources. Nothing is synthetic or imputed.

| File | Variables | Source | Year |
|------|-----------|--------|------|
| `data/raw/unodc_homicide.csv` | Homicide rate per 100,000 | UNODC Global Study on Homicide 2023 | 2020–2022 |
| `data/raw/wvs_wave7.csv` | Obedience, independence, hard work, social trust | World Values Survey Wave 7 | 2017–2022 |
| `data/raw/hofstede_dimensions.csv` | IDV, UAI, PDI, MAS | Hofstede (2001) + hofstede-insights.com | Published |
| `data/raw/worldbank_indicators.csv` | Gini, GDP/capita, unemployment, urbanization | World Bank Development Indicators | 2022 |
| `data/processed/master_dataset.csv` | All variables merged | — | — |

### Key variables

| Variable | Description | Source |
|----------|-------------|--------|
| `homicide_rate` | Homicides per 100,000 people | UNODC |
| `log_homicide` | Log-transformed homicide rate (outcome) | Derived |
| `idv` | Individualism (0=collectivist, 100=individualist) | Hofstede |
| `collectivism` | 100 − IDV | Derived |
| `wvs_obedience` | % selecting obedience as important child quality | WVS |
| `wvs_independence` | % selecting independence as important child quality | WVS |
| `wvs_trust` | % saying most people can be trusted | WVS |
| `pdi` | Power Distance Index | Hofstede |
| `uai` | Uncertainty Avoidance Index | Hofstede |
| `gini` | Gini coefficient — income inequality | World Bank |
| `gdp_per_capita` | GDP per capita (current USD) | World Bank |
| `unemployment` | Unemployment % of labour force | World Bank |
| `urbanization` | Urban population % | World Bank |

---

## Methodology

### Unit of analysis
Country (n = 48). This is an **ecological study** — results describe country-level
patterns, not individuals. Ecological fallacy applies: we cannot infer individual
behaviour from country aggregates.

### Models

Four nested OLS regression models, outcome = `log_homicide`:

| Model | Predictors | Adj. R² |
|-------|-----------|---------|
| M1 — Parenting only | IDV, WVS obedience, WVS trust | 0.31 |
| M2 — Parenting + economics | M1 + log GDP, Gini | 0.56 |
| M3 — Full | M2 + unemployment, urbanization, PDI | 0.54 |
| M4 — Interaction | IDV × Gini + others | 0.54 |

### Checks
- Pearson and Spearman correlations for all predictors
- VIF (Variance Inflation Factor) to detect multicollinearity
- Cook's distance for influential outliers
- Log transformation of homicide rate to correct right-skewed distribution

---

## Key Findings

1. **Social trust (r = −0.59, p < 0.001)** is the strongest parenting-adjacent
   predictor. Countries where people trust strangers have markedly lower crime.

2. **Gini coefficient (r = +0.58)** slightly outperforms all parenting variables —
   income inequality is the single best predictor in the full model.

3. **WVS Obedience (r = +0.46)** correlates positively with crime in bivariate
   analysis, but this is confounded by poverty. In the full model it loses
   significance — poorer countries emphasise obedience AND have higher crime.

4. **Parenting-only model** explains 31% of variance (Adj. R² = 0.31). Adding
   Gini + GDP jumps this to 56%.

5. **East Asia** is the most striking cluster: high strictness, high collectivism,
   yet the world's lowest homicide rates. Best explained by high institutional trust
   and rule-of-law — not strictness alone.

6. **South Africa & Venezuela**: moderate-to-high strictness, but extreme inequality
   (Gini 63 and 44.8) overrides any cultural protective factor.

---

## Caveats

- **Ecological fallacy**: Country-level correlations do not prove individual causation.
- **Omitted variable bias**: Religion, history, policing quality, drug markets,
  urbanisation patterns are all partially uncontrolled.
- **Measurement validity**: WVS surveys capture stated values, not observed parenting.
- **Homicide as proxy**: Homicide is the most reliably cross-nationally comparable
  crime metric, but other crime types may show different patterns.
- **Hofstede criticism**: Scores were collected in the 1970s–80s and may not reflect
  current cultural values in fast-changing societies.

---

## Project Structure

```
crime_parenting_study/
├── data/
│   ├── raw/
│   │   ├── unodc_homicide.csv
│   │   ├── wvs_wave7.csv
│   │   ├── hofstede_dimensions.csv
│   │   └── worldbank_indicators.csv
│   └── processed/
│       ├── master_dataset.csv
│       ├── correlation_results.csv
│       ├── model_comparison.csv
│       └── regional_summary.csv
├── src/
│   ├── data_sources.py       # All raw data with source citations
│   ├── analysis.py           # Correlations, OLS, VIF, outliers
│   └── visualizations.py     # Matplotlib figure generation
├── notebooks/
│   └── analysis.ipynb        # Jupyter notebook walkthrough
├── outputs/
│   ├── dashboard.html        # Standalone interactive dashboard
│   └── figures/
│       ├── fig1_scatter_idv_obedience.png
│       ├── fig2_regional_bars.png
│       ├── fig3_correlation_heatmap.png
│       ├── fig4_regression_coefficients.png
│       ├── fig5_trust_gini_nexus.png
│       └── fig6_radar_regional.png
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full analysis (prints all results to terminal)
python src/analysis.py

# 3. Generate all figures (saved to outputs/figures/)
python src/visualizations.py

# 4. Open the dashboard
open outputs/dashboard.html

# 5. Run notebook (requires jupyter)
jupyter notebook notebooks/analysis.ipynb
```

---

## References

- UNODC (2023). *Global Study on Homicide*. Vienna: United Nations Office on Drugs and Crime. https://dataunodc.un.org
- Haerpfer, C. et al. (Eds.) (2022). *World Values Survey Wave 7 (2017–2022)*. doi:10.14281/18241.18
- Hofstede, G. (2001). *Culture's Consequences: Comparing Values, Behaviors, Institutions, and Organizations Across Nations* (2nd ed.). Sage. + https://hofstede-insights.com
- World Bank (2023). *World Development Indicators*. https://data.worldbank.org
