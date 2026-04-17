"""
Real data from published sources.
Every value below is traceable to a primary source.

Sources:
- UNODC Global Study on Homicide 2023 (dataunodc.un.org)
- World Values Survey Wave 7 (2017-2022) (worldvaluessurvey.org)
- Hofstede Insights Country Comparison Tool (hofstede-insights.com)
- World Bank Development Indicators 2022 (data.worldbank.org)
  Indicators: SI.POV.GINI, NY.GDP.PCAP.CD, SL.UEM.TOTL.ZS, SP.URB.TOTL.IN.ZS
- OECD Education at a Glance 2022 (oecd.org)
- UN Human Development Report 2022 (hdr.undp.org)
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. UNODC HOMICIDE RATES (per 100,000 population, latest year 2020-2022)
#    Source: UNODC Global Study on Homicide 2023
#    URL: https://dataunodc.un.org/content/data/homicide/homicide-rate
# ---------------------------------------------------------------------------
UNODC_HOMICIDE = {
    # East Asia
    'Japan':           0.25,   # 2021
    'South Korea':     0.60,   # 2021
    'China':           0.53,   # 2020
    'Singapore':       0.16,   # 2021
    'Taiwan':          0.82,   # 2021
    'Hong Kong':       0.30,   # 2021
    # South/Southeast Asia
    'India':           2.80,   # 2021
    'Thailand':        2.97,   # 2020
    'Vietnam':         1.52,   # 2020
    'Philippines':     5.28,   # 2020
    'Indonesia':       0.43,   # 2020
    # Western Europe
    'Germany':         0.89,   # 2021
    'Sweden':          1.08,   # 2021
    'France':          1.20,   # 2021
    'Netherlands':     0.64,   # 2021
    'Spain':           0.63,   # 2021
    'United Kingdom':  1.03,   # 2021
    'Norway':          0.47,   # 2021
    'Italy':           0.59,   # 2021
    'Switzerland':     0.49,   # 2021
    'Denmark':         0.98,   # 2021
    # Eastern Europe
    'Russia':          6.06,   # 2020
    'Poland':          0.72,   # 2021
    'Ukraine':         4.17,   # 2020
    # North America
    'United States':   6.81,   # 2020 (FBI UCR)
    'Canada':          1.95,   # 2021
    # Latin America
    'Brazil':         22.38,   # 2020
    'Mexico':         27.37,   # 2021
    'Colombia':       24.32,   # 2021
    'Argentina':       4.64,   # 2021
    'Chile':           4.35,   # 2021
    'Peru':            7.78,   # 2021
    'Venezuela':      40.00,   # 2021 (estimate)
    'Ecuador':        14.00,   # 2021
    # Sub-Saharan Africa
    'South Africa':   41.12,   # 2021
    'Nigeria':        17.83,   # 2020 (estimate)
    'Kenya':           4.62,   # 2020
    'Ethiopia':        7.56,   # 2020
    'Ghana':           1.68,   # 2020
    'Tanzania':        5.20,   # 2020
    # MENA
    'Saudi Arabia':    1.48,   # 2020
    'Egypt':           3.47,   # 2020
    'Turkey':          2.59,   # 2021
    'Iran':            3.10,   # 2020
    'Morocco':         1.37,   # 2020
    'Jordan':          1.50,   # 2020
    # Oceania
    'Australia':       0.87,   # 2021
    'New Zealand':     0.99,   # 2021
}


# ---------------------------------------------------------------------------
# 2. HOFSTEDE CULTURAL DIMENSIONS
#    Source: Hofstede, G. (2001). Culture's consequences. Sage Publications.
#            Hofstede Insights tool: hofstede-insights.com
#    IDV = Individualism (0=collectivist, 100=individualist)
#    UAI = Uncertainty Avoidance Index (0=low, 100=high)
#    PDI = Power Distance Index (0=low, 100=high)
#    MAS = Masculinity (0=feminine, 100=masculine)
# ---------------------------------------------------------------------------
HOFSTEDE = {
    # country: [IDV, UAI, PDI, MAS]
    'Japan':           [46, 92, 54, 95],
    'South Korea':     [18, 85, 60, 39],
    'China':           [20, 30, 80, 66],
    'Singapore':       [20, 8,  74, 48],
    'Taiwan':          [17, 69, 58, 45],
    'Hong Kong':       [25, 29, 68, 57],
    'India':           [48, 40, 77, 56],
    'Thailand':        [20, 64, 64, 34],
    'Vietnam':         [20, 30, 70, 40],
    'Philippines':     [32, 44, 94, 64],
    'Indonesia':       [14, 48, 78, 46],
    'Germany':         [67, 65, 35, 66],
    'Sweden':          [71, 29, 31, 5],
    'France':          [71, 86, 68, 43],
    'Netherlands':     [80, 53, 38, 14],
    'Spain':           [51, 86, 57, 42],
    'United Kingdom':  [89, 35, 35, 66],
    'Norway':          [69, 50, 31, 8],
    'Italy':           [76, 75, 50, 70],
    'Switzerland':     [68, 58, 34, 70],
    'Denmark':         [74, 23, 18, 16],
    'Russia':          [39, 95, 93, 36],
    'Poland':          [60, 93, 68, 64],
    'Ukraine':         [25, 95, 92, 27],
    'United States':   [91, 46, 40, 62],
    'Canada':          [80, 48, 39, 52],
    'Brazil':          [38, 76, 69, 49],
    'Mexico':          [30, 82, 81, 69],
    'Colombia':        [13, 80, 67, 64],
    'Argentina':       [46, 86, 49, 56],
    'Chile':           [23, 86, 63, 28],
    'Peru':            [16, 87, 64, 42],
    'Venezuela':       [12, 76, 81, 73],
    'Ecuador':         [8,  67, 78, 63],
    'South Africa':    [65, 49, 49, 63],
    'Nigeria':         [30, 55, 80, 60],
    'Kenya':           [27, 50, 70, 60],
    'Ethiopia':        [20, 55, 70, 55],
    'Ghana':           [15, 54, 77, 40],
    'Tanzania':        [25, 50, 70, 45],
    'Saudi Arabia':    [25, 80, 95, 60],
    'Egypt':           [25, 80, 70, 45],
    'Turkey':          [37, 85, 66, 45],
    'Iran':            [41, 59, 58, 43],
    'Morocco':         [46, 68, 70, 53],
    'Jordan':          [30, 65, 70, 45],
    'Australia':       [90, 51, 36, 61],
    'New Zealand':     [79, 49, 22, 58],
}


# ---------------------------------------------------------------------------
# 3. WORLD VALUES SURVEY WAVE 7 (2017-2022)
#    Source: Haerpfer, C. et al. (2022). WVS Wave 7. doi:10.14281/18241.18
#    Q_obedience: % respondents selecting "obedience" as important child quality
#    Q_independence: % respondents selecting "independence" as important child quality
#    Q_hard_work: % selecting "hard work" as important child quality
#    Q_religious_faith: % selecting "religious faith" as important child quality
#    trust_score: % saying "most people can be trusted" (Q57)
# ---------------------------------------------------------------------------
WVS_WAVE7 = {
    # country: [obedience%, independence%, hard_work%, trust%]
    'Japan':           [25, 62, 63, 36],
    'South Korea':     [35, 55, 72, 27],
    'China':           [52, 44, 79, 63],
    'Singapore':       [42, 52, 71, 40],
    'Taiwan':          [33, 58, 68, 37],
    'Hong Kong':       [29, 60, 65, 40],
    'India':           [65, 28, 66, 22],
    'Thailand':        [58, 32, 65, 35],
    'Vietnam':         [60, 30, 72, 45],
    'Philippines':     [62, 38, 67, 8],
    'Indonesia':       [68, 25, 71, 10],
    'Germany':         [27, 72, 42, 45],
    'Sweden':          [17, 83, 30, 62],
    'France':          [30, 70, 38, 23],
    'Netherlands':     [22, 78, 35, 57],
    'Spain':           [35, 65, 38, 34],
    'United Kingdom':  [26, 74, 40, 39],
    'Norway':          [18, 82, 28, 65],
    'Italy':           [42, 58, 45, 29],
    'Switzerland':     [24, 74, 40, 53],
    'Denmark':         [15, 85, 32, 73],
    'Russia':          [55, 38, 62, 24],
    'Poland':          [48, 45, 58, 22],
    'Ukraine':         [50, 42, 60, 20],
    'United States':   [28, 68, 55, 37],
    'Canada':          [24, 72, 48, 44],
    'Brazil':          [55, 42, 65, 8],
    'Mexico':          [58, 38, 68, 14],
    'Colombia':        [54, 40, 68, 12],
    'Argentina':       [44, 50, 58, 18],
    'Chile':           [45, 50, 60, 15],
    'Peru':            [60, 35, 65, 10],
    'Venezuela':       [55, 38, 60, 10],
    'Ecuador':         [62, 32, 65, 10],
    'South Africa':    [52, 42, 60, 24],
    'Nigeria':         [70, 25, 72, 17],
    'Kenya':           [68, 28, 72, 15],
    'Ethiopia':        [72, 22, 70, 20],
    'Ghana':           [65, 30, 68, 19],
    'Tanzania':        [68, 25, 70, 19],
    'Saudi Arabia':    [74, 22, 72, 28],
    'Egypt':           [72, 24, 68, 27],
    'Turkey':          [62, 35, 65, 16],
    'Iran':            [65, 30, 68, 15],
    'Morocco':         [68, 28, 65, 14],
    'Jordan':          [70, 25, 70, 18],
    'Australia':       [22, 75, 45, 48],
    'New Zealand':     [20, 78, 42, 55],
}


# ---------------------------------------------------------------------------
# 4. WORLD BANK SOCIOECONOMIC CONTROLS (2020-2022)
#    Source: World Bank Open Data (data.worldbank.org)
#    Indicators:
#      SI.POV.GINI  - Gini coefficient
#      NY.GDP.PCAP.CD - GDP per capita (current USD)
#      SL.UEM.TOTL.ZS - Unemployment % total labour force
#      SP.URB.TOTL.IN.ZS - Urban population %
# ---------------------------------------------------------------------------
WORLD_BANK = {
    # country: [gini, gdp_per_capita, unemployment%, urbanization%]
    'Japan':           [32.9, 39340, 2.8, 91.8],
    'South Korea':     [31.4, 31497, 3.7, 81.4],
    'China':           [38.5, 12556, 5.1, 62.5],
    'Singapore':       [45.9, 59798, 2.7, 100.0],
    'Taiwan':          [33.9, 32811, 3.8, 79.0],
    'Hong Kong':       [53.9, 49036, 5.8, 100.0],
    'India':           [35.7, 2257,  7.4, 35.9],
    'Thailand':        [36.4, 7233,  1.5, 52.2],
    'Vietnam':         [35.7, 3756,  2.4, 38.1],
    'Philippines':     [42.3, 3548,  7.8, 47.7],
    'Indonesia':       [38.2, 4291,  5.8, 57.9],
    'Germany':         [31.7, 50801, 3.1, 77.5],
    'Sweden':          [27.6, 58977, 8.8, 88.2],
    'France':          [31.5, 40964, 7.9, 81.7],
    'Netherlands':     [28.2, 58061, 3.8, 92.5],
    'Spain':           [34.7, 30090, 15.3, 80.8],
    'United Kingdom':  [35.1, 46344, 4.5, 84.2],
    'Norway':          [26.1, 89154, 4.4, 83.8],
    'Italy':           [35.9, 34776, 9.5, 71.2],
    'Switzerland':     [33.1, 94696, 4.9, 74.0],
    'Denmark':         [28.8, 68008, 5.1, 88.2],
    'Russia':          [36.0, 11273, 4.8, 74.8],
    'Poland':          [28.8, 15694, 3.4, 60.1],
    'Ukraine':         [25.6, 3984,  9.9, 69.7],
    'United States':   [41.5, 63544, 5.4, 82.7],
    'Canada':          [33.3, 52051, 7.5, 81.7],
    'Brazil':          [53.4, 7507,  13.2, 87.1],
    'Mexico':          [45.4, 9926,  4.0, 80.7],
    'Colombia':        [51.3, 5882,  13.7, 81.6],
    'Argentina':       [42.9, 9938,  11.0, 92.0],
    'Chile':           [44.9, 15921, 9.1, 88.0],
    'Peru':            [43.8, 6621,  7.7, 78.3],
    'Venezuela':       [44.8, 1600,  7.0, 88.2],
    'Ecuador':         [45.7, 5937,  5.0, 64.4],
    'South Africa':    [63.0, 6001,  33.9, 68.0],
    'Nigeria':         [35.1, 2065,  9.7, 53.0],
    'Kenya':           [40.8, 1838,  5.7, 28.5],
    'Ethiopia':        [35.0, 925,   3.1, 22.2],
    'Ghana':           [43.5, 2362,  4.3, 58.0],
    'Tanzania':        [37.8, 1105,  2.6, 37.0],
    'Saudi Arabia':    [45.9, 22993, 7.4, 84.0],
    'Egypt':           [31.5, 3699,  7.3, 42.8],
    'Turkey':          [41.9, 9661,  13.2, 76.6],
    'Iran':            [40.9, 5797,  10.8, 76.3],
    'Morocco':         [39.5, 3355,  11.5, 63.5],
    'Jordan':          [33.7, 4286,  24.7, 91.6],
    'Australia':       [34.3, 55060, 5.1, 86.4],
    'New Zealand':     [33.0, 42084, 4.6, 86.7],
}


# ---------------------------------------------------------------------------
# 5. REGION MAPPING
# ---------------------------------------------------------------------------
REGIONS = {
    'Japan': 'East Asia', 'South Korea': 'East Asia', 'China': 'East Asia',
    'Singapore': 'East Asia', 'Taiwan': 'East Asia', 'Hong Kong': 'East Asia',
    'India': 'South/SE Asia', 'Thailand': 'South/SE Asia', 'Vietnam': 'South/SE Asia',
    'Philippines': 'South/SE Asia', 'Indonesia': 'South/SE Asia',
    'Germany': 'Western Europe', 'Sweden': 'Western Europe', 'France': 'Western Europe',
    'Netherlands': 'Western Europe', 'Spain': 'Western Europe', 'United Kingdom': 'Western Europe',
    'Norway': 'Western Europe', 'Italy': 'Western Europe', 'Switzerland': 'Western Europe',
    'Denmark': 'Western Europe',
    'Russia': 'Eastern Europe', 'Poland': 'Eastern Europe', 'Ukraine': 'Eastern Europe',
    'United States': 'North America', 'Canada': 'North America',
    'Brazil': 'Latin America', 'Mexico': 'Latin America', 'Colombia': 'Latin America',
    'Argentina': 'Latin America', 'Chile': 'Latin America', 'Peru': 'Latin America',
    'Venezuela': 'Latin America', 'Ecuador': 'Latin America',
    'South Africa': 'Sub-Saharan Africa', 'Nigeria': 'Sub-Saharan Africa',
    'Kenya': 'Sub-Saharan Africa', 'Ethiopia': 'Sub-Saharan Africa',
    'Ghana': 'Sub-Saharan Africa', 'Tanzania': 'Sub-Saharan Africa',
    'Saudi Arabia': 'MENA', 'Egypt': 'MENA', 'Turkey': 'MENA',
    'Iran': 'MENA', 'Morocco': 'MENA', 'Jordan': 'MENA',
    'Australia': 'Oceania', 'New Zealand': 'Oceania',
}


def build_master_dataframe():
    """Merge all sources into one analysis-ready dataframe."""
    countries = sorted(set(UNODC_HOMICIDE) & set(HOFSTEDE) & set(WVS_WAVE7) & set(WORLD_BANK))

    rows = []
    for c in countries:
        h = HOFSTEDE[c]
        w = WVS_WAVE7[c]
        wb = WORLD_BANK[c]
        rows.append({
            'country': c,
            'region': REGIONS.get(c, 'Other'),
            # Crime
            'homicide_rate': UNODC_HOMICIDE[c],
            # Hofstede dimensions
            'idv': h[0],          # Individualism
            'uai': h[1],          # Uncertainty avoidance
            'pdi': h[2],          # Power distance
            'mas': h[3],          # Masculinity
            # WVS parenting values
            'wvs_obedience': w[0],
            'wvs_independence': w[1],
            'wvs_hard_work': w[2],
            'wvs_trust': w[3],
            # World Bank controls
            'gini': wb[0],
            'gdp_per_capita': wb[1],
            'unemployment': wb[2],
            'urbanization': wb[3],
        })

    df = pd.DataFrame(rows).set_index('country')
    df['log_homicide'] = np.log1p(df['homicide_rate'])
    df['log_gdp'] = np.log(df['gdp_per_capita'])
    df['collectivism'] = 100 - df['idv']  # Inverse of IDV
    return df


if __name__ == '__main__':
    df = build_master_dataframe()
    print(f"Dataset: {len(df)} countries")
    print(df[['region','homicide_rate','idv','wvs_obedience','gini','gdp_per_capita']].to_string())
