# meteorite_functions.py

import pandas as pd
import numpy as np
import folium
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------
# Filtrer les tautologies géographiques
# -----------------------------
def is_geographic_tautology(row):
    """
    Détecte les règles tautologiques comme {continent_X} → {country_X}
    ou {country_X} → {continent_X}
    """
    antecedents = row['antecedents']
    consequents = row['consequents']
    
    has_continent_ant = any('continent_' in str(item) for item in antecedents)
    has_country_cons = any('country_' in str(item) for item in consequents)
    has_country_ant = any('country_' in str(item) for item in antecedents)
    has_continent_cons = any('continent_' in str(item) for item in consequents)
    
    return (has_continent_ant and has_country_cons) or (has_country_ant and has_continent_cons)

# -----------------------------
# Vérifier si règle prédit un type
# -----------------------------
def is_type_prediction_rule(row):
    """
    Vérifie si la règle prédit un TYPE de météorite (recclass_clean)
    """
    return any('recclass_clean_' in str(item) for item in row['consequents'])

# -----------------------------
# Filtrer règles selon critères
# -----------------------------
def filter_rules(rules, years=None, mass_bins=None, continents=None, YEAR_TO_PERIOD=None, MASS_BIN_RANGES=None):
    """
    Filtre les règles d'association selon les critères utilisateur.
    Élimine les tautologies géographiques et priorise les règles de TYPE.
    """
    user_criteria = set()

    # Traitement des années
    if years:
        for y in years:
            if isinstance(y, (list, tuple)) and len(y) == 2:
                relevant_years = [yr for yr in YEAR_TO_PERIOD.keys() if y[0] <= yr <= y[1]]
                user_criteria.update(f'year_period_{YEAR_TO_PERIOD[yr]}' for yr in relevant_years)
            else:
                if y in YEAR_TO_PERIOD:
                    user_criteria.add(f'year_period_{YEAR_TO_PERIOD[y]}')

    # Traitement des masses
    if mass_bins:
        for m in mass_bins:
            if isinstance(m, (list, tuple)) and len(m) == 2:
                for mb, (low, high) in MASS_BIN_RANGES.items():
                    if m[0] <= low and high <= m[1]:
                        user_criteria.add(f'mass_bin_{mb}')
            else:
                user_criteria.add(f'mass_bin_{m}')

    # Traitement des continents
    if continents:
        user_criteria.update(f'continent_{c}' for c in continents)

    if not user_criteria:
        filtered = rules[~rules.apply(is_geographic_tautology, axis=1)]
        filtered = filtered[filtered.apply(is_type_prediction_rule, axis=1)]
        return filtered

    filtered = rules[rules['antecedents'].apply(lambda x: user_criteria.issubset(x))]
    filtered = filtered[~filtered.apply(is_geographic_tautology, axis=1)]
    type_rules = filtered[filtered.apply(is_type_prediction_rule, axis=1)]
    
    if not type_rules.empty:
        return type_rules
    else:
        return filtered

# -----------------------------
# Type le plus probable
# -----------------------------
def get_most_probable_type(filtered_rules, df_subset):
    type_scores = {}
    
    if not filtered_rules.empty:
        for _, row in filtered_rules.iterrows():
            for item in row['consequents']:
                if 'recclass_clean_' in item:
                    type_name = item.replace('recclass_clean_', '')
                    type_scores[type_name] = type_scores.get(type_name, 0) + row['confidence'] * row['support']

    if type_scores:
        top_type = max(type_scores, key=type_scores.get)
        prob = type_scores[top_type] / sum(type_scores.values())
    else:
        if not df_subset.empty and 'recclass_clean' in df_subset.columns:
            top_type = df_subset['recclass_clean'].value_counts().idxmax()
            prob = df_subset['recclass_clean'].value_counts(normalize=True).max()
        else:
            top_type = "Unknown"
            prob = 0.0

    return top_type, prob, type_scores

# -----------------------------
# Prédire valeurs manquantes
# -----------------------------
def predict_missing_criteria(df, top_type, user_years=None, user_mass=None, user_continents=None):
    df_type = df[df['recclass_clean'] == top_type].copy()
    
    if user_years:
        years_flat = []
        for y in user_years:
            if isinstance(y, (list, tuple)) and len(y) == 2:
                years_flat.extend(range(y[0], y[1] + 1))
            else:
                years_flat.append(y)
        df_type = df_type[df_type['year'].isin(years_flat)]
    
    if user_mass:
        mass_mask = pd.Series(False, index=df_type.index)
        for m in user_mass:
            if isinstance(m, (list, tuple)) and len(m) == 2:
                mass_mask |= df_type['mass_cleaned'].between(m[0], m[1])
            else:
                mass_mask |= (df_type['mass_bin'] == m)
        df_type = df_type[mass_mask]
    
    if user_continents:
        df_type = df_type[df_type['continent'].isin(user_continents)]
    
    year_pred = user_years if user_years else (df_type['year_period'].mode()[0] if not df_type.empty else None)
    mass_pred = user_mass if user_mass else (df_type['mass_bin'].mode()[0] if not df_type.empty else None)
    continent_pred = user_continents if user_continents else (df_type['continent'].mode()[0] if not df_type.empty else None)

    return year_pred, mass_pred, continent_pred

# -----------------------------
# Infos selon critères
# -----------------------------
def get_type_info(df, top_type, user_years=None, user_mass=None, user_continents=None,
                  pred_year=None, pred_mass=None, pred_continent=None):
    df_type = df[df['recclass_clean'] == top_type].copy()
    final_years = user_years
    final_mass = user_mass
    final_continents = user_continents if user_continents else ([pred_continent] if pred_continent else None)

    if final_years:
        years_flat = []
        for y in final_years:
            if isinstance(y, (list, tuple)) and len(y) == 2:
                years_flat.extend(range(y[0], y[1] + 1))
            else:
                years_flat.append(y)
        df_type = df_type[df_type['year'].isin(years_flat)]

    if final_mass:
        mass_mask = pd.Series(False, index=df_type.index)
        for m in final_mass:
            if isinstance(m, (list, tuple)) and len(m) == 2:
                mass_mask |= df_type['mass_cleaned'].between(m[0], m[1])
            else:
                mass_mask |= (df_type['mass_bin'] == m)
        df_type = df_type[mass_mask]

    if final_continents:
        df_type = df_type[df_type['continent'].isin(final_continents)]

    names = df_type['name'].tolist()
    countries = df_type['country'].dropna().unique().tolist()
    sample_years = df_type['year'].dropna().tolist()
    mass_bin = df_type['mass_bin'].mode()[0] if not df_type.empty else None

    return names, countries, sample_years, mass_bin, df_type

# -----------------------------
# Statistiques de qualité des règles
# -----------------------------
def get_rules_statistics(filtered_rules):
    if filtered_rules.empty:
        return {
            'total': 0,
            'type_rules': 0,
            'geo_rules': 0,
            'mean_confidence': 0,
            'mean_lift': 0
        }
    
    type_rules = filtered_rules[filtered_rules.apply(is_type_prediction_rule, axis=1)]
    
    return {
        'total': len(filtered_rules),
        'type_rules': len(type_rules),
        'geo_rules': len(filtered_rules) - len(type_rules),
        'mean_confidence': filtered_rules['confidence'].mean(),
        'mean_lift': filtered_rules['lift'].mean()
    }

# -----------------------------
# Traitement d'une sélection utilisateur
# -----------------------------
def process_user_selection(sel, rules, df):
    filtered_rules = filter_rules(rules, sel.get('years'), sel.get('mass'), sel.get('continents'))
    top_type, prob, scores = get_most_probable_type(filtered_rules, df)
    year_pred, mass_pred, cont_pred = predict_missing_criteria(df, top_type, sel.get('years'), sel.get('mass'), sel.get('continents'))
    names, countries, sample_years, mass_bin, df_points = get_type_info(df, top_type, sel.get('years'), sel.get('mass'), sel.get('continents'), year_pred, mass_pred, cont_pred)
    stats = get_rules_statistics(filtered_rules)
    
    return {
        'selection': sel,
        'filtered_rules': filtered_rules,
        'top_type': top_type,
        'probability': prob,
        'type_scores': scores,
        'year_pred': year_pred,
        'mass_pred': mass_pred,
        'continent_pred': cont_pred,
        'names': names,
        'countries': countries,
        'sample_years': sample_years,
        'mass_bin': mass_bin,
        'df_points': df_points,
        'stats': stats
    }

# -----------------------------
# Carte interactive
# -----------------------------
def plot_examples_on_map(examples_info, colors=None, map_file='map.html'):
    if colors is None:
        colors = ['hotpink', 'purple', 'orange', 'blue', 'green', 'darkred', 'yellow', 'red']
    
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    for i, info in enumerate(examples_info):
        df_points = info['df_points']
        color = colors[i % len(colors)]
        
        for _, row in df_points.iterrows():
            if pd.notna(row.get('reclat')) and pd.notna(row.get('reclong')):
                folium.CircleMarker(
                    location=[row['reclat'], row['reclong']],
                    radius=4,
                    popup=f"<b>{row['name']}</b><br>{row.get('mass_cleaned', 'N/A')} g<br>{row.get('country', 'N/A')}",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)
    
    m.save(map_file)
    return m

# -----------------------------
# Évaluation qualité des règles
# -----------------------------
def evaluate_rules_quality(rules):
    eval_df = rules[~rules.apply(is_geographic_tautology, axis=1)].copy()
    conditions = [
        (eval_df['support'] >= 0.01) & (eval_df['confidence'] >= 0.7) & (eval_df['lift'] >= 1.2),
        (eval_df['support'] >= 0.005) & (eval_df['confidence'] >= 0.5) & (eval_df['lift'] >= 1.0),
        (eval_df['support'] < 0.005) | (eval_df['confidence'] < 0.5) | (eval_df['lift'] < 1.0)
    ]
    labels = ['Forte', 'Moyenne', 'Faible']
    eval_df['quality'] = np.select(conditions, labels, default='Faible')
    return eval_df
