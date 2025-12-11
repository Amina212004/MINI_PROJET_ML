# generate_rules_complete.py
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import pickle
import os

# -----------------------------
# Charger dataset
# -----------------------------
dataset_path = os.path.join('..', 'data', 'meteorites_final_rebalanced.csv')
df = pd.read_csv(dataset_path)

print(f"Dataset chargé : {len(df)} lignes")

# -----------------------------
# Colonnes à utiliser pour apriori
# -----------------------------
columns = ["year_period", "mass_bin", "continent", "recclass_clean"]
df_small = df[columns].dropna()  # Supprimer les lignes avec valeurs manquantes

print(f"Données après nettoyage : {len(df_small)} lignes")
print(f"Valeurs uniques par colonne :")
for col in columns:
    print(f"  - {col}: {df_small[col].nunique()} valeurs")

# -----------------------------
# Encodage one-hot
# -----------------------------
df_encoded = pd.get_dummies(df_small).astype(bool)

print(f"Colonnes encodées : {len(df_encoded.columns)}")

# -----------------------------
# AMÉLIORATION 1: Support encore plus bas pour capturer les petites masses
# -----------------------------
# Support minimal TRÈS BAS (0.0005 = 0.05% = ~16 météorites minimum)
# Cela permet de capturer des règles pour les catégories rares comme <1g
frequent_itemsets = apriori(df_encoded, min_support=0.0005, use_colnames=True)

print(f"Itemsets fréquents trouvés : {len(frequent_itemsets)}")

# -----------------------------
# AMÉLIORATION 2: Générer règles avec lift plus bas pour plus de diversité
# -----------------------------
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filtrer les règles de qualité (lift > 1 signifie corrélation positive)
rules = rules[rules['lift'] > 1.0]

print(f"Règles générées (avant filtrage) : {len(rules)}")

# -----------------------------
# Filtrer pour garder les règles qui prédisent un type
# -----------------------------
def has_type_in_consequents(row):
    return any('recclass_clean_' in str(item) for item in row['consequents'])

type_rules = rules[rules.apply(has_type_in_consequents, axis=1)]
other_rules = rules[~rules.apply(has_type_in_consequents, axis=1)]

print(f"Règles prédisant un type : {len(type_rules)}")
print(f"Autres règles : {len(other_rules)}")

# -----------------------------
# AMÉLIORATION 3: Équilibrer les règles par type pour plus de diversité
# -----------------------------
# Compter les règles par type
type_to_rules = {}
for idx, row in type_rules.iterrows():
    for item in row['consequents']:
        if 'recclass_clean_' in str(item):
            type_name = str(item).replace('recclass_clean_', '')
            if type_name not in type_to_rules:
                type_to_rules[type_name] = []
            type_to_rules[type_name].append(idx)

print(f"\nTypes avec règles : {len(type_to_rules)}")

# Garder un nombre équilibré de règles par type (max 50 par type)
# mais garder TOUTES les règles pour les types rares
MAX_RULES_PER_TYPE = 50
balanced_indices = []
for type_name, indices in type_to_rules.items():
    if len(indices) <= MAX_RULES_PER_TYPE:
        # Type rare → garder toutes ses règles
        balanced_indices.extend(indices)
    else:
        # Type fréquent → garder les meilleures règles (par confidence)
        type_df = type_rules.loc[indices].nlargest(MAX_RULES_PER_TYPE, 'confidence')
        balanced_indices.extend(type_df.index.tolist())

balanced_type_rules = type_rules.loc[list(set(balanced_indices))]
print(f"Règles de type après équilibrage : {len(balanced_type_rules)}")

# Garder toutes les règles mais PRIORISER celles qui prédisent un type (en premier)
rules = pd.concat([balanced_type_rules, other_rules])

# -----------------------------
# Ajouter colonnes utiles pour filtrage
# -----------------------------
# Convertir antecedents et consequents en set de chaînes
rules['antecedents'] = rules['antecedents'].apply(lambda x: set(x))
rules['consequents'] = rules['consequents'].apply(lambda x: set(x))

# -----------------------------
# Statistiques finales
# -----------------------------
print("\n=== STATISTIQUES DES RÈGLES ===")
print(f"Total règles : {len(rules)}")
print(f"Confidence moyenne : {rules['confidence'].mean():.3f}")
print(f"Lift moyen : {rules['lift'].mean():.3f}")
print(f"Support moyen : {rules['support'].mean():.4f}")

# Top types les plus prédits
type_counts = {}
for _, row in balanced_type_rules.iterrows():
    for item in row['consequents']:
        if 'recclass_clean_' in str(item):
            type_name = str(item).replace('recclass_clean_', '')
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

print("\nDistribution des règles par type (équilibré) :")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  - {t}: {c} règles")

# -----------------------------
# AMÉLIORATION 4: Vérifier les règles pour les petites masses
# -----------------------------
small_mass_rules = balanced_type_rules[
    balanced_type_rules['antecedents'].apply(
        lambda x: any('<1g' in str(item) or '1-10g' in str(item) for item in x)
    )
]
print(f"\nRègles pour petites masses (<1g et 1-10g) : {len(small_mass_rules)}")

# -----------------------------
# Sauvegarder règles
# -----------------------------
rules_path = os.path.join(os.path.dirname(__file__), 'rules.pkl')
with open(rules_path, 'wb') as f:
    pickle.dump(rules, f)

print(f"\n✅ Fichier rules.pkl créé avec succès ! ({len(rules)} règles)")
