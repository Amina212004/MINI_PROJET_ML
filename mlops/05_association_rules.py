import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import os

# --- CONFIGURATION ---
PROJECT_NAME = "MINI_PROJET_ML-mlops"
ENTITY = "meriemkhedir"

run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    name="Association_Rules_Mining",
    job_type="analysis",
    tags=["rules", "apriori", "final"]
)

print("📥 Chargement des données...")
# On reprend ta logique de chargement robuste
target_names = ["meteorites_final_rebalanced.csv", "meteorite_data_cleaned.csv"]
df = None

for name in target_names:
    if os.path.exists(name):
        print(f"✅ Fichier trouvé : {name}")
        df = pd.read_csv(name)
        break

if df is None:
    # Mode secours si fichier pas trouvé (pour la démo)
    print("⚠️ Génération de données simulées pour l'Association Rules...")
    data = {
        'mass': np.random.exponential(500, 1000),
        'year': np.random.randint(1900, 2020, 1000),
        'hazardous': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)

# --- 1. PRÉPARATION (DISCRÉTISATION) ---
# Pour faire des règles, il faut transformer les chiffres en "Mots"
print("🔄 Transformation des données en catégories...")

data_rules = pd.DataFrame()

# A. On découpe la MASSE en catégories
# Exemple : Petit (<100g), Moyen, Grand (>1000g)
data_rules['Mass_Level'] = pd.cut(df['mass'], 
                                  bins=[-1, 100, 1000, 10000000], 
                                  labels=['Mass_Small', 'Mass_Medium', 'Mass_Large'])

# B. On transforme HAZARDOUS en texte
if 'hazardous' in df.columns:
    data_rules['Hazard'] = df['hazardous'].apply(lambda x: 'Hazardous' if x == 1 else 'Safe')
elif 'class' in df.columns:
     data_rules['Hazard'] = df['class'].apply(lambda x: 'Hazardous' if x == 1 else 'Safe')

# C. On découpe l'ANNÉE (Siècles)
if 'year' in df.columns:
    data_rules['Era'] = pd.cut(df['year'], 
                               bins=[0, 1900, 2000, 2100], 
                               labels=['Ancient', '20th_Century', 'Modern'])

print("Aperçu des catégories :")
print(data_rules.head())

# --- 2. ONE-HOT ENCODING ---
# Apriori a besoin de 0 et 1 partout
df_encoded = pd.get_dummies(data_rules)
# Conversion en booléens (True/False) pour mlxtend
df_encoded = df_encoded.astype(bool)

# --- 3. ALGORITHME APRIORI ---
print("🕵️ Recherche des items fréquents (Apriori)...")
# min_support = 0.05 signifie qu'on garde les règles qui apparaissent dans au moins 5% des cas
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

print(f"✅ Trouvé {len(frequent_itemsets)} groupes fréquents.")

# --- 4. GÉNÉRATION DES RÈGLES ---
print("📜 Génération des règles d'association...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# On trie par "Confidence" (Fiabilité de la règle)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# Sélection des colonnes utiles pour l'affichage
cols_to_keep = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
display_rules = rules[cols_to_keep].head(20).copy()

# On convertit les sets en string pour l'affichage WandB
display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))

print("Top 5 Règles :")
print(display_rules.head())

# Envoi du tableau à WandB
wandb.log({"Association Rules Table": wandb.Table(dataframe=display_rules)})

# --- 5. VISUALISATIONS (Comme tes amies !) ---

# Graphe 1 : Scatter Plot (Support vs Confidence)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="support", y="confidence", size="lift", hue="lift", data=rules, palette="viridis", sizes=(20, 200))
plt.title("Association Rules: Support vs Confidence")
plt.xlabel("Support (Fréquence)")
plt.ylabel("Confidence (Fiabilité)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
wandb.log({"Rules Scatter Plot": wandb.Image(plt)})
plt.close()

# Graphe 2 : Distribution du Lift
plt.figure(figsize=(8, 5))
sns.histplot(rules['lift'], bins=10, kde=True, color='purple')
plt.title("Distribution du Lift (Force des règles)")
plt.xlabel("Lift")
wandb.log({"Lift Distribution": wandb.Image(plt)})
plt.close()

print("✅ Terminé ! Va voir tes règles sur WandB.")
run.finish()