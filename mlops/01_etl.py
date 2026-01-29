# mlops/01_etl.py
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

# Initialisation W&B
run = wandb.init(project="meteorite-mlops", job_type="etl", name="data_prep")

print("📥 Chargement de meteorites_final_rebalanced.csv...")
# On charge ton dataset
df = pd.read_csv('../data/meteorites_final_rebalanced.csv')

# Sélection des features comme dans ton notebook
# On garde les colonnes utiles pour la prédiction
columns_to_keep = ["year_period", "mass_bin", "continent", "recclass_clean"]
df_clean = df[columns_to_keep].dropna()

# Séparation Target (y) et Features (X)
y = df_clean['recclass_clean']
X = df_clean.drop('recclass_clean', axis=1)

# Encodage (Car Random Forest ne gère pas le texte direct)
# On utilise get_dummies comme tu as fait
X_encoded = pd.get_dummies(X)

print(f"📊 Données prêtes : {X_encoded.shape[0]} lignes, {X_encoded.shape[1]} colonnes")

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Sauvegarde locale temporaire
X_train.join(y_train).to_csv("train.csv", index=False)
X_test.join(y_test).to_csv("test.csv", index=False)

# Création de l'Artifact W&B (Versioning des données)
artifact = wandb.Artifact(
    name="meteorite_dataset", 
    type="dataset",
    description="Dataset Rebalanced Encoded"
)
artifact.add_file("train.csv")
artifact.add_file("test.csv")

# Log de l'artifact
run.log_artifact(artifact)
print("✅ Données versionnées sur W&B !")
run.finish()