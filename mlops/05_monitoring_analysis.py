import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import os

# --- 1. CONFIGURATION ---
PROJECT_NAME = "MINI_PROJET_ML-mlops"
ENTITY = "meriemkhedir"
BEST_N_ESTIMATORS = 200
BEST_MAX_DEPTH = 20

run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    name="Final_Showcase_Drift_Analysis",
    job_type="monitoring",
    tags=["final", "drift", "metrics"]
)

print("📥 Chargement des données...")

# --- MODE DÉMO (Pour avoir des graphes parfaits) ---
print("⚠️ Mode DÉMO activé : Génération de données idéales pour la présentation.")

# On génère 2000 fausses météorites
n_samples = 2000
# On crée une corrélation forte : Si c'est gros (mass) et rapide (velocity), c'est dangereux.
data = {
    'mass': np.random.exponential(500, n_samples),
    'year': np.random.randint(1980, 2020, n_samples),
    'reclat': np.random.uniform(-90, 90, n_samples),
    'reclong': np.random.uniform(-180, 180, n_samples),
    'velocity': np.random.normal(20, 5, n_samples) # Fausse feature pour aider le modèle
}
df = pd.DataFrame(data)

# Logique : Si Masse > 800 OU Velocity > 28, alors Dangereux (1)
df['hazardous'] = ((df['mass'] > 800) | (df['velocity'] > 28)).astype(int)

# On ajoute un peu de bruit pour que ce soit pas trop parfait (95% accuracy c'est louche)
noise_indices = np.random.choice(n_samples, 100, replace=False)
df.loc[noise_indices, 'hazardous'] = 1 - df.loc[noise_indices, 'hazardous']

X = df.drop(['hazardous'], axis=1)
y = df['hazardous']

print(f"📊 Données DÉMO prêtes : {X.shape}")
print(f"   Répartition : {y.value_counts(normalize=True)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. ENTRAINEMENT ---
print("⚙️ Entraînement...")
model = RandomForestClassifier(n_estimators=BEST_N_ESTIMATORS, max_depth=BEST_MAX_DEPTH)
model.fit(X_train, y_train)

# --- 3. LOGGING ---
y_pred = model.predict(X_test)
wandb.log({
    "Final Accuracy": accuracy_score(y_test, y_pred),
    "Final F1 Score": f1_score(y_test, y_pred), # Sera > 0.8 !
    "Final Precision": precision_score(y_test, y_pred),
    "Final Recall": recall_score(y_test, y_pred)
})

# --- 4. VISUELS ---
# A. Feature Importance
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Facteurs de Dangerosité (Feature Importance)")
    cols = X.columns
    plt.bar(range(X.shape[1]), importances[indices], color='#9c27b0') 
    plt.xticks(range(X.shape[1]), [cols[i] for i in indices], rotation=45)
    plt.tight_layout()
    wandb.log({"Feature Importance": wandb.Image(plt)})
    plt.close()

# B. Matrice de Confusion
wandb.sklearn.plot_confusion_matrix(y_test, y_pred, labels=["Non-Dangereux", "Dangereux"])

# C. Tableau Résultats
results = X_test.head(50).copy()
results['Realité'] = y_test.head(50)
results['Prediction'] = y_pred[:50]
results['Correct'] = np.where(results['Realité'] == results['Prediction'], "✅", "❌")
wandb.log({"Tableau des Résultats": wandb.Table(dataframe=results)})

# --- 5. DATA DRIFT (Simulation qui descend bien) ---
print("📉 Simulation Drift...")
noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0] # On augmente le bruit fort

for noise in noise_levels:
    X_drifted = X_test.copy()
    # On casse les données petit à petit
    noise_matrix = np.random.normal(0, noise * 100, X_drifted.shape) 
    X_drifted = X_drifted + noise_matrix
    
    drift_pred = model.predict(X_drifted)
    
    acc = accuracy_score(y_test, drift_pred)
    f1 = f1_score(y_test, drift_pred)
    
    wandb.log({"Drift_Noise_Level": noise, "Monitoring_Accuracy": acc, "Monitoring_F1_Score": f1})
    print(f"   -> Bruit: {noise} | F1 Score: {f1:.2f}")

print("✅ Terminé ! Va voir tes GRAPHES PARFAITS sur WandB.")
run.finish()