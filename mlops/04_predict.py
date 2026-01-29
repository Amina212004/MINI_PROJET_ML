import wandb
import joblib
import os

# --- Configuration ---
# Remarque : Ici, on ne met PAS l'ID du run. On demande juste "production".
ENTITY = "meriemkhedir"
PROJECT = "MINI_PROJET_ML-mlops"
REGISTRY_NAME = "model-registry-meteorites"

print("🤖 Démarrage du système de prédiction...")

# On initialise un run "job_type=inference" (juste pour l'utilisation, pas l'entraînement)
run = wandb.init(project=PROJECT, job_type="inference", entity=ENTITY)

# MAGIE : On demande le modèle par son ALIAS "production"
# Peu importe quel run a gagné avant, on récupère toujours le meilleur ici.
artifact_path = f"{ENTITY}/{PROJECT}/{REGISTRY_NAME}:production"
print(f"📥 Téléchargement du modèle depuis : {artifact_path}")

try:
    artifact = run.use_artifact(artifact_path, type='model')
    model_dir = artifact.download()
    
    # Chargement du fichier .pkl
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    
    print("\n✅ Modèle chargé avec succès !")
    print(f"🧠 Type de modèle : {type(model)}")
    print("🚀 Le système est prêt à faire des prédictions (Inférence).")
    
except Exception as e:
    print(f"\n❌ Erreur lors du chargement : {e}")

run.finish()