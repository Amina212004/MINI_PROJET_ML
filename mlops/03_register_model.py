import wandb

# --- Configuration ---
# On reprend les infos de tes logs précédents
ENTITY = "meriemkhedir"            # Ton nom d'utilisateur ou organisation
PROJECT = "MINI_PROJET_ML-mlops"   # Ton projet
SWEEP_ID = "821bmi5k"              # L'ID du sweep que tu viens de finir
REGISTRY_NAME = "model-registry-meteorites" # Le nom du "dossier" où on rangera le modèle

api = wandb.Api()

print(f"🕵️‍♀️ Connexion au sweep {SWEEP_ID}...")
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

# 1. Trouver le meilleur run
# WandB va chercher celui qui a maximisé la métrique définie (f1_score ou accuracy)
best_run = sweep.best_run()
print(f"🏆 Le meilleur run est : {best_run.name}")
print(f"📊 Ses performances : {best_run.summary.get('accuracy')}")

# 2. Récupérer l'artefact (le fichier modèle) de ce run
# On cherche un artefact de type 'model'
artifacts = best_run.logged_artifacts()
model_artifact = None

for artifact in artifacts:
    if artifact.type == "model":
        model_artifact = artifact
        break

if model_artifact:
    print(f"📦 Modèle trouvé : {model_artifact.name}")
    
    # 3. Enregistrer dans le Model Registry avec le tag 'production'
    # Cela permet à l'étape suivante (monitoring) de savoir quel modèle charger
    print(f"🔗 Enregistrement dans le registre '{REGISTRY_NAME}'...")
    
    # On lie l'artefact au registre et on lui donne l'alias "production"
    model_artifact.link(
        target_path=f"{ENTITY}/{PROJECT}/{REGISTRY_NAME}", 
        aliases=["production", "latest"]
    )
    
    print("✅ Succès ! Le meilleur modèle est maintenant tagué 'production'.")
else:
    print("❌ Erreur : Aucun modèle trouvé dans les artefacts du meilleur run.")