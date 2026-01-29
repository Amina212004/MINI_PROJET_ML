# mlops/02_train.py
import pandas as pd
import wandb
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Init W&B (La config viendra du Sweep)
run = wandb.init(project="meteorite-mlops", job_type="train")
config = wandb.config

print("📥 Récupération des données...")
artifact = run.use_artifact('meriemkhedir/meteorite-mlops/meteorite_dataset:latest')
data_dir = artifact.download()

# Chargement
train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")

# Séparation X et y
target_col = 'recclass_clean'
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# Entraînement avec Hyperparamètres du Sweep
print(f"🌲 Training RF: n_estimators={config.n_estimators}, max_depth={config.max_depth}")

model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    min_samples_split=config.min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

# Évaluation
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

print(f"📈 Résultats -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

# Log des métriques pour W&B
wandb.log({"accuracy": acc, "f1_score": f1})

# Sauvegarde du modèle
joblib.dump(model, "model.pkl")

# Versioning du modèle
model_artifact = wandb.Artifact(
    name=f"rf_model_{run.id}", 
    type="model",
    metadata=dict(config)
)
model_artifact.add_file("model.pkl")
run.log_artifact(model_artifact)

run.finish()