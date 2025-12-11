# 🌠 Météorite Type Predictor

> Système intelligent de prédiction du type de météorites basé sur des règles d'association

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table des Matières

- [Description](#-description)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API Reference](#-api-reference)
- [Exemples](#-exemples)
- [Dataset](#-dataset)
- [Algorithme](#-algorithme)
- [Résultats](#-résultats)
- [Auteurs](#-auteurs)

---

## 📖 Description

Ce projet implémente un **système de prédiction de types de météorites** utilisant l'algorithme **Apriori** pour générer des règles d'association. À partir de critères fournis par l'utilisateur (période, masse, localisation), le modèle prédit le type de météorite le plus probable.

### 🎯 Objectif

Permettre aux chercheurs et passionnés de météorites de :
- **Identifier** le type probable d'une météorite en fonction de ses caractéristiques
- **Explorer** les corrélations entre localisation, période et types de météorites
- **Découvrir** des exemples de météorites similaires dans la base de données

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|----------------|-------------|
| 🔮 **Prédiction de type** | Prédit le type de météorite (IRON, LL, H5, etc.) |
| 📊 **Probabilité** | Indique le niveau de confiance de la prédiction |
| 🌍 **Géolocalisation** | Retourne les pays où ce type de météorite a été trouvé |
| 📅 **Prédiction temporelle** | Prédit la période si non spécifiée |
| ⚖️ **Prédiction de masse** | Prédit la catégorie de masse si non spécifiée |
| 📝 **Exemples** | Fournit des noms de météorites correspondantes |

---

## 🏗️ Architecture

```
MINI_PROJET_ML/
├── Backend/
│   ├── app.py              # API Flask
│   ├── generate_rules.py   # Génération des règles Apriori
│   ├── rules.pkl           # Règles d'association sauvegardées
│   └── test.py             # Script de test
├── data/
│   ├── meteorites_final.csv
│   └── meteorites_final_rebalanced.csv
├── training/
│   └── REGLES.py           # Fonctions de traitement des règles
└── README.md
```

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes

1. **Cloner le repository**
```bash
git clone https://github.com/Amina212004/MINI_PROJET_ML.git
cd MINI_PROJET_ML
```

2. **Installer les dépendances**
```bash
pip install flask pandas numpy mlxtend folium
```

3. **Générer les règles d'association**
```bash
cd Backend
python generate_rules.py
```

4. **Lancer le serveur**
```bash
python app.py
```

Le serveur sera accessible sur `http://127.0.0.1:5000`

---

## 📡 Utilisation

### Via Python (requests)

```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "years": [[1990, 2000]],
    "mass": ["10-100g"],
    "continents": ["Africa"]
}

response = requests.post(url, json=data)
print(response.json())
```

### Via cURL

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"years": [[1990, 2000]], "mass": ["10-100g"], "continents": ["Africa"]}'
```

### Via Postman

1. Méthode : `POST`
2. URL : `http://127.0.0.1:5000/predict`
3. Body (raw JSON) :
```json
{"years": [[1990, 2000]], "mass": ["10-100g"], "continents": ["Africa"]}
```

---

## 📚 API Reference

### POST /predict

Prédit le type de météorite basé sur les critères fournis.

#### Paramètres d'entrée

| Paramètre | Type | Description | Valeurs possibles |
|-----------|------|-------------|-------------------|
| `years` | `array` | Plage d'années | `[[1900, 2000]]` ou `null` |
| `mass` | `array` | Catégorie de masse | `["<1g", "1-10g", "10-100g", "100-1kg", "1-10kg", ">10kg"]` ou `null` |
| `continents` | `array` | Continent | `["Africa", "Europe", "Asia", "North America", "South America", "Antarctica", "Oceania"]` ou `null` |

#### Réponse

| Champ | Type | Description |
|-------|------|-------------|
| `top_type` | `string` | Type de météorite prédit |
| `probability` | `float` | Probabilité de la prédiction (0-1) |
| `countries` | `array` | Liste des pays correspondants |
| `names` | `array` | Noms de météorites exemples |
| `sample_years` | `array` | Années des exemples |
| `predicted_years` | `string` | Période prédite (si non fournie) |
| `predicted_mass` | `array` | Masse prédite (si non fournie) |
| `predicted_continent` | `array` | Continent prédit (si non fourni) |

> ⚠️ Les champs `predicted_*` n'apparaissent que si l'utilisateur n'a pas fourni la valeur correspondante.

---

## 🧪 Exemples

### Exemple 1 : Recherche par continent
**Input :**
```jsons
{"years": null, "mass": null, "continents": ["Africa"]}
```

**Output :**
```json
{
    "top_type": "IRON",
    "probability": 0.431,
    "countries": ["Egypt", "Morocco", "Mali", "Nigeria", "South Africa", ...],
    "predicted_mass": [">10kg"],
    "predicted_years": "20th Century"
}
```

### Exemple 2 : Tous les critères spécifiés
**Input :**
```json
{"years": [[1950, 1980]], "mass": ["1-10kg"], "continents": ["North America"]}
```

**Output :**
```json
{
    "top_type": "IRON",
    "probability": 0.95,
    "countries": ["United States of America", "Mexico"],
    "names": ["Hope", "Harlowton", "Sombrerete", ...]
}
```
> Note : Pas de `predicted_*` car tous les critères sont fournis.

### Exemple 3 : Petites météorites
**Input :**
```json
{"years": null, "mass": ["<1g"], "continents": null}
```

**Output :**
```json
{
    "top_type": "E3",
    "probability": 0.381,
    "countries": ["Antarctica"],
    "predicted_continent": ["Antarctica"],
    "predicted_years": "20th Century"
}
```

---

## 📊 Dataset

Le dataset contient **31 905 météorites** avec les caractéristiques suivantes :

| Colonne | Description | Valeurs |
|---------|-------------|---------|
| `name` | Nom de la météorite | Ex: "Yamato 793153" |
| `recclass_clean` | Type de météorite | 12 types (IRON, LL, H5, L6, etc.) |
| `mass_bin` | Catégorie de masse | 6 catégories |
| `year_period` | Période de découverte | 4 périodes |
| `continent` | Continent de découverte | 7 continents |
| `country` | Pays de découverte | ~100 pays |

### Distribution des types

```
L6              3827
H5              3827
OTHER           3192
H4              3190
L5              3188
CARBONACEOUS    2554
ACHONDRITE      2554
H6              2551
L4              1915
IRON            1915
L3              1596
H3              1596
```

---

## 🧠 Algorithme

### 1. Génération des règles (Apriori)

```python
# Paramètres optimisés
min_support = 0.0005      # Capture les cas rares
min_lift = 1.0            # Corrélation positive
max_rules_per_type = 50   # Équilibrage des types
```

### 2. Processus de prédiction

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRÉE UTILISATEUR                        │
│         (années, masse, continent - optionnels)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  FILTRAGE DES RÈGLES                         │
│    - Mode strict (tous les critères correspondent)           │
│    - Mode non-strict (au moins un critère correspond)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SCORING DES TYPES                         │
│         Score = Confidence × Lift × Bonus                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PRÉDICTION + COMPLÉTION                      │
│    - Type le plus probable                                   │
│    - Prédiction des critères manquants                       │
│    - Recherche d'exemples dans la base                       │
└─────────────────────────────────────────────────────────────┘
```

### 3. Métriques utilisées

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Support** | P(A ∩ B) | Fréquence de la règle |
| **Confidence** | P(B\|A) | Probabilité conditionnelle |
| **Lift** | P(B\|A) / P(B) | Corrélation (>1 = positive) |

---

## 📈 Résultats

### Performance du modèle

| Métrique | Valeur |
|----------|--------|
| Nombre de règles | 3103 |
| Confidence moyenne | 28.8% |
| Lift moyen | 3.419 |
| Types couverts | 12/12 (100%) |

### Tests de validation

| Scénario | Probabilité moyenne |
|----------|---------------------|
| Critères complets | 95% |
| 2 critères | 60-80% |
| 1 critère | 35-55% |


---

## 🛠️ Technologies

- **Python 3.8+** - Langage principal
- **Flask** - API REST
- **Pandas** - Manipulation de données
- **MLxtend** - Algorithme Apriori
- **Folium** - Visualisation cartographique
- **NumPy** - Calculs numériques



## 📄 License

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- Dataset original : [NASA Meteorite Landings](https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh)
- Inspiration : Cours de Machine Learning - IASD Semestre 1

---

<p align="center">
  <i>Fait avec ❤️ pour le Mini Projet ML - IASD 2024/2025</i>
</p>


