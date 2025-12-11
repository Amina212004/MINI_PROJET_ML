import pandas as pd
import numpy as np
import os

# Obtenir le répertoire du script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'meteorites_final.csv')

# Charger les données
df = pd.read_csv(csv_path)

print("=== ANALYSE INITIALE ===")
print(f"Nombre total de météorites: {len(df)}")
print("\nDistribution recclass_clean AVANT rééquilibrage:")
print(df['recclass_clean'].value_counts())
print(f"\nProportion L6: {(df['recclass_clean'] == 'L6').sum() / len(df) * 100:.1f}%")

# Définir les proportions cibles pour un meilleur équilibre
# L6 passe de ~25% à ~12% pour permettre aux autres types d'émerger
TARGET_PROPORTIONS = {
    'H5': 0.12,      # ~12%
    'H4': 0.10,      # ~10%
    'L6': 0.12,      # ~12% (au lieu de 25%)
    'L5': 0.10,      # ~10%
    'H6': 0.08,      # ~8%
    'L4': 0.06,      # ~6%
    'L3': 0.05,      # ~5%
    'H3': 0.05,      # ~5%
    'CARBONACEOUS': 0.08,  # ~8%
    'IRON': 0.06,    # ~6%
    'ACHONDRITE': 0.08,    # ~8%
    'OTHER': 0.10    # ~10% (autres types)
}

def rebalance_dataset(df, target_props, target_size=None):
    """
    Rééquilibre le dataset selon les proportions cibles
    """
    if target_size is None:
        target_size = len(df)
    
    rebalanced_dfs = []
    
    for type_name, target_prop in target_props.items():
        # Filtrer les lignes de ce type
        type_df = df[df['recclass_clean'] == type_name].copy()
        current_count = len(type_df)
        target_count = int(target_size * target_prop)
        
        if current_count == 0:
            continue
        
        if current_count > target_count:
            # Sous-échantillonner (réduire)
            # Stratifier par continent pour garder la diversité géographique
            if 'continent' in type_df.columns and type_df['continent'].notna().sum() > 0:
                sampled = type_df.groupby('continent', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), max(1, int(target_count * len(x) / current_count))), 
                                      random_state=42)
                )
            else:
                sampled = type_df.sample(n=target_count, random_state=42)
        else:
            # Sur-échantillonner (augmenter) avec remplacement
            sampled = type_df.sample(n=target_count, replace=True, random_state=42)
        
        rebalanced_dfs.append(sampled)
        print(f"{type_name}: {current_count} → {len(sampled)} ({len(sampled)/target_size*100:.1f}%)")
    
    # Combiner tous les types
    result = pd.concat(rebalanced_dfs, ignore_index=True)
    
    # Mélanger aléatoirement
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return result

# Appliquer le rééquilibrage
print("\n=== RÉÉQUILIBRAGE EN COURS ===")
df_rebalanced = rebalance_dataset(df, TARGET_PROPORTIONS, target_size=len(df))

print("\n=== ANALYSE APRÈS RÉÉQUILIBRAGE ===")
print(f"Nombre total de météorites: {len(df_rebalanced)}")
print("\nDistribution recclass_clean APRÈS rééquilibrage:")
print(df_rebalanced['recclass_clean'].value_counts())
print(f"\nProportion L6: {(df_rebalanced['recclass_clean'] == 'L6').sum() / len(df_rebalanced) * 100:.1f}%")

# Vérifier la cohérence recclass → recclass_clean
print("\n=== VÉRIFICATION COHÉRENCE ===")
mapping_check = df_rebalanced.groupby('recclass')['recclass_clean'].nunique()
inconsistent = mapping_check[mapping_check > 1]
if len(inconsistent) > 0:
    print("⚠️ Attention: Incohérences détectées:")
    print(inconsistent)
else:
    print("✓ Cohérence recclass → recclass_clean préservée")

# Vérifier la diversité géographique
print("\n=== DIVERSITÉ GÉOGRAPHIQUE ===")
print("Distribution par continent:")
print(df_rebalanced['continent'].value_counts())

# Sauvegarder le nouveau dataset dans le même répertoire
output_file = os.path.join(script_dir, 'meteorites_final_rebalanced.csv')
df_rebalanced.to_csv(output_file, index=False)
print(f"\n✓ Dataset rééquilibré sauvegardé dans: {output_file}")

# Créer un rapport comparatif
print("\n=== RAPPORT COMPARATIF ===")
comparison = pd.DataFrame({
    'Avant': df['recclass_clean'].value_counts(),
    'Après': df_rebalanced['recclass_clean'].value_counts()
})
comparison['Différence'] = comparison['Après'] - comparison['Avant']
comparison['% Avant'] = (comparison['Avant'] / len(df) * 100).round(1)
comparison['% Après'] = (comparison['Après'] / len(df_rebalanced) * 100).round(1)
print(comparison.sort_values('% Après', ascending=False))

print("\n=== IMPACT SUR LES RÈGLES D'ASSOCIATION ===")
print("Avantages du rééquilibrage:")
print("1. Réduction de L6: 25% → 12% permet à d'autres types d'émerger")
print("2. Distribution plus uniforme → plus de règles significatives")
print("3. Diversité géographique préservée")
print("4. Cohérence recclass → recclass_clean maintenue")
