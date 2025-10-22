import pandas as pd

# Remplace 'ton_fichier.csv' par le nom de ton fichier
df = pd.read_csv('courtinm_submitty2_2_2.csv')

# Lire les 5 premières lignes "brutes" du fichier
with open('courtinm_submitty2_2.csv', 'r', encoding='utf-8') as f:
    for i in range(50):
        line = f.readline()
        print(f"Ligne {i+1}: {repr(line)}")
