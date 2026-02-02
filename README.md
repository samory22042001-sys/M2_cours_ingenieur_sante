## Description du projet

Ce projet implémente un modèle de prédiction du risque temporel (time-to-event) basé sur un réseau de neurones de type LSTM, adapté à des données cliniques longitudinales.  
L’objectif est d’estimer un risque instantané ou une fonction de survie à partir de séries temporelles multivariées, en tenant compte de la censure des données.

## Dépendances principales

Le projet repose notamment sur les librairies suivantes :

- Python >= 3.10
- PyTorch
- NumPy
- uv (gestion des environnements et dépendances)

L’ensemble des dépendances est défini dans `pyproject.toml` et figé via `uv.lock` afin de garantir la reproductibilité des expériences.

## Pour mettre à jour les packages dans votre environnement virtuel par rapport au pyproject.toml et au uv.lock
```bash
uv sync
```
## Structure du projet

clinical_cool_etud/
├── process_data.py        # Point d’entrée principal (script CLI)
├── correct_data.py        # Nettoyage et correction des données
├── prepa_data_model.py    # Préparation des tenseurs pour le modèle
├── model.py               # Définition du modèle LSTM
├── NLLsurv.py             # Fonction de loss pour données de survie
├── training.py            # Boucle d’entraînement
├── Calcul.py              # Fonctions utilitaires de calcul du C-index
├── ODE.py                 # Modélisation par équations différentielles (le cas échéant)
└── Comparaison.py         # Comparaisons de modèles LSTM vs ODE


## Pour lancer la fonction main du fichier process_data.py

```bash
uv run process_data
```

Le script `process_data` constitue le point d’entrée principal du projet.  
Il orchestre les différentes étapes du pipeline :

1. Chargement et nettoyage des données
2. Préparation des données pour le modèle
3. Initialisation du modèle
4. Calcul des prédictions de risque

(Le pyproject.toml contient :

```bash
[project.scripts]
process_data = "clinical_cool_etud.process_data:main"
```
ce qui signifie : si je lance la commande process_data, alors je lance la fonction main du module process_data du package clinical_cool_etud)

## Pour tester le modèle et la fonction de loss

La fonction `NLLSurvLoss` implémente une log-vraisemblance négative adaptée aux données censurées, couramment utilisées en analyse de survie.  
Elle prend en compte à la fois :
- le temps jusqu’à l’événement
- l’indicateur de censure (événement observé ou non)
Vous pouvez vérifier que le code pour votre modèle et votre fonction de loss fonctionne en créant une instance d'environnement virtuel dans votre terminal (équivalent à créer un notebook temporaire dans le terminal):

```bash
uv run python
```

Puis ensuite entrer les lignes :

```bash
import numpy as np
import torch

from clinical_cool_etud.model import LSTM_risk_estimator
from clinical_cool_etud.NLLsurv import NLLSurvLoss

input_tensor = torch.ones(8).reshape(2,2,2)
model = LSTM_risk_estimator(2,2,1,10)
loss_fn = NLLSurvLoss()

risk_estimation = model(input_tensor)
list_times_to_event = [6,8]
list_status_event = [0,1]
target = np.stack((list_times_to_event, list_status_event), axis = 1)
target = torch.tensor(target)

loss = loss_fn(risk_estimation, target)
print(loss)
```
## Pistes d’amélioration

- Des analyses d’ablation ou de sensibilité peuvent être mises en place afin d’évaluer l’impact des différentes variables d’entrée sur la prédiction du risque (afin de mieux comprendre l'interprétabilité du modèle)
- Comparaison avec des modèles classiques (Cox, RSF)
- Validation croisée temporelle


