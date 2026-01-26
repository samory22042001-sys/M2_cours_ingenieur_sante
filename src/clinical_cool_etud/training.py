from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

from clinical_cool_etud.Calcul import manual_concordance_index
from clinical_cool_etud.NLLsurv import NLLSurvLoss
from clinical_cool_etud.config import DATA_DIR
from clinical_cool_etud.model import LSTM_risk_estimator
from clinical_cool_etud.prepa_data_model import build_lstm_tensor, split_tensors_stratified


def main():

    # Charger les données

    data_pbc = pd.read_csv(DATA_DIR / "clinical_data_pbc_cleaned.csv")

    list_features_continuous = ["age", "edema", "serBilir", "serChol", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "histologic"]
    list_features_binary = ["drug", "sex", "ascites", "hepatomegaly", "spiders"]

    time_to_event_column = "tte"
    event_column = "label"

    number_features = len(list_features_continuous) + len(list_features_binary)

    # Construction des tenseurs
    X_tensor, y_tensor, all_ids = build_lstm_tensor(
        data_pbc,  # Ton dataframe longitudinal complet (pas le baseline !)
        id_col='id',
        tte_col=time_to_event_column,
        event_col=event_column,
        feature_continuous_cols=list_features_continuous,
        features_binary_cols=list_features_binary,
    )

    # Split et datasets

    X_train, X_test, Y_train, Y_test = split_tensors_stratified(X_tensor, y_tensor)

    # Tensordataset : necessaire pour utiliser le dataloader (création des batchs). Spécifie la façon de prendre 1 élement

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    # DataLoader
    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Pour instancier l'architecture du modèle, on doit définir le nombre de neurones en sortie (nombre de pas de temps)
    # On va prendre le temps maximum observé dans les labels d'entraînement (et ajouter 1 pour bien couvrir les temps)
    MAX_TIME_HORIZON = int(Y_train[:, 0].max()) +1 

    # Instanciation du modèle, de la loss et de l'optimizer

    model = LSTM_risk_estimator(input_size=number_features, hidden_size=64, num_layers=2, number_time_discrete=MAX_TIME_HORIZON)
    criterion = NLLSurvLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam est le choix classique encore aujourd'hui


    # Boucle d'entraînement

    loss_history = []
    
    model.train()  # Met le modèle en mode "Apprentissage" 

    number_epochs = 100

    for epoch in range(number_epochs):
        epoch_loss = 0.0 # On calcule la loss sur toute l'époch

        # On itère sur les batchs (paquets de patients)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):

            # Toujours remettre à zéro avant chaque étape, sinon PyTorch cumule les gradients dans le graphe de calcul
            optimizer.zero_grad()

            # Le modèle sort les probabilités de risques de décès pour chaque jour
            probs = model(X_batch)

            # On calcule la loss que l'on a définit (negative log likelihood)
            loss = criterion(probs, y_batch)

            # Rétroprogation du gradients pour calculer la correction à apporté aux poids du réseau
            loss.backward()

            # Optimisation des poids
            optimizer.step()

            # On ajoute la loss du batch a celle de l'epoch
            epoch_loss += loss.item() 

        # Calcul de la loss moyenne sur toute l'époque
        epoch_loss /= (train_dataset.__len__())
        loss_history.append(epoch_loss)

        # Affichage régulier (toutes les 5 époques)
        if (epoch + 1) % 5 == 0:
            print(f"Example Epoch [{epoch + 1}/{number_epochs}], Loss: {epoch_loss:.4f}")


    # Affichage graphique de la courbe d'apprentissage 

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.title("Loss ")
    plt.xlabel("Époch")
    plt.ylabel("Loss (Negative Log-Likelihood)")
    plt.legend()
    plt.grid(True)
    plt.show()

    model.eval() 
    with torch.no_grad():
        # Obtenir les probabilités sur le test
        probs_test = model(X_test) 
        # Calcul du risque cumulé : Matrix [n_patients, n_time_points]
        risk_scores_cumulative = torch.cumsum(probs_test, dim=1).numpy() 

    # --- CORRECTION ICI (Indentation et appel) ---
    # On passe Y_test.numpy() et la matrice complète
    c_index_time_dep = manual_concordance_index(Y_test.numpy(), risk_scores_cumulative)
    
    print(f"\nC-Index dépendant du temps final : {c_index_time_dep:.4f}")

    #Le modèle affiche un C-Index dépendant du temps autour des 0.8 , ce qui indique une forte capacité discriminatoire. 
    #Le modèle parvient à ordonner correctement la survie des patients dans 83,8% des paires comparables.
    #Cela confirme que l'intégration des trajectoires temporelles via le LSTM apporte une précision prédictive élevée dans ce contexte clinique.
