import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

from clinical_cool_etud.config import DATA_DIR
from clinical_cool_etud.NLLsurv import NLLSurvLoss
from clinical_cool_etud.model import LSTM_risk_estimator
from clinical_cool_etud.ODE import NeuralODESurvival 
from clinical_cool_etud.prepa_data_model import build_lstm_tensor, split_tensors_stratified

def train_and_get_history(model, criterion, optimizer, epochs, X_train, Y_train):
    """Fonction pour entraîner un modèle et enregistrer l'évolution de la loss."""
    history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Passage avant (forward)
        probs = model(X_train)
        
        # Calcul de la perte (Negative Log-Likelihood)
        loss = criterion(probs, Y_train)
        
        # Rétropropagation
        loss.backward()
        optimizer.step()
        
        # On stocke la perte moyenne par patient
        history.append(loss.item() / len(X_train))
        
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}/{epochs} terminée")
            
    return history

def main():
    # 1. Chargement et préparation des données
    data_pbc = pd.read_csv(DATA_DIR / "clinical_data_pbc_cleaned.csv")
    
    list_features_continuous = ["age", "edema", "serBilir", "serChol", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "histologic"]
    list_features_binary = ["drug", "sex", "ascites", "hepatomegaly", "spiders"]
    number_features = len(list_features_continuous) + len(list_features_binary)

    # Construction des tenseurs
    X_tensor, y_tensor, _ = build_lstm_tensor(
        data_pbc, id_col='id', tte_col="tte", event_col="label",
        feature_continuous_cols=list_features_continuous,
        features_binary_cols=list_features_binary,
    )

    # Split Train/Test
    X_train, _, Y_train, _ = split_tensors_stratified(X_tensor, y_tensor)
    
    # Définition de l'horizon temporel maximal
    MAX_TIME = int(Y_train[:, 0].max()) + 1
    epochs = 200

    # 2. Initialisation du LSTM (Modèle Discret)
    model_lstm = LSTM_risk_estimator(
        input_size=number_features, hidden_size=64, num_layers=2, number_time_discrete=MAX_TIME
    )
    
    # 3. Initialisation de la Neural ODE (Modèle Continu)
    model_ode = NeuralODESurvival(
        input_size=number_features, hidden_size=32, number_time_discrete=MAX_TIME
    )

    criterion = NLLSurvLoss() #
    
    # Optimiseurs
    opt_lstm = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    opt_ode = torch.optim.Adam(model_ode.parameters(), lr=0.001)

    # 4. Phase d'entraînement
    print(f"--- Entraînement du LSTM ({epochs} epochs) ---")
    history_lstm = train_and_get_history(model_lstm, criterion, opt_lstm, epochs, X_train, Y_train)
    
    print(f"\n--- Entraînement de la Neural ODE ({epochs} epochs) ---")
    history_ode = train_and_get_history(model_ode, criterion, opt_ode, epochs, X_train, Y_train)

    # 5. Création du graphique comparatif
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_lstm, label='Loss : LSTM (Discret)', color='royalblue', linewidth=2)
    plt.plot(history_ode, label='Loss : Neural ODE (Continu)', color='darkorange', linewidth=2)
    
    plt.title("Comparaison de la convergence : LSTM vs Neural ODE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (Negative Log-Likelihood Moyenne)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Sauvegarde dans le dossier DATA_DIR
    save_path = DATA_DIR / "comparaison_loss_complete.png"
    plt.savefig(save_path)
    plt.close()

    print(f"\n" + "="*40)
    print(f"COMPARAISON TERMINÉE")
    print(f"Le graphique est disponible ici : {os.path.abspath(save_path)}")
    print("="*40)

if __name__ == "__main__":
    main()

""" 
Méthodologie et Données :
Pour garantir une comparaison équitable, les deux modèles ont été entraînés sur un dataset strictement identique :
* Population : 312 patients au total, avec un suivi longitudinal allant jusqu'à 16 visites par patient.
* Features : 15 variables cliniques (10 continues comme la bilirubine et l'albumine, et 5 binaires).
* Répartition : Un split stratifié a été utilisé pour maintenir un taux d'événements (décès) cohérent entre l'entraînement (45,0%) et le test (44,4%).


Configuration des Hyperparamètres :
"Les deux modèles ont partagé le même cadre d'apprentissage :
* Optimiseur avec un taux d'apprentissage (learning rate) de 0.001.
* Fonction de perte : La Negative Log-Likelihood (NLL) adaptée à la survie discrète.
* Durée : 200 épochs d'entraînement pour observer la stabilisation complète des courbes.
* Architecture : Le LSTM utilisait 64 neurones cachés sur 2 couches, tandis que le Neural ODE utilisait un espace latent de 32 dimensions pour modéliser la dynamique continue.

Interprétation du Graphique :
    * Le LSTM (Bleu) : Malgré sa complexité (64 neurones, 2 couches), il plafonne rapidement. On observe une chute très brutale de la perte (Loss) dès les premières époques. 
Le modèle identifie rapidement les corrélations majeures dans les données séquentielles. Cependant, il atteint un plateau très vite. 
Cela suggère que le LSTM, peine à capturer toute la complexité des données, probablement à cause de sa nature discrète qui ignore le temps réel s'écoulant entre deux visites médicales.

    * Le Neural ODE (Orange) : Avec seulement quelques dimensions latentes, il surpasse le LSTM. La descente est beaucoup plus régulière et ne présente pas de cassure brutale. 
C'est le signe que le solveur d'équations différentielles optimise continuellement la fonction de transition du patient. Contrairement au LSTM, le Neural ODE ne stagne pas. 
Il continue de minimiser la perte tout au long des 200 epochs, finissant à un niveau proche de 0.3. 

Conclusion Technique :
En résumé, avec les mêmes données et le même budget d'optimisation, le Neural ODE parvient à une perte finale d'environ 0.3 contre environ 2.7 pour le LSTM. 
Cette capacité à mieux ajuster les trajectoires de santé confirme que les modèles à base d'équations différentielles sont l'outil de choix pour la médecine personnalisée et le suivi à long terme.

Limites et Perspectives :
Cependant, cette analyse se base uniquement sur la convergence de la fonction de perte. Pour une évaluation complète, il serait essentiel d'examiner les performances prédictives en utilisant des métriques comme le C-Index ou l'AUC.
Car oui la Loss est un indicateur d'ajustement, mais ne garantit pas une meilleure capacité de généralisation sur des données inédites. (cas de surajustement)
"""