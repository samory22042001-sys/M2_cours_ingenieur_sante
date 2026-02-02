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
    print(f"--- Entraînement du LSTM ({epochs} époques) ---")
    history_lstm = train_and_get_history(model_lstm, criterion, opt_lstm, epochs, X_train, Y_train)
    
    print(f"\n--- Entraînement de la Neural ODE ({epochs} époques) ---")
    history_ode = train_and_get_history(model_ode, criterion, opt_ode, epochs, X_train, Y_train)

    # 5. Création du graphique comparatif
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_lstm, label='Loss : LSTM (Discret)', color='royalblue', linewidth=2)
    plt.plot(history_ode, label='Loss : Neural ODE (Continu)', color='darkorange', linewidth=2)
    
    plt.title("Comparaison de la convergence : LSTM vs Neural ODE")
    plt.xlabel("Époques")
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