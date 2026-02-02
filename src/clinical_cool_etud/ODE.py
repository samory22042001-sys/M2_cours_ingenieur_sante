import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torchdiffeq import odeint

from clinical_cool_etud.Calcul import manual_concordance_index
from clinical_cool_etud.NLLsurv import NLLSurvLoss
from clinical_cool_etud.config import DATA_DIR
from clinical_cool_etud.prepa_data_model import build_lstm_tensor, split_tensors_stratified

# --- 1. Définition de la fonction ODE (la dérivée du système) ---
# Ce réseau définit dh/dt, soit la dynamique d'évolution de l'état du patient.
class ODEFunc(nn.Module):
    def __init__(self, hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t, h):
        return self.net(h)

# --- 2. Modèle Neural ODE pour la Survie ---
class NeuralODESurvival(nn.Module):
    def __init__(self, input_size, hidden_size, number_time_discrete):
        super(NeuralODESurvival, self).__init__()
        self.hidden_size = hidden_size
        
        # Projection des caractéristiques cliniques vers l'espace latent
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        
        # La dynamique ODE continue
        self.ode_func = ODEFunc(hidden_size)
        
        # Couche de sortie pour prédire la probabilité de décès à chaque instant t
        self.fc = nn.Linear(hidden_size, number_time_discrete)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialisation de l'état latent (h0)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # Passage à travers la séquence de visites
        for t_step in range(seq_len):
            # Mise à jour de l'état latent avec les données de la visite actuelle
            obs = self.input_to_hidden(x[:, t_step, :])
            h = h + obs 
            
            # Évolution "naturelle" entre les visites via le solver ODE
            # On simule un petit pas de temps symbolique (0.1)
            h = odeint(self.ode_func, h, torch.tensor([0., 0.1]).to(x.device))[-1]

        # Transformation de l'état latent final en probabilités discrètes de survie
        risk_logits = self.fc(h)
        return self.softmax(risk_logits)

# --- 3. Boucle principale d'exécution ---
def main():
    # 1. Chargement et préparation des données
    data_pbc = pd.read_csv(DATA_DIR / "clinical_data_pbc_cleaned.csv")
    
    list_features_continuous = ["age", "edema", "serBilir", "serChol", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "histologic"]
    list_features_binary = ["drug", "sex", "ascites", "hepatomegaly", "spiders"]
    number_features = len(list_features_continuous) + len(list_features_binary)

    X_tensor, y_tensor, _ = build_lstm_tensor(
        data_pbc, id_col='id', tte_col="tte", event_col="label",
        feature_continuous_cols=list_features_continuous,
        features_binary_cols=list_features_binary,
    )

    # Split Train/Test (80/20)
    X_train, X_test, Y_train, Y_test = split_tensors_stratified(X_tensor, y_tensor)
    
    MAX_TIME_HORIZON = int(Y_train[:, 0].max()) + 1
    
    # 2. Initialisation
    model = NeuralODESurvival(input_size=number_features, hidden_size=32, number_time_discrete=MAX_TIME_HORIZON)
    criterion = NLLSurvLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    
    # 3. Entraînement
    model.train()
    epochs = 200 # Augmenté pour une meilleure convergence
    
    print(f"Lancement de l'entraînement Neural ODE sur {epochs} époques...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        probs = model(X_train)
        loss = criterion(probs, Y_train)
        
        loss.backward()
        optimizer.step()
        
        avg_loss = loss.item() / len(X_train)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # --- 4. Génération et Sauvegarde du Graphique ---
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss (NLL)', color='firebrick', linewidth=2)
    plt.title("Convergence de l'apprentissage : Neural ODE")
    plt.xlabel("Époques")
    plt.ylabel("Loss Moyenne")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Chemin de sauvegarde précis
    plot_filename = "neural_ode_loss_final.png"
    plot_path = DATA_DIR / plot_filename
    plt.savefig(plot_path)
    plt.close()

    # --- 5. Évaluation et C-Index ---
    model.eval()
    with torch.no_grad():
        probs_test = model(X_test)
        # On cumule les probabilités de décès pour obtenir le score de risque
        risk_cumulative = torch.cumsum(probs_test, dim=1).numpy()
        
    c_index = manual_concordance_index(Y_test.numpy(), risk_cumulative)
    
    print("\n" + "="*30)
    print(f"RÉSULTATS FINAUX")
    print(f"C-Index : {c_index:.4f}")
    print(f"Graphique sauvegardé dans : {os.path.abspath(plot_path)}")
    print("="*30)

if __name__ == "__main__":
    main()