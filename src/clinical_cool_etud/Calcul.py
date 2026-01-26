import numpy as np


def manual_concordance_index(y_true, risk_matrix):
    times = y_true[:, 0]
    events = y_true[:, 1]
    concordant_pairs = 0
    total_comparable_pairs = 0
    n = len(times)
    
    for i in range(n):
        if events[i] == 1: # Patient i est décédé
            t_i = int(times[i])
            # On s'assure de ne pas sortir de la matrice de risque
            t_idx = min(t_i, risk_matrix.shape[1] - 1)
            
            for j in range(n):
                if times[j] > times[i]: # j a survécu plus longtemps que i
                    total_comparable_pairs += 1
                    
                    # On compare les risques AU MOMENT t_i
                    risk_i = risk_matrix[i, t_idx]
                    risk_j = risk_matrix[j, t_idx]
                    
                    if risk_i > risk_j:
                        concordant_pairs += 1
                    elif risk_i == risk_j:
                        concordant_pairs += 0.5
                        
    return concordant_pairs / total_comparable_pairs if total_comparable_pairs > 0 else 0