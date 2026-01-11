import numpy as np

# Dados los datos X queremos encontrar los parámetros de la funcion que mejor los aproximen.


# SIN regularizacion. Es decir, nos importa el problema así como viene. No imponemos más restricciones para β

# Ej 2a
def cml(X, y):
    U, S, Vh = np.linalg.svd(X, full_matrices=False) # Si se setea en True se tiene que usar Sigma = np.zeros((n,m)); np.fill_diagonal(Sigma, S) para que pueda reconstruirse X.

    Sigma = np.diag(1/S)

    return Vh.T@Sigma@U.T@y  # Devolvemos el beta, con este se puede calcular el ECM


#Ej 2b
def cml_reg(X,y, lam):
    U, S, Vh = np.linalg.svd(X, full_matrices=False) # Si se setea en True se tiene que usar Sigma = np.zeros((n,m)); np.fill_diagonal(Sigma, S) para que pueda reconstruirse X.


    S_hat = np.diag(S/(np.square(S) + (lam * np.ones(len(S)))))

    return Vh.T@S_hat@U.T@y

