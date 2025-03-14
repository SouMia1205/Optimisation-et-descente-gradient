# pylint: disable=all
import numpy as np  # Pour les calculs numériques

# Définir la fonction objectif
def f(x):
    """
    Fonction objectif à minimiser.
    Args:
        x (numpy array): Vecteur d'entrée [x1, x2, x3].
    Returns:
        float: Valeur de la fonction en x.
    """
    x1, x2, x3 = x
    return x1**4 + x1**2 + 2*x2**2 + x3**2 - 6*x1 + 3*x2 - 2*x3*(x1 + x2 + 1)

# Définir le gradient de la fonction objectif
def grad_f(x):
    """
    Gradient de la fonction objectif.
    Args:
        x (numpy array): Vecteur d'entrée [x1, x2, x3].
    Returns:
        numpy array: Vecteur gradient [df/dx1, df/dx2, df/dx3].
    """
    x1, x2, x3 = x
    df_dx1 = 4*x1**3 + 2*x1 - 6 - 2*x3  # Dérivée partielle par rapport à x1
    df_dx2 = 4*x2 + 3 - 2*x3             # Dérivée partielle par rapport à x2
    df_dx3 = 2*x3 - 2*(x1 + x2 + 1)      # Dérivée partielle par rapport à x3
    return np.array([df_dx1, df_dx2, df_dx3])