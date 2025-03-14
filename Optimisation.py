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

# 1.1. Méthode de descente de gradient avec pas fixe
def gradient_descent_fixed_step(x0, alpha, max_iter=1000, tol=1e-6):
    """
    Descente de gradient avec un pas fixe.
    Args:
        x0 (numpy array): Point initial.
        alpha (float): Pas fixe.
        max_iter (int): Nombre maximal d'itérations.
        tol (float): Tolérance pour la convergence.
    Returns:
        tuple: (Point final, valeur de la fonction, trajectoire, valeurs de la fonction).
    """
    x = x0  # Initialiser le point de départ
    traj = [x0]  # Stocker la trajectoire des points
    f_values = [f(x0)]  # Stocker les valeurs de la fonction à chaque étape
    for i in range(max_iter):
        grad = grad_f(x)  # Calculer le gradient au point actuel
        x_new = x - alpha * grad  # Mettre à jour le point avec le pas fixe
        traj.append(x_new)  # Ajouter le nouveau point à la trajectoire
        f_values.append(f(x_new))  # Ajouter la nouvelle valeur de la fonction
        if np.linalg.norm(x_new - x) < tol:  # Vérifier la convergence
            break
        x = x_new  # Mettre à jour le point actuel
    return x, f(x), np.array(traj), np.array(f_values)  # Retourner les résultats