# pylint: disable=all
import numpy as np  # Pour les calculs numériques
import sympy as sp  # Pour les calculs symboliques
import matplotlib.pyplot as plt  # Pour les visualisations
from scipy.optimize import line_search  # Pour la recherche linéaire (non utilisé ici)
from scipy.optimize import approx_fprime  # Pour approximer le gradient (non utilisé ici)

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

# 1.2. Méthode de descente de gradient avec pas optimal
def gradient_descent_optimal_step(x0, max_iter=1000, tol=1e-6):
    """
    Descente de gradient avec un pas optimal calculé à chaque itération.
    Args:
        x0 (numpy array): Point initial.
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
        alpha = pas_optimal(x, grad)  # Calculer le pas optimal
        x_new = x - alpha * grad  # Mettre à jour le point avec le pas optimal
        traj.append(x_new)  # Ajouter le nouveau point à la trajectoire
        f_values.append(f(x_new))  # Ajouter la nouvelle valeur de la fonction
        if np.linalg.norm(x_new - x) < tol or np.linalg.norm(grad) < tol:  # Vérifier la convergence
            break
        x = x_new  # Mettre à jour le point actuel
    return x, f(x), np.array(traj), np.array(f_values)  # Retourner les résultats

def pas_optimal(x, gradient):
    """
    Calculer le pas optimal pour la descente de gradient en utilisant la différenciation symbolique.
    Args:
        x (numpy array): Point actuel.
        gradient (numpy array): Gradient au point actuel.
    Returns:
        float: Pas optimal.
    """
    alpha = sp.Symbol('alpha', real=True, positive=True)  # Variable symbolique pour le pas
    # Define h(alpha) = f(x - alpha * gradient)
    x_vars = sp.symbols('x1 x2 x3')  # Variables symboliques pour x1, x2, x3
    new_x = [x[i] - alpha * gradient[i] for i in range(len(x_vars))]  # Nouveau point en fonction de alpha
    # Substitute the new_x into the function f
    h_alpha = f(new_x)  # Fonction objectif en fonction de alpha
    # Compute derivative: h'(alpha)
    h_prime = sp.diff(h_alpha, alpha)  # Dérivée de h par rapport à alpha
    alpha_solution = sp.solve(h_prime, alpha)  # Résoudre h'(alpha) = 0
    valid_alphas = [sol.evalf() for sol in alpha_solution if sol.is_real and sol > 0]  # Filtrer les solutions valides
    if valid_alphas:
        return float(min(valid_alphas))  # Retourner le plus petit pas valide
    else:
        return 0.01  # Pas par défaut si aucune solution valide n'est trouvée
    
# Méthode de Newton
def hessian_f(x):
    """
    Calculer la matrice hessienne de la fonction objectif.
    Args:
        x (numpy array): Vecteur d'entrée [x1, x2, x3].
    Returns:
        numpy array: Matrice hessienne.
    """
    x1, x2, x3 = x
    H = np.zeros((3, 3))  # Initialiser la matrice hessienne
    H[0, 0] = 12*x1**2 + 2  # d²f/dx1²
    H[0, 1] = 0             # d²f/dx1dx2
    H[0, 2] = -2            # d²f/dx1dx3
    H[1, 0] = 0             # d²f/dx2dx1
    H[1, 1] = 4             # d²f/dx2²
    H[1, 2] = -2            # d²f/dx2dx3
    H[2, 0] = -2            # d²f/dx3dx1
    H[2, 1] = -2            # d²f/dx3dx2
    H[2, 2] = 2             # d²f/dx3²
    return H

def newton_method(x0, max_iter=1000, tol=1e-6):
    """
    Méthode de Newton en utilisant explicitement l'inverse de la matrice hessienne.
    
    Args:
        x0 (numpy array): Point initial.
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
        H = hessian_f(x)  # Calculer la matrice hessienne
        
        try:
            H_inv = np.linalg.inv(H)  # Calculer explicitement l'inverse de la hessienne
            delta_x = np.dot(H_inv, -grad)  # Calculer la mise à jour
        except np.linalg.LinAlgError:
            print("La matrice hessienne est singulière, arrêt de l'algorithme.")
            break
        
        x_new = x + delta_x  # Mettre à jour le point
        traj.append(x_new)  # Ajouter le nouveau point à la trajectoire
        f_values.append(f(x_new))  # Ajouter la nouvelle valeur de la fonction
        
        if np.linalg.norm(delta_x) < tol:  # Vérifier la convergence
            break
        
        x = x_new  # Mettre à jour le point actuel
    
    return x, f(x), np.array(traj), np.array(f_values)  # Retourner les résultats


# Point initial
x0 = np.array([0.0, 0.0, 0.0])  # Point de départ pour l'optimisation
alpha = 0.01  # Pas fixe pour la descente de gradient

# Exécuter les méthodes d'optimisation
x_min_fixed, f_min_fixed, traj_fixed, f_fixed = gradient_descent_fixed_step(x0, alpha)
x_min_opt, f_min_opt, traj_optimal, f_optimal = gradient_descent_optimal_step(x0)
x_min_newton, f_min_newton, traj_newton, f_newton = newton_method(x0)

# Afficher les résultats
print(f"Minimum trouvé (Descente de gradient avec pas fixe) à: {x_min_fixed} avec valeur: {f_min_fixed}")
print(f"Minimum trouvé (Descente de gradient avec pas optimal) à: {x_min_opt} avec valeur: {f_min_opt}")
print(f"Minimum trouvé (Newton) à: {x_min_newton} avec valeur: {f_min_newton}")

# Visualisation
def visualize_optimization_with_contours(f, x0, traj_fixed, traj_optimal, traj_newton, f_fixed, f_optimal, f_newton, x_min_fixed, x_min_opt, x_min_newton):
    """
    Visualiser le processus d'optimisation avec des lignes de contour et des courbes de convergence.
    Args:
        f (function): Fonction objectif.
        x0 (numpy array): Point initial.
        traj_fixed (numpy array): Trajectoire de la descente de gradient avec pas fixe.
        traj_optimal (numpy array): Trajectoire de la descente de gradient avec pas optimal.
        traj_newton (numpy array): Trajectoire de la méthode de Newton.
        f_fixed (numpy array): Valeurs de la fonction pour la descente de gradient avec pas fixe.
        f_optimal (numpy array): Valeurs de la fonction pour la descente de gradient avec pas optimal.
        f_newton (numpy array): Valeurs de la fonction pour la méthode de Newton.
        x_min_fixed (numpy array): Point minimum trouvé par la descente de gradient avec pas fixe.
        x_min_opt (numpy array): Point minimum trouvé par la descente de gradient avec pas optimal.
        x_min_newton (numpy array): Point minimum trouvé par la méthode de Newton.
    """
    variables = sp.symbols('x1 x2 x3')  # Variables symboliques pour x1, x2, x3
    f_symbolic = f(variables)  # Représentation symbolique de la fonction objectif
    f_numeric = lambda x: f(x)  # Fonction numérique pour l'évaluation

    # Créer une grille pour la visualisation
    x1_min, x1_max = min(min(traj_fixed[:, 0]), min(traj_optimal[:, 0]), min(traj_newton[:, 0])) - 0.5, max(max(traj_fixed[:, 0]), max(traj_optimal[:, 0]), max(traj_newton[:, 0])) + 0.5
    x2_min, x2_max = min(min(traj_fixed[:, 1]), min(traj_optimal[:, 1]), min(traj_newton[:, 1])) - 0.5, max(max(traj_fixed[:, 1]), max(traj_optimal[:, 1]), max(traj_newton[:, 1])) + 0.5

    # Ajuster les bornes de la grille si elles sont trop étroites
    if x1_max - x1_min < 2:
        x1_center = (x1_max + x1_min) / 2
        x1_min, x1_max = x1_center - 1, x1_center + 1
    if x2_max - x2_min < 2:
        x2_center = (x2_max + x2_min) / 2
        x2_min, x2_max = x2_center - 1, x2_center + 1

    # Créer une grille pour les lignes de contour
    x1_grid = np.linspace(x1_min, x1_max, 100)
    x2_grid = np.linspace(x2_min, x2_max, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    x3_value = (x_min_fixed[2] + x_min_opt[2] + x_min_newton[2]) / 3  # Valeur moyenne de x3 pour la visualisation

    # Évaluer la fonction sur la grille
    f_symbolic_lambda = sp.lambdify(variables, f_symbolic, "numpy")
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f_symbolic_lambda(X1[i, j], X2[i, j], x3_value)

    # Tracer les lignes de contour et les trajectoires
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(121)
    contour_levels = 20
    contour = ax1.contour(X1, X2, Z, contour_levels, colors='k', alpha=0.4)
    filled_contour = ax1.contourf(X1, X2, Z, contour_levels, cmap='viridis', alpha=0.7)
    plt.colorbar(filled_contour, ax=ax1, label='f(x1, x2, x3_opt)')
    ax1.plot(traj_fixed[:, 0], traj_fixed[:, 1], 'r.-', linewidth=1.5, label='Pas Fixe', markersize=6)
    ax1.plot(traj_optimal[:, 0], traj_optimal[:, 1], 'b.-', linewidth=1.5, label='Pas Optimal', markersize=6)
    ax1.plot(traj_newton[:, 0], traj_newton[:, 1], 'g.-', linewidth=1.5, label='Newton', markersize=6)
    ax1.plot(x0[0], x0[1], 'yo', markersize=10, label='Point Initial')
    ax1.plot(x_min_fixed[0], x_min_fixed[1], 'ro', markersize=10, label='Min (Pas Fixe)')
    ax1.plot(x_min_opt[0], x_min_opt[1], 'bo', markersize=10, label='Min (Pas Optimal)')
    ax1.plot(x_min_newton[0], x_min_newton[1], 'go', markersize=10, label='Min (Newton)')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Lignes de niveau et trajectoires d\'optimisation')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Tracer la convergence des méthodes
    ax2 = fig.add_subplot(122)
    iterations_fixed = range(len(f_fixed))
    iterations_optimal = range(len(f_optimal))
    iterations_newton = range(len(f_newton))
    ax2.plot(iterations_fixed, f_fixed, 'r.-', linewidth=2, label='Pas Fixe', markersize=4)
    ax2.plot(iterations_optimal, f_optimal, 'b.-', linewidth=2, label='Pas Optimal', markersize=4)
    ax2.plot(iterations_newton, f_newton, 'g.-', linewidth=2, label='Newton', markersize=4)
    ax2.set_xlabel('Itérations')
    ax2.set_ylabel('Valeur de la fonction objectif')
    ax2.set_title('Convergence des méthodes d\'optimisation')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Appeler la fonction de visualisation
visualize_optimization_with_contours(f, x0, traj_fixed, traj_optimal, traj_newton, f_fixed, f_optimal, f_newton, x_min_fixed, x_min_opt, x_min_newton)