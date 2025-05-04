# pylint: disable=all
import numpy as np  # Pour les calculs numériques
import sympy as sp  # Pour les calculs symboliques
import matplotlib.pyplot as plt  # Pour les visualisations
from scipy.optimize import line_search  # Pour la recherche linéaire (non utilisé ici)
from scipy.optimize import approx_fprime  # Pour approximer le gradient (non utilisé ici)
np.seterr(all='raise')
# Définir la fonction objectif
def f(x):
    x1, x2, x3 = x
    return x1**4 + x1**2 + 2*x2**2 + x3**2 - 6*x1 + 3*x2 - 2*x3*(x1 + x2 + 1)

# Définir le gradient de la fonction objectif
def grad_f(x):
    x1, x2, x3 = x
    df_dx1 = 4*x1**3 + 2*x1 - 6 - 2*x3  # Dérivée partielle par rapport à x1
    df_dx2 = 4*x2 + 3 - 2*x3             # Dérivée partielle par rapport à x2
    df_dx3 = 2*x3 - 2*(x1 + x2 + 1)      # Dérivée partielle par rapport à x3
    return np.array([df_dx1, df_dx2, df_dx3])

# 1.1. Méthode de descente de gradient avec pas fixe
def gradient_descent_fixed_step(x0, alpha, max_iter=1000, tol=1e-6):
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
    x1, x2, x3 = x

    H = np.array([[12*x1**2 +2, 0, -2],[0, 4,-2],
                  [-2, -2,2]])
    return H
#Methode de GRCONJU
def conjugate_gradient_optimal(x0, max_iter=1000, tol=1e-6):
    x = x0
    traj=[x0]
    f_values = [f(x0)]
    r = -grad_f(x)
    d=r
    for i in range(max_iter):
        alpha = pas_optimal(x,d)
        x_new = x + alpha * d
        traj.append(x_new)
        f_values.append(f(x_new))
        r_new = -grad_f(x_new)
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new)/np.dot(r,r)
        d = r_new+beta*d
        x, r = x_new, r_new

    return x, f(x), np.array(traj), np.array(f_values)


def gradient_conjuguee(x0,max_iter=1000, tol=1e-6):
    x=x0  # Initialisation du point de départ
    traj = [x0] # Liste pour stocker les points parcourus
    f_values = [f(x0)] # Liste pour stocker les valeurs de la fonction
    grad = grad_f(x)  # Calcul du gradient initial
    d = -grad #Direction initiale
    for i in range(max_iter):
        if np.linalg.norm(grad)<tol: #si la norme du gradient est inférieure à la tolérance, on arrête.
            break
        Q = hessian_f(x)
        # Calcul du pas optimal alpha selon la méthode de Fletcher-Reeves
        alpha = -np.dot(grad,d)/np.dot(d,Q @ d)#@ pour effectuer la multiplication matricielle
        x_new = x+alpha*d # Mise à jour de la solution
        grad_new = grad_f(x_new)
        #print(x_new)
        # Calcul du coefficient beta de Fletcher-Reeves
        beta = np.dot(grad_new, grad_new)/np.dot(grad, grad)
        d = -grad_new + beta*d  # Mise à jour de la direction de descente conjuguée
        # Mise à jour de la direction de descente conjuguée
        x = x_new
        grad = grad_new
        traj.append(x)
        f_values.append(f(x))
        # Retourner le point optimal, la valeur optimale de la fonction, la trajectoire et les valeurs de la fonction
    return x, f(x), np.array(traj), np.array(f_values)


"""
def newton_method(x0, max_iter=1000, tol=1e-6):
   
    Méthode de Newton pour l'optimisation.
    Args:
        x0 (numpy array): Point initial.
        max_iter (int): Nombre maximal d'itérations.
        tol (float): Tolérance pour la convergence.
    Returns:
        tuple: (Point final, valeur de la fonction, trajectoire, valeurs de la fonction).
    
    x = x0  # Initialiser le point de départ
    traj = [x0]  # Stocker la trajectoire des points
    f_values = [f(x0)]  # Stocker les valeurs de la fonction à chaque étape
    for i in range(max_iter):
        grad = grad_f(x)  # Calculer le gradient au point actuel
        H = hessian_f(x)  # Calculer la matrice hessienne
        delta_x = np.linalg.solve(H, -grad)  # Résoudre pour la direction de mise à jour
        x_new = x + delta_x  # Mettre à jour le point
        traj.append(x_new)  # Ajouter le nouveau point à la trajectoire
        f_values.append(f(x_new))  # Ajouter la nouvelle valeur de la fonction
        if np.linalg.norm(x_new - x) < tol:  # Vérifier la convergence
            break
        x = x_new  # Mettre à jour le point actuel
    return x, f(x), np.array(traj), np.array(f_values)  # Retourner les résultats
"""


def newton_method_with_inverse(x0, max_iter=1000, tol=1e-6):
    x = x0  # Initialiser le point de départ
    traj = [x0]  # Stocker la trajectoire des points
    f_values = [f(x0)]  # Stocker les valeurs de la fonction à chaque étape

    for i in range(max_iter):
        grad = grad_f(x)  # Calculer le gradient au point actuel
        H = hessian_f(x)  # Calculer la matrice hessienne

        try:
            H_inv = np.linalg.inv(H)  # Calculer explicitement l'inverse de la hessienne
            delta_x = np.dot(H_inv, -grad)  # Calculer la mise à jour de points
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
x_min_newton, f_min_newton, traj_newton, f_newton = newton_method_with_inverse(x0)#newton_method
x_min_cg, f_min_cg, traj_cg, f_cg =gradient_conjuguee (x0) #gradient_conjuguee conjugate_gradient_optimal
# Afficher les résultats
print(f"Minimum trouvé (Descente de gradient avec pas fixe) à: {x_min_fixed} avec valeur: {f_min_fixed}")
print(f"Minimum trouvé (Descente de gradient avec pas optimal) à: {x_min_opt} avec valeur: {f_min_opt}")
print(f"Minimum trouvé (Newton) à: {x_min_newton} avec valeur: {f_min_newton}")
print(f"Minimum trouvé (Gradient conjugué) à: {x_min_cg} avec valeur: {f_min_cg}")

# Visualisation
def visualize_optimization_with_contours(f, x0, traj_fixed, traj_optimal, traj_newton,traj_cg, f_fixed, f_optimal, f_newton, f_cg, x_min_fixed, x_min_opt, x_min_newton,x_min_cg):
    variables = sp.symbols('x1 x2 x3')  # Variables symboliques pour x1, x2, x3
    f_symbolic = f(variables)  # Représentation symbolique de la fonction objectif
    f_numeric = lambda x: f(x)  # Fonction numérique pour l'évaluation

    # Créer une grille pour la visualisation
    x1_min, x1_max = min(min(traj_fixed[:, 0]), min(traj_optimal[:, 0]), min(traj_newton[:, 0]), min(traj_cg[:, 0])) - 0.5, max(max(traj_fixed[:, 0]), max(traj_optimal[:, 0]), max(traj_newton[:, 0]), max(traj_cg[:, 0])) + 0.5
    x2_min, x2_max = min(min(traj_fixed[:, 1]), min(traj_optimal[:, 1]), min(traj_newton[:, 1]), min(traj_cg[:, 0])) - 0.5, max(max(traj_fixed[:, 1]), max(traj_optimal[:, 1]), max(traj_newton[:, 1]), max(traj_cg[:, 1])) + 0.5

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
    x3_value = (x_min_fixed[2] + x_min_opt[2] + x_min_newton[2]+ x_min_cg[2]) / 4  # Valeur moyenne de x3 pour la visualisation

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
    ax1.plot(traj_cg[:, 0], traj_cg[:, 1], 'm.-', linewidth=1.5, label='Gradient Conjugué', markersize=6)
    ax1.plot(x0[0], x0[1], 'yo', markersize=10, label='Point Initial')
    ax1.plot(x_min_fixed[0], x_min_fixed[1], 'ro', markersize=10, label='Min (Pas Fixe)')
    ax1.plot(x_min_opt[0], x_min_opt[1], 'bo', markersize=10, label='Min (Pas Optimal)')
    ax1.plot(x_min_newton[0], x_min_newton[1], 'go', markersize=10, label='Min (Newton)')
    ax1.plot(x_min_cg[0], x_min_cg[1], 'mo', markersize=10, label='Min (Gradient Conjugué)')
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
    iterations_cg = range(len(f_cg))
    ax2.plot(iterations_fixed, f_fixed, 'r.-', linewidth=2, label='Pas Fixe', markersize=4)
    ax2.plot(iterations_optimal, f_optimal, 'b.-', linewidth=2, label='Pas Optimal', markersize=4)
    ax2.plot(iterations_newton, f_newton, 'g.-', linewidth=2, label='Newton', markersize=4)
    ax2.plot(iterations_cg, f_cg, 'm.-', linewidth=2, label='Gradient Conjugué', markersize=4)
    ax2.set_xlabel('Itérations')
    ax2.set_ylabel('Valeur de la fonction objectif')
    ax2.set_title('Convergence des méthodes d\'optimisation')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Appeler la fonction de visualisation
visualize_optimization_with_contours(f, x0, traj_fixed, traj_optimal, traj_newton,traj_cg, f_fixed, f_optimal, f_newton,f_cg, x_min_fixed, x_min_opt, x_min_newton,x_min_cg)


