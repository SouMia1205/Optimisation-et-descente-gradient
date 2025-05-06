# Comparaison de Méthodes d'Optimisation

Ce projet Python implémente et compare plusieurs algorithmes d'optimisation pour minimiser une fonction multivariée. Les méthodes incluent la Descente de Gradient (avec pas fixe et optimal), la Méthode de Newton, et le Gradient Conjugué. Le code comprend également des outils de visualisation pour comparer les performances et la convergence de ces méthodes.

## Fonctionnalités

- **Descente de Gradient avec Pas Fixe** : Descente de gradient classique avec un taux d'apprentissage constant.
- **Descente de Gradient avec Pas Optimal** : Descente de gradient où la taille du pas est optimisée à chaque itération.
- **Méthode de Newton** : Optimisation du second ordre utilisant la matrice hessienne.
- **Méthode du Gradient Conjugué** : Méthode itérative adaptée pour l'optimisation.
- **Vérification Symbolique de Convexité** : Vérifie si la fonction objectif est convexe en utilisant le calcul symbolique.
- **Visualisation** : Compare les trajectoires d'optimisation et les taux de convergence.

## Prérequis

- Python 3.x
- NumPy
- SymPy
- Matplotlib
- SciPy

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votrenom/methodes-optimisation.git
   cd methodes-optimisation
   ```

2. Installez les packages requis :
   ```bash
   pip install numpy sympy matplotlib scipy
   ```

3. Utilisation : Exécutez simplement le script Python :
   ```bash
   python main.py
   ```

## Fonction Objectif

La fonction à minimiser est :   f(x1, x2, x3) = x1⁴ + x1² + 2x2² + x3² - 6x1 + 3x2 - 2x3(x1 + x2 + 1)
