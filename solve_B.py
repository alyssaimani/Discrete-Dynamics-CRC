import numpy as np
from scipy.optimize import fsolve

# Define the equation to solve
def equation(B_star, r_B, K_B, S, gamma, K, r_c, alpha, mu_2, g, B_0):
    term1 = r_B * (1 - B_star / K_B) * S
    sigmoid = 1 / (1 + np.exp(-g * (B_star - B_0)))
    term2 = gamma * (K - (K / r_c) * (alpha - mu_2 * sigmoid))
    return term1 + term2

# Example parameter values (adjust as needed)
params = {
    "r_B": 1.0,
    "K_B": 10.0,
    "S": 1.0,
    "gamma": 0.5,
    "K": 5.0,
    "r_c": 1.5,
    "alpha": 1.2,
    "mu_2": 0.8,
    "g": 1.0,
    "B_0": 2.0
}

# Solve for B*
B_star_guess = 1.0  # initial guess
B_star_solution = fsolve(equation, B_star_guess, args=tuple(params.values()))

print(f"Solution B* = {B_star_solution[0]:.4f}")
