import numpy as np
from scipy.optimize import fsolve
import math
from dataclasses import dataclass
from datatypes.loader import get_params

# Define the equation to solve
def equation(r_C, r_B, alpha, gamma, m_2, g, K, K_B, B_0, B_star):
    C = K - K/r_C*(alpha - m_2*(1/(1+math.exp(-g*(B_star-B_0)))))
    B_condition = r_B*(1-B_star/K_B) + gamma*C
    return B_condition

# Example parameter values (adjust as needed)
params = {
    "r_B": 1,
    "r_C": 1.5,
    "alpha": 1.2,
    "gamma": 0.5,
    "m_2": 0.8,
    "g": 1.0,
    "K": 5.0,
    "K_B": 10,
    "B_0": 2
}

# Solve for B*
B_star_guess = 0.001  # initial guess
B_star_solution = fsolve(equation, B_star_guess, args=tuple(params.values()))

print(f"Solution B* ≈ {B_star_solution[0]:.4f}")
