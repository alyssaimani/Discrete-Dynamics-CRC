import numpy as np
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.rcParams.update({'font.size': 15, 'font.family': 'serif'})


def equilibrium(point, param, B_star):
    r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0 = param

    if point == 2:
        return 0, 0, K_B
      
    if point == 3:
        C_eq = K - K/r_C*(alpha - (m_2/(1 + math.exp(g*B_0))))
        return 0, C_eq, 0

    if point == 4:
        C_eq = K - K/r_C*(alpha - m_2*(1/(1+math.exp(-g*(B_star-B_0)))))
        return 0, C_eq, B_star
    
    raise ValueError("Invalid equilibrium point. Only 2, 3, or 4 are supported.")

def find_jacobian(H, C, B, param):
    r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0 = param
    A11 = r_H*(1-(2*H+C)/K) - beta*C - m_1/(1+math.exp(-g*(K_B-B_0)))
    A12 = -H*(beta + r_H/K)
    A13 = -(m_1*H*g*math.exp(-g*(B-B_0))/(1+math.exp(-g*(B-B_0)))**2)
    A21 = -(r_C*C/K) + beta*C
    A22 = -(r_C*C/K) + r_C*(1 - (H + C)/K) + beta*H - alpha + (m_2/(1+math.exp(-g*(B-B_0))))
    A23 = m_2*C*g*math.exp(-g*(B-B_0))/(1+math.exp(-g*(B-B_0)))**2
    A31 = 0
    A32 = gamma*B
    A33 = r_B*(1-2*B/K_B) + gamma*C

    J = np.array([[A11, A12, A13],
                  [A21, A22, A23], 
                  [A31, A32, A33]]) # type: ignore
    
    eigenvalues = np.linalg.eigvals(J)
    print('A11, A22, A33', [A11, A22, A33])
    re = eigenvalues.real
    im = eigenvalues.imag
    return eigenvalues, re, im

def find_phi( eigenvalues, h):
    q = np.max([
        (lam**2) / (2 * abs(np.real(lam)))
        for lam in eigenvalues if np.real(lam) != 0
    ])
    phi = (1-math.exp(-q*h))/q
    return phi

def find_new(phi, H_n, C_n, B_n, param):
    r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0 = param
    sigmoid = (1/(1+math.exp(-g*(B_n - B_0))))
    H_new = phi * (r_H*H_n*(1-(H_n + C_n)/K) - beta*H_n*C_n - m_1*H_n*sigmoid) + H_n
    C_new = phi * (r_C*C_n*(1-(H_n + C_n)/K) + beta*H_n*C_n - alpha*C_n + m_2*C_n*sigmoid) + C_n
    B_new = phi * (r_B*B_n*(1-(B_n/K_B)) + gamma*C_n*B_n) + B_n
    return H_new, C_new, B_new

def model(t, state, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0):
    H, C, B = state
    sigmoid = (1/(1+math.exp(-g*(B - B_0))))
    dH = r_H*H*(1-(H + C)/K) - beta*H*C - m_1*H*sigmoid
    dC = r_C*C*(1-(H + C)/K) + beta*H*C - alpha*C + m_2*C*sigmoid
    dB = r_B*B*(1-(B/K_B)) + gamma*C*B
    return [dH, dC, dB]

def main():
    # initialize parameter
    h = 0.002
    t0, th = 0, 100
    t_val = np.arange(t0, th, h)
    # Eq2
    point = 2
    r_H, r_C, r_B = 0.41, 0.1, 0.01
    alpha, beta, gamma = 0.2, 1e-3, 0.6
    m_1, m_2 = 0.8, 0.1
    k, k_B = 1e6, 10
    g = 0.1
    b_0 = 3

    param = (r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0)

    # initialize point
    # H_nol, C_nol, B_nol = 1e-1, 35e-2, 1e-1
    H_nol, C_nol, B_nol = 1,1,1

    # get equilibrium point and eigenvalues
    H_eq, C_eq, B_eq = equilibrium(point, param, B_nol)
    eigenvalues, re, im = find_jacobian(H_eq, C_eq, B_eq, param)
    phi = find_phi(eigenvalues, h)
    print("equilibrium point:", f"({H_eq}, {C_eq}, {B_eq})")
    print("eigenvalues: ", eigenvalues)
    print("real parts:", re)
    print("imaginary parts:", im)

    B_val = np.arange(t0, th, h)

    def H_model1(param, B_val):
        r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0 = param
        H_value = k - k/r_H*((m_1/(1+math.exp(-g*(B_val-b_0))) - (r_B*beta/gamma)*(1-beta/k_B))) + r_B/gamma * (1 - B_val/k_B)
        return H_value
    def H_model2(param, B_val):
        r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0 = param
        H_value = (alpha/beta - r_C/beta - r_C*r_B/beta*k*gamma * (1- beta/k_B) - m_2/beta*(1/(1+math.exp(-g*(B_val-b_0)))))*(beta*k/(beta*k - r_C)) 
        return H_value
    
    H_1, H_2 = [], []
    for i in B_val:
        H_1.append(H_model1(param, i))
        H_2.append(H_model2(param, i))

    # plot 2 population growth
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(B_val, H_1, color="r")
    ax1.plot(B_val, H_2, color="b")
    ax1.set_xlabel('B')
    ax1.set_ylabel('C')
    ax1.set_title('B vs C')
    ax1.grid()
    ax1.legend()

    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.plot(dH, dB, color="b")
    # ax2.set_xlabel('dH')
    # ax2.set_ylabel('dB')
    # ax2.set_title('dH vs dB')
    # ax2.grid()
    # ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()