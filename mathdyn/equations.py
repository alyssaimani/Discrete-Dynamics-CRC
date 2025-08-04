
import numpy as np
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15, 'font.family': 'serif'})

def equilibrium(point, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0, B_star):
    if point == 2:
        H_eq = 0
        C_eq = 0
        B_eq = K_B
        return H_eq, C_eq, B_eq
    
    elif point == 3:
        H_eq = 0
        C_eq = K - K/r_C*(alpha - (m_2/(1 + math.exp(g*B_0))))
        B_eq = 0
        return H_eq, C_eq, B_eq

    elif point == 4:
        H_eq = 0
        C_eq = K - K/r_C*(alpha - m_2*(1/(1+math.exp(-g*(B_star-B_0)))))
        B_eq = B_star
        return H_eq, C_eq, B_eq

    else:
        raise ValueError(f"unknown equilibrium point:{point}")

def find_jacobian(H, C, B, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0):
    sigmoid = math.exp(-g*(B-B_0))
    A11 = r_H*(1-(2*H+C)/K) - beta*C - m_1/(1+sigmoid)
    A12 = -H*(beta + r_H/K)
    A13 = -(m_1*H*g*sigmoid/(1+sigmoid)**2)
    A21 = -(r_C*C/K) + beta*C
    A22 = -(r_C*C/K) + r_C*(1 - (H + C)/K) + beta*H - alpha + (m_2/(1+sigmoid))
    A23 = m_2*C*g*sigmoid/(1+sigmoid)**2
    A31 = 0
    A32 = gamma*B
    A33 = r_B*(1-(2*B/K_B)) + gamma*C

    J = np.array([[A11, A12, A13],
                  [A21, A22, A23], 
                  [A31, A32, A33]])
    eigenvalues = np.linalg.eigvals(J)
    # print('A11, A22, A33', [A11, A22, A33])
    re = eigenvalues.real
    im = eigenvalues.imag
    return eigenvalues, re, im

def find_phi( eigenvalues, h):
    q = np.max([
        (lam**2) / (2 * abs(np.real(lam)))
        for lam in eigenvalues if np.real(lam) != 0
    ])
    phi = (1-math.exp(-q*h))/q
    # print(f"phi: {phi}, q: {q}")
    return phi

def stable_sigmoid(x):
    # Numerically stable sigmoid function
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

        # S = -r_C - 2*r_C*C/K - alpha + m_2/(1+math.exp(-g*(B_star-B_0)))
        # T = r_B*(1 - 2*B_star/K_B) + gamma*C
        # U = (m_2*C*g*math.exp(-g*(B_star-B_0))/(1+math.exp(-g*(B_star-B_0)))**2)*(gamma*B_star)
        # lam1 = (r_H+beta*K)/r_C*(alpha - (m_2/(1 + math.exp(-g*(B_star)))))
        # lam2 = ((S + T) + math.sqrt((S+T)**2 - 4*(S*T-U)))/2
        # lam3 = ((S + T) - math.sqrt((S+T)**2 - 4*(S*T-U)))/2
def get_lambda(point, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0, B_star):
    if point == 2:
        lam1 = r_H - (m_1/(1 + math.exp(-g*(K_B-B_0))))
        lam2 = r_C - alpha + (m_2/(1 + math.exp(-g*(K_B-B_0))))
        lam3 = -r_B
        return lam1, lam2, lam3
    elif point == 3:
        lam1 = ((r_H+beta*K)/r_C)*(alpha - m_2/(1 + math.exp(g*B_0))) - beta*K - m_1/(1+math.exp(g*B_0))
        lam2 = -r_C + alpha - m_2/(1 + math.exp(g*B_0))
        lam3 = r_B + gamma*K - (gamma*K/r_C)*(alpha - m_2/(1+math.exp(g*B_0)))
        return lam1, lam2, lam3
    elif point == 4:
        sigmoid = math.exp(-g*(B_star-B_0))
        C = K - K/r_C*(alpha - (m_2/(1+sigmoid)))
        lam1 = (r_H + beta*K)/r_C * (alpha - (m_2/(1+sigmoid))) - beta*K - (m_1/(1+sigmoid))
        a1 = -r_C*(1 - 2*C/K) + alpha - m_2/(1 + sigmoid)*r_B - (1 - 2*B_star/K_B) - gamma*C
        a2 = (r_C*(1 - 2*C/K) - alpha + m_2/(1 + sigmoid)) * (r_B*(1 - 2*B_star/K_B) + gamma*C) - (gamma*m_2*C*B_star*g*sigmoid / (1 + sigmoid)**2)

        lam2 = (-a1 + math.sqrt(a1**2 - 4*a2))/2
        lam3 = (-a1 - math.sqrt(a1**2 - 4*a2))/2
        
        condition_B = r_B*(1-B_star/K_B) + gamma*C

        print("B value:", condition_B)
        if abs(condition_B) > 1e-3:
            print("Warning: B condition residual is not negligible:", condition_B)
        
        return lam1, lam2, lam3
    else:
        raise ValueError(f"unknown equilibrium point:{point}")

def stability_check(point, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0, B_star):
    if point == 2:
        bound_r_H = m_1/(1+math.exp(-g*(K_B-B_0)))
        bound_r_C = alpha - m_2/(1+math.exp(-g*(K_B-B_0)))
        bound_r_B = r_B
        return {
            "boundrH": bound_r_H, 
            "boundrC": bound_r_C, 
            "boundrB": bound_r_B  
        } 
    elif point == 3:
        bound_r_H = (r_C*(beta*K + m_1/(1+ math.exp(g*B_0)))/(alpha - m_2/(1+ math.exp(g*B_0)))) - beta*K
        bound_r_C = alpha - m_2/(1+math.exp(g*B_0))
        bound_r_B = gamma*K/r_C*(alpha - m_2/(1+ math.exp(g*B_0))) - gamma*K
        return {
            "boundrH": bound_r_H, 
            "boundrC": bound_r_C, 
            "boundrB": bound_r_B  
        } 
    elif point == 4:
        C = K - K/r_C*(alpha - m_2*(1/(1+math.exp(-g*(B_star-B_0)))))
        bound_r_H = (r_C*(beta*K+m_1/(1+math.exp(-g*(B_star-B_0))))/(alpha - (m_2/(1+math.exp(-g*(B_star-B_0)))))) - beta*K
        # bound_r_H = (beta*C + m_1/(1+math.exp(-g*(B_star-B_0))))*(K/(K-C))
        S = -r_C - 2*r_C*C/K - alpha + m_2/(1+math.exp(-g*(B_star-B_0)))
        T = r_B*(1 - 2*B_star/K_B) + gamma*C
        U = (m_2*C*g*math.exp(-g*(B_star-B_0))/(1+math.exp(-g*(B_star-B_0)))**2)*(gamma*B_star)
        bound_A = S + T
        bound_B = S * T - U
        B_condition = r_B*(1-B_star/K_B) + gamma*C
        if abs(B_condition) > 1e-3:
            print("Warning: B condition residual is not negligible:", B_condition)
        
        return {
            "boundrH": bound_r_H, 
            "boundA": bound_A, 
            "boundB": bound_B,
            "B_condition": B_condition  
        } 
    else:
        raise ValueError(f"unknown equilibrium point:{point}")

def find_new(phi, H_n, C_n, B_n, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0):
    sigmoid = (1/(1+math.exp(-g*(B_n - B_0))))
    H_new = phi * (r_H*H_n*(1-(H_n + C_n)/K) - beta*H_n*C_n - m_1*H_n*sigmoid) + H_n
    C_new = phi * (r_C*C_n*(1-(H_n + C_n)/K) + beta*H_n*C_n - alpha*C_n + m_2*C_n*sigmoid) + C_n
    B_new = phi * (r_B*B_n*(1-(B_n/K_B)) + gamma*C_n*B_n) + B_n
    return H_new, C_new, B_new

def find_nsfd(H_nol, C_nol, B_nol, phi, t_val, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0):
        H_nsfd = []
        C_nsfd = []
        B_nsfd = []
        
        H_n = H_nol
        C_n = C_nol
        B_n = B_nol

        for t in t_val:
            H_nsfd.append(H_n)
            C_nsfd.append(C_n)
            B_nsfd.append(B_n)
            H_new, C_new, B_new = find_new(phi, H_n, C_n, B_n, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0)

            H_n = H_new
            C_n = C_new
            B_n = B_new
        return H_nsfd, C_nsfd, B_nsfd

def find_rk(H_nol, C_nol, B_nol, t0, th, t_val, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0):
    y0 = [H_nol, C_nol, B_nol]           # initial value
    p = (r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0)
    sol = solve_ivp(fun=model, t_span=[t0, th], y0=y0, args=p, method='RK45', t_eval=t_val)
    rk_H, rk_C, rk_B = sol.y  
    return rk_H, rk_C, rk_B

def find_diff(H_nsfd, C_nsfd, B_nsfd, rk_H, rk_C, rk_B, t_val):
    diff_H = []
    diff_C = []
    diff_B = []

    for t in range(len(t_val)):
        temp_H = abs(H_nsfd[t]-rk_H[t])
        temp_C = abs(C_nsfd[t]-rk_C[t])
        temp_B = abs(B_nsfd[t]-rk_B[t])
        diff_H.append(temp_H)
        diff_C.append(temp_C)
        diff_B.append(temp_B)
    
    return diff_H, diff_C, diff_B

def get_endpoint(model_name, H_model, C_model, B_model, th):
    print(f"{model_name} H:{H_model[th]}")
    print(f"{model_name} C:{C_model[th]}")
    print(f"{model_name} B:{B_model[th]}")

def model(t, state, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0):
    H, C, B = state
    sigmoid = (1/(1+math.exp(-g*(B - B_0))))
    dH = r_H*H*(1-(H + C)/K) - beta*H*C - m_1*H*sigmoid
    dC = r_C*C*(1-(H + C)/K) + beta*H*C - alpha*C + m_2*C*sigmoid
    dB = r_B*B*(1-(B/K_B)) + gamma*C*B
    return [dH, dC, dB]


def plot_all(t_val, H_nsfd, C_nsfd, B_nsfd, rk_H, rk_C, rk_B, diff_H, diff_C, diff_B):
    fig = plt.figure(figsize=(12, 10))

    # Subplot 1: NSFD time series
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t_val, H_nsfd, color='green', label='H(t): Healthy Cells')
    ax1.plot(t_val, C_nsfd, color='red', label='C(t): Colorectal Cancer Cells')
    ax1.plot(t_val, B_nsfd, color='blue', label='B(t): Pro-inflammatory Bacteria')
    ax1.set_title('NSFD')
    ax1.set_ylabel('Growth')
    ax1.set_xlabel('Time (t)')
    ax1.grid()
    ax1.legend()

    # Subplot 2: Runge-Kutta time series
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_val, rk_H, color='green', label='H(t): Healthy Cells')
    ax2.plot(t_val, rk_C, color='red', label='C(t): Colorectal Cancer Cells')
    ax2.plot(t_val, rk_B, color='blue', label='B(t): Pro-inflammatory Bacteria')
    ax2.set_title('Runge-Kutta')
    ax2.set_xlabel('Time (t)')
    ax2.grid()
    ax2.legend()

    # Subplot 3: Difference
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t_val, diff_H, color='green', label='ΔH: Healthy Cells')
    ax3.plot(t_val, diff_C, color='red', label='ΔC: Colorectal Cancer Cells')
    ax3.plot(t_val, diff_B, color='blue', label='ΔB: Pro-inflammatory Bacteria')
    ax3.set_title('Difference')
    ax3.set_xlabel('Time (t)')
    ax3.grid()
    ax3.legend()

    # Subplot 4: 3D Phase Portrait
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(H_nsfd, C_nsfd, B_nsfd, color='blue', label='NSFD')
    ax4.plot(rk_H, rk_C, rk_B, color='red', label='Runge Kutta')
    ax4.set_title('Phase Portrait')
    ax4.set_xlabel('H(t)')
    ax4.set_ylabel('C(t)')
    ax4.set_zlabel('B(t)')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.savefig("eq4 test.jpg")
    plt.show()
