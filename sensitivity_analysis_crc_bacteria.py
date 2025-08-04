import numpy as np
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.rcParams.update({'font.size': 15, 'font.family': 'serif'})


def equilibrium(K, r_H, K_B, m_1,g, B_0):
# equilibrium point E1
    # H_eq = 0
    # C_eq = 0
    # B_eq = 0

# equilibrium point E2
    H_eq = 0
    C_eq = 0
    B_eq = K_B
    
# # equilibrium point E3
    # H_eq = K - (K/r_H)*(m_1*(1/(1+math.exp(g*B_0))))
    # C_eq = 0
    # B_eq = 0

# # equilibrium point E4   
    # H_eq = K - (m_1*K/r_H)*(1/(1+math.exp(-g*(K_B-B_0))))
    # C_eq = 0
    # B_eq = K_B
    
    return H_eq, C_eq, B_eq

def find_jacobian(H, C, B, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0):
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
                  [A31, A32, A33]])
    eigenvalues = np.linalg.eigvals(J)
    return eigenvalues

def find_phi( eigenvalues, h):
    q = np.max([
        (lam**2) / (2 * abs(np.real(lam)))
        for lam in eigenvalues if np.real(lam) != 0
    ])
    phi = (1-math.exp(-q*h))/q
    # print(f"phi: {phi}, q: {q}")
    return phi

def find_new(phi, H_n, C_n, B_n, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, K, K_B, B_0):
    sigmoid = (1/(1+math.exp(-g*(B_n - B_0))))
    H_new = phi * (r_H*H_n*(1-(H_n + C_n)/K) - beta*H_n*C_n - m_1*H_n*sigmoid) + H_n
    C_new = phi * (r_C*C_n*(1-(H_n + C_n)/K) + beta*H_n*C_n - alpha*C_n + m_2*C_n*sigmoid) + C_n
    B_new = phi * (r_B*B_n*(1-(B_n/K_B)) + gamma*C_n*B_n) + B_n
    return H_new, C_new, B_new

def find_nsfd(H_nol, C_nol, B_nol, t_val, phi, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B, b_0):
    #NSFD Scheme
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

def plot_all(t_val, H_nsfd1, C_nsfd1, B_nsfd1, H_nsfd2, C_nsfd2, B_nsfd2, H_nsfd3, C_nsfd3, B_nsfd3, var, vc1, vc2, vc3):
    fig, axs = plt.subplots(1,1, figsize=(6,5))
    axs.plot(t_val, H_nsfd1, color='red', linestyle = 'dashed', label=f'{var}_H={vc1}')
    axs.plot(t_val, H_nsfd2, color='blue', linestyle = 'dashed', label=f'{var}_H={vc2}')
    axs.plot(t_val, H_nsfd3, color='green', linestyle = 'dashed', label=f'{var}_H={vc3}')
    axs.plot(t_val, C_nsfd1, color='red', label=f'{var}_C={vc1}')
    axs.plot(t_val, C_nsfd2, color='blue', label=f'{var}_C={vc2}')
    axs.plot(t_val, C_nsfd3, color='green', label=f'{var}_C={vc3}')
    axs.plot(t_val, B_nsfd1, color='red',linestyle = 'dotted', label=f'{var}_B={vc1}')
    axs.plot(t_val, B_nsfd2, color='blue',linestyle = 'dotted', label=f'{var}_B={vc2}')
    axs.plot(t_val, B_nsfd3, color='green',linestyle = 'dotted', label=f'{var}_B={vc3}')
    axs.set_title(f'Different {var}')
    axs.set_xlabel('Time (t)')
    axs.set_ylabel('Growth')
    axs.grid()
    axs.legend(loc='center right')
    
    fig.tight_layout()
    # plt.savefig('Sensitivity Analysis mu1.jpg')
    plt.show()

def plot_H(t_val, H_nsfd1,  H_nsfd2, H_nsfd3, var, vc1, vc2, vc3):
    fig, axs = plt.subplots(1,1, figsize=(6,5))
    axs.plot(t_val, H_nsfd1, color='red', label=f'{var}={vc1}')
    axs.plot(t_val, H_nsfd2, color='blue', linestyle = 'dashed', label=f'{var}={vc2}')
    axs.plot(t_val, H_nsfd3, color='green', linestyle = 'dashed', label=f'{var}={vc3}')
    axs.set_title(f'Different {var}')
    axs.set_xlabel('Time (t)')
    axs.set_ylabel('Growth H')
    axs.grid()
    axs.legend(loc='center right')
    
    fig.tight_layout()
    plt.savefig(f'SA kB diff to H growth.jpg')
    # plt.show()

def plot_C(t_val, C_nsfd1,  C_nsfd2, C_nsfd3, var, vc1, vc2, vc3):
    fig, axs = plt.subplots(1,1, figsize=(6,5))
    axs.plot(t_val, C_nsfd1, color='red', label=f'{var}={vc1}')
    axs.plot(t_val, C_nsfd2, color='blue', linestyle = 'dashed', label=f'{var}={vc2}')
    axs.plot(t_val, C_nsfd3, color='green', linestyle = 'dashed', label=f'{var}={vc3}')
    axs.set_title(f'Different {var}')
    axs.set_xlabel('Time (t)')
    axs.set_ylabel('Growth C')
    axs.grid()
    axs.legend(loc='center right')
    
    fig.tight_layout()
    plt.savefig(f'SA kB diff to C growth.jpg')
    # plt.show()

def plot_B(t_val, B_nsfd1,  B_nsfd2, B_nsfd3, var, vc1, vc2, vc3):
    fig, axs = plt.subplots(1,1, figsize=(6,5))
    axs.plot(t_val, B_nsfd1, color='red', label=f'{var}={vc1}')
    axs.plot(t_val, B_nsfd2, color='blue', linestyle = 'dashed', label=f'{var}={vc2}')
    axs.plot(t_val, B_nsfd3, color='green', linestyle = 'dashed', label=f'{var}={vc3}')
    axs.set_title(f'Different {var}')
    axs.set_xlabel('Time (t)')
    axs.set_ylabel('Growth B')
    axs.grid()
    axs.legend(loc='center right')
    
    fig.tight_layout()
    plt.savefig(f'SA kB diff to B growth.jpg')
    # plt.show()

def main():
    h = 0.002
    t0 = 0
    th = 100
    t_val = np.arange(t0, th, h)

    H_nol = 1e-1
    C_nol = 35e-2
    B_nol = 1e-1

    # Eq2 (gamma)
    # r_H = 0.41
    # r_C = 0.1
    # r_B = 0.01
    # alpha = 0.2
    # beta = 1e-3
    # gamma1 = 0.6
    # gamma2 = 0.5
    # gamma3 = 0.4
    # g = 0.1
    # m_1 = 0.8
    # m_2 = 0.1
    # k = 1e6
    # k_B = 10
    # b_0 = 3

    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1, g, b_0)

    # eigenvalues1 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma1, m_1, m_2, g, k, k_B, b_0)
    # eigenvalues2 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma2, m_1, m_2, g, k, k_B, b_0)
    # eigenvalues3 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma3, m_1, m_2, g, k, k_B, b_0)
    
    # phi1 = find_phi(eigenvalues1, h)
    # phi2 = find_phi(eigenvalues2, h)
    # phi3 = find_phi(eigenvalues3, h)
    
    
    # # #NSFD Scheme
    
    # H_nsfd1, C_nsfd1, B_nsfd1 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi1, r_H, r_C, r_B, alpha, beta, gamma1, m_1, m_2, g, k, k_B, b_0)
    # H_nsfd2, C_nsfd2, B_nsfd2 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi2, r_H, r_C, r_B, alpha, beta, gamma2, m_1, m_2, g, k, k_B, b_0)
    # H_nsfd3, C_nsfd3, B_nsfd3 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi3, r_H, r_C, r_B, alpha, beta, gamma3, m_1, m_2, g, k, k_B, b_0)

    # plot_all(t_val, H_nsfd1, C_nsfd1, B_nsfd1, H_nsfd2, C_nsfd2, B_nsfd2, H_nsfd3, C_nsfd3, B_nsfd3, var=r"$\gamma$", vc1=gamma1, vc2=gamma2, vc3=gamma3)

    #300
    # plot_H(t_val, H_nsfd1, H_nsfd2, H_nsfd3, var=r"$\gamma$", vc1=gamma1, vc2=gamma2, vc3=gamma3)
    # plot_B(t_val, B_nsfd1, B_nsfd2, B_nsfd3, var=r"$\gamma$", vc1=gamma1, vc2=gamma2, vc3=gamma3)
    #100
    # plot_C(t_val, C_nsfd1, C_nsfd2, C_nsfd3, var=r"$\gamma$", vc1=gamma1, vc2=gamma2, vc3=gamma3)
    

    # Eq2 (mu1)
    # r_H = 0.41
    # r_C = 0.1
    # r_B = 0.01
    # alpha = 0.2
    # beta = 1e-3
    # gamma = 0.6
    # g = 0.1
    # m_1_1 = 0.8
    # m_1_2 = 0.7
    # m_1_3 = 0.9
    # m_2 = 0.1
    # k = 1e6
    # k_B = 10
    # b_0 = 3

    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1_1, g, b_0)
    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1_2, g, b_0)
    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1_3, g, b_0)

    # eigenvalues1 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1_1, m_2, g, k, k_B, b_0)
    # eigenvalues2 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1_2, m_2, g, k, k_B, b_0)
    # eigenvalues3 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1_3, m_2, g, k, k_B, b_0)
    
    # phi1 = find_phi(eigenvalues1, h)
    # phi2 = find_phi(eigenvalues2, h)
    # phi3 = find_phi(eigenvalues3, h)
    
    # #NSFD Scheme
    
    # H_nsfd1, C_nsfd1, B_nsfd1 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi1, r_H, r_C, r_B, alpha, beta, gamma, m_1_1, m_2, g, k, k_B, b_0)
    # H_nsfd2, C_nsfd2, B_nsfd2 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi2, r_H, r_C, r_B, alpha, beta, gamma, m_1_2, m_2, g, k, k_B, b_0)
    # H_nsfd3, C_nsfd3, B_nsfd3 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi3, r_H, r_C, r_B, alpha, beta, gamma, m_1_3, m_2, g, k, k_B, b_0)

    # plot_all(t_val, H_nsfd1, C_nsfd1, B_nsfd1, H_nsfd2, C_nsfd2, B_nsfd2, H_nsfd3, C_nsfd3, B_nsfd3, var=r"$\mu_1$", vc1=m_1_1, vc2=m_1_2, vc3=m_1_3)

    #300
    # plot_H(t_val, H_nsfd1, H_nsfd2, H_nsfd3,  var=r"$\mu_1$", vc1=m_1_1, vc2=m_1_2, vc3=m_1_3)
    #100
    # plot_C(t_val, C_nsfd1, C_nsfd2, C_nsfd3,  var=r"$\mu_1$", vc1=m_1_1, vc2=m_1_2, vc3=m_1_3)
    # plot_B(t_val, B_nsfd1, B_nsfd2, B_nsfd3,  var=r"$\mu_1$", vc1=m_1_1, vc2=m_1_2, vc3=m_1_3)

    # Eq2 (mu2)
    # r_H = 0.41
    # r_C = 0.1
    # r_B = 0.01
    # alpha = 0.2
    # beta = 1e-3
    # gamma = 0.6
    # g = 0.1
    # m_1 = 0.8
    # m_2_1 = 0.1
    # m_2_2 = 0.01
    # m_2_3 = 0.05
    # k = 1e6
    # k_B = 10
    # b_0 = 3

    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1, g, b_0)
    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1, g, b_0)
    # H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B, m_1, g, b_0)

    # eigenvalues1 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2_1, g, k, k_B, b_0)
    # eigenvalues2 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2_2, g, k, k_B, b_0)
    # eigenvalues3 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2_3, g, k, k_B, b_0)
    
    # phi1 = find_phi(eigenvalues1, h)
    # phi2 = find_phi(eigenvalues2, h)
    # phi3 = find_phi(eigenvalues3, h)
    
    # #NSFD Scheme
    
    # H_nsfd1, C_nsfd1, B_nsfd1 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi1, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2_1, g, k, k_B, b_0)
    # H_nsfd2, C_nsfd2, B_nsfd2 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi2, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2_2, g, k, k_B, b_0)
    # H_nsfd3, C_nsfd3, B_nsfd3 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi3, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2_3, g, k, k_B, b_0)

    # plot_all(t_val, H_nsfd1, C_nsfd1, B_nsfd1, H_nsfd2, C_nsfd2, B_nsfd2, H_nsfd3, C_nsfd3, B_nsfd3, var=r"$\mu_2$", vc1=m_2_1, vc2=m_2_2, vc3=m_2_3)

    #300
    # plot_H(t_val, H_nsfd1, H_nsfd2, H_nsfd3,  var=r"$\mu_2$", vc1=m_2_1, vc2=m_2_2, vc3=m_2_3)
    # plot_B(t_val, B_nsfd1, B_nsfd2, B_nsfd3,  var=r"$\mu_2$", vc1=m_2_1, vc2=m_2_2, vc3=m_2_3)
    #100
    # plot_C(t_val, C_nsfd1, C_nsfd2, C_nsfd3,  var=r"$\mu_2$", vc1=m_2_1, vc2=m_2_2, vc3=m_2_3)

    # Eq2 (kB)
    r_H = 0.41
    r_C = 0.1
    r_B = 0.01
    alpha = 0.2
    beta = 1e-3
    gamma = 0.6
    g = 0.1
    m_1 = 0.8
    m_2 = 0.1
    k = 1e6
    k_B_1 = 10
    k_B_2 = 15
    k_B_3 = 20
    b_0 = 3

    H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B_1, m_1, g, b_0)
    H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B_2, m_1, g, b_0)
    H_eq, C_eq, B_eq = equilibrium(k, r_H, k_B_3, m_1, g, b_0)

    eigenvalues1 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B_1, b_0)
    eigenvalues2 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B_2, b_0)
    eigenvalues3 = find_jacobian(H_eq, C_eq, B_eq, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B_3, b_0)
    
    phi1 = find_phi(eigenvalues1, h)
    phi2 = find_phi(eigenvalues2, h)
    phi3 = find_phi(eigenvalues3, h)
    
    #NSFD Scheme
    
    H_nsfd1, C_nsfd1, B_nsfd1 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi1, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B_1, b_0)
    H_nsfd2, C_nsfd2, B_nsfd2 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi2, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B_2, b_0)
    H_nsfd3, C_nsfd3, B_nsfd3 = find_nsfd(H_nol, C_nol, B_nol, t_val, phi3, r_H, r_C, r_B, alpha, beta, gamma, m_1, m_2, g, k, k_B_3, b_0)

    plot_all(t_val, H_nsfd1, C_nsfd1, B_nsfd1, H_nsfd2, C_nsfd2, B_nsfd2, H_nsfd3, C_nsfd3, B_nsfd3, var=r"$K_B$", vc1=k_B_1, vc2=k_B_2, vc3=k_B_3)

    #300
    # plot_B(t_val, B_nsfd1, B_nsfd2, B_nsfd3,  var=r"$K_B$", vc1=k_B_1, vc2=k_B_2, vc3=k_B_3)
    #100
    plot_H(t_val, H_nsfd1, H_nsfd2, H_nsfd3,  var=r"$K_B$", vc1=k_B_1, vc2=k_B_2, vc3=k_B_3)
    plot_C(t_val, C_nsfd1, C_nsfd2, C_nsfd3,  var=r"$K_B$", vc1=k_B_1, vc2=k_B_2, vc3=k_B_3)

    print("eigenvalues sim1", eigenvalues1)
    print("eigenvalues sim2", eigenvalues2)
    print("eigenvalues sim3", eigenvalues3)
   
main()