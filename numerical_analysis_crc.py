from mathdyn import *
import yaml
from dataclasses import dataclass

@dataclass
class ParameterSet:
    point: int
    r_H: float
    r_C: float
    r_B: float
    alpha: float
    beta: float
    gamma: float
    g: float
    m_1: float
    m_2: float
    k: float
    k_B: float
    b_0: float

@dataclass
class InitialSet:
    H_nol: float
    C_nol: float
    B_nol: float

@ dataclass
class IntervalSet:
    h : float
    t0 : int
    th : int 

def get_params(param_path, equilibrium):
    with open(param_path, "r") as f:
        config = yaml.safe_load(f)
    return ParameterSet(**config[equilibrium]), InitialSet(**config['initial']), IntervalSet(**config['interval'])


def main():
    params, initial, interval = get_params("params.yaml", "eq4")
    t_val = np.arange(interval.t0, interval.th, interval.h)

    # lower_bound = alpha - m_2/(1+math.exp(g*b_0))
    # upper_bound = gamma*k*(alpha-m_2)/(r_B+gamma*k)*(1+math.exp(g*b_0))
    # print('check bound:', lower_bound, upper_bound)

    H_eq, C_eq, B_eq = equilibrium(params.point, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0, initial.B_nol)
    eigenvalues, re, im = find_jacobian(H_eq, C_eq, B_eq, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0)
    phi = find_phi(eigenvalues, interval.h)
    print("equilibrium point:", f"({H_eq}, {C_eq}, {B_eq})")
    
    # #NSFD Scheme
    find_nsfd(initial.H_nol, initial.C_nol, initial.B_nol, phi, t_val, params.r_H, params.r_C, params. r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0)

    lam1, lam2, lam3 = get_lambda(params.point, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0, initial.B_nol)
    print("lambda 1:", lam1)
    print("lambda 2:", lam2)
    print("lambda 3:", lam3)
    print("eigenvalues: ", eigenvalues)

    
    # print("real parts:", re)
    # print("imaginary parts:", im)

    if(params.point != 4):
        
        results = stability_check(params.point, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0, initial.B_nol)

        print("condition bound rH:", results["boundrH"])
        print("condition bound rC:", results["boundrC"])
        print("condition bound rB:", results["boundrB"])
    else :
        results = stability_check(params.point, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0, initial.B_nol)
        print("bound rH:", results["boundrH"])
        print("S+T:", results["boundA"])
        print("ST-U:", results["boundB"])
        print("B condition:",results["B_condition"])


    # get nsfd
    H_nsfd, C_nsfd, B_nsfd = find_nsfd(initial.H_nol, initial.C_nol, initial.B_nol, phi, t_val, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params. m_1, params.m_2, params.g, params.k, params.k_B, params.b_0)
    
    # get runge kutta
    rk_H, rk_C, rk_B = find_rk(initial.H_nol, initial.C_nol, initial.B_nol, interval.t0, interval.th, t_val, params.r_H, params.r_C, params.r_B, params.alpha, params.beta, params.gamma, params.m_1, params.m_2, params.g, params.k, params.k_B, params.b_0)
    
    # # end point
    # get_endpoint("runge kutta", rk_H, rk_C, rk_B, interval.th)
    # get_endpoint("nsfd",H_nsfd, C_nsfd, B_nsfd, interval.th)

    diff_H, diff_C, diff_B = find_diff(H_nsfd, C_nsfd, B_nsfd, rk_H, rk_C, rk_B, t_val)

    # get max diff
    # max_H = max(diff_H)
    # max_C = max(diff_C)
    # max_B = max(diff_B)
    # print("max diff H, C, B", [max_H, max_C, max_B])
    # plot_all(t_val, H_nsfd, C_nsfd, B_nsfd, rk_H, rk_C, rk_B, diff_H, diff_C, diff_B)
   
if __name__ == "__main__":
    main()