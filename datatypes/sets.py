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
