import kwant
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.signal
from scipy.optimize import curve_fit
import scipy.signal
from scipy.stats import multivariate_normal
import argparse
import kwant.continuum
import systems
import scipy.sparse.linalg as sl
from scipy.ndimage import shift
from pfapack import pfaffian as pfa

def get_default_finite_params():
    params = dict(hbar=1,
                    mu=0.014823834205238105,
                    V=0,
                    mu_m=1.6,
                    vF=3,#for Bi2Se3-based 3DTI, 10^15 canceled out with hbar
                    m0=-0.005,
                    m1=15,
                    Msysz=0.015,
                    Msysx=.0,
                    Msysy=.0,
                    Mz=0.1,
                    Mx=.0,
                    My=.0,
                    a=10,
                    phi_0=1.0,
                    exp=np.exp,
                    Smag_imp=0.0,
                    # Smag_mimp=.0,
                    # Stheta_mimp=np.pi/2,
                    # Sphi_mimp=.0,
                    disorder_imp=np.zeros([1, 1]),
                    # correlation_length=35,
                    re=lambda x: x.real,
                    im=lambda x: x.imag,
                    Delta=lambda x: 0.005 if x >= 1000 else 0,
                    S_imp=systems.get_S_imp("")
                    )
    return params

def get_default_lead_params():
    params = dict(hbar=1,
                    mu=0.014823834205238105,
                    V=0,
                    mu_m=1.6,
                    vF=3,#for Bi2Se3-based 3DTI, 10^15 canceled out with hbar
                    m0=-0.005,
                    m1=15,
                    Msysz=0.015,
                    Msysx=.0,
                    Msysy=.0,
                    Mz=0.1,
                    Mx=.0,
                    My=.0,
                    a=10,
                    phi_0=1.0,
                    exp=np.exp,
                    Smag_imp=0.0,
                    # Smag_mimp=.0,
                    # Stheta_mimp=np.pi/2,
                    # Sphi_mimp=.0,
                    disorder_imp=np.zeros([1, 1]),
                    # correlation_length=35,
                    re=lambda x: x.real,
                    im=lambda x: x.imag,
                    Delta=0.005,
                    S_imp=systems.get_S_imp("")
                    )
    return params

def normal_conductance(energy, fsyst, params):
    smatrix = kwant.smatrix(fsyst, energy=energy, params = params)
    return smatrix.transmission((1, 0), (0,0))

def calc_energies(fsyst, params, num_orbitals, num_states):
    ham = fsyst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
    energies, states = sl.eigsh(ham, sigma=0, k=num_states)
    densities = (np.linalg.norm(states.reshape(-1, num_orbitals, num_states), axis=1) ** 2)
    return energies, states, densities

def tun_con(fsyst, energy, params, lead_in=0, lead_out=0):
      m_eff=0.1
      params.update(dict(C_m=3.81/m_eff))
      smatrix = kwant.smatrix(fsyst, energy, params=params)
      N_e = smatrix.submatrix((lead_in, 0), (lead_in, 0)).shape[0]
      r_ee = smatrix.transmission((lead_out, 0), (lead_in, 0))
      r_he = smatrix.transmission((lead_out, 1), (lead_in, 0))

      return N_e-r_ee+r_he


# def normal_conductance(energy, fsyst, params, L, W):
#     ham=get_mti_hamiltonian(material='mti', ph_symmetry=False)
#     shape=systems.shape(L, W, v=np.array([0,0])
#     return shape 


def majorana_num(lead, params):
    h_k = get_h_k(lead, params)

    skew_h0 = make_skew_symmetric(h_k(0))
    skew_h_pi = make_skew_symmetric(h_k(np.pi))

    pf_0 = np.sign(cpf(1j * skew_h0, avoid_overflow=True).real)
    pf_pi = np.sign(cpf(1j * skew_h_pi, avoid_overflow=True).real)
    pfaf = pf_0 * pf_pi

    return pfaf

def get_h_k(lead, params, bias=0, sparse=False):
    h, t = cell_mats(lead, params, bias, sparse)

    def h_k(k):
        return h + t * np.exp(1j * k) + t.T.conj() * np.exp(-1j * k)
    return h_k

def cell_mats(lead, params, bias=0, sparse=False):
    h = lead.cell_hamiltonian(params=params, sparse=sparse)
    if sparse:
        h -= bias * np.identity(h.shape[0])
    else:
        h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params, sparse=sparse)
    return h, t

def make_skew_symmetric(ham):
    W = ham.shape[0] // 8
    I = np.eye(4, dtype=complex)
    U = np.bmat([[I, I], [-1j * I, 1j * I]])
    U = np.kron(np.eye(W, dtype=complex), U)

    skew_ham = U @ ham @ U.H

    assert is_antisymmetric(skew_ham)

    return skew_ham

def is_antisymmetric(H):
    return np.allclose(-H, H.T)


def majorana_num2(lead, params):
    h_k = get_h_k(lead, params)

    skew_h0 = make_skew_symmetric(h_k(0))
    skew_h_pi = make_skew_symmetric(h_k(np.pi))

    pf_0 = np.sign(pfa.pfaffian(1j * skew_h0).real)
    pf_pi = np.sign(pfa.pfaffian(1j * skew_h_pi).real)
    pfaf = pf_0 * pf_pi
    
    return pfaf