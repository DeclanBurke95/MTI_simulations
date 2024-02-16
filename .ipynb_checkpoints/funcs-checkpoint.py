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
from systems import make_lead
import warnings


MTI_params = dict(
    hbar=1,
    V=0,
    mu_m=1.6,
    vF=3,#for Bi2Se3-based 3DTI, 10^15 canceled out with hbar
    m0=-0.005,
    m1=15,
    sysx=.0,
    Msysy=.0,
    Mz=0.1,
    Mx=.0,
    My=.0,
    a=10,
    phi_0=1.0,
    exp=np.exp,
    C_m=38.1
)

def get_default_params2(finite=True):
    if finite:
        params =dict(
            mu=0.014823834205238105,
            Msysz=0.015,
            re=lambda x: x.real,
            im=lambda x: x.imag,
            delta=0.005,
            L_normal=1000,
            Delta=lambda x: params['delta'] if x >= params['L_normal'] else 0,
            S_imp=systems.get_S_imp(""),
            Smag_imp=0.0,
            **MTI_params)
    else:
         params =dict(
            mu=0.014823834205238105,
            Msysz=0.015,
            Smag_imp=0.0,
            re=lambda x: x.real,
            im=lambda x: x.imag,
            Delta=0.05,
            S_imp=systems.get_S_imp(""),
            **MTI_params)   
    return params

def normal_conductance(energy, fsyst, params):
    smatrix = kwant.smatrix(fsyst, energy=energy, params = params)
    return smatrix.transmission((1, 0), (0,0))

def calc_energies(fsyst, params, num_orbitals, num_states):
    ham = fsyst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
    energies, states = sl.eigsh(ham, sigma=0, k=num_states)
    return energies, states

def calc_densities(fsyst, params, num_orbitals, num_states):
    ham = fsyst.hamiltonian_submatrix(params=params, sparse=True).tocsc()
    energies, states = sl.eigsh(ham, sigma=0, k=num_states)
    densities = (np.linalg.norm(states.reshape(-1, num_orbitals, num_states), axis=1) ** 2)
    return densities

def tun_con(fsyst, energy, params, lead_in=0, lead_out=0):
    smatrix = kwant.smatrix(fsyst, energy, params=params)
    N_e = smatrix.submatrix((lead_in, 0), (lead_in, 0)).shape[0]
    r_ee = smatrix.transmission((lead_out, 0), (lead_in, 0))
    r_he = smatrix.transmission((lead_out, 1), (lead_in, 0))

    return N_e-r_ee+r_he

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

def bands(k, lead, params, num_bands=20, sigma=0, sort=True):
    h_k = get_h_k(lead, params, bias=0, sparse=True)
    ham = h_k(k)

    if np.isnan(ham.data).any():
        raise Exception(f'{params}')

    if num_bands is None:
        energies, wfs = np.linalg.eig(ham.todense())
    else:
        energies, wfs = sl.eigs(ham, k=num_bands, sigma=sigma)

    if sort:
        energies, wfs = sort_spectrum(energies, wfs)
        energies, wfs = fix_shift(energies, wfs)

    return energies, wfs

def gap_search_k(xy, Delta, W):
    params = get_default_params2(finite=False)
    import time
    from scipy import optimize
    import importlib
    # importlib.reload(funcs)
   
    lead = make_lead(W)
    
    params["Msysz"], params["mu"] = xy
    params["Delta"] = 0
    
    prop_modes, stab_modes = lead.modes(energy=0, params=params)
    momenta = prop_modes.momenta
    
    params['Delta'] = Delta
    
    def E_k(k):
        Es, wfs = bands(k, lead, num_bands=20, sort=False, params=params)
        return np.min(np.abs(Es.real))
    
    def f(energy):
        if energy < 0:
            return 1
        else:
            return gap_minimizer(lead, params, energy) - 1e-13
    
    start = time.perf_counter()
    
    num_modes = momenta.shape[0]
    
    gaps = []
    res = []
    
    E_k0 = E_k(0)
    
    if num_modes == 0:
        #k_start = params['mu_ti']/3*a
        #E_cone = E_k(k_start)

        sol = optimize.root_scalar(f,  x0=E_k0, bracket=[-1e-13, 0.3], method='bisect')

        data = dict(
            gap=abs(sol.root),
            gaps=[],
            momenta=[],
            res=[],
            sol=sol,
            time=time.perf_counter()-start
        )
    else: 
        for i in np.arange(num_modes//4 + (num_modes//2)%2):
            k = momenta[i]
            delta_k = 2.5e-2
            bounds = (k-delta_k, k+delta_k)
            _res = optimize.minimize_scalar(E_k, bounds=bounds, method='bounded')
            res.append(_res)
            gaps.append(_res.fun)
            
        if E_k0 < min(gaps):
            gaps.append(E_k0)
           
        data = dict(
            gap=min(gaps),
            gaps=gaps,
            momenta=momenta,
            res=res,
            time=time.perf_counter()-start
        )
    
    return data

def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.
    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.
    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))

def get_rhos(syst, sum=False):
    p = np.kron(np.kron((np.eye(2) + sigma_z)//2, np.eye(2)), np.eye(2))
    h = np.kron(np.kron((np.eye(2) - sigma_z)//2, np.eye(2)), np.eye(2))
    sx = np.kron(np.kron(np.eye(2), sigma_x), np.eye(2))
    sy = np.kron(np.kron(np.eye(2), sigma_y), np.eye(2))
    sz = np.kron(np.kron(np.eye(2), sigma_z), np.eye(2))

    rohs = dict(
        all=kwant.operator.Density(syst, np.eye(8), sum=sum),
        p=kwant.operator.Density(syst, p, sum=sum),
        h=kwant.operator.Density(syst, h, sum=sum),

        sx=kwant.operator.Density(syst, sx, sum=sum),
        sy=kwant.operator.Density(syst, sy, sum=sum),
        sz=kwant.operator.Density(syst, sz, sum=sum),
    )

    return rohs

def majorana_num(xy, W, Delta):
    warnings.filterwarnings("ignore", category=np.ComplexWarning)
    # params = default_params.copy()
    params=get_default_params2(finite=False)
    import importlib
    import time    
    parmas=get_default_params2()
    # importlib.reload(funcs)

    a=10    
    lead = make_lead(W)
    params["Delta"] = Delta
    params["Msysz"], params["mu"] = xy

    
    return dict(mn=majorana_num2(lead, params))
