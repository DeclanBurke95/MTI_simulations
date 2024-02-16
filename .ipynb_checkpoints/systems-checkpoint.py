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

def get_mti_hamiltonian(ph_symmetry=True, disorder=True):
    a=10
    if ph_symmetry:
        if disorder:
            ham = (
            "-mu * kron(sigma_z, sigma_0, sigma_0) +"
            "hbar * vF * k_y * kron(sigma_0, sigma_x, sigma_z) - "
            "hbar * vF * k_x * kron(sigma_z, sigma_y, sigma_z) + "
            "{m} * kron(sigma_z, sigma_0, sigma_x) + " # top-bottom surface coupling
            "Msysz * kron(sigma_z, sigma_z, sigma_0) + " # magnetization term
            " - re({Delta}) * (kron(sigma_y, sigma_y,sigma_0) + kron(sigma_y, sigma_y,sigma_z))/2 - "
            "im({Delta}) * (kron(sigma_x, sigma_y,sigma_0) + kron(sigma_x, sigma_y,sigma_z))/2 +"
            "S_imp(site, Smag_imp, disorder_imp) * kron(sigma_z, sigma_0, sigma_0)")  
        
        else:
            ham=(
            "-mu * kron(sigma_z, sigma_0, sigma_0) +"
            "hbar * vF * k_y * kron(sigma_0, sigma_x, sigma_z) - "
            "hbar * vF * k_x * kron(sigma_z, sigma_y, sigma_z) + "
            "{m} * kron(sigma_z, sigma_0, sigma_x) + " # top-bottom surface coupling
            "Msysz * kron(sigma_z, sigma_z, sigma_0) + " # magnetization term
            " - re({Delta}) * (kron(sigma_y, sigma_y,sigma_0) + kron(sigma_y, sigma_y,sigma_z))/2 - "
            "im({Delta}) * (kron(sigma_x, sigma_y,sigma_0) + kron(sigma_x, sigma_y,sigma_z))/2")
            
    else:
        if disorder:
            ham = (
            "hbar * vF * k_y * kron(sigma_x, sigma_z) - "
            "hbar * vF * k_x * kron(sigma_y, sigma_z) + "
            "{m} * kron(sigma_0, sigma_x) + " 
            "Msysz * kron(sigma_z, sigma_0) +"
            "S_imp(site, Smag_imp, disorder_imp) * kron(sigma_0, sigma_0)") 
        else:
            ham = (
            "hbar * vF * k_y * kron(sigma_x, sigma_z) - "
            "hbar * vF * k_x * kron(sigma_y, sigma_z) + "
            "{m} * kron(sigma_0, sigma_x) + " 
            "Msysz * kron(sigma_z, sigma_0)")
            
    return ham


def hamiltonian_tip():
    hamiltonian_tip =  (
    "(-mu_m + V) * kron(sigma_z, sigma_0, sigma_0) + "
    "C_m * (k_x**2 + k_y**2) * kron(sigma_z, sigma_0, sigma_0)"
    )
    return hamiltonian_tip

def shape(L, W, v=np.array([0,0])):
    L_start, W_start = 0, 0
    L_stop, W_stop = L, W
    
    def _shape(site):
        (x, y) = site.pos - v
        return (0 <= x <= L and 0 <= y <= W)
    
    return _shape, v + np.array([L_stop, W_stop])

def shape_pointcontact(W):
    def _shape(site):
        (x, y) = site.pos
        return y==W//2

    return _shape, (0,W//2)

def syst_create_mti(L, W, dis):
    a=10
    W=W
    L=L
    syst = kwant.Builder()
    L_start, W_start = 0, 0
    L_stop, W_stop = L, W
    ham=get_mti_hamiltonian(ph_symmetry=Flase, disorder=bool(dis))
    m = "(m0 - m1 * (k_x**2 + k_y**2))"
    hamiltonian_MTI = ham.format(m=m)
    ham_MTI_sys_discretized = kwant.continuum.discretize(hamiltonian_MTI, grid=a)
    ham_MTI_leads_discretized = kwant.continuum.discretize(hamiltonian_MTI, grid=a)
    syst.fill(ham_MTI_sys_discretized, *shape(L,W))
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    # lead_left = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((a, 0)))  
    lead_left.fill(ham_MTI_leads_discretized, *shape(L, W))
    lead_right.fill(ham_MTI_leads_discretized, *shape(L, W))

    print(syst)
    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)
    fsyst=syst.finalized()
    
    return fsyst

def syst_create_smti_tunnelling(L, W, dis):
    a=10
    W=W
    L=L
    tau_z = np.kron(np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)), np.eye(2))
    syst = kwant.Builder()
    L_start, W_start = 0, 0
    L_stop, W_stop = L, W
    ham=get_mti_hamiltonian(ph_symmetry=True, disorder=bool(dis))
    ham_tip_discretized = kwant.continuum.discretize(hamiltonian_tip(), grid=a)
    m = "(m0 - m1 * (k_x**2 + k_y**2))"
    hamiltonian_SMTI = ham.format(m=m, Delta="Delta(x)")
    ham_SMTI_sys_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    ham_SMTI_leads_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    syst.fill(ham_SMTI_sys_discretized, *shape(L,W))
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), conservation_law=-tau_z)
    # lead_left = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((a, 0)))  
    lead_left.fill(ham_tip_discretized, *shape_pointcontact(W))
    lead_right.fill(ham_SMTI_leads_discretized, *shape(L, W))


#     print(syst)
    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)
    fsyst=syst.finalized()
    
    return fsyst

    
def make_mti_lead(W, ph_symmetry=True):
    a = 10
    if ph_symmetry:
        ham = (
            "-mu * kron(sigma_z, sigma_0, sigma_0) +"
            "hbar * vF * k_y * kron(sigma_0, sigma_z, sigma_x) - "
            "hbar * vF * k_x * kron(sigma_z, sigma_z, sigma_y) + "
            "{m} * kron(sigma_z, sigma_x, sigma_0) + " # top-bottom surface coupling
            "Msysz * kron(sigma_z, sigma_0, sigma_z) + " # magnetization term
            " - re(Delta) * (kron(sigma_y, sigma_0,sigma_y) + kron(sigma_y, sigma_z,sigma_y))/2 - "
            "im(Delta) * (kron(sigma_x, sigma_0,sigma_y) + kron(sigma_x, sigma_z,sigma_y))/2"
        )

        m = "(m0 - m1 * (k_x**2 + k_y**2))"

        hamiltonian_SMTI = ham.format(m=m)
        ham_SMTI_sys_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)


        def shape(W, v=np.array([0,0])):
            W_start = 0
            W_stop = W

            def _shape(site):
                (x, y) = site.pos - v
                return (0 <= y <= W)

            return _shape, v + np.array([W_stop])

        lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
        lead.fill(ham_SMTI_sys_discretized, *shape(W))
    
    else:
        ham = (
            "hbar * vF * k_y * kron(sigma_x, sigma_z) - "
            "hbar * vF * k_x * kron(sigma_y, sigma_z) + "
            "{m} * kron(sigma_0, sigma_x) + " 
            "Msysz * kron(sigma_z, sigma_0)"
        )
        
        m = "(m0 - m1 * (k_x**2 + k_y**2))"
        hamiltonian_MTI = ham.format(m=m)
        ham_MTI_sys_discretized = kwant.continuum.discretize(hamiltonian_MTI, grid=a)


        def shape(W, v=np.array([0,0])):
            W_start = 0
            W_stop = W

            def _shape(site):
                (x, y) = site.pos - v
                return (0 <= y <= W)

            return _shape, v + np.array([W_stop])

        lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
        lead.fill(ham_MTI_sys_discretized, *shape(W))
        
    return lead.finalized()


def syst_create_smti(L, W, dis):
    a=10
    W=W
    L=L
    syst = kwant.Builder()
    L_start, W_start = 0, 0
    L_stop, W_stop = L, W
    ham=get_mti_hamiltonian(ph_symmetry=True, disorder=bool(dis))
    m = "(m0 - m1 * (k_x**2 + k_y**2))"
    hamiltonian_SMTI = ham.format(m=m, Delta="Delta(x)")
    ham_SMTI_sys_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    syst.fill(ham_SMTI_sys_discretized, *shape(L,W))
    fsyst=syst.finalized()
    return fsyst

def make_lead(W):
    a = 10
    hamiltonian_SMTI = (
        "-mu * kron(sigma_z, sigma_0, sigma_0) +"
        "hbar * vF * k_y * kron(sigma_0, sigma_z, sigma_x) - "
        "hbar * vF * k_x * kron(sigma_z, sigma_z, sigma_y) + "
        "{m} * kron(sigma_z, sigma_x, sigma_0) + " # top-bottom surface coupling
        "Msysz * kron(sigma_z, sigma_0, sigma_z) + " # magnetization term
        " - re(Delta) * (kron(sigma_y, sigma_0,sigma_y) + kron(sigma_y, sigma_z,sigma_y))/2 - "
        "im(Delta) * (kron(sigma_x, sigma_0,sigma_y) + kron(sigma_x, sigma_z,sigma_y))/2"
    )
    
    m = "(m0 - m1 * (k_x**2 + k_y**2))"
    
    hamiltonian_SMTI = hamiltonian_SMTI.format(m=m)
    ham_SMTI_sys_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    
    
    def shape(W, v=np.array([0,0])):
        W_start = 0
        W_stop = W
    
        def _shape(site):
            (x, y) = site.pos - v
            return (0 <= y <= W)
    
        return _shape, v + np.array([W_stop])

    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead.fill(ham_SMTI_sys_discretized, *shape(W))
    return lead.finalized()

def get_disorder(n_dis, correlation_length, L, W, L_normal):

    correlation_length=correlation_length
    a=10
    n_dis = n_dis
    W=W
    L=L
    L_normal=L_normal

    if (correlation_length >= 1):
        x, y = np.mgrid[-3*correlation_length:3*correlation_length+1:1, -3*correlation_length:3*correlation_length+1:1]
        pos = np.stack((x, y), axis=2)
        rv = multivariate_normal([0, 0], [[correlation_length**2, 0], [0, correlation_length**2]])
        filter_kernel = rv.pdf(pos)


        disorder = np.array([[kwant.digest.gauss('('+str(ind_x) + ',' + str(ind_y) +')', salt=str(n_dis))
                               for ind_x in range(L//a+1)]
                             for ind_y in range(W//a+1)])
        disorder = scipy.signal.fftconvolve(disorder, filter_kernel, mode='same')
        disorder_imp = disorder/np.std(disorder.flatten())


    else:
        disorder_imp = np.array([[kwant.digest.gauss('('+str(ind_x) + ',' + str(ind_y) +')', salt=str(n_dis))
                               for ind_x in range(L//a+1)]
                            for ind_y in range(W//a+1)])
       
    return disorder_imp


def get_S_imp(salt):
    a=10
    def S_imp(site, Smag_imp, disorder_imp):
        a=10
        ind_x = int(site.pos[0]/a)
        ind_y = int(site.pos[1]/a)
        if (ind_x >= disorder_imp.shape[1]) or (ind_y >= disorder_imp.shape[0]):
            #print('Out of bounds. No disorder applied to site.')
            return 0
        else:
            return Smag_imp * disorder_imp[ind_y, ind_x]
    return S_imp



def syst_create_smti(L, W, dis):
    a=10
    W=W
    L=L
    syst = kwant.Builder()
    L_start, W_start = 0, 0
    L_stop, W_stop = L, W
    ham=get_mti_hamiltonian(ph_symmetry=True, disorder=bool(dis))
    m = "(m0 - m1 * (k_x**2 + k_y**2))"
    hamiltonian_SMTI = ham.format(m=m, Delta="Delta(x)")
    ham_SMTI_sys_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    syst.fill(ham_SMTI_sys_discretized, *shape(L,W))
    fsyst=syst.finalized()
    return fsyst


def syst_create_smti_tunnelling(L, W, dis):
    a=10
    W=W
    L=L
    tau_z = np.kron(np.kron(np.array([[1, 0], [0, -1]]), np.eye(2)), np.eye(2))
    syst = kwant.Builder()
    L_start, W_start = 0, 0
    L_stop, W_stop = L, W
    ham=get_mti_hamiltonian(ph_symmetry=True, disorder=bool(dis))
    ham_tip_discretized = kwant.continuum.discretize(hamiltonian_tip(), grid=a)
    m = "(m0 - m1 * (k_x**2 + k_y**2))"
    hamiltonian_SMTI = ham.format(m=m, Delta="Delta(x)")
    ham_SMTI_sys_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    ham_SMTI_leads_discretized = kwant.continuum.discretize(hamiltonian_SMTI, grid=a)
    syst.fill(ham_SMTI_sys_discretized, *shape(L,W))
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)), conservation_law=-tau_z)
    # lead_left = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead_right = kwant.Builder(kwant.TranslationalSymmetry((a, 0)))  
    lead_left.fill(ham_tip_discretized, *shape_pointcontact(W))
    lead_right.fill(ham_SMTI_leads_discretized, *shape(L, W))


#     print(syst)
    syst.attach_lead(lead_left)
    syst.attach_lead(lead_right)
    fsyst=syst.finalized()
    
    return fsyst