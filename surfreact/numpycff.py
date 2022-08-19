import numpy as np
import time
import logging
import surfreact.reactions as rx
from joblib import Parallel, delayed
from surfreact import utils as ut
import pickle
import pandas as pd
import mpmath as mp

def flux(x, Ham, x_div_surf=0):
    """
    Computes the flux operator for the given hamiltonian and barrier location. The flux operator is defined as the commutator [H,theta]. Theta is the heaviside function about the location of the dividing surface
    
    Parameters:
    -----------
    x (np array) - 1D spatial grid
    Ham (np array, NxN) - Hamiltonian represented in DVR 
    x_div_surf - location of the dividing surface between products/reactants. Default zero for eckart barriers. 
    
    Returns:
    --------
    flux - flux operator matrix for system
    """
    
    N = len(x)
    
    h_vec = np.zeros(N, dtype= np.float64)
    heavyH = np.zeros((N,N), dtype= np.float64)
    Hheavy = np.zeros((N,N), dtype= np.float64)
    flux = np.zeros((N,N), dtype= np.float64)

    for i in range(0, N):
        if x[i] < x_div_surf:
            h_vec[i] = np.float64(0.0)
        else:
            h_vec[i] = np.float64(1.0)

    for i in range(0, N):
        if h_vec[i] > 0.0:
            for j in range(0, N):
                Hheavy[j,i] = Ham[j,i]
                heavyH[i,j] = Ham[i,j]

    flux = Hheavy - heavyH

    return flux



def boltzmann(beta, evecs, evals):
    """
    Boltzmann operator computer using mpmath library functions to get a higher precision. I think that this computation is causing errors at low temperatures, so improving accuracy here may help

    Takes in evecs and evals as np arrays, returns bmn as an mpmath matrix object.
    """
    N = len(evecs)
    evalexp = np.eye(N,N)*np.exp(-0.5*beta*evals)

    bmnRight = np.matmul(evalexp, evecs.transpose())
    
    bmn = np.matmul(evecs, bmnRight)
    return bmn


def boltflux(flux, bmn):
    """
    Compute the boltzmannized flux matrix
    """
    boltflux = bmn @ (flux @ bmn )
    return boltflux


def time_evolution(t, evecs, evals, N, hbar=1):
    """
    Computes the time evolution operator U for hamiltonian H using the eigenvectors and eigenvalues associated with H.
    Time evolution operator is defined as exp(-iHt/hbar)
    Computes exponential using eigenvalues
    
    Parameters:
    -----------
    t (float) - time to compute U at
    evecs (np array) - eigenvectors associated with H
    evals (np array) - eigenvalues associated with H
    N - length of H
    hbar - Reduced Planck's constant. Equal to one for Atomic Units (default 1)
    
    Returns:
    --------
    U - time evolution operator (matrix)
    """

    N = evecs.shape[0]

    evalexp = np.eye(N,N)*np.exp(-1.j*t*evals/hbar)
    Uright = np.matmul(evalexp, evecs.transpose())
    U = np.matmul(evecs, Uright)

    return U


def fluxU(U, flux):
    """
    Computes the time-evolution component of the flux operator, U^*FU
    Parameters:
    -----------
    U - time evolution operator, returned by time_evolution
    flux - flux operator matrix

    Returns:
    --------
    evolvedFlux - time evolved flux

    """
    evolvedFlux = np.matmul(U.conjugate().transpose(), np.matmul(flux, U))

    return evolvedFlux

def Cff_np(U, flux, fluxbmn, N):
    #re-write to save CFF cumulative sum as it is generated. Also want to use to-be-written function to calculate valid significant figures at each step of the sum, save to list/ar
    """
    Compute CFF using numpy and a manually implemented trace calculation
    """
    mp.mp.dps = 30
    flux_t = fluxU(U, flux)

    Cff = complex(0)

    for i in range(0, N):
        Cfftemp = complex(0)
        for j in range(0, N):
            # Perform matrix multiplication and trace sum using numpy, at ~15 dps
            add = fluxbmn[i,j]*flux_t[j,i]
            Cfftemp = Cfftemp + add
    
        Cff = Cff - Cfftemp

    Cff = float(Cff.real)
    return Cff


def CFFsingletime(t_now, evecs, evalsmatrix, flux_mat, bolt_flux):
    """
    Evaluates the CFF value at a single time point. For use with joblib parallelization.

    Parameters:
    -----------
    t_now - current time, iterable. Pass this from joblib
    evecs
    evalsmatrix
    flux_mat
    bolt_flux
    N - number of grid points, hamiltonian dimension

    Returns:
    --------
    Cff - value of CFF at one point
    """
    N = bolt_flux.shape[0]
    U = time_evolution(t_now, evecs, evalsmatrix, N)

    Cff = Cff_np(U, flux_mat, bolt_flux, N)
  
    return Cff


def geteckartparams(reactionID, pathtodata):
    """
    Function to get eckart parameters from database for reactionID

    Parameters:
    -----------
    reactionID - cathub reaction ID 
    pathtodata (str) - path to where database is stored

    Returns:
    --------
    V1 - activation energy
    w - eckart w parameters
    alpha - eckart alpha parameter
    """

    # 0. Load sample eckart dataset
    reactionDataset = pd.read_json(str(pathtodata + '/cff_dataset/cff_dataset.json'))
    #1. Pull eckart grid parameters from dataset
    reaction = reactionDataset[reactionDataset.reaction_id==reactionID]
    params = reaction.eckart_params_w_alpha_xshift

    w=params.iloc[0][0]
    alpha = params.iloc[0][1]
    xshift = params.iloc[0][2]
    V1 = reaction.activation_energy.iloc[0]

    return V1, w, alpha, xshift

def getskewparams(reactionID, pathtodata):
    """
    Function to get eckart parameters from database for reactionID

    Parameters:
    -----------
    reactionID - cathub reaction ID 
    pathtodata (str) - path to where database is stored

    Returns:
    --------
    V1 - activation energy
    w - eckart w parameters
    alpha - eckart alpha parameter
    """

    # 0. Load sample eckart dataset
    reactionDataset = pd.read_json(str(pathtodata + '/cff_dataset/cff_dataset.json'))
    #1. Pull eckart grid parameters from dataset
    reaction = reactionDataset[reactionDataset.reaction_id==reactionID]
    params = reaction['skewLogParams_skewA_skewscale_L_k_vscale']

    skewA=params.iloc[0][0]
    skewscale = params.iloc[0][1]
    L = params.iloc[0][2]
    k = params.iloc[0][3]
    vscale = params.iloc[0][4]

    return skewA, skewscale, L, k, vscale

def setnumpts(L, dx):
    """
    Defines an odd integer number of points 

    Parameters:
    -----------
    Density (int) - density of points in pts/AU
    w - eckart parameters w
    windwidth - buffer to add to each side of w - default 5

    Returns:
    --------
    numpts (int) - odd integer number of points, 
    xmax
    xmin
    """

    numpts = 2*L/dx
    
    numpts = round(numpts)
    # make odd:
    if numpts%2==0:
        numpts+=1

    return numpts
    
def getDVR_eckartSingleAsymm(numpts, L, alpha, w, V1, xshift, m):
    """
    Calculates a grid q and hamiltonian H, then eigenstates, using the DVR representation.

    Parameters:
    -----------
    n_points (int) - number of points to use
    xmin and xmax - min and max x values
    alpha, w - eckart parameters
    V1 - activation energy, units eV
    m - mass, units g/mol

    Returns:
    --------
    H - hamiltonian for system
    evecs - eigenvector matrix
    evals - eigenvalues vector
    """

    # Convert to AU
    V1 = V1*ut.ev2hartree
    m = m*ut.gmol2me  

    q = np.linspace(-L, L, numpts)
    dq = q[1] - q[0]

    V = rx.eckartSingleAsymm((q+xshift), alpha, w, V1)

    #np.save('Potential.npy', V)
    H = rx.hamiltonian(V-min(V), dq, numpts, m)

    evals, evecs = np.linalg.eigh(H)
    evalsmatrix = np.reshape(evals, (numpts,1))

    return q, H, evecs, evalsmatrix


def getDVR_skewLogistic(numpts, Lgrid, skewA, skewscale, Lskew, k, vscale, m):
    """
    Calculates a grid q and hamiltonian H, then eigenstates, using the DVR representation. This function uses a skew logistic barrier, developed for narrow reactions Feb 2022

    Parameters:
    -----------
    n_points (int) - number of points to use
    m - mass, units g/mol

    Returns:
    --------
    H - hamiltonian for system
    evecs - eigenvector matrix
    evals - eigenvalues vector
    """
    m = m*ut.gmol2me

    q = np.linspace(-Lgrid, Lgrid, numpts)
    dq = q[1] - q[0]

    V = rx.skewLogistic(q, skewA, skewscale, Lskew, k, vscale)
    #np.save('Potential.npy', V)
    H = rx.hamiltonian(V, dq, numpts, m)
    #np.save('Hamiltonian.npy', H)
    evals, evecs = np.linalg.eigh(H)
    evalsmatrix = np.reshape(evals, (numpts,1))

    #np.save('Evals.npy', evalsmatrix)

    return q, H, evecs, evalsmatrix

def CFFsetup(reactionID, Lgrid, dx, pathtodata, barrier='eckartSingleAsymm', m = 'usedf'):
    """
    Runs all setup calculations to do a CFF calculation, up to calculating the boltzman matrix
    """

    # define mass. this chunk of code allows for either user defined or tabulated mass
    logging.info('Cff Calculation for Reaction {}'.format(reactionID))
    if m == 'usedf':
        logging.info('Using Mass tabulated in data')
        m = getMass(reactionID, pathtodata)

    else:
        assert isinstance(m, int) or isinstance(m, float), "If mass is user provided, must be int or float"

    logging.info('Reactant Mass: {} g/mol'.format(m))

    numpts = setnumpts(Lgrid, dx)
    logging.info('Number of Points: {}'.format(numpts))

    if barrier == 'eckartSingleAsymm':
        V1, w, alpha, xshift = geteckartparams(reactionID, pathtodata)
        logging.info('Eckart Barrier Parameters: w={}, alpha={}, V1={}'.format(w, alpha, V1))
        q, H, evecs, evals = getDVR_eckartSingleAsymm(numpts, Lgrid, alpha, w, V1, xshift, m)

    if barrier == 'skewedLogistic':
        skewA, skewscale, Lskew, k, vscale = getskewparams(reactionID, pathtodata)
        logging.info('Skew Logistic Barrier used')
        logging.info('Skew parameters: SkewA = {}, skewscale = {}, Lskew = {}, k = {}, vscale = {}'.format(skewA, skewscale, Lskew, k, vscale))
        q, H, evecs, evals = getDVR_skewLogistic(numpts, Lgrid, skewA, skewscale, Lskew, k, vscale, m)

    fluxMatrix = flux(q, H)

    return H, evecs, evals, fluxMatrix

def getMass(reactionID, pathtodata):
    """
    Gets the reactant mass from the pickles dataframe in gmols and returns it.
    """
    reactionDataset = pd.read_json(str(pathtodata + '/cff_dataset/cff_dataset.json'))
    #1. Pull eckart grid parameters from dataset
    reaction = reactionDataset[reactionDataset.reaction_id==reactionID]
    m = reaction.reactant_mass_gmol
 
    return float(m)

def computeBoltzmann(evecs, evals, T, save=False):
    """
    Wrapper function to handle computing and saving the boltzmann operator

    Parameters:
    -----------
    reactionID - Cathub reaction ID. Used to call eckart parameters from database
    m [g/mol] - mass of reactants in g/mol
    density [pts/AU] - grid density in pts/AU
    wingwidth [AU] - extra buffer to add to grid. Added to eckart w to set grid width 
    pathtodata - path to 'data' directory
    dprec - decimal precision for mpmath
    T [K] - temperature
    save (bool) - save bmn matrix to npy. Default True 

    Returns:
    -------
    bmnMatrix - np array - boltzmann matrix
    Also saves bmn matrix to directory called from
    """

    beta = getBeta(T)
    logging.info('Starting Boltzmann Matrix Calculation')
    tic = time.time()
    bmn = boltzmann(beta, evecs, evals)
    toc = time.time()
    logging.info('Computed Boltzmann Matrix in {} s'.format(toc-tic))

    if save:
        np.save('BoltzmannMatrix.npy', bmn)
        print('saved file')

    return bmn



def computeBoltFlux(fluxMatrix, bmnMatrix, save=False):
    """
    Computes and saved the boltzmannized flux matrix using mpmath.

    Parameters:
    -----------
    reactionID - Cathub reaction ID. Used to call eckart parameters from database
    m [g/mol] - mass of reactants in g/mol
    density [pts/AU] - grid density in pts/AU
    wingwidth [AU] - extra buffer to add to grid. Added to eckart w to set grid width 
    pathtodata - path to 'data' directory
    dprec - decimal precision for mpmath
    T [K] - temperature
    bmnMatrix - boltzmann matrix as np array
    save (bool) - save bmn matrix to npy. Default True 

    Returns:
    --------
    BoltFlux - bolt flux matrix
    also saves to disk
    """

    logging.info('Starting Boltzmannized Flux Matrix Calculation')
    tic = time.time()
    boltFlux = boltflux(fluxMatrix, bmnMatrix)
    toc = time.time()
    logging.info('Computed BoltFlux in {} s'.format(toc-tic))

    if save:
        np.save('BoltFluxMatrix.npy', boltFlux)

    return boltFlux


def computeCFF(evals, evecs, fluxMatrix, boltFlux, times, reactionID, titlestring='', save=True):
    """
    Given a boltFlux matrix and other reaction parameters, compute CFF

    Parameters:
    -----------
    evals - Hamiltonian eigenvalues
    evecs - Hamiltonian eigenvectors
    fluxmatrix - DVR flux matrix
    boltFlux - boltzmannized DVR flux matrix
    times - array of times to evaluate at. Set up this way to work with naive bash-based parallelism
    reactionID - Cathub reaction ID. Used to call eckart parameters from database
    titlestring - additional strings to pas to title
    save (bool) - save CFF values to CSV. Default True 

    Returns:
    ---------
    CFFlist - Array of CFF values
    Also saves output to pickle by default

    """

    logging.info('Starting CFF Calculation')
    tic  = time.time()

    Cff_valarray = np.zeros(len(times))

    for i, t in enumerate(times):
        Cff_valarray[i] = CFFsingletime(t, evecs, evals, fluxMatrix, boltFlux)
        logging.info('CFF value at time {}: {}'.format(t, Cff_valarray[i]))

    toc =time.time()
    logging.info('Completed CFF calculation in {} s'.format(toc-tic))
    
    if save:
        Cffvals = pd.Series(np.real(Cff_valarray), name="CFFval")
        thyme = pd.Series(times, name = 'time_AU')
        data = pd.concat([thyme, Cffvals], axis=1)
        data.to_csv('CFFvalues_numpy_reaction{}_{}.csv'.format(reactionID, titlestring))

    return Cff_valarray


def getBeta(T):
    """"
    Computes thermodynamic temperature beta
    Parameters:
    -----------
    T - temperature in K

    Returns:
    --------
    beta - beta in AU
    """

    beta = 1./(ut.kbAU*T)

    return beta


def getTimes(tmax, ntimes, nNodes, nodenum):
    """
    Assume nodenum starts from 1
    """

    time = np.linspace(0, tmax, ntimes)
    #print(nNodes)
    #print(time)
    timeChunks = np.array_split(time, nNodes)
    #print(timeChunks)

    return timeChunks[nodenum-1]

