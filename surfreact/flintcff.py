import numpy as np
import time
import logging
import surfreact.reactions as rx
from joblib import Parallel, delayed
import mpmath as mp
from surfreact import utils as ut
import pickle
import pandas as pd
import flint

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



def boltzmann_flint(beta, evecs, evals, dprec):
    """
    Boltzmann operator computer using mpmath library functions to get a higher precision. I think that this computation is causing errors at low temperatures, so improving accuracy here may help

    Takes in evecs and evals as np arrays, returns bmn as an mpmath matrix object.
    """
    mp.mp.dps = dprec
    flint.ctx.dps = dprec

    print(flint.ctx.threads)

    # Convert stuff to mpmath:
    evecs = ut.np2flint(evecs, dprec)
    evals = ut.np2flint(evals, dprec)


    for i in range(0, evals.nrows()):
        evals[i,0] = (-0.5*beta*evals[i,0]).exp()
    
    #evalexp = mp.diag(evals)
    # turn eigenvalues into a diagonal matrix
    eye = np.eye(evals.nrows())

    evalexp = ut.np2flint(eye, dprec)
    for i in range(0, evalexp.nrows()):
        evalexp[i,i] = evals[i,0]
    

    print('evecs rows', evecs.nrows())
    print('evecs cols, ', evecs.ncols())

    bmnRight = evalexp*evecs.transpose()
    
    bmn = evecs*bmnRight

    return bmn





def boltflux_flint(flux, bmn, dprec):
    """
    Compute the boltzmannized flux matrix to arbitrary precision with mpmath

    Pass flux as a np array, bmn as a mpmath matrix
    """
    #mp.mp.dps = dprec
    flint.ctx.dps = dprec

    flux = ut.np2flint(flux, dprec)

    bf_right = flux*bmn

    boltflux = bmn*bf_right

    return boltflux




def time_evolution_flint(t, evecs, evals, N, dprec, hbar=1):
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
    flint.ctx.dps = dprec

    evals = ut.np2flint(evals, dprec)
    evecs = ut.np2flint(evecs, dprec)

    N = evecs.nrows()
    evalsexp = np.zeros((N, N))
    evalsexp = ut.np2flint(evalsexp, dprec)

    for i in range(0, N):
        evalsexp[i,i] = (-1.j*t*evals[i,0]/hbar).exp()
    
    Uright = evalsexp*evecs.transpose()

    U = evecs*Uright

    return U

def fluxU_flint(U, flux, dprec):
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
    flint.ctx.dps = dprec

    flux = ut.np2flint(flux, dprec)

    evolvedright = flux*U
    evolvedFlux = U.conjugate().transpose()*evolvedright

    return evolvedFlux

def Cff_flint(U, flux, fluxbmn, N, dprec):
    #re-write to save CFF cumulative sum as it is generated. Also want to use to-be-written function to calculate valid significant figures at each step of the sum, save to list/ar
    """
    Compute CFF using mpmath
    """
    flint.ctx.dps = dprec

    flux_t = fluxU_flint(U, flux, dprec)

    Cff = flint.acb('0.0')

    for i in range(0, N):
        Cfftemp = flint.acb(0)
        for j in range(0, N):
            add = fluxbmn[i,j]*flux_t[j,i]    
            Cfftemp = Cfftemp+add
        Cff = Cff - Cfftemp
        

        # get the most sig figs trusted. Just doing this for real part for now
        

    Cff = float(Cff.real.str(radius=False))

    return Cff


def CFFsingletime(t_now, evecs, evalsmatrix, flux_mat, boltFlux_list, dprec):
    """
    Evaluates the CFF value at a single time point. For use with joblib parallelization.

    Parameters:
    -----------
    t_now - current time, iterable. Pass this from joblib
    evecs
    evalsmatrix
    flux_mat
    bolt_flux [list] - boltzmann. flux matrix as a list
    dprec - decimal precision

    Returns:
    --------
    Cff - value of CFF at one point
    """
    bolt_flux = boltFlux_list#flint.acb(boltFlux_list)
    N = bolt_flux.nrows()
    U = time_evolution_flint(t_now, evecs, evalsmatrix, N, dprec)

    Cff = Cff_flint(U, flux_mat, bolt_flux, N, dprec)
  
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

    np.save('Potential.npy', V)
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
    elif barrier == 'skewedLogistic':
        skewA, skewscale, Lskew, k, vscale = getskewparams(reactionID, pathtodata)
        logging.info('Skew Logistic Barrier used')
        logging.info('Skew parameters: SkewA = {}, skewscale = {}, Lskew = {}, k = {}, vscale = {}'.format(skewA, skewscale, Lskew, k, vscale))
        q, H, evecs, evals = getDVR_skewLogistic(numpts, Lgrid, skewA, skewscale, Lskew, k, vscale, m)
    else:
        raise AssertionError('Valid barrier not specified. Must be "eckartSingleAsymm" or "skewLogistic"')

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

def getT(reactionID, pathtodata):
    """
    Gets the selected temperature list from the pickles dataframe in K and returns it.
    """
    reactionDataset = pd.read_json(str(pathtodata + '/cff_dataset/cff_dataset.json'))
    #1. Pull eckart grid parameters from dataset
    reaction = reactionDataset[reactionDataset.reaction_id==reactionID]
    temps = reaction.temperatures

    return temps

def computeBoltzmann(evecs, evals, dprec, T, save=False):
    """
    Wrapper function to handle computing and saving the boltzmann operator

    Parameters:
    -----------
    reactionID - Cathub reaction ID. Used to call eckart parameters from database
    m [g/mol] - mass of reactants in g/mol
    density [pts/AU] - grid density in pts/AU
    wingwidth [AU] - extra buffer to add to grid. Added to eckart w to set grid width 
    pathtodata - path to 'data' directory in repo
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
    bmn = boltzmann_flint(beta, evecs, evals, dprec)
    toc = time.time()
    logging.info('Computed Boltzmann Matrix in {} s'.format(toc-tic))

    if save:
        bmn_save = bmn.tolist()
        ut.save_object(bmn_save, 'bmnMatrix.pkl')
        print('saved file')

    return bmn



def computeBoltFlux(fluxMatrix, bmnMatrix, dprec, save=False):
    """
    Computes and saved the boltzmannized flux matrix using mpmath.

    Parameters:
    -----------
    reactionID - Cathub reaction ID. Used to call eckart parameters from database
    m [g/mol] - mass of reactants in g/mol
    density [pts/AU] - grid density in pts/AU
    wingwidth [AU] - extra buffer to add to grid. Added to eckart w to set grid width 
    pathtodata - path to 'data' directory in repo for getting eckart.
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
    boltFlux = boltflux_flint(fluxMatrix, bmnMatrix, dprec)
    toc = time.time()
    logging.info('Computed BoltFlux in {} s'.format(toc-tic))

    if save:
        boltFlux_save = boltFlux.tolist()
        ut.save_object(boltFlux_save, 'boltFluxMatrix.pkl')

    return boltFlux

def computeCFF(evals, evecs, fluxMatrix, boltFlux, times, reactionID, dprec, titlestring='', save=True):
    """
    Given a boltFlux matrix and other reaction parameters, compute CFF

    Parameters:
    -----------
    reactionID - Cathub reaction ID. Used to call eckart parameters from database
    m [g/mol] - mass of reactants in g/mol
    density [pts/AU] - grid density in pts/AU
    wingwidth [AU] - extra buffer to add to grid. Added to eckart w to set grid width 
    pathtodata - path to 'data' directory in repo
    dprec - decimal precision for mpmath
    T [K] - temperature
    boltFlux - np array of boltFlux from boltflux function.
    times - array of times (in AU) to run CFF computation over
    numjobs - number of parallel jobs to run with. Suggestion 20 for 180 GB RAM. 
    titlestring - string to use in naming saved output. Example '1000K_narrow'. Combined with reaction ID, math package used to make title
    save (bool) - save CFF values to CSV. Default True 

    Returns:
    ---------
    CFFlist - Array of CFF values
    Also saves output to CSV by default

    """ 
    logging.info('Starting CFF Calculation')
    logging.info('Min Time: {}'.format(times[0]))
    logging.info('Max Time: {}'.format(times[-1]))
    # to make parallel calculation work, need to convert boltflux to list:
    boltFlux_list = boltFlux.tolist()
    ntimes = len(times)

    tic  = time.time()
    
    Cff_valarray = np.zeros(len(times))

    logging.info('Starting CFF Calculation')
    for i, t in enumerate(times):
        Cff_valarray[i] = CFFsingletime(t, evecs, evals, fluxMatrix, boltFlux_list, dprec)
        logging.info('CFF value at time {}: {}'.format(t, Cff_valarray[i]))

    toc =time.time()
    logging.info('Completed CFF calculation in {} s'.format(toc-tic))

    #re-format output to be in right shape:
    
    if save:
        Cffvals = pd.Series(np.real(Cff_valarray), name="CFFval")
        thyme = pd.Series(times, name = 'time_AU')
        data = pd.concat([thyme, Cffvals], axis=1)
        data.to_csv('CFFvalues.csv'.format(reactionID, titlestring))

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



########### SV code - use to verify against SV barrier

def getDVR_SV(numpts, L, V1, V2, Fstar, m):
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

    q = np.linspace(-L, L, numpts)
    dq = q[1] - q[0]

    V = rx.eckartSingleSVThesis(q, V1, V2, Fstar)
    H = rx.hamiltonian(V-min(V), dq, numpts, m)
    
    np.save('Potential.npy', V)
    evals, evecs = np.linalg.eigh(H)
    evalsmatrix = np.reshape(evals, (numpts,1))

    return q, H, evecs, evalsmatrix

def CFFsetup_SV(V1, V2, Fstar, L, numpts, m):
    """
    Runs all setup calculations to do a CFF calculation, up to calculating the boltzman matrix
    """

    q, H, evecs, evals = getDVR_SV(numpts, L, V1, V2, Fstar, m)

    fluxMatrix = flux(q, H)

    return H, evecs, evals, fluxMatrix


def computeCFF_SV(evals, evecs, fluxMatrix, boltFlux, times, dprec, save=True):
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
        Cff_valarray[i] = CFFsingletime(t, evecs, evals, fluxMatrix, boltFlux, dprec)
        logging.info('CFF value at time {}: {}'.format(t, Cff_valarray[i]))

    toc =time.time()
    logging.info('Completed CFF calculation in {} s'.format(toc-tic))
    
    if save:
        Cffvals = pd.Series(np.real(Cff_valarray), name="CFFval")
        thyme = pd.Series(times, name = 'time_AU')
        data = pd.concat([thyme, Cffvals], axis=1)
        data.to_csv('CFFvalues.csv')

    return Cff_valarray
