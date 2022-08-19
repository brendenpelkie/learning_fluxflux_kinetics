import numpy as np
import ase
from scipy.optimize import curve_fit
import surfreact.numpycff as cff
import surfreact.utils as ut
import os
from scipy.stats import skewnorm
import matplotlib.pyplot as plt


"""
Module with functions to compute reaction-related properties such as reaction
 coordinate, energies, etc
"""
def exportGeometry(filepath, dataset):
    """
    Exports original (not processed) NEB and non-neb geometries from catalysis-hub derived dataset. Exports all geometries for each reaction into their own folder named with reactionID. Geometries are named with name stored in atoms_name column. Exports as xyz file.

    Parameters:
    -----------
    filepath (str) - Path where folders for each reaction should be deposited
    dataset (pandas df from cathub) - dataset with geometries to be exported.

    Returns:
    --------
    Nothing
    """
    with ut.cd(filepath) as location:

        for i in range(0, len(dataset)):
            reaction = dataset.iloc[i]

            atoms = reaction.atoms
            ID = reaction.reaction_id

            if reaction.NEBstatus:
                os.chdir('NEB')
                os.mkdir('{}'.format(ID))
                os.chdir('{}'.format(ID))
                for i, atom in enumerate(atoms):
                    name = reaction.atoms_name[i]
                    ase.io.write('{}.xyz'.format(name), atom)
                os.chdir('../..')
            else:
                os.chdir('nonNEB')
                os.mkdir('{}'.format(ID))
                os.chdir('{}'.format(ID))
                for i, atom in enumerate(atoms):
                    name = reaction.atoms_name[i]

                    ase.io.write('{}.xyz'.format(name), atom)
                os.chdir('../..')




def atomstoarray(MEPImage_list, metal, get_energy=True):
    """
    Extracts an array of positions and an array of energies from an ordered
    list of images along on MEP.
    Parameters:
    -----------
    MEPImage_list - list of ase atoms objects. List must be ordered
                    (ie, MEPImage_list[0] is first image along MEP,
                    MEPImage_list[1] is second image, etc).
    Metal (str) - The metal atomic symbol of the surface the geometry is for
    get_energy - default True. If get_energy, retrieve energy as a 
    Returns: positions: numatoms x numimages x 3 array with cartesian position
                        coordinates of each image
             energy: numimages dimensional vector of total energy of each image
    """

    # Verify that all images are of same length:

    numimages = len(MEPImage_list)
    target_num = 0

    #for i, atom in enumerate(MEPImage_list):
    #    numatoms = len(atom.get_atomic_numbers())
    #    if i == 0:
    #        target_num = numatoms
    #    else:
    #        assert numatoms == target_num, "Inconsistent number of atoms in an image suggests non-MEP images"
            
    #Calculate how many non-metal atoms are in the geometry: This is how many positions will be kept
    count = 0
    symbols = MEPImage_list[0].get_chemical_symbols()
    for symbol in symbols:
        if symbol != metal:
            count+=1

    # Now initialize arrays:
    positions = np.zeros((count, numimages, 3))
    energy = np.zeros((numimages, 1))

    for i in range(0, numimages):
        image = MEPImage_list[i]
        positions[:, i, :] = image.positions[-count:,:]
        if get_energy:
            energy[i] = image.get_total_energy()

    return positions, energy


def reactioncoordinate(positions):
    """
    Calculated the euclidean norm reaction coordinate from an array that
    contains cartesian coordinates for each atom at each image.

    Parameters:
    ----------
    positions: numpy array. Should be formatted such that the dimensions are
               atoms x images x 3

    Returns:
    ----------
    rxncoord - np vector containing cumulative reactions coordinates, dimension
               number of images
    """
    numatoms = positions.shape[0]
    diff_ID = np.diff(positions, axis=1)
    diff = np.zeros((numatoms, 1, 3))
    diff = np.append(diff, diff_ID, axis=1)

    deltas = np.zeros(positions.shape[1])

    for i in range(0, positions.shape[1]):
        diff_image = diff[:, i, :]
        diff_vect = np.reshape(diff_image, -1)
        deltas[i] = np.linalg.norm(diff_vect)

    rxncoord = np.cumsum(deltas)

    return rxncoord

def align_geom(refgeomFull, geomFull, numFix = 15):
    """Find translation/rotation that moves a given geometry to maximally
    overlap
    with a reference geometry. Implemented with Kabsch algorithm. Modified from function from SV to 
    work specifically for slab geometries with fixed bottom layers.

    Args:
        refgeomFull:    The reference geometry to be rotated to
        geomFull:       The geometry to be rotated and shifted   
        numFix:     The number of metal atoms on top of geometry array to treat as reference points

    Returns:
        init_rmsdFull: initial RMSD on all common atoms between reference and target geometry
        init_rmsdFix: intial RMSD between first numFix atoms in reference and target geoms
        RMSD:       Root-mean-squared difference between the rotated geometry
                    and the reference
        
        new_geom:   The rotated geometry that maximumally overal with the reference
    """
    # Setup/Data Cleaning:
    refLen  = refgeomFull.shape[0]
    init_rmsdFull = np.sqrt(np.mean((geomFull[0:refLen,:] - refgeomFull[0:refLen,:]) ** 2))
    init_rmsdFix = np.sqrt(np.mean((geomFull[0:numFix,:] - refgeomFull[0:numFix,:]) ** 2))
    
    refgeom = refgeomFull[0:numFix, :]
    geom = geomFull[0:numFix, :]
    
    # Translation:
    center = np.mean(refgeom, axis=0)   # Find the geometric center
    ref2 = refgeom - center
    geom2 = geom - np.mean(geom, axis=0)
    geomFull2 = geomFull - np.mean(geom, axis=0)

    #print(center)
    #print(geomFull2)
    
    # Calculation of Covariance Matrix:
    cov = np.dot(geom2.T, ref2)
    
    # Calculate Rotation Matrix:
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    
    #Rotate full geometry including non-reference atoms:
    new_geom = np.dot(geomFull2, u) + center
    
    #Calculate RMSD:
    rmsd = np.sqrt(np.mean((new_geom[0:refLen,:] - refgeomFull[0:refLen,:]) ** 2))
    return init_rmsdFull, init_rmsdFix, rmsd, new_geom

def eckartFit(coordinates, energy, activationEnergy, NEBstatus, p0):
    """
    Function to fit single assymmetric eckart barrier to reaction coordinate data. Calculates alpha based on reaction and activation energies, and uses activation energy for V1. If NEB images available, fits w to NEB. If only reactant/product/TS images available, calculates w as average reactant or product to TS distance divided by 2. 

    Parameters:
    -----------
    coordinates (np array, nx1) - RMSD reaction coordinates calculated on reactants only (no metal atoms, unless part of reaction). Coordinates must be shifted so that the zero                   coordinate corresponds to the maximum energy of the reaction.
    energy (np array, nx1) - reaction energies corresponding to coordinates
    activationEnergy (float) - Activation energy of reaction. Use same units as energy
    NEBstatus (bool) - NEB status of reaction (True of False). If NEBstatus, barrier width is fit. If not NEBstatus, barrier width is calculated from reaction coordinates

    Returns:
    --------
    params (list) - list of parameters for eckart barrier. If NEBstatus, returns [w, alpha]. If not NEBstatus, returns [alpha]

    """
    # Input Cleaning and Verification:
    # Verify coordinates and energy are same size
    assert coordinates.shape[0] == energy.shape[0], 'Coordinates and Energy must be same length and nx1'
    # Verify coordinates have been shifted so that max E is at coordinates=0
    assert bool(np.where(coordinates==0)[0][0]==int(np.where(energy == max(energy))[0][0])), "Coordinates should be shifted so x=0 occurs at maximum energy/Transition state"

    coords = coordinates
    V1=activationEnergy
    reactionEnergy = energy[-1] - energy[0]
    alpha = (activationEnergy - reactionEnergy)/activationEnergy

    if NEBstatus:
        p0 = ((coords[2]-coords[1])+(coords[1]-coords[0]))/2
        w, cov = curve_fit(lambda x, w: eckartSingleAsymm(x, alpha, w, V1), coords, energy.flatten())
        params = [w, alpha]
    # If non-NEB reaction, fit alpha and feed w
    if not NEBstatus:
        w = min((coords[2]-coords[1]), (coords[1]-coords[0]))
        #w, cov = curve_fit(lambda x, w: eckartSingleAsymm(x, alpha, w, V1), coords, energy.flatten(), p0 = [w0])
        
        params = [w, alpha]

    return(params)

def findLbound(reactionID, xmax=30, filepath = '/mmfs1/gscratch/davinci/usr/bgpelkie/git_repos/surface_reactions/data'):
    """
    Calculates a first-guess value for DVR grid width L based on the reaction barrier derivative dV/dx=0.00025, which seemed to be the optimal grid width from initial tests.

    Parameters:
    -----------
    ReactionID - cathub reaction ID for use pulling values from dataset
    xmax (default 30): maximum x to search over. If you need to go over 30, you're gonna have a bad time runnning CFF

    Returns:
    --------
    X_bound (int) - integer value of estimated optimal L
    
    """



    x = np.linspace(-xmax, xmax, 2001)
    V1, w, alpha, xshift = cff.geteckartparams(reactionID, filepath)
    V = eckartSingleAsymm((x+xshift), alpha, w, V1)

    for i in range(int(len(x)/2+20), len(x)):
        Vderiv = abs(V[i]-V[i-1])/(x[i]-x[i-1])
        if Vderiv<0.00025:
            x_bound = x[i]
            break
        if i==len(x)-1:
            raise Exception('Barrier does not flatten enough within xmax range. Try a larger range or find a better reaction')
    
    x_bound = np.ceil(x_bound)
    
    return x_bound

def findLrange(Lbound):
    """
    Given an integer L bound, returns an array of L values to test, above and below the bound.

    Parameters: 
    -----------
    Lbound (int/float) - Estimate of optimal L

    Returns:
    --------
    Lrange - 3 L guesses: Lbound - 2, -0, +2
    """

    Lrange = np.zeros(3)
    Lrange[0] = int(Lbound-2)
    Lrange[1] = int(Lbound)
    Lrange[2] = int(Lbound+2)
    
    return Lrange

def eckartSingleAsymm(x, alpha, w, V1):
    """
    Compute the single asymmetric Eckart barrier given x and parameters

    Parameters:
    -----------
    x (np array - n x 1)- reaction coordinate [atomic units]
    alpha - reaction energy [atomic units]
    w - barrier width [atomic units]
    V1 - barrier height/activation energy [atomic units]

    Returns:
    --------
    V - single asymmetric Eckart barrier 
    """

    V=V1*(1-alpha)/(1+np.exp(-2*np.pi*x/w))+(V1*(1+np.sqrt(alpha))**2)/(4*(np.cosh(np.pi*x/w))**2) 

    return V.flatten()

def eckartSingleSVThesis(x, V1, V2, Fstar):
    """
    Computes eckart barrier describe in Stephanie Valleau bachelor's thesis equation A1. Used for benchmarking mpmath cff code to make sure results are right.
    """
    A = V1-V2
    B = (np.sqrt(V1)+np.sqrt(V2))**2
    L = 2*np.pi*np.sqrt(-2/Fstar)*((1/np.sqrt(V1))+(1/np.sqrt(V2)))**-1
    zeta = -np.exp(2*np.pi*x/L)
    V = (A*zeta)/(1-zeta)-(B*zeta)/(1-zeta)**2

    return V

######################## Skew Logistic Barrier Functions #############################
def logistic(x, L, x0, k):
    """
    Logistic function. For use in generating a logistic skew barrier
    
    Parameters:
    -----------
    x: x grid
    x0: value of the sigmoid's midpoint;
    L: the curve's maximum value;
    k: the logistic growth rate or steepness of the curve. Pass negative for negative curve
    
    """
    func = L/(1+np.exp(-k*(x-x0)+1))
    return func

def logisticSkew(x, skewA, skewscale, L, k):
    """
    Returns a barrier (array) using the logistic skew barrier functional form. Adds a logistic to a skewed normal distribution
    Does not do any sort of peak zeroing - use the getBarrier function for a polished result
    
    Parameters:
    -----------
    
    x - x grid
    skewA - skew a parameter for distribution
    skewscale - 'standard deviation'/width parameter for distribution
    L - height for logistic function
    k - width of logistic function. Make negative for inverse logistic
    
    note: logsitic x0/centering is hard-coded as 0
    """
    logfunc = logistic(x, L, 0, k)
    
    skewdist = skewnorm.pdf(x, skewA, scale = skewscale)
    xshift = x[np.where(skewdist == max(skewdist))[0][0]]
    skewdist = skewnorm.pdf(x+xshift, skewA, scale = skewscale)
    
    barrier = logfunc+skewdist
    
    return barrier

def skewLogistic(x, skewA, skewscale, Lskew, k, vscale):
    """
    Function to return barrier and shift scales for logistic skew function. Call this function to get a barrier returned.
    Returns a cleaned up barrier with x-shifting so peak is at 0. 
    """

    initialbarrier = logisticSkew(x, skewA, skewscale, Lskew, k)
    # handle xshifting
    #shift returned barrier so max is at 0
    final_xshift = x[np.where(initialbarrier == max(initialbarrier))[0][0]]

    barrier = logisticSkew(x+final_xshift, skewA, skewscale, Lskew, k)

    #scale height or barrier
    barrier = barrier*vscale
    # shift so energy zero is at minimum energy
    barrier = barrier-min(barrier)

    # debug plotting

    #plt.figure()

    #plt.plot(x, barrier)
    #plt.xlabel('x')
    #plt.ylabel('Potential [should be hartrees]')
    #plt.savefig('skewLogBarrier.png', dpi=300)
    
    return barrier





##########Discrete Variable Representation Functions ####################
def hamiltonian(V, dq, n_points, m=1):
    """
    Computes the DVR hamiltonian for a 1D system given potential V. Uses reduced planck's constant.
    
    Parameters:
    -----------
    V (n_points x 1 array) - array providing potential at each spatial location
    dq (float) - spacing between spatial points in q, [atomic units]
    m - mass of system in atomic units
    
    Returns:
    -------
    H - hamiltonian, n_points x n_points array
    """
    # Check Input
    assert V.shape[0]==n_points, "length of V and number of points n_points must match"
    assert V.shape==(n_points,) or V.shape==(n_points, 1), "V must be a 1D potential"

    hbar = 1 # Reduced Planck's Constant

    V = np.eye(n_points)*V
    T=np.zeros((n_points,n_points))

    for i in range(0,n_points):
        for j in range(0, n_points):
            if i==j:
                T[i,j] = ((hbar**2*(-1)**(i-j))/(2*m*dq**2))*(np.pi**2)/3
            else:
                T[i,j] = ((hbar**2*(-1)**(i-j))/(2*m*dq**2))*(2/((i-j)**2))


    H = T+V
    
    return H


def CFFintegrator(CFFarr, dt, threshFrac=0.00001):
    """
    Integrates CFF to get a rate constant kQ. This integrator is written assuming that the curve converges directly with no recrossing, so that the first 0 slope point corresponds to reflection.
    This would need to be changed to make it general. uses a rectangular integration

    Need to figure out how to make this track a change in first derivative and 

    Parameters:
    -----------
    CFF: 2D array with CFF values
    dt: spacing between time points
    threshFrac: Fraction of CFF[0] that CFF[t] must be before reflection will be tested for. Default 0.00001, but this is up for discussion

    Returns:
    --------
    CFFint (float): value of integral
    i (int): Index of CFFarr for which reflection calcualated to have occured.
    """

    CFFint = 0
    thresh = CFFarr[0]*threshFrac
    print(thresh)
    
    lastDeriv = 0

    for i in range(1, len(CFFarr)):
        CFF = CFFarr[i]
        Deriv = (CFF-CFFarr[i-1])/dt
        #print(Deriv)
        if np.sign(Deriv) != np.sign(lastDeriv) and abs(CFF) < abs(thresh):
            print('CFF reflected at time {}. CFF integral is {}'.format(dt*i, CFFint))
            break 

        CFFint+=CFF*dt
        lastDeriv = Deriv

    return CFFint, i

def CFFintegrator_trap(CFFarr, dt, threshFrac=0.00001):
    """
    Integrates CFF to get a rate constant kQ. This integrator is written assuming that the curve converges directly with no recrossing, so that the first 0 slope point corresponds to reflection.
    This would need to be changed to make it general. uses a rectangular integration

    Need to figure out how to make this track a change in first derivative and 

    Parameters:
    -----------
    CFF: 2D array with CFF values
    dt: spacing between time points
    threshFrac: Fraction of CFF[0] that CFF[t] must be before reflection will be tested for. Default 0.00001, but this is up for discussion

    Returns:
    --------
    CFFint (float): value of integral
    i (int): Index of CFFarr for which reflection calcualated to have occured.
    """

    CFFint = 0
    thresh = CFFarr[0]*threshFrac
    print(thresh)
    
    lastDeriv = 0

    for i in range(1, len(CFFarr)):
        CFF = CFFarr[i]
        Deriv = (CFF-CFFarr[i-1])/dt
        #print(Deriv)
        if np.sign(Deriv) != np.sign(lastDeriv) and abs(CFF) < abs(thresh):
            print('CFF reflected at time {}. CFF integral is {}'.format(dt*i, CFFint))
            break 

        CFFint+=((CFFarr[i-1]+CFFarr[i])/2)*dt
        lastDeriv = Deriv

    return CFFint, i


