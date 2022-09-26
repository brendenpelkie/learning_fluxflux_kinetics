import pickle
import matplotlib
import mpmath as mp
import numpy as np
from joblib import Parallel, delayed
from pathlib import Path
import os
import flint

from typing import Type
import bidict

import ase
import molml.utils

# patch elements which molml does not have all of
molml.utils.ELE_TO_NUM = bidict.bidict(ase.data.atomic_numbers)

Atoms = Type[ase.Atoms]

def atoms_to_molmllist(atoms: Atoms, bonds: bool = False):
    """Constructs a molml list of lists from an atoms object.

    By default, will consider Atomic numbers and atom distances, but can be
    directed to also compute bonds. See the reaxnet.io.rdkit for bond
    determination.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms to compute on.
    bonds : bool, default False
        Whether to compute bonds for the atoms
    """
    positions = atoms.positions
    atomic_numbers = atoms.get_atomic_numbers()
    if bonds:
        raise NotImplementedError(
            'Calculation of molml bond dictionary not implemented yet.'
        )
    else:
        return [atomic_numbers, positions]


def save_object(obj, filename):
    """
    Function to save an object in a pickle. Uses pickle highest priority

    Parameters:
    -----------
    obj - python object to save
    filename (str): desired filename
    """
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def nptomp(matrix, complex=True):
    """
    Converts an np array to an mpmath matrix
    matrix must be 2D
    """
    assert len(matrix.shape)==2, "Input matrix must be 2D"
    mpmat = mp.matrix(matrix.shape[0], matrix.shape[1])
    
    for i in range(0,matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            if complex==True:
                mpmat[i,j] = mp.mpc(matrix[i,j])
            else:
                mpmat[i,j] = matrix[i,j]
            
    return mpmat


def mptonp(mpmat):
    """
    Convert mp matrix to an np array
    """
    #nparray = np.array(mpmat.tolist(), dtype=float64)

    nparray = np.zeros((mpmat.rows, mpmat.cols), dtype = complex)

    for i in range(0, mpmat.rows):
        for j in range(0, mpmat.cols):
            elem = mpmat[i,j]
        
            nparray[i,j] = complex(elem.real, elem.imag)
            
    return nparray

def np2flint(nparray, dprec):
    """
    Converts a np array to a flint matrix of same shape with precision dprec
    """
    
    flint.ctx.dps = dprec
    
    stringList = list(map(lambda sl:list(map(str, sl)), nparray.tolist()))
    return flint.acb_mat(stringList)


def matrow_mp(A, B, i, jmax, kmax, dprec):
    """
    Function to be used in combination with mpmatmul to calculate high-precision matrix multiplications in mpmath in parallel.
    This function takes np arrays as inmputs, then converts the elements to mpc to perform arithmetic with mpmath. The conversion is necessary because the parallel framework used, joblib,
    needs to be able to pickle objects to move them around (can't be done with mpmath). Since this function is not expected to be called on its own, sanitized input is assumed.
    
    Parameters:
    -----------
    A - np array - first matrix 
    B - np array - second matrix
    i - current row index for first matrix, passed from joblib for loop
    jmax - number of columnns in B, computed in mpmatmul
    kmax - number of cols in A/rows in B
    dprec  - decimal precision to use for mpmath calculations
    
    Returns:
    --------
    Crow = np array of complex elements, represents row i in the final product matrix C\
    
    """
    
    mp.mp.dps = dprec

    A = mp.matrix(A)
    B = mp.matrix(B)
    
    Crow = mp.matrix(jmax, 1)
    
    for j in range(0,jmax):
        Celem=mp.mpc(0)
        for k in range(0,kmax):
            Aelem = A[i,k]
            Belem = B[k,j]
            Cadd = mp.fmul(Aelem, Belem)
            Celem = mp.fadd(Cadd, Celem)
        
        # I think this code was some attempt at incrasing accuracy - should be unnneeded now that dps is set right
        #Celem_imag = Celem.imag
        #if Celem_imag<0:
        #    Celem = np.complex128(str(mp.nstr(Celem.real, n=dprec)+'-'+mp.nstr(Celem.imag, n=dprec)+'j'))
        #else:
        #    Celem = np.complex128(str(mp.nstr(Celem.real, n=dprec)+'+'+mp.nstr(Celem.imag, n=dprec)+'j'))
        #print(Celem.imag)
        
        Crow[j] = Celem
    
    Crow = Crow.transpose().tolist()

    return Crow[0]

def mpmatmul(A, B, numjobs=40, dprec=70):
    """
    Computes the matrix multiplication between A and B using mpmath in parallel.
    Uses a naive matrix multiplication implementation, probably not faster than native methods unless running with big matrix and lots of cores.
    Converts elements to mpmath format as they are accessed for calculation, then performs calculation at dprec using mpmath, before converting back to numpy.
    The conversion is necessary to use the parallel processing libaray Joblib
    Should be safe for complex numbers
    
    Parameters:
    -----------
    A - mpmath matrix - first matrix
    B - mpmath matrix - second matrix
    numjobs (int - default 40): Number of jobs/cores to run with. Default to 40 for use on Klone
    dprec (int) - decimal precision for use in mpmath
    
    Returns:
    --------
    C - mpmath matrix - product matrix, np array of complex elements
    
    """
    mp.mp.dps = dprec

    imax = A.rows
    kmax = A.cols
    jmax = B.cols

    print(imax, kmax, jmax)

    A = A.tolist()
    B = B.tolist()
    
    
    #assert A.shape[1]==B.shape[0], "Matrices A and B do not have compatible dimensions for multiplication"
    
    outlist = Parallel(n_jobs = 40, verbose=50)(delayed(matrow_mp)(A, B, i, jmax, kmax, dprec) for i in range(0, imax))

    C = mp.matrix(outlist)
    return C 

def mp_diagonal(passedMatrix):
    """
    Returns the diagonal of a mpmath matrix as a list of mpmath float objects. Use to inspect diagonal elements. Supports complex numbers
    """

    diagonal_real = np.zeros(len(passedMatrix))
    diagonal_complex = np.zeros(len(passedMatrix))

    for i in range(len(passedMatrix)):
        diagonal_real[i]=float(passedMatrix[i,i].real)
        diagonal_complex[i] = float(passedMatrix[i,i].imag)

    return diagonal_real, diagonal_complex


def trustedSigFigs(maxvalue, currentvalue, dps=15):
    """
    Calculates the number of significant digits of currentvalue that we can have faith in for a cumulative sum operation. Assumes that during 
    the calculation, when the max value was added to another value, dps digits of max value were maintained, and everything of smaller 
    significance was lost. This is used to calculate a max trusted magnitude, which then is mapped into a number of significant digits that
    can be trusted of the current number.
    
    Parameters:
    ---------
    maxvalue - the maximum (absolute) value that occured during addition calculations
    currentvalue - the current value of the cumulative sum
    dps - the decimal precision, number of decimal places kept in precision. Default is 15 for numpy float64.
    
    Returns:
    -------
    sigfigs - Number of significant digits of currentvalue that can be trusted
    """
    
    #calculate maximum sig figs we can trust. 
    ordertrusted = magnitude(abs(maxvalue)) - abs(dps)
    sigfigs = magnitude(abs(currentvalue)) - ordertrusted
    
    return sigfigs

def magnitude(x):
    """
    Returns the order of magnitude of x
    """
    return int(np.log10(x))

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, path):
        path = Path(path)
        self.path = path.expanduser()

    def __enter__(self):
        self.prevPath = Path(os.getcwd())
        os.chdir(str(self.path))

    def __exit__(self, etype, value, traceback):
        os.chdir(str(self.prevPath))



##### Unit Conversion Definitions

ev2hartree = 0.0367493 # Convert eV units of energy to Hartree units of energy (AU)
angs2bohr = 1.8897259886 # Convert angstroms to bohr radii
gmol2me = 1822.93 # Convert grams/mol to mass electrons

kcal2hartree = 0.0016 #Convert kcal to hartrees


##### Physical Constants

kbAU = 3.16709*10**-6 # kB in AU (hartrees/kelvin)
hAU = 2*np.pi # I think. because hbar is one in au right?


kb_eVK = 8.617333262145*10**-5 #wikipedia

h_evS = 6.582119569*10**-16 #wikipedia

