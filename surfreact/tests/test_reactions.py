
import pytest
import surfreact.reactions as rx
from ase.io import read
import numpy as np
import pickle



def test_atomstoarray():
    """ Test the atoms to array function """
    #Assume:
    imagelist = pickle.load(open('data/reactions/testimagelist_rxn432473.pkl', 'rb'))
    truePositions = pickle.load(open('data/reactions/testpositionsarray_rxn432473.pkl', 'rb'))
    trueEnergy = pickle.load(open('data/reactions/testenergy_rxn432743.pkl', 'rb'))
    inconsistentnumberofatomslist = pickle.load(open('data/reactions/testimagelist_inconsistentimagelengths.pkl', 'rb'))

    #Test:
    calculatedPositions, calculatedEnergy = rx.atomstoarray(imagelist)

    # Check:
    
    # 1. Positions match test case
    assert 0.0001 > np.sum(calculatedPositions - truePositions), "Alignment not working - final geometry not aligned with target"
    # 2. Energies match test case
    assert 0.0001 > np.sum(calculatedEnergy - trueEnergy), "Energies do not match test case"
    # 3. AssertionError for calling with inconsistent number of atoms
    with pytest.raises(AssertionError) as err:
        rx.atomstoarray(inconsistentnumberofatomslist)
  


def test_reactioncoordinate():
    """Test the reactioncoordinate function on an MEP with known reaction coordinates, provided by Nida. This test uses the atomstoarray function and assumes that it works"""
    #Assume:
    atoms = read('data/reactions/testMEP.xyz', index=':')
    positions, energy = rx.atomstoarray(atoms, get_energy=False)

    trueCoords = np.array([0., 0.28938468, 0.57867296, 0.87527824, 1.18094048, 1.52703523, 1.89360292, 2.25781743, 2.61328356, 2.96800835, 3.31149986, 3.72489164, 4.07660476, 4.41345921, 4.75055512, 5.11596584, 5.44954843, 5.80489202, 6.24393405, 6.73848069])

    #Test:
    calculatedCoords = rx.reactioncoordinate(positions)

    #check:
    assert 0.00001 > np.sum(calculatedCoords - trueCoords), "Incorrect reaction coordinates calculated"

def test_eckartFit():
    """
    Tests the eckartFit function against what is believed to be correct output at the time of writing
    """

    #Things to test:
    # 1. Raises assertion for mismatched coordinate and energy lengths
    # 2. Raises assertion for non-shifted coordinates
    # 3. Returns correct parameters for a test reaction (test both NEB and non-NEBreactions)

    # Define Test cases:

    # 1. Mismatched coordinate/reactant Length:

    coords1 = np.array([0.        , 0.52256492, 1.07850326, 1.40664791, 1.75666362,
       2.07775968, 2.44467664, 2.75571639, 2.98706398, 3.20109566,
       4.27938637])
    energy1 = np.array([ 0.        ,  0.03696218,  0.11059972,  0.1387065 ,  0.06997938,
       -0.07662658, -0.15553307, -0.2453437 , -0.25635887, -0.25213471])
    act1 = 5

    # 2. Non-shifted coordinates:
    coords2 = np.array([0.        , 0.52256492, 1.07850326, 1.40664791, 1.75666362,
       2.07775968, 2.44467664, 2.75571639, 2.98706398, 3.20109566,
       4.27938637])
    energy2 = np.array([ 0.        ,  0.03696218,  0.11059972,  0.1387065 ,  0.06997938,
       -0.07662658, -0.15553307, -0.2453437 , -0.25635887, -0.25213471,
       -0.19868284])
    act2 = 5

    # 3a. Returns correct parameters (NEB):
    coords3a = np.array([-1.40664791, -0.88408298, -0.32814465,  0.        ,  0.35001572,
        0.67111177,  1.03802874,  1.34906848,  1.58041608,  1.79444776,
        2.87273846])
    energy3a = np.array([ 0.        ,  0.03696218,  0.11059972,  0.1387065 ,  0.06997938,
       -0.07662658, -0.15553307, -0.2453437 , -0.25635887, -0.25213471,
       -0.19868284])
    act3a = 0.13870649781165412
    params3a = np.array([2.35918857, 2.74185145])

    # 3b. Returns correct parameters (non-NEB):
    coords3b = np.array([-3.19440598,  0.        ,  3.87132635])
    energy3b = np.array([0.        , 1.17409446, 0.09207829])
    act3b = 1.17409446077
    params3b = np.array([3.5328661653058404, 0.9253849747747331])

    # Test 1:
    with pytest.raises(AssertionError) as err:
        rx.eckartFit(coords1, energy1, act1, True)
    
    # Test 2:
    with pytest.raises(AssertionError) as err:
        rx.eckartFit(coords2, energy2, act2, True)

    # Test 3a:
    calcparams3a = rx.eckartFit(coords3a, energy3a, act3a, True)
    assert np.allclose(calcparams3a, params3a), "NEB eckart fit failed"

    # Test 3b:
    calcparams3b = rx.eckartFit(coords3b, energy3b, act3b, False)
    assert np.allclose(calcparams3b, params3b), "non-NEB Eckart fit failed"

def test_hamiltonian():
    """Test hamiltonian function
    """

    ####1. Check that correct output is given for a given potential
    ## A. Define parameters
    V1 = np.array([0.5  , 0.125, 0.   , 0.125, 0.5  ])
    dq1 = 0.5
    n_points1 = 5
    m1 = 1
    H1 = np.array([[ 7.07973627, -3.875     ,  1.        , -0.31944444,  0.75      ],
       [-3.5       ,  6.70473627, -4.        ,  1.125     ,  0.05555556],
       [ 1.5       , -3.875     ,  6.57973627, -3.875     ,  1.5       ],
       [ 0.05555556,  1.125     , -4.        ,  6.70473627, -3.5       ],
       [ 0.75      , -0.31944444,  1.        , -3.875     ,  7.07973627]])
    H1 = np.array([[ 7.07973627, -4.,          1.,         -0.44444444,  0.25      ],
 [-4.,          6.70473627, -4.,          1.,         -0.44444444],
 [ 1.,         -4.,          6.57973627, -4.,          1.        ],
 [-0.44444444,  1.,         -4.,          6.70473627, -4.        ],
 [ 0.25,       -0.44444444,  1.,         -4.,          7.07973627]])

    ## B. Run calculation
    H = rx.hamiltonian(V1, dq1, n_points1, m1)

    ## C. Compare

    assert np.sum(H1-H)<10**-6, "Incorrect Hamiltonian returned"
    ####2. Check that raises assertion for insonsistent n_points, len(V)
    V2 = np.array([0.5, 1, 1])
    n_points2 = 5

    with pytest.raises(AssertionError) as err:
        rx.hamiltonian(V2, dq1, 6)

    #3. Check that raises assertion for not 1D V
    V3 = np.array([[0.5, 1, 1], [0.5, 1, 1]])
    with pytest.raises(AssertionError) as err:
        rx.hamiltonian(V3, dq1, 5)

def test_boltzmann():
    # 1. Check that # of eigenvectors equals number of eigenvalues
    evals1 = np.array([1])
    evecs1 = np.array([[1,1], [2,2]])

    with pytest.raises(AssertionError) as err:
        rx.boltzmann(1, evecs1, evals1, 1)

    # 2. Check that boltzmann matrix calculated correctly for small system
    evals = np.array([-0.50435499, -0.06497893,  0.01918774,  0.05780807,  1.00923528])
    evecs = np.array([[ 7.37347544e-04,  1.76005793e-02,  9.28942725e-01,
         3.69784929e-01,  3.76769761e-03],
       [ 3.22408889e-02, -1.17247635e-01,  3.68560086e-01,
        -9.20755304e-01,  3.98327733e-02],
       [ 4.13557953e-01, -4.95983765e-01, -3.27208916e-02,
         9.72724211e-02,  7.56595892e-01],
       [ 2.10262948e-02,  8.34084099e-01,  1.25549581e-02,
        -7.68413529e-02,  5.45709987e-01],
       [-9.09663507e-01, -2.10349785e-01, -7.69891602e-04,
         1.01122942e-02,  3.57997795e-01]])
    beta = 1

    bmn_correct = np.array([[ 9.87869673e-01,  6.31066225e-03, -2.06498690e-03,
         3.72679690e-04, -9.48843025e-04],
       [ 6.31066225e-03,  9.74672609e-01, -3.53071576e-03,
        -1.37085933e-02, -1.29803991e-02],
       [-2.06498690e-03, -3.53071576e-03,  8.30063398e-01,
        -1.74560470e-01, -2.11817743e-01],
       [ 3.72679690e-04, -1.37085933e-02, -1.74560470e-01,
         9.04923997e-01, -8.86729252e-02],
       [-9.48843025e-04, -1.29803991e-02, -2.11817743e-01,
        -8.86729252e-02,  1.18801583e+00]])

    bmn = rx.boltzmann(beta, evecs, evals, 5)

    assert np.sum(bmn - bmn_correct)<1e-6, "Incorrect Boltzmann matrix returned"
    
def test_time_evolution():
    """ Test that the time evolution function computes the matrix exponential correctly.
    """
    # 1. Check that bad evec/eval pair caught

    evals1 = np.array([1])
    evecs1 = np.array([[1,1], [2,2]])

    with pytest.raises(AssertionError) as err:
        rx.time_evolution(1, evecs1, evals1, 1)

    # 2. Check for correct return:

    evals = np.array([-0.50435499, -0.06497893,  0.01918774,  0.05780807,  1.00923528])
    evecs = np.array([[ 7.37347544e-04,  1.76005793e-02,  9.28942725e-01,
         3.69784929e-01,  3.76769761e-03],
       [ 3.22408889e-02, -1.17247635e-01,  3.68560086e-01,
        -9.20755304e-01,  3.98327733e-02],
       [ 4.13557953e-01, -4.95983765e-01, -3.27208916e-02,
         9.72724211e-02,  7.56595892e-01],
       [ 2.10262948e-02,  8.34084099e-01,  1.25549581e-02,
        -7.68413529e-02,  5.45709987e-01],
       [-9.09663507e-01, -2.10349785e-01, -7.69891602e-04,
         1.01122942e-02,  3.57997795e-01]])

    U_correct = np.array([[ 9.99605379e-01-0.02444871j,  4.36958802e-04+0.01285315j,
        -1.40667916e-03-0.00432732j, -9.48789993e-04+0.00063834j,
        -5.45351210e-04-0.00190853j],
       [ 4.36958802e-04+0.01285315j,  9.97658642e-01-0.05153609j,
        -1.57200432e-02-0.00988358j, -1.01589988e-02-0.02859791j,
        -3.05111676e-03-0.02409794j],
       [-1.40667916e-03-0.00432732j, -1.57200432e-02-0.00988358j,
         7.10559460e-01-0.38646974j, -1.93216046e-01-0.37169409j,
        -8.00046504e-02-0.40434045j],
       [-9.48789993e-04+0.00063834j, -1.01589988e-02-0.02859791j,
        -1.93216046e-01-0.37169409j,  8.59248084e-01-0.20702154j,
        -8.85774763e-02-0.18595041j],
       [-5.45351210e-04-0.00190853j, -3.05111676e-03-0.02409794j,
        -8.00046504e-02-0.40434045j, -8.85774763e-02-0.18595041j,
         8.36958052e-01+0.29426483j]])

    U = rx.time_evolution(1, evecs, evals, 5)

    assert np.sum(U-U_correct)<1e-6, "Incorrect U returned"

def test_Cff():
    """ Test the Cff function. Use mocked inputs, make sure value comes out right. 
    """
    #1. Make sure U, flux, fluxbmn each have the same dimensions (assert for test case w/o)
    U1 = np.zeros(4)
    flux_mat1 = np.zeros(2)
    fluxbmn1 = np.zeros(3)

    with pytest.raises(AssertionError) as err:
        rx.Cff(U1, flux_mat1, fluxbmn1, 4)
    
    flux_mat2 = np.zeros(4)
    with pytest.raises(AssertionError) as err:
        rx.Cff(U1, flux_mat2, fluxbmn1, 4)

    # 2. Check that result comes out right
    flux_mat = np.array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         2.30703520e-01, -3.13184545e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         2.35795531e-01, -3.15632628e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         2.18169338e-01, -3.10540616e-01],
       [ 3.05494668e-04, -3.21293877e-02, -4.38750654e-01,
         0.00000000e+00,  0.00000000e+00],
       [-2.14258763e-03, -2.70373766e-02, -4.56376846e-01,
         0.00000000e+00,  0.00000000e+00]])
    fluxbmn = np.array([[ 3.81812584e-04,  9.07394456e-04,  2.60787692e-02,
         2.34682226e-01, -3.89517679e-01],
       [ 3.82320011e-04,  1.57409552e-03,  3.50188727e-02,
         2.33855549e-01, -3.89503814e-01],
       [ 4.19326255e-04,  1.12976943e-02,  1.66617087e-01,
         1.54993977e-01, -3.56929264e-01],
       [ 9.57942551e-04, -2.49327094e-02, -3.01057458e-01,
         2.01182033e-02,  1.48931049e-01],
       [-1.77180155e-03, -2.70021505e-02, -4.23868077e-01,
         3.72440384e-02,  1.94739807e-01]])
    U = np.array([[ 9.99605379e-01-0.02444871j,  4.36958802e-04+0.01285315j,
            -1.40667916e-03-0.00432732j, -9.48789993e-04+0.00063834j,
            -5.45351210e-04-0.00190853j],
        [ 4.36958802e-04+0.01285315j,  9.97658642e-01-0.05153609j,
            -1.57200432e-02-0.00988358j, -1.01589988e-02-0.02859791j,
            -3.05111676e-03-0.02409794j],
        [-1.40667916e-03-0.00432732j, -1.57200432e-02-0.00988358j,
            7.10559460e-01-0.38646974j, -1.93216046e-01-0.37169409j,
            -8.00046504e-02-0.40434045j],
        [-9.48789993e-04+0.00063834j, -1.01589988e-02-0.02859791j,
            -1.93216046e-01-0.37169409j,  8.59248084e-01-0.20702154j,
            -8.85774763e-02-0.18595041j],
        [-5.45351210e-04-0.00190853j, -3.05111676e-03-0.02409794j,
            -8.00046504e-02-0.40434045j, -8.85774763e-02-0.18595041j,
            8.36958052e-01+0.29426483j]])

    Cff_correct = -0.10026163797879997+2.7755575615628914e-17j

    Cff = rx.Cff(U, flux_mat, fluxbmn, 5)

    assert Cff - Cff_correct < 1e-12, "Incorrect CFF"

if __name__ == '__main__':
    test_atomstoarray()
    test_reactioncoordinate()
    test_eckartFit()
    test_hamiltonian()
    test_boltzmann()
    test_time_evolution()
    test_Cff()
    print('All Checks Pass')