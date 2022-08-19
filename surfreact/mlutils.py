from operator import ge
from re import M
import pandas as pd
import numpy as np
import pickle
import surfreact.utils as ut
import os
import ase.io
import sys
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ExpSineSquared, ConstantKernel, Matern, PairwiseKernel
import sklearn.metrics

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import joblib

def load_MLdataset(path_to_root):
    """
    Loads/reconstructs ML dataset (ie dataset of select values relevant to ML phase of project). Requires a directory of files created from compileMLdataset notebook to work right
    
    Parameters:
    ----------
    path_to_root: (str) path to root of ML dataset filestructure. this directory should contain one 'MLdataset.json' file and one 'Cff_ReactantProductGeoms' directory.
    
    Returns:
    --------
    ML dataset: Pandas dataframe of ML relevant Cff data
    
    """
    
    if path_to_root[-1] != '/':
        path_to_root+='/'
        
    MLdatatest = pd.read_json(path_to_root+'MLdataset.json')

    zeros = pd.Series(np.zeros(len(MLdatatest)))

    MLdatatest['reactant_geom'] = zeros
    MLdatatest['product_geom'] = zeros

    MLdatatest['reactant_geom'] = MLdatatest['reactant_geom'].astype(object)
    MLdatatest['product_geom'] = MLdatatest['product_geom'].astype(object)

    for i, reaction in MLdatatest.iterrows():
        reaction_id = reaction.reaction_id

        reactant_geom, product_geom = read_geometry(path_to_root, reaction_id)

        MLdatatest.at[i, 'reactant_geom'] = reactant_geom
        MLdatatest.at[i, 'product_geom'] = product_geom
        
    return MLdatatest


def read_geometry(path_to_root, reaction_ID):
    """
    Helper function for load_MLdataset
    Loads reactant and product geometries stored in reaction-specific directories named with format "rxn<rxnID>" from path_to_root. Returns reactant and product geoms as ASE objects
    """
    if path_to_root[-1] != '/':
        path_to_root+='/'
    
    with ut.cd(path_to_root+'Cff_ReactantProductGeoms/rxn{}'.format(reaction_ID)) as loc:
        reactant_geom = ase.io.read('rxn{}_reactant.xyz'.format(reaction_ID))
        product_geom = ase.io.read('rxn{}_product.xyz'.format(reaction_ID))

    return reactant_geom, product_geom


def get_moving_atoms(react, prod, N_atoms):
    """
    Finds the N_atoms atoms out of reactive system that move most from react to prod, and returns ase atoms object with only these atoms
    
    Parameters:
    ----------
    react (ase atoms) - reactant geoms
    prod (ase atoms) - product geoms
    N_atoms (int): Number of atoms to select
    
    Returns:
    -------
    react_selected (ase atoms)
    prod_selected(ase atoms)
    """
    diffs = np.linalg.norm(prod.get_positions() - react.get_positions(), axis = 1)
    
    #print(diffs)
    
    # get indices of atoms that move most, sorted most move to least 
    top_movers = np.argsort(diffs)[::-1]
    #print(top_movers)
    
    # find the first organic atom in that list ( there are some PBC f-ups that have metal atoms move like 8)
    organic = ['H','O','C','N','S','P','Cl','Si','I','Br']
    
    sorted_atoms = np.array(react.get_chemical_symbols())[top_movers]
    
    for i, atom in enumerate(sorted_atoms):
        if atom in organic:
            start_ind = i
            break
    
    selected_indices = top_movers[start_ind:start_ind+N_atoms]
    
    print(selected_indices)
    
    react_selected = react[selected_indices]
    prod_selected = prod[selected_indices]
    
    return react_selected, prod_selected


def concatenate_features_timeseries(geom_features, MLdataset, cff_values, cff0_values, time_values, time_count, normalize_cff = True):
    """
    Concatenate geometry features matrix and other reaction features into a single X and y matrices for a time-series dataset

    Parameters:
    -----------
    geom_features (array): n_rxns x  n_features array of geometry features
    MLdataset (pd dataframe): dataframe with one entry per reaction including rxn ID column titled reaction_id, temperature list titled Temperature_list, and reaction energy titled reaction_energy
    cff_values (dict): nested dict with key:value pairs of rxnID:dict of T:Cff value list pairings
    cff0_values (dict): nested dict of {rxnID:{T:cff0}}
    time_values: nested dict like cffvals but for time
    time_count (int): number of cff(t) values
    normalize_cff (bool): normalize cff(t) by cff(0)

    Returns:
    --------
    X
    y
    y0 - vector of y0 values for use in fancy scaling or input schemes
    """

    # Define X and y result matrices
    num_extra_feats = 4
    X = np.zeros((time_count, geom_features.shape[1]+num_extra_feats))
    y = np.zeros(time_count)
    y0 = np.zeros(time_count)    
    rowcount = 0

    num_geom_feats = geom_features.shape[1]

    for i, row in MLdataset.iterrows():

        print(i)
        
        rxnID = row.reaction_id
        key = str(rxnID)
        geom_features_rxn = geom_features[i, :]
        temp_list = row.Temperature_list
        RE  = row.reaction_energy
        
        rxn_cff_vals = cff_values[key]
        rxn_time_vals = time_values[key]
        rxn_cfft0 = cff0_values[key]
        
        for j, T in enumerate(temp_list):
            T_cff_vals = rxn_cff_vals[str(T)]
            T_time_vals = rxn_time_vals[str(T)]
            Cff0 = rxn_cfft0[j]
            
            for k, t in enumerate(T_time_vals):
                cff = T_cff_vals[k]
                
                X[rowcount, 0] = rxnID
                X[rowcount, 1:num_geom_feats+1] = geom_features_rxn
                X[rowcount, num_geom_feats+1] = t
                X[rowcount, num_geom_feats+2] = 1/T
                X[rowcount, num_geom_feats+3] = RE
                

                if normalize_cff:
                    y[rowcount] = cff/Cff0
                else:
                    y[rowcount] = cff 

                y0[rowcount] = Cff0
                        
                rowcount += 1

    return X, y, y0

def concatenate_features_kQ(geom_features, MLdataset, log_cff = True, include_metalmass = False):
    """
    Concatenate geometry and other features for prediction of kQ_cff

    Parameters:
    ----------
    geom_features (np array): Array of shape n_rxnsx n_geomfeatures 
    MLdataset: pd DataFrame of reaction info including temps, cff kQ values, and RE with appropriate titles
    numcff: number of cff values in MLdataset
    """

    numcff = get_numcff(MLdataset)


    if include_metalmass:
        num_extras = 4
    else:
        num_extras = 3
    X = np.zeros((numcff, geom_features.shape[1]+num_extras))
    y = np.zeros(numcff)

    rxncount = 0

    for i, row in MLdataset.iterrows():
        cfflist = row['kQ_Cff_unitstimeau']
        TempList = row['Temperature_list']
        rxnID = row['reaction_id']
        metal_mass = get_metal_mass(row['metal_name'])
        
        # for each temperature, we need to add an entry to X with geometry features and T, and add corresponding cff to y
        for cff, temp in zip(cfflist, TempList):
            X[rxncount, 0] = rxnID
            X[rxncount, 1:geom_features.shape[1]+1] = geom_features[i,:]
            X[rxncount, geom_features.shape[1]+1] = 1/temp
            X[rxncount, geom_features.shape[1]+2] = row.reaction_energy
            if include_metalmass:
                X[rxncount, geom_features.shape[1]+3] = metal_mass
            
            if log_cff:
                y[rxncount] = np.log(float(cff))
            else:
                y[rxncount] = float(cff)
            
            # update the rxnindex, or row in feature dataset we are working on 
            rxncount += 1

    return X, y 

def get_metal_mass(atom):
    """
    Definitions of metal masses. Sourced from SV notebook
    """
    
    mass = 0.
    
    if atom == 'Rh':
        mass = 102.906
    elif atom == 'Cu':
        mass = 63.546
    elif atom == 'Ag':
        mass = 107.868
    elif atom == 'Pd':
        mass = 106.49
    elif atom == 'Ir':
        mass = 192.22
    elif atom == 'Pt':
        mass = 195.08
        
    return mass


def get_numcff(MLdataset):
    """
    Helper function to get number of cff values in MLdataset
    """

    numcff = 0
    for i, row in MLdataset.iterrows():
        
        numcff_row =0
        numtemps_row = 0
        
        for cff in row['kQ_Cff_unitstimeau']:
            numcff_row += 1
        for temp in row['Temperature_list']:
            numtemps_row += 1
            
        assert numcff_row == numtemps_row, 'Mismatch temps and cff for reaction {}'.format(row.reaction_id)
        
        numcff += numcff_row

    return numcff

def reaction_split(X, y, test_reactions, return_inds = False):
    """
    Performs a per-reaction split on data such that all cff examples for one reaction are in the test set and no data is leaked

    Parameters:
    -----------
    X - np array of test data. Must have first column be reaction IDs
    y - np array of target data.
    test_reactions: List or array of reaction indices to include in test set

    Returns:
    --------
    X_train, X_test, y_train, y_test. X arrays have reaction ID stripped. 
    
    """
    X_test_list = []
    X_test_shape = X.shape[1]
    y_test_list = []

    delete_indices = []

    for ind, rxn in enumerate(test_reactions):
        for i in range(0, len(X)):
            if X[i,0] == rxn:
                X_test_list.append(X[i,:])
                y_test_list.append(y[i])
                delete_indices.append(i)
        
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list).reshape(-1,1)
                
    X_train = np.delete(X, delete_indices, axis = 0)
    y_train = np.delete(y, delete_indices, axis = 0)

    # strip reaction ID from X
    #X_train = X_train[:,1:]
    #X_test = X_test[:,1:]

    if return_inds:
        return X_train, X_test, y_train, y_test, delete_indices
    else:
        return X_train, X_test, y_train, y_test

def temperature_split(X, y, MLdataset, return_inds = False):
    """
    Perform a per-temperature split so that all cff examples at one temperature are placed in test set for each reaction
    Split is done by randomly selecting one temperature for each reaction and witholding all values that correspond to that temp
    Reaction ID is used as seed for temperature selection
    
    Parameters:
    ----------
    X (np array): np array of X features. First column must be reaction ID. inverse temperature MUST be in the -2 columnn.
    y - array of targets
    MLdataset: dataset with temperature list for each reactionID

    Returns:
    --------
    X_train, X_test, y_train, y_test. X vectors not stripped to make tracking what went where easier
    
    """
    test_ind_list = []

    for i, row in MLdataset.iterrows():
        
        rxnID = row.reaction_id
        temp_list = row.Temperature_list

        rxnIndices = []
        rng = np.random.default_rng(rxnID)
        T = rng.choice(temp_list, size = 1)
        for j in range(0, len(X)):
            # get all indices for one temp and reaction
            if X[j,0] == rxnID:
                if np.isclose(X[j, -2], 1/T):
                    rxnIndices.append(j)
                    
        test_ind_list.extend(rxnIndices)

    # select examples according to indices
    X_test = X[test_ind_list, :]
    y_test = y[test_ind_list]

    X_train = np.delete(X, test_ind_list, axis = 0)
    y_train = np.delete(y, test_ind_list, axis = 0)

    # strip reaction ID 
    #X_train = X_train[:,1:]
    #X_test = X_test[:,1:]
    if return_inds:
        return X_train, X_test, y_train, y_test, test_ind_list
    else:
        return X_train, X_test, y_train, y_test

### Cross val on temp and reaction split functions


def crossval_split_reaction(X_train, y_train):
    """
    Generates cross-val splits by reaction-splitting methodology
    """
    reaction_ids = np.unique(X_train[:,0])
    
    X_train_val_list = []
    y_train_val_list = []
    X_val_list = []
    y_val_list = []
    
    
    for rxnID in reaction_ids:
        X_train_cv, X_val, y_train_cv, y_val = reaction_split(X_train, y_train, [int(rxnID)])
        X_train_val_list.append(X_train_cv)
        y_train_val_list.append(y_train_cv)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
        
    return X_train_val_list, y_train_val_list, X_val_list, y_val_list

def crossval_split_temperature(X_train, y_train):
    """
    Function to generate cross-val splits using temperature split methodology. Generates 3 splits
    Parameters:
    ----------
    X_train - train set from initial temp split, must include reaction IDs
    y_train - train y from initial split
    
    Returns:
    -------
    X_train,
    X_val
    y_train
    y_val
    """
    val_ind_1 = []
    val_ind_2 = []
    val_ind_3 = []

    # get temperatures or rows in X available
    rx_ids = np.unique(X_train[:,0])

    for rxnID in rx_ids:
        indices = []

        # iterate over X to get locations of examples for each reaction
        for j in range(0, len(X_train)):
            if X_train[j,0] == rxnID:
                indices.append(j)

        # shuffle indices in place
        rng = np.random.default_rng(int(rxnID))
        rng.shuffle(indices)

        # append shuffled indices to respective folds
        val_ind_1.append(indices[0])
        val_ind_2.append(indices[1])
        if len(indices)==3:
            val_ind_3.append(indices[2])


    # assign each reaction/temp example to val set 1, 2, 3
    X_train_list = []
    X_val_list = []
    y_train_list = []
    y_val_list = []

    # val ind 1
    y_val_list.append(y_train[val_ind_1])
    y_train_list.append(np.delete(y_train, val_ind_1, axis=0))

    X_val_list.append(X_train[val_ind_1])
    X_train_list.append(np.delete(X_train, val_ind_1, axis = 0))

    # val set 2
    y_val_list.append(y_train[val_ind_2])
    y_train_list.append(np.delete(y_train, val_ind_2, axis=0))

    X_val_list.append(X_train[val_ind_2])
    X_train_list.append(np.delete(X_train, val_ind_2, axis = 0))
    # val set 3
    y_val_list.append(y_train[val_ind_3])
    y_train_list.append(np.delete(y_train, val_ind_3, axis=0))

    X_val_list.append(X_train[val_ind_3])
    X_train_list.append(np.delete(X_train, val_ind_3, axis = 0))
    
    return X_train_list, y_train_list, X_val_list, y_val_list


def crossval_evaluate(X_train_val_list, y_train_val_list, X_val_list, y_val_list, include_compositekernels = True, verbose_dump = False):
    """
    Function to handle leave-one out cross val on arbitrary (reaction or temperature) split.
    
    Trains suite of GPRs with different kernels for each provided cross-val fold using train_gprs function. Then evaluates train and val MAE, r2, and norm std.
    
    Input lists should have reaction ID values in column 0. No splitting is done but these are stripped in function
    Inpout features will be scaled with minmax scaler

    Parameters:
    -----------
    Each '_list' input is a list of values for cv folds, generated from a crossval_split function
    X_train_val_list: train
    y_train_val_list: train
    X_val_list: X_val
    y_val_list: y_val
    include_composite_kernels: default True (train_gprs param)
    verbose_dump: default False (train_gprs param)
    
    Returns:
    --------
    kernels: list of kernels. While this list displays fitted kernel values, it is taken from one cv fold and should only be used to compare kernel types
    val_out: list of metrics (MAE, MAE-norm, r2, r2_norm, std, std-norm) for aggregate validation set for each kernel in kernels
    train_out: list of metrics (see val_out0 for aggregate train set for each kernel in kernels
    """

# write a function to scale, train GPR, evaluate, and return results for the cross-validated 

# initialize empty lists to store y_pred values for each kernel type
    y_pred_val_lists = []
    for i in range(0,30):
        y_pred_val_lists.append([])

    y_pred_val_lists_norm = []
    for i in range(0,30):
        y_pred_val_lists_norm.append([])

    y_pred_std_lists = []
    for i in range(0,30):
        y_pred_std_lists.append([])

    y_pred_std_lists_norm = []
    for i in range(0,30):
        y_pred_std_lists_norm.append([])

    # train list
    y_pred_train_lists = []
    for i in range(0,30):
        y_pred_train_lists.append([])

    y_pred_train_lists_norm = []
    for i in range(0,30):
        y_pred_train_lists_norm.append([])

    y_pred_train_std_lists = []
    for i in range(0,30):
        y_pred_train_std_lists.append([])

    y_pred_train_std_lists_norm = []
    for i in range(0,30):
        y_pred_train_std_lists_norm.append([])

    # track y_val values
    y_val_composite = []
    y_train_composite = []

    # iterate over each cv split, train models, and keep predictions to evaluate later
    for X_train_cv, y_train_cv, X_val, y_val in zip(X_train_val_list, y_train_val_list, X_val_list, y_val_list):

        #print(X_val)

        # strip reaction ID
        print(X_train_cv.shape)
        print(y_train_cv.shape)

        X_train_cv = X_train_cv[:, 1:]
        X_val = X_val[:, 1:]

        # scale X
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(X_train_cv)
        X_train_cv = scaler.transform(X_train_cv)
        X_val = scaler.transform(X_val)

        # train gpr with train gpr function

        gpr_output, gpr_output_normalized = train_GPRs(X_train_cv, y_train_cv, X_val, y_val, include_compositeKernels=include_compositekernels, verbose_dump=verbose_dump)

        #store predicted values and std values
        for i in range(0,len(gpr_output[0])):
            y_pred = gpr_output[2][i]
            y_pred_val_lists[i].extend(list(y_pred))

            y_pred_norm = gpr_output_normalized[2][i]
            y_pred_val_lists_norm[i].extend(list(y_pred_norm))

            y_pred_std = gpr_output[3][i]
            y_pred_std_lists[i].extend(list(y_pred_std))

            y_pred_std_norm = gpr_output_normalized[3][i]
            y_pred_std_lists_norm[i].extend(list(y_pred_std_norm))

            # also get training predictions:

            y_pred_train, y_std_train = gpr_output[1][i].predict(X_train_cv, return_std=True)
            y_pred_train_norm, y_std_train_norm = gpr_output_normalized[1][i].predict(X_train_cv, return_std=True)

            y_pred_train_lists[i].extend(list(y_pred_train))
            y_pred_train_lists_norm[i].extend(list(y_pred_train_norm))

            y_pred_train_std_lists[i].extend(list(y_std_train))
            y_pred_train_std_lists_norm[i].extend(list(y_std_train_norm))

        kernels = gpr_output[0]

        y_val_composite.extend(y_val)
        y_train_composite.extend(y_train_cv)




    # generate list of MAE, r2, std norm values
    val_out = cv_metric_compute(y_pred_val_lists, y_pred_val_lists_norm, y_pred_std_lists, y_pred_std_lists_norm, y_val_composite)

    train_out = cv_metric_compute(y_pred_train_lists, y_pred_train_lists_norm, y_pred_train_std_lists, y_pred_train_std_lists_norm, y_train_composite)

    return kernels, val_out, train_out

def cv_metric_compute(y_pred_val_lists, y_pred_val_lists_norm, y_pred_std_lists, y_pred_std_lists_norm, y_true):
    """
    Wrapper function to compute metrics for cross validation work. Works closely with crossval_evaluate function. Can be used on either train or val predictions.
    
    Inputs are lists of lists of values for each kernel. IE input lists should have one list for each kernel tested, and sub-lists should either be the size of the overall test set or overall train set.
    
    Parameters:
    ----------
    y_pred_val_lists: predicted y values from non-normalized models
    y_pred_val_lists_norm: Predicted y values from normalized models
    y_pred_std_lists: Standard deviations of predictions
    y_pred_std_lists_norm: standard devs of predictions from norm models
    y_true: the true y values
    
    Returns: A list of 6 sublists. Each sublist contains n_kernels entries and corresponds to, in order:
    -MAE
    -MAE norm
    -r2 
    -r2 norm
    -std
    -std norm
    """

    MAE_list = []
    MAE_norm_list = []

    r2_list = []
    r2_norm_list = []

    stderror_list = []
    stderror_norm_list = []


    for i in range(0,30):

        y_pred = y_pred_val_lists[i]
        y_pred_norm = y_pred_val_lists_norm[i]

        y_std = y_pred_std_lists[i]
        y_std_norm = y_pred_std_lists_norm[i]

        if len(y_pred) > 1:
            MAE = sklearn.metrics.mean_absolute_error(y_true, y_pred)
            MAE_norm = sklearn.metrics.mean_absolute_error(y_true, y_pred_norm)

            r2 = sklearn.metrics.r2_score(y_true, y_pred)
            r2_norm = sklearn.metrics.r2_score(y_true, y_pred_norm)

            stderror = np.linalg.norm(y_std)
            stderror_norm = np.linalg.norm(y_std_norm)

            MAE_list.append(MAE)
            MAE_norm_list.append(MAE_norm)

            r2_list.append(r2)
            r2_norm_list.append(r2_norm)

            stderror_list.append(stderror)
            stderror_norm_list.append(stderror_norm)

    return [MAE_list, MAE_norm_list, r2_list, r2_norm_list, stderror_list, stderror_norm_list] 



def train_GPRs(X_train, y_train, X_test, y_test, Nr=50, RS=27, LS=1., scale_min=1e-5, scale_max=1e5, include_compositeKernels = True, verbose_dump = True):
    """
    Function written by SV to evaluate a bunch of GPR kernels on a train and test set

    Parameters:
    -----------
    X_train
    y_train
    X_Test
    y_test
    Nr
    RS
    LS
    scale_min
    scale_max
    include_compositeKernels (bool) - If false, kernels made of composites of kernel functions will not be evalutated
    verbose_dump (Bool) - if true, gpr output objects will be written to pwd at each kernel iteration. Useful for checkpointing progress.

    Returns:
    --------
    gpr_nonnormalized and gpr_normalized result nested liss
    """

    kernels = [1 * RBF(length_scale=LS, length_scale_bounds=(scale_min, scale_max)),
               1 * Matern(length_scale=LS,
                          length_scale_bounds=(scale_min, scale_max)),
               1 * PairwiseKernel(), 1*RationalQuadratic()]
    
    gp_kernels = []
    gp_n_kernels = []
    
    gps = []
    gps_n = []
    
    mean_y_preds = []
    std_y_preds = []
    
    mean_y_n_preds = []
    std_y_n_preds = []
    
    for k in kernels:

        print('Starting model for kernel {}'.format(k))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=Nr, random_state=RS)
        print('starting training')
        gp.fit(X_train, y_train)
        mean_y, std_y = gp.predict(X_test, return_std=True)
        
        gp_kernels.append(gp.kernel_)
        gps.append(gp)
        mean_y_preds.append(mean_y)
        std_y_preds.append(std_y)

        print('Non normalized results')
        print('Test MAE: ', sklearn.metrics.mean_absolute_error(y_test, mean_y))
        print('Test R2: ', sklearn.metrics.r2_score(y_test, mean_y))
        
        gp_n = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=Nr, random_state=RS)
        gp_n.fit(X_train, y_train)
        mean_y_n, std_y_n = gp_n.predict(X_test, return_std=True)
        
        gp_n_kernels.append(gp.kernel_)
        gps_n.append(gp_n)
        mean_y_n_preds.append(mean_y_n)
        std_y_n_preds.append(std_y_n)

        print('Normalized results')
        print('Val MAE: ', sklearn.metrics.mean_absolute_error(y_test, mean_y_n))
        print('Val R2: ', sklearn.metrics.r2_score(y_test, mean_y_n))
        
        if include_compositeKernels:
            for k2 in kernels:
                gp = GaussianProcessRegressor(kernel=k+k2, normalize_y=False, n_restarts_optimizer=Nr, random_state=RS)
                gp.fit(X_train, y_train)
                mean_y, std_y = gp.predict(X_test, return_std=True)
                
                gp_kernels.append(gp.kernel_)
                gps.append(gp)
                mean_y_preds.append(mean_y)
                std_y_preds.append(std_y)
            
                gp_n = GaussianProcessRegressor(kernel=k+k2, normalize_y=True, n_restarts_optimizer=Nr, random_state=RS)
                gp_n.fit(X_train, y_train)
                mean_y_n, std_y_n = gp_n.predict(X_test, return_std=True)
            
                gp_n_kernels.append(gp.kernel_)
                gps_n.append(gp_n)
                mean_y_n_preds.append(mean_y_n)
                std_y_n_preds.append(std_y_n)

        if verbose_dump:
            joblib.dump([gp_kernels, gps, mean_y_preds, std_y_preds], 'gpr_output_nonnormalized.joblib')
            joblib.dump([gp_n_kernels, gps_n, mean_y_n_preds, std_y_n_preds], 'gpr_output_normalized.joblib')
        
    return [gp_kernels, gps, mean_y_preds, std_y_preds] , [gp_n_kernels, gps_n, mean_y_n_preds, std_y_n_preds] 


def evaluate_results(gpr_output, gpr_output_normalized, y_test, plot_title_base):
    """
    
    Convenience function for processing and making sense of SV gpr traning results
    
    Parameters:
    ----------
    gpr_output (list) - output of SV gpr train function
    gpr_output_normalized (list) - normalized sv gpr output
    plot_title_base - string for titling plots
    
    Returns:
    -------
    
    [best_r2_index, best_MAE_index, best_r2_index_norm, best_MAE_index_norm] - indices in GPR output and gpr output norm to get best results
    """
    Ng=7
    nC = 1
    loc = (1, 1.1)
    ms=90
    markers = ["+","*","o","v","^","1","8","p","s","P","4","h","H","x","X","d","D",
            "+","*","o","v","^","1","8","p","s","P","4","h","H","x","X","d","D",
            "+","*","o","v","^","1","8","p","s","P","4","h","H","x","X","d","D",
            "+","*","o","v","^","1","8","p","s","P","4","h","H","x","X","d","D"]
    color = cm.plasma(np.linspace(0, 1, len(gpr_output[2])))

    MAElist = []
    r2list = []

    plt.figure(figsize = (Ng, Ng))
    for i in range(len(gpr_output[1])):
        r2 = sklearn.metrics.r2_score(y_test, gpr_output[2][i])
        MAE = sklearn.metrics.mean_absolute_error(y_test, gpr_output[2][i])
        r2list.append(r2)
        MAElist.append(MAE)
        if i == 0:
            max_r2 = r2
            min_MAE = MAE
            best_r2_index = i
            best_MAE_index = i
        else: 
            if r2 > max_r2:
                max_r2 = r2
                best_r2_index = i
            if MAE < min_MAE:
                min_MAE = MAE
                best_MAE_index = i

    plt.scatter(y_test, gpr_output[2][best_r2_index], alpha = 0.8, s=ms, color = color[1], marker = markers[1], label = "Best R2 kernel")
    plt.scatter(y_test, gpr_output[2][best_MAE_index], alpha = 0.3, s=ms, color = color[20], marker = markers[2], label = "Best MAE kernel")

    mi_y = min(y_test)
    ma_y = max(y_test)
    x1 = [mi_y, ma_y]
    y1 = [mi_y, ma_y]

    plt.plot(x1, y1, '-k')

    #plt.xlim([min(y_test)-2,max(y_test)+2])
    #plt.ylim([min(y_test)-2,max(y_test)+2])
    plt.legend()

    plt.title('{}, non-normalized y'.format(plot_title_base))

    #### Normalized y results
    MAE_normlist = []
    r2normlist = []
    plt.figure(figsize = (Ng, Ng))
    for i in range(len(gpr_output_normalized[1])):
        r2 = sklearn.metrics.r2_score(y_test, gpr_output_normalized[2][i])
        MAE = sklearn.metrics.mean_absolute_error(y_test, gpr_output_normalized[2][i])
        MAE_normlist.append(MAE)
        r2normlist.append(r2)
        if i == 0:
            max_r2_norm = r2
            min_MAE_norm = MAE
            best_r2_index_norm = i
            best_MAE_index_norm = i
        else:
            if r2 > max_r2_norm:
                max_r2_norm = r2
                best_r2_index_norm = i
            if MAE < min_MAE_norm:
                min_MAE_norm = MAE
                best_MAE_index_norm = i


    plt.scatter(y_test, gpr_output_normalized[2][best_r2_index_norm], alpha = 0.8, s=ms, color = color[1], marker = markers[1], label = "Best R2 kernel")
    plt.scatter(y_test, gpr_output_normalized[2][best_MAE_index_norm], alpha = 0.8, s=ms, color = color[10], marker = markers[2], label = "Best MAE kernel")

    mi_y = min(y_test)
    ma_y = max(y_test)
    x1 = [mi_y, ma_y]
    y1 = [mi_y, ma_y]

    plt.plot(x1, y1, '-k')

    #plt.xlim([min(y_test)-2,max(y_test)+2])
    #plt.ylim([min(y_test)-2,max(y_test)+2])
    plt.legend()

    plt.title('{}, normalized y'.format(plot_title_base))
    
    # get best non normalized model results
    print('#### Non Normalized y Results ####')
    print('Best R2 kernel:')
    print(gpr_output[0][best_r2_index])
    print('R2: ', r2list[best_r2_index])
    print('MAE: ', MAElist[best_r2_index])

    print('Best MAE kernel:')
    print(gpr_output[0][best_MAE_index])
    print('R2: ', r2list[best_MAE_index])
    print('MAE: ', MAElist[best_MAE_index])
    
    # get best normalized model results
    print('#### Normalized y Results ####')
    print('Best R2 kernel:')
    print(gpr_output_normalized[0][best_r2_index_norm])
    print('R2: ', r2normlist[best_r2_index_norm])
    print('MAE: ', MAE_normlist[best_r2_index_norm])

    print('Best MAE kernel:')
    print(gpr_output_normalized[0][best_MAE_index_norm])
    print('R2: ', r2normlist[best_MAE_index_norm])
    print('MAE: ', MAE_normlist[best_MAE_index_norm])
    
    return [best_r2_index, best_MAE_index, best_r2_index_norm, best_MAE_index_norm]

def evaluate_trainfit(gpr_output, gpr_output_normalized, X_train, y_train):
    """
    Helper function to evalute fit to train set of models trained from train_gprs function
    """
    train_MAE = []
    train_r2 = []
    train_MAE_norm = []
    train_r2_norm = []
    train_std = []
    train_std_norm = []

    for i in range(len(gpr_output[1])):
        model = gpr_output[1][i]

        y_pred, std_pred = model.predict(X_train, return_std=True)

        MAE = sklearn.metrics.mean_absolute_error(y_train, y_pred)
        r2 = sklearn.metrics.r2_score(y_train, y_pred)

        stdnorm = np.linalg.norm(std_pred)

        train_MAE.append(MAE)
        train_r2.append(r2)
        train_std.append(stdnorm)

    for i in range(len(gpr_output_normalized[1])):
        model = gpr_output[1][i]

        y_pred, std_pred = model.predict(X_train, return_std=True)

        MAE = sklearn.metrics.mean_absolute_error(y_train, y_pred)
        r2 = sklearn.metrics.r2_score(y_train, y_pred)

        stdnorm = np.linalg.norm(std_pred)

        train_MAE_norm.append(MAE)
        train_r2_norm.append(r2)
        train_std_norm.append(stdnorm)


    print('Best non-norm models')
    mae_ind = np.argmin(train_MAE)
    print('Best MAE Model:')
    print('Index: ', mae_ind)
    print('Kernel :', gpr_output[0][mae_ind])
    print('MAE :', train_MAE[mae_ind])
    print('r2: ', train_r2[mae_ind])
    print('std norm: ', train_std[mae_ind])



    print('Best normalized y models')
    mae_ind = np.argmin(train_MAE_norm)
    print('Best MAE Model:')
    print('Index: ', mae_ind)
    print('Kernel :', gpr_output_normalized[0][mae_ind])
    print('MAE :', train_MAE_norm[mae_ind])
    print('r2: ', train_r2_norm[mae_ind])
    print('std norm: ', train_std_norm[mae_ind])
    
    return


def evaluate_crossval(val_out):
    """
    Evaluate cross val results
    """

    MAEscores = val_out[0]
    MAEscores_norm = val_out[1]

    r2_scores = val_out[2]
    r2_scores_norm = val_out[3]

    # get best model for MAE on val

    r2_minind = np.argmax(r2_scores)
    r2_minind_norm = np.argmax(r2_scores_norm)

    r2_min = np.max(r2_scores)
    r2_min_norm = np.max(r2_scores_norm)

    if r2_min >= r2_min_norm:
        print('Best r2 score model:')
        print('non-normalized y')
        print('index: ', r2_minind)
        print('MAE score: ', MAEscores[r2_minind])
        print('r2 score: ', r2_scores[r2_minind])

    if r2_min < r2_min_norm:
        print('Best r2 score model:')
        print('normalized y')
        print('index: ', r2_minind_norm)
        print('MAE score: ', MAEscores_norm[r2_minind_norm])
        print('r2 score: ', r2_scores_norm[r2_minind_norm])

    # get best model for r2 on val
    mae_minind = np.argmin(MAEscores)
    mae_minind_norm = np.argmin(MAEscores_norm)

    mae_min = np.min(MAEscores)
    mae_min_norm = np.min(MAEscores_norm)

    if mae_min <= mae_min_norm:
        print('Best MAE score model:')
        print('non-normalized y')
        print('index: ', mae_minind)
        print('MAE score: ', MAEscores[mae_minind])
        print('r2 score: ', r2_scores[mae_minind])

    if mae_min > mae_min_norm:
        print('Best MAE score model:')
        print('normalized y')
        print('index: ', mae_minind_norm)
        print('MAE score: ', MAEscores_norm[mae_minind_norm])
        print('r2 score: ', r2_scores_norm[mae_minind_norm])

    return
