import numpy as np
import ase
import surfreact.numpycff as cff
import surfreact.utils as ut
import os

def get_nearest_metals(reactant_atom, product_atom, metal_symbol, n = 5):
    """
    Finds the nearest n metal atoms to the adsorbates, and returns an atom object containing the adsorbate and these metals for the reactants and products.
    
    For use with representation methods that require including a few metal atoms along with organic adsorbates. Assumes single metal type in surface, and that adsorbate
    atoms all occur at end of atom. (last index positions). Distances will be calculated using the reactant positions.
    
    Parameters:
    ----------
    reactant_atom (ase atom): ase atom containing reactant positions
    product_atom (ase atom): atom object containing product positions
    metal_symbol (str): Chemical symbol of metal making up surface in reactant and product. Ex 'Cu'
    n (int, default 5): Number of metal atoms to include in output geometry
    
    Returns:
    -------
    cleaned_reactant (ase atom): ase atom object with n nearest metals and adsorbates
    cleaned_product (ase atom): ase atom object with n nearest metals and adsorbates
    """
    # cleaning assert statements:
    assert reactant_atom.get_chemical_symbols()[0] == metal_symbol, 'First atom symbol in reactants does not match metal symbol. Check metal name and ensure atoms objects are ordered so metals are listed before adsorbates'
    assert len(reactant_atom) == len(reactant_atom), 'Ensure reactant and product atoms have the same length'
    assert len(reactant_atom) > n, 'n must be less than len(reactant_atom)'
    
    # 1. get needed values
    count = count_nonmetalatoms(reactant_atom, metal_symbol)
    interatomic_distances = reactant_atom.get_all_distances()
    numatoms = reactant_atom.get_global_number_of_atoms()
    
    # 2. Calculate average distances between metal atoms and all adsorbates
    # set up staging array for avg distance between each metal atom and adsorbate atoms
    avg_adsorbate_distance = np.zeros((numatoms-count, 1))
    #loop over each metal atom (should be first n-count atoms), and calculate mean distance between that atom and the non-metal atoms.
    for i in range(0, numatoms - count):
        #print(i)
        distances = interatomic_distances[i]
        mean_distance = np.mean(distances[-count:])
        #print(mean_distance)
        avg_adsorbate_distance[i] = mean_distance
    
    #3. Rank distances and find the indices of the closest metals, create list of indices of all atoms to keep on export
    closest_atoms = np.argsort(avg_adsorbate_distance, axis = 0)[:n]
    adsorbate_indices = np.arange(numatoms - count, numatoms, 1)
    select_atoms = np.concatenate((closest_atoms[:,0], adsorbate_indices))
    
    # 4. create new atoms objects to return that only have selected atoms
    
    cleaned_reactant = reactant_atom[select_atoms]
    cleaned_product = product_atom[select_atoms]
    
    return [cleaned_reactant, cleaned_product]
    

def count_nonmetalatoms(atom, metal_symbol):
    """
    counts the number of atoms that don't have the symbol 'metal symbol' in atom. Intended to count number of adsorbates in slab geometries
    
    Parameters:
    ----------
    atom: ase atom 
    metal_symbol: str, ex. Cu
    
    Returns:
    -------
    metal_count (int)
    """
    
    count = 0
    symbols = atom.get_chemical_symbols()
    for symbol in symbols:
        if symbol != metal_symbol:
            count+=1
            
    return count
