cff_dataset.json contains reaction information and the parameters needed to run flux-flux correlation calculations. See the zenodo dataset that accompanied this publication for a more complete description of data. The schema of this dataset is as follows:

- reaction_number: Reaction identifier number used in this work
- reaction_id: catalysis-hub reaction ID. This identifier is used internally to refer to reactions within the code
- reactants: Python dictionary object of reactants and their quantities
- products: Python dictionary object of products and their quantities
- metal_name: atomic symbol of metal surface
- facet_number: Miller indices of surface
- reaction_energy: reaction energy in electron-volts
- activation_energy: activation energy of reaction in electron-volts
- reactant_mass_gmol: Mass of reactants in grams/mol
- reaction_coordinates: 1d reaction coordinate in units Angstroms
- eckart_params_w_alpha_xshift: Fit parameters for 1d assymetric eckart barrier. Parameters in order are w, alpha, and an x-shift value used to center the barrier peak.
- barrier_type: type of barrier used in cff(t) calculation. Either eckart or skew-logistic.
- skewLogParams_skewA_skewscale_L_k_vscale: Fit parameters for skew logistic barrier, if applicable. In order, A, scake, K, k, and vertical scale parameters.
- Temp_min: Minimum temperature in Kelvin that DVR grid parameters have been verified to. Values less than 300 require the flintcff module to avoid numerical stability issues
- dx_au: grid spacing of DVR grid in au (space between grid points)
- L_au: half-width of DVR grid (measured -L -> 0 or 0 -> L) in au
- Temperature_list: list of temperatures for Cff(t) calculations in K assigned to that reaction 
