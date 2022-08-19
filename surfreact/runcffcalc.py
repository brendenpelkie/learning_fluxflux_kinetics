import logging
import os 
import pandas as pd
import surfreact.reactions as rx
import surfreact.utils as ut
import surfreact.numpycff as numpycff
import surfreact.submitManagers as submit
import numpy as np

# hard coded things

reactionID = 4105
temperature_index = 0 # index of assigned temperature from list in cff_dataset.json table. can be 0, 1, 2, 3
dt = 70
path_to_data = 'data/cff_dataset/'
m = 'usedf'
tmin = 0
tmax = 20000

# get values from dataframe
cff_data = pd.read_json(path_to_data + 'cff_dataset.json')

reaction = cff_data[cff_data['reaction_id'] == reactionID]

T = reaction.Temperature_list[temperature_index]
dx = reaction.dx_au
L = reaction.L_au
barrier = reaction.barrier_type



# set up logging
logging.basicConfig(filename="Computelog_CFF_reaction{}.log".format(reactionID), level=logging.INFO, filemode='w')

logging.info('Using numpy Cff methods')
logging.info('Grid width: {}'.format(L))
logging.info('Temperature: {}'.format(T))
logging.info('dx: {}'.format(dx))



times = submit.get_times_dt(tmax, dt, 1, 1, tmin = tmin)
# run setup/bmn/etc calculations
H, evecs, evals, fluxmatrix = numpycff.CFFsetup(reactionID, L, dx, path_to_data, barrier = barrier, m = m)

bmn = numpycff.computeBoltzmann(evecs, evals, T, save=False)
boltFlux = numpycff.computeBoltFlux(fluxmatrix, bmn, save=False)
# run cff calculation


Cff = numpycff.computeCFF(evals, evecs, fluxmatrix, boltFlux, times, reactionID, titlestring = '')
# write results to concatenated outputs file

numpts = numpycff.setnumpts(L, dx)


resultsdf = pd.DataFrame({'time [au]':times, 'Cff values':Cff})
resultsdf.to_csv(f'Cffresults_reaction{reactionID}_temperature{T}K.csv')


logging.info('Recorded Cff value to main CSV file')
