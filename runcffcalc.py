import logging
import pandas as pd
import surfreact.numpycff as numpycff
import surfreact.flintcff as flintcff
import surfreact.submitManagers as submit

### User modifiable variables

reaction_number = 8 # reaction number to run cff calculation for
T = 150 # temperature in K
path_to_data = 'data' #filepath to the data directory

# 
timeOffset = True # whether to offset time steps by 1/2 to fill in gaps at early times. 
tmax = 20000 # max time in au time to run cff calculation for


### Values below here are not intended for user modification
m = 'usedf' # get mass from dataset
dt = 70 # time step, au
tmin = 0 # start at time 0
dprec = 200 # decimal precision for flint


# get values from dataframe
cff_data = pd.read_json(path_to_data + '/cff_dataset/cff_dataset.json')
reaction = cff_data[cff_data['reaction_number'] == reaction_number]
reactionID = int(reaction.reaction_id)
dx = 1#float(reaction.dx_au)
L = float(reaction.L_au)
barrier = reaction.barrier_type.values[0]


logging.basicConfig(filename="Computelog_CFF_reaction{}.log".format(reactionID), level=logging.INFO, filemode='w')

if T >= 300:
    # run with numpy cff module
    logging.info('Using numpy Cff methods')
    logging.info('Grid width: {}'.format(L))
    logging.info('Temperature: {}'.format(T))
    logging.info('dx: {}'.format(dx))

    if timeOffset:
        times = submit.get_times_dt_offset(tmax, dt, 1, 1)
    else:
        times = submit.get_times_dt(tmax, dt, 1, 1, tmin = tmin)

    # run setup/bmn/etc calculations
    H, evecs, evals, fluxmatrix = numpycff.CFFsetup(reactionID, L, dx, path_to_data, barrier = barrier, m = m)
    bmn = numpycff.computeBoltzmann(evecs, evals, T, save=False)
    boltFlux = numpycff.computeBoltFlux(fluxmatrix, bmn, save=False)
    # run cff calculation
    Cff = numpycff.computeCFF(evals, evecs, fluxmatrix, boltFlux, times, reactionID, titlestring = '')

    # write results to concatenated outputs file
    resultsdf = pd.DataFrame({'time [au]':times, 'Cff values':Cff})
    resultsdf.to_csv(f'Cffresults_reaction{reactionID}_temperature{T}K.csv')
    logging.info('Recorded Cff value to main CSV file')

else:
    # run using flint high precision methods

    logging.info('Using flint Cff Methods')
    logging.info('Grid width: {}'.format(L))
    logging.info('Temperature: {}'.format(T))
    logging.info('dx: {}'.format(dx))

    #calculate times
    if timeOffset:
        times = submit.get_times_dt_offset(tmax, dt, 1, 1)
    else:
        times = submit.get_times_dt(tmax, dt, 1, 1, tmin = tmin)

    # run setup/bmn/etc calculations
    H, evecs, evals, fluxmatrix = flintcff.CFFsetup(reactionID, L, dx, path_to_data, barrier = barrier, m = m)
    bmn = flintcff.computeBoltzmann(evecs, evals, dprec, T, save=False)
    boltFlux = flintcff.computeBoltFlux(fluxmatrix, bmn, dprec, save=False)
    # run cff calculation
    Cff = flintcff.computeCFF_SV(evals, evecs, fluxmatrix, boltFlux, times, dprec)
    
    # write results to file
    resultsdf = pd.DataFrame({'time [au]':times, 'Cff values':Cff})
    resultsdf.to_csv(f'Cffresults_reaction{reactionID}_temperature{T}K.csv')
    logging.info('Recorded Cff value to main CSV file')