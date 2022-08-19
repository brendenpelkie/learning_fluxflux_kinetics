import pandas as pd
import os
import pickle
import surfreact.utils as ut
import shutil
import numpy as np
import surfreact.reactions as rx
import surfreact.utils as ut
import surfreact.flintcff as flintcff
import surfreact.numpycff as numpycff
import logging


def dxsubmithelper(ctxfile_template):
    """
    Function to manage submissions of a dx sweep for one reaction and temperature
    This is to be run on the head node, and will set up directories and submit jobs for each dx value in the sweep. Currently the list of dx values is hard coded, this could be modified to pass via ctxfile_template. 

    Parameters:
    -----------
    ctxfile_template: template ctxfile.py file with everything needed for cff specced, except dx which should be called 'Q'

    Also requires submitCFF_template.sh and calcCFF.py to be in directory called from

    Returns:
    --------
    Nothing. makes a dx folder, with subdirectories to hold calculation results from each dx value
    """

    dxlist = ctxfile_template.dxlist
    
    # Future Feature: look at T and decide if numpy or flint shuold be used. Not sure how to change this - find and replace? Separate prep/calculate wrapper functions?
    # come back and add this feature later - not needed right away

    #make a directory to hold all dx results, move into there
    try:
        os.mkdir('dx')
    except FileExistsError:
        userinput = input('dx Directory exists. To continue, enter "y" This will nuke anything in this directory. To exit, enter anything else')
        if userinput == 'y':
            os.system('rm -r dx/*') #nuke contents. May be a better way.
        else:
            raise Exception('Directory exists and overwrite not authorized. Exiting')

    shutil.copy('submitCFF_template.sh', 'dx')
    shutil.copy('ctxfile_template.py', 'dx')
    shutil.copy('calcCFF.py', 'dx')

    # get pwd to pass to each cff instance to write results
    submit_dir = os.getcwd()

    with ut.cd('dx') as dir:
        #assert len(os.listdir('.')==0), 'Directory not empty, something went wrong' # not working b/c i added copying above this. Might fix later, probably not needed
         # initialize a file to store all the results. Should probably be a csv file 
        results = np.zeros((len(dxlist), 3))
        results_df = pd.DataFrame(results, columns = ['dx [au]', 'Number Points', 'Cff(t=0)'])
        results_df['dx [au]'] = dxlist
        results_df.to_csv('dxConvergenceResults_rxn{}_{}K.csv'.format(ctxfile_template.reactionID, ctxfile_template.T))
    
    # Looping through each dx value:
        
        for dx in dxlist:
            #1. make a directory/'move' there
            os.mkdir('dx_{}'.format(dx))
            # 2.  copy needed scripts into there: ctx file, a submit script, python script to run\
            shutil.copy('submitCFF_template.sh', 'dx_{}'.format(dx))
            shutil.copy('ctxfile_template.py', 'dx_{}'.format(dx))
            shutil.copy('calcCFF.py', 'dx_{}'.format(dx))

            with ut.cd('dx_{}'.format(dx)):
            # 3. make edits to scripts to make them run
                with open('ctxfile_template.py', 'rt') as file: #should make a ctxfile template file, delete template after copying. 
                    filedata = file.read()

                filedata = filedata.replace("'Q'", '{}'.format(dx))
                filedata = filedata.replace("'nan'", '{}'.format(submit_dir))

                with open('ctxfile.py', 'wt') as fileout:
                    fileout.write(filedata)
        
                with open('submitCFF_template.sh', 'rt') as file: #should make a ctxfile template file, delete template after copying. 
                    with open('submitCFF.sh', 'wt') as fileout:
                        for line in file:
                            fileout.write(line.replace('zzz', '{}_{}'.format(ctxfile_template.reactionID, dx)))

             #4. Submit jobs
                
                os.system('sbatch submitCFF.sh')
                print('Submitted dx {}'.format(dx))
                os.remove('ctxfile_template.py')
                os.remove('submitCFF_template.sh')
            
            # clean up submit files
        os.remove('ctxfile_template.py')
        os.remove('submitCFF_template.sh')
        os.remove('calcCFF.py')
    # delete any template files or other extras

    return
    
def Lsubmithelper(ctxfile_template):
    """
    Function to manage submissions of an L/grid width sweep for one reaction and temperature
    This is to be run on the head node, and will set up directories and submit jobs for each L value in the sweep. Currently the list of L values is hard coded, this could be modified to pass via ctxfile_template. 

    Parameters:
    -----------
    ctxfile_template: template ctxfile.py file with everything needed for cff specced, except L which should be called 'Q'

    Also requires submitCFF_template.sh and calcCFF.py to be in directory called from

    Returns:
    --------
    Nothing. makes a L folder, with subdirectories to hold calculation results from each L value
    """

    Llist = ctxfile_template.Llist
    
    # Future Feature: look at T and decide if numpy or flint shuold be used. Not sure how to change this - find and replace? Separate prep/calculate wrapper functions?
    # come back and add this feature later - not needed right away

    #make a directory to hold all L results, move into there
    try:
        os.mkdir('L')
    except FileExistsError:
        userinput = input('L Directory exists. To continue, enter "y" This will nuke anything in this directory. To exit, enter anything else')
        if userinput == 'y':
            os.system('rm -r L/*') #nuke contents. May be a better way.
        else:
            raise Exception('Directory exists and overwrite not authorized. Exiting')

    shutil.copy('submitCFF_template.sh', 'L')
    shutil.copy('ctxfile_template.py', 'L')
    shutil.copy('calcCFF.py', 'L')

    # get pwd to pass to each cff instance to write results
    submit_dir = os.getcwd()

    with ut.cd('L') as dir:
        #assert len(os.listdir('.')==0), 'Directory not empty, something went wrong' # not working b/c i added copying above this. Might fix later, probably not needed
         # initialize a file to store all the results. Should probably be a csv file 
        results = np.zeros((len(Llist), 3))
        results_df = pd.DataFrame(results, columns = ['L [au]', 'Number Points', 'Cff(t=0)'])
        results_df['L [au]'] = Llist
        results_df.to_csv('LConvergenceResults_rxn{}_{}K.csv'.format(ctxfile_template.reactionID, ctxfile_template.T))
    
    # Looping through each L value:
        
        for L in Llist:
            #1. make a directory/'move' there
            os.mkdir('L_{}'.format(L))
            # 2.  copy needed scripts into there: ctx file, a submit script, python script to run\
            shutil.copy('submitCFF_template.sh', 'L_{}'.format(L))
            shutil.copy('ctxfile_template.py', 'L_{}'.format(L))
            shutil.copy('calcCFF.py', 'L_{}'.format(L))

            with ut.cd('L_{}'.format(L)):
            # 3. make edits to scripts to make them run
                with open('ctxfile_template.py', 'rt') as file: #should make a ctxfile template file, delete template after copying. 
                    filedata = file.read()

                filedata = filedata.replace("'Q'", '{}'.format(L))
                filedata = filedata.replace("'nan'", '{}'.format(submit_dir))

                with open('ctxfile.py', 'wt') as fileout:
                    fileout.write(filedata)
        
                with open('submitCFF_template.sh', 'rt') as file: #should make a ctxfile template file, delete template after copying. 
                    with open('submitCFF.sh', 'wt') as fileout:
                        for line in file:
                            fileout.write(line.replace('zzz', '{}_L{}'.format(ctxfile_template.reactionID, L)))

             #4. Submit jobs
                
                os.system('sbatch submitCFF.sh')
                print('Submitted L {}'.format(L))
                os.remove('ctxfile_template.py')
                os.remove('submitCFF_template.sh')
            
            # clean up submit files
        os.remove('ctxfile_template.py')
        os.remove('submitCFF_template.sh')
        os.remove('calcCFF.py')
    # delete any template files or other extras

    return


def run_cff_calculation(ctxfile):
    """
    Function to manage submissions of a dx sweep for one reaction and temperature
    This is to be run on the head node, and will set up directories and submit jobs for each dx value in the sweep. Currently the list of dx values is hard coded, this could be modified to pass via ctxfile_template. 

    Parameters:
    -----------
    ctxfile_template: template ctxfile.py file with everything needed for cff specced, except dx which should be called 'Q'
    Tmax: Max temperature to submit jobs for. Allows for numpy jobs that have already been run to not be re-submitted. default 1000K

    Also requires submitCFF_template.sh and calcCFF.py to be in directory called from

    Returns:
    --------
    Nothing. makes a dx folder, with subdirectories to hold calculation results from each dx value
    """
    pathtodata = ctxfile_template.path_to_data
    tmax = ctxfile_template.tmax
    reactionID = ctxfile_template.reactionID

    dataset = pd.read_json(str(pathtodata + '/cff_dataset/cff_dataset.json'))
    Tlist = list(dataset[dataset.reaction_id == reactionID]['Temperature_list'])[0]
    dx = float(dataset[dataset.reaction_id == reactionID]['dx_au'])
    L = int(dataset[dataset.reaction_id == reactionID]['L_au'])
    index = dataset[dataset['reaction_id']==reactionID].index[0]
    barrier = dataset.loc[index, 'barrier_type']

    #make a directory to hold all dx results, move into there
    try:
        os.mkdir(f'results_reaction{reactionID}')
    except FileExistsError:
        userinput = input('production_results directory exists. To continue, enter "y" This will nuke anything in this directory. To exit, enter anything else')
        if userinput == 'y':
            os.system('rm -r production_results/*') #nuke contents. May be a better way.
        else:
            raise Exception('Directory exists and overwrite not authorized. Exiting')



    # get pwd to pass to each cff instance to write results
    submit_dir = os.getcwd()

    ###### Need to loop over each T and make a job for each one 
    

    with ut.cd(f'results_reaction{reactionID}') as dir:
         # initialize a file to store all the results. Should probably be a csv file 
        for T in Tlist:

            T = int(T)
            os.mkdir('T{}K'.format(T))
            # 2.  copy needed scripts into there: ctx file, a submit script, python script to run\
            shutil.copy('submitCFF_template.sh', 'T{}K'.format(T))
            shutil.copy('ctxfile_template.py', 'T{}K'.format(T))
            # one janky way to handle the flint/numpy divide
            if T >= 300:
                shutil.copy('calcCFF_numpy.py', 'T{}K/calcCFF.py'.format(T))
            if T < 300:
                shutil.copy('calcCFF_flint.py', 'T{}K/calcCFF.py'.format(T))

            with ut.cd('T{}K'.format(T)):
    # Looping through each job number, set up dir for that, and submit:
        
                for nodenum in range(0, int(njobs)):
                    #1. make a directory/'move' there
                    os.mkdir('job_{}'.format(nodenum))
                    # 2.  copy needed scripts into there: ctx file, a submit script, python script to run\
                    shutil.copy('submitCFF_template.sh', 'job_{}'.format(nodenum))
                    shutil.copy('ctxfile_template.py', 'job_{}'.format(nodenum))
                    shutil.copy('calcCFF.py', 'job_{}'.format(nodenum))

                    with ut.cd('job_{}'.format(nodenum)):
                    # 3. make edits to scripts to make them run
                        with open('ctxfile_template.py', 'rt') as file: #should make a ctxfile template file, delete template after copying. 
                            filedata = file.read()

                        filedata = filedata.replace("'nan'", '{}'.format(submit_dir))
                        filedata = filedata.replace("'@@@'", '{}'.format(njobs))
                        filedata = filedata.replace("'$$'", '{}'.format(nodenum))
                        filedata = filedata.replace("'Q'", '{}'.format(T))
                        filedata = filedata.replace("'dxplaceholder'", '{}'.format(dx))
                        filedata = filedata.replace("'Lplaceholder'", '{}'.format(L))
                        filedata = filedata.replace("'barrierplaceholder'", '{}'.format(barrier))

                        with open('ctxfile.py', 'wt') as fileout:
                            fileout.write(filedata)
                
                        with open('submitCFF_template.sh', 'rt') as file: #should make a ctxfile template file, delete template after copying. 
                            with open('submitCFF.sh', 'wt') as fileout:
                                for line in file:
                                    fileout.write(line.replace('zzz', '{}_{}K_job{}'.format(reactionID, T, nodenum)))

                    #4. Submit jobs
                        
                        os.system('sbatch submitCFF.sh')
                        print('Submitted node {}'.format(nodenum))
                        os.remove('ctxfile_template.py')
                        os.remove('submitCFF_template.sh')

                os.remove('ctxfile_template.py')
                os.remove('submitCFF_template.sh')
                os.remove('calcCFF.py')
            # clean up submit files
        os.remove('ctxfile_template.py')
        os.remove('submitCFF_template.sh')
        try:
            os.remove('calcCFF_flint.py')
        except FileNotFoundError:
            os.remove('calcCFF_numpy.py')
    # delete any template files or other extras

    return


def flintCff_calculationManger(ctxfile, barrier = 'eckartSingleAsymm', timeOffset = False):
    """
    Calculation manager function for Cff calculations. Takes a completed ctxfile module? object? whatever as input, uses this to set up ctx calculation. 
    """
    # compile variables from ctx file and reaction database

    dprec = ctxfile.dprec
    T = ctxfile.T
    dx = ctxfile.dx
    reactionID = ctxfile.reactionID
    tmax = ctxfile.tmax
    dt = ctxfile.dt
    nodenum = int(ctxfile.nodenum)
    njobs = int(ctxfile.njobs)
    submitdir = ctxfile.submitdir
    L = ctxfile.L
    try:
        tmin = ctxfile.tmin
    except:
        tmin = 0

    # danger: hard coded file path. I will probably regret this sometime
    filepath = '/mmfs1/gscratch/davinci/usr/bgpelkie/git_repos/surface_reactions/data'

    logging.info('Using flint Cff Methods')
    logging.info('Grid width: {}'.format(L))
    logging.info('Temperature: {}'.format(T))
    logging.info('dx: {}'.format(dx))

    # code to let user assign mass. Adding this on 2/15/22 to do some testing. Can remove once onto production
    try:
        m = ctxfile.m
        logging.info('Using user defined mass {} g/mol'.format(m))
    except NameError:
        m = 'usef'

    #calculate times
    if timeOffset:
        times = get_times_dt_offset(tmax, dt, njobs, nodenum+1)
    else:
        times = get_times_dt(tmax, dt, njobs, nodenum+1, tmin = tmin)
    # run setup/bmn/etc calculations
    H, evecs, evals, fluxmatrix = flintcff.CFFsetup(reactionID, L, dx, filepath, barrier = barrier, m = m)

    bmn = flintcff.computeBoltzmann(evecs, evals, dprec, T, save=False)
    boltFlux = flintcff.computeBoltFlux(fluxmatrix, bmn, dprec, save=False)
    # run cff calculation


    Cff = flintcff.computeCFF_SV(evals, evecs, fluxmatrix, boltFlux, times, dprec)
    # write results to concatenated outputs file

    numpts = flintcff.setnumpts(L, dx)
    # report results to a central csv file for the parameter sweep. This is risky - I'm relying on no two processes writing to the file at the same time. Python does not have a good way to lock files it seems. 
    try:
        resultsdf = pd.read_csv(str(submitdir+'/dx/dxConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)), index_col = 'dx [au]') #this might be an issue on klone since it doesn't like relative paths. Would have to keep track of submit directory somewhere
        resultsdf.loc[dx, 'Number Points'] = numpts
        resultsdf.loc[dx, 'Cff(t=0)'] = Cff
        resultsdf.to_csv(str(submitdir+'/dx/dxConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)))

        logging.info('Recorded Cff value to main CSV file')
    except FileNotFoundError:
        try:
            
            resultsdf = pd.read_csv(str(submitdir+'/L/LConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)), index_col = 'L [au]') #this might be an issue on klone since it doesn't like relative paths. Would have to keep track of submit directory somewhere
            resultsdf.loc[L, 'Number Points'] = numpts
            resultsdf.loc[L, 'Cff(t=0)'] = Cff
            resultsdf.to_csv(str(submitdir+'/L/LConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)))
            
            logging.info('Recorded Cff value to main CSV file')
        except FileNotFoundError:
            print('Results csv not found, results not saved. See compute log for Cff result')
        except:
            print('something else went wrong')
    return

def numpyCff_calculationManger(ctxfile, barrier = 'eckartSingleAsymm', numjobs = 1, timeOffset = False):
    """
    Calculation manager function for Cff calculations. Takes a completed ctxfile module? object? whatever as input, uses this to set up ctx calculation. 
    """
    # compile variables from ctx file and reaction database

    T = ctxfile.T
    dx = ctxfile.dx
    reactionID = ctxfile.reactionID
    tmax = ctxfile.tmax
    dt = ctxfile.dt
    nodenum = int(ctxfile.nodenum)
    njobs = int(ctxfile.njobs)
    submitdir = ctxfile.submitdir
    L = ctxfile.L
    filepath = ctxfile.path_to_data
    try:
        tmin = ctxfile.tmin
    except:
        tmin = 0

    logging.info('Using numpy Cff methods')
    logging.info('Grid width: {}'.format(L))
    logging.info('Temperature: {}'.format(T))
    logging.info('dx: {}'.format(dx))

    # code to let user assign mass. Adding this on 2/15/22 to do some testing. Can remove once onto production
    try:
        m = ctxfile.m
        logging.info('Using user defined mass {} g/mol'.format(m))
    except NameError:
        m = 'usef'

    # get times:
    if timeOffset:
        times = get_times_dt_offset(tmax, dt, njobs, nodenum+1)
    else:
        times = get_times_dt(tmax, dt, njobs, nodenum+1, tmin = tmin)
    # run setup/bmn/etc calculations
    H, evecs, evals, fluxmatrix = numpycff.CFFsetup(reactionID, L, dx, filepath, barrier = barrier, m = m)

    bmn = numpycff.computeBoltzmann(evecs, evals, T, save=False)
    boltFlux = numpycff.computeBoltFlux(fluxmatrix, bmn, save=False)
    # run cff calculation


    Cff = numpycff.computeCFF(evals, evecs, fluxmatrix, boltFlux, times, reactionID, titlestring = '')
    # write results to concatenated outputs file

    numpts = numpycff.setnumpts(L, dx)
    # report results to a central csv file for the parameter sweep. This is risky - I'm relying on no two processes writing to the file at the same time. Python does not have a good way to lock files it seems. 
    try:
        resultsdf = pd.read_csv(str(submitdir+'/dx/dxConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)), index_col = 'dx [au]') #this might be an issue on klone since it doesn't like relative paths. Would have to keep track of submit directory somewhere
        resultsdf.loc[dx, 'Number Points'] = numpts
        resultsdf.loc[dx, 'Cff(t=0)'] = Cff
        resultsdf.to_csv(str(submitdir+'/dx/dxConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)))

        logging.info('Recorded Cff value to main CSV file')
    except FileNotFoundError:
        try:
            
            resultsdf = pd.read_csv(str(submitdir+'/L/LConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)), index_col = 'L [au]') #this might be an issue on klone since it doesn't like relative paths. Would have to keep track of submit directory somewhere
            resultsdf.loc[L, 'Number Points'] = numpts
            resultsdf.loc[L, 'Cff(t=0)'] = Cff
            resultsdf.to_csv(str(submitdir+'/L/LConvergenceResults_rxn{}_{}K.csv'.format(ctxfile.reactionID, ctxfile.T)))
            
            logging.info('Recorded Cff value to main CSV file')
        except FileNotFoundError:
            print('Results csv not found, results not saved. See compute log for Cff result')
        except:
            print('something else went wrong')
    return


def get_times_dt(tmax, dt, nNodes, nodenum, tmin=0):
    """
    Assume nodenum starts from 1
    """

    time = np.arange(tmin, tmax, dt)
    #print(nNodes)
    #print(time)
    timeChunks = np.array_split(time, nNodes)
    #print(timeChunks)

    return timeChunks[nodenum-1]

def get_times_dt_offset(tmax_offset, dt, nNodes, nodenum):
    """
    Assume nodenum starts from 1. Calculates offset time interval to increase time resolution for calculations originally calculated with time step dt
    """
    tstart = int(dt/2)
    time = np.arange(tstart, tmax_offset, dt)
    timeChunks = np.array_split(time, nNodes)

    return timeChunks[nodenum-1]
