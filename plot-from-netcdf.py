import numpy as np
import netCDF4

ncfile = netCDF4.Dataset('output.nc', 'r')

[nsamples, nstates] = ncfile.variables['logZ'].shape

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn

testsystem_name = 'HarmonicOscillatorSimulatedTempering'
import sams.tests.testsystems
testsystem_class = getattr(sams.tests.testsystems, testsystem_name)
testsystem = testsystem_class()
nstates = len(testsystem.thermodynamic_states)

with PdfPages('alchemy.pdf') as pdf:
    # PAGE 1
    plt.figure(figsize=(6, 6))

    if hasattr(testsystem, 'logZ'):
        plt.hold(True)
        plt.plot(testsystem.logZ, 'r-')
        print testsystem.logZ

    logZ = ncfile.variables['logZ'][-2,:]
    plt.plot(logZ, 'ko')
    plt.title(testsystem.description)
    plt.xlabel('state index')
    plt.ylabel('log Z estimate')
    plt.axis([0, nstates-1, min(logZ), max(logZ)])
    pdf.savefig()  # saves the current figure into a pdf page

    # PAGE 2
    plt.figure(figsize=(6, 6))
    state_index = ncfile.variables['state_index'][:]
    plt.plot(state_index, '.')
    plt.title(testsystem.description)
    plt.xlabel('iteration')
    plt.ylabel('state index')
    plt.axis([0, nsamples, 0, nstates-1])
    pdf.savefig()  # saves the current figure into a pdf page

    # PAGE 3
    plt.figure(figsize=(6, 6))

    if hasattr(testsystem, 'logZ'):
        plt.hold(True)
        M = np.tile(testsystem.logZ, [nsamples,1])
        print M
        print M.shape
        plt.plot(M, ':')

    logZ = ncfile.variables['logZ'][:,:]
    plt.plot(logZ[:,:], '-')
    plt.title(testsystem.description)
    plt.xlabel('iteration')
    plt.ylabel('logZ estimate')
    plt.axis([0, nsamples, 0, logZ.max()])
    pdf.savefig()  # saves the current figure into a pdf page


    # FINISH
    plt.close()
