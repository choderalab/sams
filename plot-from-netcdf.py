import numpy as np
import netCDF4

ncfile = netCDF4.Dataset('output.nc', 'r')

[nsamples, nstates] = ncfile.variables['logZ'].shape

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn

#testsystem_name = 'HarmonicOscillatorSimulatedTempering'
#testsystem_name = 'AlanineDipeptideVacuumSimulatedTempering'
#testsystem_name = 'AlanineDipeptideExplicitSimulatedTempering'
#testsystem_name = 'WaterBoxAlchemical'
testsystem_name = 'AblImatinibExplicitAlchemical'
import sams.tests.testsystems
testsystem_class = getattr(sams.tests.testsystems, testsystem_name)
testsystem = testsystem_class()
nstates = len(testsystem.thermodynamic_states)

with PdfPages('alchemy.pdf') as pdf:
    # PAGE 1
    plt.figure(figsize=(6, 6))

    if hasattr(testsystem, 'logZ'):
        plt.hold(True)
        plt.plot(testsystem.logZ, 'ro')
        print testsystem.logZ

    logZ = ncfile.variables['logZ'][-2,:]
    plt.plot(logZ, 'ko')
    plt.title(testsystem.description)
    plt.xlabel('state index $j$')
    plt.ylabel('$\zeta^{(t)}$')
    plt.axis([0, nstates-1, min(logZ), max(logZ)])
    if hasattr(testsystem, 'logZ'):
        plt.axis([0, nstates-1, 0.0, max(testsystem.logZ)])
    pdf.savefig()  # saves the current figure into a pdf page

    # PAGE 2
    plt.figure(figsize=(6, 6))
    state_index = ncfile.variables['state_index'][:]
    plt.plot(state_index, '.')
    plt.title(testsystem.description)
    plt.xlabel('iteration $t$')
    plt.ylabel('state index')
    plt.axis([0, nsamples, 0, nstates-1])
    pdf.savefig()  # saves the current figure into a pdf page

    # PAGE 3
    plt.figure(figsize=(6, 6))
    if hasattr(testsystem, 'logZ'):
        plt.hold(True)
        M = np.tile(testsystem.logZ, [nsamples,1])
        plt.plot(M, ':')
    logZ = ncfile.variables['logZ'][:,:]
    plt.plot(logZ[:,:], '-')
    plt.title(testsystem.description)
    plt.xlabel('iteration $t$')
    plt.ylabel('$\zeta^{(t)}$')
    plt.axis([0, nsamples, logZ.min(), logZ.max()])
    pdf.savefig()  # saves the current figure into a pdf page

    # FINISH
    plt.close()
