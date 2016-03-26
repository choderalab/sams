import numpy as np
import netCDF4

ncfile = netCDF4.Dataset('output.nc', 'r')

logZ = ncfile.variables['logZ'][-1,:]
state_index = ncfile.variables['state_index'][:]

nsamples = len(state_index)
nstates = len(logZ)

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn

with PdfPages('alchemy.pdf') as pdf:
    plt.figure(figsize=(6, 6))
    plt.plot(logZ, '.')
    plt.title('electrostatic scaling with softcore Lennard-Jones')
    plt.xlabel('state index')
    plt.ylabel('ln Z')
    plt.axis([0, logZ.size-1, min(logZ), max(logZ)])
    pdf.savefig()  # saves the current figure into a pdf page

    plt.figure(figsize=(6, 6))
    plt.plot(state_index, '.')
    plt.title('electrostatic scaling with softcore Lennard-Jones')
    plt.xlabel('iteration')
    plt.ylabel('state index')
    plt.axis([0, nsamples, 0, nstates])

    pdf.savefig()  # saves the current figure into a pdf page


    plt.close()
