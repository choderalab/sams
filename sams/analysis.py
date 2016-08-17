"""
Analysis for self-adjusted mixture sampling (SAMS).

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import numpy as np
import netCDF4
import os, os.path
import sys, math
import copy
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn

import sams.tests.testsystems

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# Thermodynamic state description
################################################################################

def analyze(netcdf_filename, testsystem, pdf_filename):
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    testsystem_name = testsystem.__class__.__name__
    nstates = len(testsystem.thermodynamic_states)

    with PdfPages(pdf_filename) as pdf:
        # PAGE 1
        plt.figure(figsize=(6, 6))

        if hasattr(testsystem, 'logZ'):
            plt.hold(True)
            plt.plot(testsystem.logZ, 'ro')
            print testsystem.logZ

        title_fontsize = 7

        logZ = ncfile.variables['logZ'][-2,:]
        plt.plot(logZ, 'ko')
        plt.title(testsystem.description, fontsize=title_fontsize)
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
        plt.title(testsystem.description, fontsize=title_fontsize)
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
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('$\zeta^{(t)}$')
        plt.axis([0, nsamples, logZ.min(), logZ.max()])
        pdf.savefig()  # saves the current figure into a pdf page

        # FINISH
        plt.close()
