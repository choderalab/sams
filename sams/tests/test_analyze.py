"""
Test SAMS analysis code.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

from simtk import openmm, unit
from simtk.openmm import app
import os, os.path
import sys, math
import numpy as np
import logging
from functools import partial

import sams.analysis

################################################################################
# TEST SAMPLERS
################################################################################

def test_analysis():
    """
    Test analysis.

    """
    from sams.tests.testsystems import AlanineDipeptideVacuumSimulatedTempering
    netcdf_filename = 'output.nc'
    test = AlanineDipeptideVacuumSimulatedTempering(netcdf_filename=netcdf_filename)
    testsystem_name = test.__class__.__name__
    niterations = 20 # number of iterations to run
    test.mcmc_sampler.nsteps = 5

    # Test SAMSSampler.
    test.sams_sampler.run(niterations)

    # Test analysis
    from sams.analysis import analyze
    analyze(netcdf_filename, test, 'analyze.pdf')

if __name__=="__main__":
    test_analysis()
