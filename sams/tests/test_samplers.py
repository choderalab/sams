"""
Test SAMS sampler options.

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

################################################################################
# TEST SAMPLERS
################################################################################

def test_sampler_options():
    """
    Test sampler options on a single test system.

    """
    from sams.tests.testsystems import WaterBoxAlchemical
    test = WaterBoxAlchemical()
    testsystem_name = test.__class__.__name__
    niterations = 5 # number of iterations to run
    test.mcmc_sampler.nsteps = 50

    # Test MCMCSampler sampler options.
    f = partial(test.mcmc_sampler.run, niterations)
    f.description = "Testing MCMC sampler with %s" % (testsystem_name)
    yield f

    # Test ExpandedEnsembleSampler samplers.
    for update_scheme in test.exen_sampler.supported_update_schemes:
        test.exen_sampler.update_scheme = update_scheme
        f = partial(test.exen_sampler.run, niterations)
        f.description = "Testing expanded ensemble sampler with %s using expanded ensemble update scheme '%s'" % (testsystem_name, update_scheme)
        yield f

    # Test SAMSSampler samplers.
    for update_scheme in test.exen_sampler.supported_update_schemes:
        test.exen_sampler.update_scheme = update_scheme
        for update_method in test.sams_sampler.supported_update_methods:
            test.sams_sampler.update_method = update_method
            f = partial(test.sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s using expanded ensemble update scheme '%s' and SAMS update method '%s'" % (testsystem_name, update_scheme, update_method)
            yield f


if __name__=="__main__":
    test_sampler_options()
