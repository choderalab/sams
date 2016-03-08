"""
Samplers for perses automated molecular design.

TODO:
* Refactor tests into a test class so that AlanineDipeptideSAMS test system only needs to be constructed once for a battery of tests.
* Generalize tests of samplers to iterate over all PersesTestSystem subclasses

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
# TEST MCMCSAMPLER
################################################################################

#=============================================================================================
# TEST SYSTEM DEFINITIONS
#=============================================================================================

from openmmtools import testsystems

alchemical_test_systems = dict()
alchemical_test_systems['Lennard-Jones cluster'] = {
    'test' : testsystems.LennardJonesCluster(),
    'factory_args' : {'ligand_atoms' : range(0,1), 'receptor_atoms' : range(1,2) }}
alchemical_test_systems['Lennard-Jones fluid without dispersion correction'] = {
    'test' : testsystems.LennardJonesFluid(dispersion_correction=False),
    'factory_args' : {'ligand_atoms' : range(0,1), 'receptor_atoms' : range(1,2) }}
alchemical_test_systems['Lennard-Jones fluid with dispersion correction'] = {
    'test' : testsystems.LennardJonesFluid(dispersion_correction=True),
    'factory_args' : {'ligand_atoms' : range(0,1), 'receptor_atoms' : range(1,2) }}
alchemical_test_systems['TIP3P with reaction field, no charges, no switch, no dispersion correction'] = {
    'test' : testsystems.DischargedWaterBox(dispersion_correction=False, switch=False, nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }}
alchemical_test_systems['TIP3P with reaction field, switch, no dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=False, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }}
alchemical_test_systems['TIP3P with reaction field, no switch, dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=True, switch=False, nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }}
alchemical_test_systems['TIP3P with reaction field, switch, dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=True, switch=True, nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }}
alchemical_test_systems['alanine dipeptide in vacuum'] = {
    'test' : testsystems.AlanineDipeptideVacuum(),
    'factory_args' : {'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22) }}
alchemical_test_systems['alanine dipeptide in vacuum with annihilated bonds, angles, and torsions'] = {
    'test' : testsystems.AlanineDipeptideVacuum(),
    'factory_args' : {'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22),
    'alchemical_torsions' : True, 'alchemical_angles' : True, 'alchemical_bonds' : True }}
alchemical_test_systems['alanine dipeptide in vacuum with annihilated sterics'] = {
    'test' : testsystems.AlanineDipeptideVacuum(),
    'factory_args' : {'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22),
    'annihilate_sterics' : True, 'annihilate_electrostatics' : True }}
alchemical_test_systems['alanine dipeptide in OBC GBSA'] = {
    'test' : testsystems.AlanineDipeptideImplicit(),
    'factory_args' : {'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22) }}
alchemical_test_systems['alanine dipeptide in OBC GBSA, with sterics annihilated'] = {
    'test' : testsystems.AlanineDipeptideImplicit(),
    'factory_args' : {'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22),
    'annihilate_sterics' : True, 'annihilate_electrostatics' : True }}
alchemical_test_systems['alanine dipeptide in TIP3P with reaction field'] = {
    'test' : testsystems.AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,22), 'receptor_atoms' : range(22,22) }}
alchemical_test_systems['T4 lysozyme L99A with p-xylene in OBC GBSA'] = {
    'test' : testsystems.LysozymeImplicit(),
    'factory_args' : {'ligand_atoms' : range(2603,2621), 'receptor_atoms' : range(0,2603) }}
alchemical_test_systems['DHFR in explicit solvent with reaction field, annihilated'] = {
    'test' : testsystems.DHFRExplicit(nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,2849), 'receptor_atoms' : [],
    'annihilate_sterics' : True, 'annihilate_electrostatics' : True }}
alchemical_test_systems['Src in TIP3P with reaction field, with Src sterics annihilated'] = {
    'test' : testsystems.SrcExplicit(nonbondedMethod=app.CutoffPeriodic),
    'factory_args' : {'ligand_atoms' : range(0,4428), 'receptor_atoms' : [],
    'annihilate_sterics' : True, 'annihilate_electrostatics' : True }}
alchemical_test_systems['Src in GBSA'] = {
    'test' : testsystems.SrcImplicit(),
    'factory_args' : {'ligand_atoms' : range(0,4427), 'receptor_atoms' : [],
    'annihilate_sterics' : False, 'annihilate_electrostatics' : False }}
alchemical_test_systems['Src in GBSA, with Src sterics annihilated'] = {
    'test' : testsystems.SrcImplicit(),
    'factory_args' : {'ligand_atoms' : range(0,4427), 'receptor_atoms' : [],
    'annihilate_sterics' : True, 'annihilate_electrostatics' : True }}

# Problematic tests: PME is not fully implemented yet
alchemical_test_systems['TIP3P with PME, no switch, no dispersion correction'] = {
    'test' : testsystems.WaterBox(dispersion_correction=False, switch=False, nonbondedMethod=app.PME),
    'factory_args' : {'ligand_atoms' : range(0,3), 'receptor_atoms' : range(3,6) }}

alchemical_test_systems['toluene in implicit solvent'] = {
    'test' : testsystems.TolueneImplicit(),
    'factory_args' : {'ligand_atoms' : [0,1], 'receptor_atoms' : list(),
    'alchemical_torsions' : True, 'alchemical_angles' : True, 'annihilate_sterics' : True, 'annihilate_electrostatics' : True }}

# Slow tests
#alchemical_test_systems['Src in OBC GBSA'] = {
#    'test' : testsystems.SrcImplicit(),
#    'ligand_atoms' : range(0,21), 'receptor_atoms' : range(21,7208) }
#alchemical_test_systems['Src in TIP3P with reaction field'] = {
#    'test' : testsystems.SrcExplicit(nonbondedMethod=app.CutoffPeriodic),
#    'ligand_atoms' : range(0,21), 'receptor_atoms' : range(21,4091) }

accuracy_testsystem_names = [
    'Lennard-Jones cluster',
    'Lennard-Jones fluid without dispersion correction',
    'Lennard-Jones fluid with dispersion correction',
    'TIP3P with reaction field, no charges, no switch, no dispersion correction',
    'TIP3P with reaction field, switch, no dispersion correction',
    'TIP3P with reaction field, switch, dispersion correction',
    'alanine dipeptide in vacuum with annihilated sterics',
    'toluene in implicit solvent',
]

overlap_testsystem_names = [
    'Lennard-Jones cluster',
    'Lennard-Jones fluid without dispersion correction',
    'Lennard-Jones fluid with dispersion correction',
    'TIP3P with reaction field, no charges, no switch, no dispersion correction',
    'TIP3P with reaction field, switch, no dispersion correction',
    'TIP3P with reaction field, switch, dispersion correction',
    'alanine dipeptide in vacuum with annihilated sterics',
    'TIP3P with PME, no switch, no dispersion correction', # PME still lacks reciprocal space component; known energy comparison failure
    'toluene in implicit solvent',
]

def test_alchemical_samplers():
    """
    Test samplers on multiple alchemical test systems.

    """
    testsystem_names = accuracy_testsystem_names
    niterations = 5 # number of iterations to run

    # If TESTSYSTEMS environment variable is specified, test those systems.
    if 'TESTSYSTEM_RANGE' in os.environ:
        [start, stop] = os.environ['TESTSYSTEM_RANGE'].split('-')
        testsystem_names = testsystem_names[int(start):int(stop)]

    for testsystem_name in testsystem_names:
        test = alchemical_test_systems[testsystem_name]
        testsystem = test['test']
        factory_args = test['factory_args']
        factory = AbsoluteAlchemicalFactory(testsystem.system, **factory_args)
        alchemical_system = alchemical_system.createPerturbedSystem()

        # Test MCMCSampler samplers.
        for environment in testsystem.environments:
            mcmc_sampler = testsystem.mcmc_samplers[environment]
            f = partial(mcmc_sampler.run, niterations)
            f.description = "Testing MCMC sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test ExpandedEnsembleSampler samplers.
        for environment in testsystem.environments:
            exen_sampler = testsystem.exen_samplers[environment]
            f = partial(exen_sampler.run, niterations)
            f.description = "Testing expanded ensemble sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test SAMSSampler samplers.
        for environment in testsystem.environments:
            sams_sampler = testsystem.sams_samplers[environment]
            f = partial(sams_sampler.run, niterations)
            f.description = "Testing SAMS sampler with %s '%s'" % (testsystem_name, environment)
            yield f
        # Test MultiTargetDesign sampler for implicit hydration free energy
        from perses.samplers.samplers import MultiTargetDesign
        # Construct a target function for identifying mutants that maximize the peptide implicit solvent hydration free energy
        for environment in testsystem.environments:
            target_samplers = { testsystem.sams_samplers[environment] : 1.0, testsystem.sams_samplers['vacuum'] : -1.0 }
            designer = MultiTargetDesign(target_samplers)
            f = partial(designer.run, niterations)
            f.description = "Testing MultiTargetDesign sampler with %s transfer free energy from vacuum -> %s" % (testsystem_name, environment)
            yield f


if __name__=="__main__":
    for t in test_samplers():
        print(t.description)
        if(t.description) == "Testing expanded ensemble sampler with T4LysozymeInhibitorsTestSystem 'explicit'":
            t()
