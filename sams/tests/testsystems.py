"""
Test systems for perses automated design.

Examples
--------

Alanine dipeptide in various environments (vacuum, implicit, explicit):

>>> from perses.tests.testsystems import AlaninDipeptideSAMS
>>> testsystem = AlanineDipeptideTestSystem()
>>> system_generator = testsystem.system_generator['explicit']
>>> sams_sampler = testsystem.sams_sampler['explicit']

TODO
----
* Have all PersesTestSystem subclasses automatically subjected to a battery of tests.
* Add short descriptions to each class through a class property.

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
from functools import partial
from pkg_resources import resource_filename
from openeye import oechem
from openmmtools import testsystems

################################################################################
# CONSTANTS
################################################################################

################################################################################
# TEST SYSTEMS
################################################################################

class SAMSTestSystem(object):
    """
    Create a consistent set of samplers useful for testing.

    Properties
    ----------
    environments : list of str
        Available environments
    topologies : dict of simtk.openmm.app.Topology
        Initial system Topology objects; topologies[environment] is the topology for `environment`
    positions : dict of simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions corresponding to initial Topology objects
    system_generators : dict of SystemGenerator objects
        SystemGenerator objects for environments
    proposal_engines : dict of ProposalEngine
        Proposal engines
    themodynamic_states : dict of thermodynamic_states
        Themodynamic states for each environment
    mcmc_samplers : dict of MCMCSampler objects
        MCMCSampler objects for environments
    exen_samplers : dict of ExpandedEnsembleSampler objects
        ExpandedEnsembleSampler objects for environments
    sams_samplers : dict of SAMSSampler objects
        SAMSSampler objects for environments

    """
    def __init__(self):
        pass

class AlanineDipeptideVacuumSimulatedTempering(SAMSTestSystem):
    """
    Similated tempering for alanine dipeptide in implicit solvent.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from sams.tests.testsystems import AlanineDipeptideVacuumSimulatedTempering
    >>> testsystem = AlanineDipeptideVacuumSimulatedTempering()

    """
    def __init__(self):
        super(AlanineDipeptideVacuumSimulatedTempering, self).__init__()
        self.description = 'Alanine dipeptide in vacuum simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideVacuum
        testsystem = AlanineDipeptideVacuum()
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Create thermodynamic states.
        Tmin = 270 * unit.kelvin
        Tmax = 600 * unit.kelvin
        ntemps = 8 # number of temperatures
        from sams import ThermodynamicState
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature) for temperature in temperatures ]

        # Create SAMS samplers
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 500 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

class AlanineDipeptideExplicitSimulatedTempering(SAMSTestSystem):
    """
    Simulated tempering for alanine dipeptide in explicit solvent.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from sams.tests.testsystems import AlanineDipeptideExplicitSimulatedTempering
    >>> testsystem = AlanineDipeptideExplicitSimulatedTempering()

    """
    def __init__(self):
        super(AlanineDipeptideExplicitSimulatedTempering, self).__init__()
        self.description = 'Alanine dipeptide in explicit solvent simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideExplicit
        testsystem = AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Add a MonteCarloBarostat
        temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        pressure = 1.0 * unit.atmospheres
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        self.system.addForce(barostat)

        # Create thermodynamic states.
        Tmin = 270 * unit.kelvin
        Tmax = 600 * unit.kelvin
        ntemps = 32 # number of temperatures
        from sams import ThermodynamicState
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature, pressure=pressure) for temperature in temperatures ]

        # Create SAMS samplers
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 50 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

class AlanineDipeptideVacuumAlchemical(SAMSTestSystem):
    """
    Alchemical free energy calculation for alanine dipeptide in vacuum.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from sams.tests.testsystems import AlanineDipeptideExplicitSimulatedTempering
    >>> testsystem = AlanineDipeptideExplicitSimulatedTempering()

    """
    def __init__(self):
        super(AlanineDipeptideVacuumAlchemical, self).__init__()
        self.description = 'Alanine dipeptide in vacuum alchemical free energy calculation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideVacuum
        testsystem = AlanineDipeptideVacuum()
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Create thermodynamic states.
        temperature = 300 * unit.kelvin
        alchemical_atoms = range(0,22) # alanine dipeptide
        from alchemy import AbsoluteAlchemicalFactory
        factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False)
        self.system = factory.createPerturbedSystem()
        nlambda = 32 # number of alchemical intermediates
        from sams import ThermodynamicState
        alchemical_lambdas = np.linspace(1.0, 0.0, nlambda)
        self.thermodynamic_states = list()
        for alchemical_lambda in alchemical_lambdas:
            parameters = {'lambda_sterics' : alchemical_lambda, 'lambda_electrostatics' : alchemical_lambda}
            self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )

        # Create SAMS samplers
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 50 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

class AlanineDipeptideExplicitAlchemical(SAMSTestSystem):
    """
    Alchemical free energy calculation for alanine dipeptide in explicit solvent.

    Properties
    ----------
    topology : simtk.openmm.app.Topology
        The system Topology
    system : simtk.openmm.System
        The OpenMM System to simulate
    positions : simtk.unit.Quantity of [nparticles,3] with units compatible with nanometers
        Initial positions
    thermodynamic_states : list of ThermodynamicState
        List of thermodynamic states to be used in expanded ensemble sampling

    Examples
    --------

    >>> from sams.tests.testsystems import AlanineDipeptideExplicitSimulatedTempering
    >>> testsystem = AlanineDipeptideExplicitSimulatedTempering()

    """
    def __init__(self):
        super(AlanineDipeptideExplicitAlchemical, self).__init__()
        self.description = 'Alanine dipeptide in explicit solvent alchemical free energy calculation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideExplicit
        testsystem = AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Add a MonteCarloBarostat
        #temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        #pressure = 1.0 * unit.atmospheres
        #barostat = openmm.MonteCarloBarostat(pressure, temperature)
        #self.system.addForce(barostat)

        # Create thermodynamic states.
        temperature = 300 * unit.kelvin
        pressure = 1.0 * unit.atmospheres
        alchemical_atoms = range(0,22) # alanine dipeptide
        from alchemy import AbsoluteAlchemicalFactory
        factory = AbsoluteAlchemicalFactory(testsystem.system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False)
        self.system = factory.createPerturbedSystem()
        nlambda = 512 # number of alchemical intermediates
        from sams import ThermodynamicState
        alchemical_lambdas = np.linspace(1.0, 0.0, nlambda)
        self.thermodynamic_states = list()
        for alchemical_lambda in alchemical_lambdas:
            parameters = {'lambda_sterics' : alchemical_lambda, 'lambda_electrostatics' : alchemical_lambda}
            #parameters = {'lambda_electrostatics' : alchemical_lambda}
            #self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, pressure=pressure, parameters=parameters) )
            self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )

        # Create SAMS samplers
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 50 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

def test_testsystems():
    np.set_printoptions(linewidth=130, precision=3)
    # TODO: Automatically discover subclasses of SAMSTestSystem
    niterations = 5
    import sams
    for testsystem_name in ['AlanineDipeptideVacuumSimulatedTempering', 'AlanineDipeptideExplicitSimulatedTempering', 'AlanineDipeptideVacuumAlchemical', 'AlanineDipeptideExplicitAlchemical']:
        testsystem = getattr(sams.tests.testsystems, testsystem_name)
        test = testsystem()
        f = partial(test.mcmc_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' MCMC simulation'
        yield f
        f = partial(test.exen_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' expanded ensemble simulation'
        yield f
        f = partial(test.sams_sampler.run, niterations)
        f.description = 'Testing ' + test.description + ' SAMS simulation'
        yield f

if __name__ == '__main__':
    testsystem = AlanineDipeptideExplicitAlchemical()
    niterations = 1000
    testsystem.mcmc_sampler.nsteps = 50
    testsystem.mcmc_sampler.pdbfile = None
    testsystem.exen_sampler.update_scheme = 'local'
    testsystem.exen_sampler.locality = 5    
    testsystem.sams_sampler.run(niterations)
