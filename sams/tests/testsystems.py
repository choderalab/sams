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
import mdtraj as md

################################################################################
# CONSTANTS
################################################################################

################################################################################
# SUBROUTINES
################################################################################

def minimize(testsystem):
    """
    Minimize all structures in test system.

    Parameters
    ----------
    testystem : PersesTestSystem
        The testsystem to minimize.

    """
    print("Minimizing '%s'..." % testsystem.description)
    collision_rate = 20.0 / unit.picoseconds
    temperature = 300 * unit.kelvin
    integrator = openmm.LangevinIntegrator(1.0 * unit.femtoseconds, collision_rate, temperature)
    context = openmm.Context(testsystem.system, integrator)
    context.setPositions(testsystem.positions)
    print ("Initial energy is %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    TOL = 1.0
    MAX_STEPS = 100
    openmm.LocalEnergyMinimizer.minimize(context, TOL, MAX_STEPS)
    print ("Final energy is   %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    # Take some steps.
    nsteps = 500
    integrator.step(nsteps)
    print ("After %d steps    %12.3f kcal/mol" % (nsteps, context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    # Update positions.
    testsystem.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    testsystem.mcmc_sampler.sampler_state.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Clean up.
    del context, integrator

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
    def __init__(self, netcdf_filename=None):
        import netCDF4
        self.ncfile = None
        if netcdf_filename is not None:
            self.ncfile = netCDF4.Dataset(netcdf_filename, mode='w')

class HarmonicOscillatorSimulatedTempering(SAMSTestSystem):
    """
    Similated tempering for 3D harmonic oscillator.

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

    >>> from sams.tests.testsystems import HarmonicOscillatorSimulatedTempering
    >>> testsystem = HarmonicOscillatorSimulatedTempering()

    """
    def __init__(self, **kwargs):
        super(HarmonicOscillatorSimulatedTempering, self).__init__(**kwargs)
        self.description = 'Harmonic oscillator simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import HarmonicOscillator
        K = 1.0 * unit.kilocalories_per_mole / unit.angstroms**2 # 3D harmonic oscillator spring constant
        mass = 39.948 * unit.amu # 3D harmonic oscillator particle mass
        period = 2.0 * np.pi * unit.sqrt(mass / K) # harmonic oscillator period
        timestep = 0.01 * period
        testsystem = HarmonicOscillator(K=K, mass=mass)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # Create thermodynamic states.
        Tmin = 100 * unit.kelvin
        Tmax = 1000 * unit.kelvin
        ntemps = 8 # number of temperatures
        from sams import ThermodynamicState
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature) for temperature in temperatures ]

        # Compute analytical logZ for each thermodynamic state.
        self.logZ = np.zeros([ntemps], np.float64)
        for (index, thermodynamic_state) in enumerate(self.thermodynamic_states):
            beta = thermodynamic_state.beta
            self.logZ[index] = - 1.5 * np.log(beta * K * unit.angstrom**2)
        self.logZ[:] -= self.logZ[0]

        # Create SAMS samplers
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.timestep = timestep
        self.mcmc_sampler.collision_rate = 1.0 / (100 * timestep)
        self.mcmc_sampler.nsteps = 1000
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler, update_stages='two-stage', update_method='optimal')
        self.sams_sampler.verbose = True


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
    def __init__(self, **kwargs):
        super(AlanineDipeptideVacuumSimulatedTempering, self).__init__(**kwargs)
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
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
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
    def __init__(self, **kwargs):
        super(AlanineDipeptideExplicitSimulatedTempering, self).__init__(**kwargs)
        self.description = 'Alanine dipeptide in explicit solvent simulated tempering simulation'

        # Create topology, positions, and system.
        from openmmtools.testsystems import AlanineDipeptideExplicit
        testsystem = AlanineDipeptideExplicit(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # DEBUG: Write PDB
        from simtk.openmm.app import PDBFile
        outfile = open('initial.pdb', 'w')
        PDBFile.writeFile(self.topology, self.positions, outfile)
        outfile.close()

        # Add a MonteCarloBarostat
        temperature = 270 * unit.kelvin # will be replaced as thermodynamic state is updated
        pressure = 1.0 * unit.atmospheres
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        self.system.addForce(barostat)

        # Create thermodynamic states.
        Tmin = 270 * unit.kelvin
        Tmax = 600 * unit.kelvin
        ntemps = 256 # number of temperatures
        from sams import ThermodynamicState
        temperatures = unit.Quantity(np.logspace(np.log10(Tmin / unit.kelvin), np.log10(Tmax / unit.kelvin), ntemps), unit.kelvin)
        self.thermodynamic_states = [ ThermodynamicState(system=self.system, temperature=temperature, pressure=pressure) for temperature in temperatures ]

        # Create SAMS samplers
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        #self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 500
        self.mcmc_sampler.timestep = 2.0 * unit.femtoseconds
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
    def __init__(self, **kwargs):
        super(AlanineDipeptideVacuumAlchemical, self).__init__(**kwargs)
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
        thermodynamic_state_index = 0 # thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
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
    def __init__(self, **kwargs):
        super(AlanineDipeptideExplicitAlchemical, self).__init__(**kwargs)
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
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 50 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

class AblImatinibExplicitAlchemical(SAMSTestSystem):
    """
    Alchemical free energy calculation for Abl:imatinib in explicit solvent.

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

    >>> from sams.tests.testsystems import AblImatinibExplicitAlchemical
    >>> testsystem = AblImatinibExplicitAlchemical()

    """
    def __init__(self, **kwargs):
        super(AblImatinibExplicitAlchemical, self).__init__(**kwargs)
        self.description = 'Abl:imatinib in explicit solvent alchemical free energy calculation'

        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/abl-imatinib'

        # Create topology, positions, and system.
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('sams', 'data/gaff.xml')
        imatinib_xml_filename = resource_filename('sams', 'data/abl-imatinib/imatinib.xml')
        system_generators = dict()
        ffxmls = [gaff_xml_filename, imatinib_xml_filename, 'amber99sbildn.xml', 'tip3p.xml']
        forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : app.HBonds, 'rigidWater' : True }

        # Load topologies and positions for all components
        print('Creating Abl:imatinib test system...')
        forcefield = app.ForceField(*ffxmls)
        from simtk.openmm.app import PDBFile, Modeller
        pdb_filename = resource_filename('sams', os.path.join(setup_path, '%s.pdb' % 'inhibitor'))
        #pdb_filename = resource_filename('sams', os.path.join(setup_path, '%s.pdb' % 'complex'))
        pdbfile = PDBFile(pdb_filename)
        modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
        print('Adding solvent...')
        modeller.addSolvent(forcefield, model=explicit_solvent_model, padding=padding)
        self.topology = modeller.getTopology()
        self.positions = modeller.getPositions()
        print('Creating system...')
        self.system = forcefield.createSystem(self.topology, **forcefield_kwargs)

        # DEBUG: Write PDB
        outfile = open('initial.pdb', 'w')
        PDBFile.writeFile(self.topology, self.positions, outfile)
        outfile.close()

        # Add a MonteCarloBarostat
        #temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        #pressure = 1.0 * unit.atmospheres
        #barostat = openmm.MonteCarloBarostat(pressure, temperature)
        #self.system.addForce(barostat)

        # Create thermodynamic states.
        print('Creating alchemically-modified system...')
        temperature = 300 * unit.kelvin
        pressure = 1.0 * unit.atmospheres
        #alchemical_atoms = range(4266,4335) # Abl:imatinib
        alchemical_atoms = range(0,69) # Abl:imatinib
        from alchemy import AbsoluteAlchemicalFactory
        factory = AbsoluteAlchemicalFactory(self.system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False, softcore_beta=0.0) # turn off softcore electrostatics
        self.system = factory.createPerturbedSystem()
        print('Setting up alchemical intermediates...')
        from sams import ThermodynamicState
        self.thermodynamic_states = list()
        for state in range(251):
            parameters = {'lambda_sterics' : 1.0, 'lambda_electrostatics' : (1.0 - float(state)/250.0) }
            self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )
        for state in range(1,251):
            parameters = {'lambda_sterics' : (1.0 - float(state)/250.0), 'lambda_electrostatics' : 0.0 }
            self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )

        # Create SAMS samplers
        print('Setting up samplers...')
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        #self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

        # This test case requires minimization to not explode.
        #minimize(self)

class WaterBoxAlchemical(SAMSTestSystem):
    """
    Alchemical free energy calculation for TIP3P water in TIP3P water.

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

    >>> from sams.tests.testsystems import AblImatinibExplicitAlchemical
    >>> testsystem = WaterBoxAlchemical()

    """
    def __init__(self, alchemical_protocol='fused', **kwargs):
        super(WaterBoxAlchemical, self).__init__(**kwargs)
        self.description = 'TIP3P water in TIP3P water NPT alchemical free energy calculation with %s protocol' % alchemical_protocol

        # Create topology, positions, and system.
        from openmmtools.testsystems import WaterBox
        testsystem = WaterBox(nonbondedMethod=app.CutoffPeriodic)
        self.topology = testsystem.topology
        self.positions = testsystem.positions
        self.system = testsystem.system

        # DEBUG: Write PDB
        from simtk.openmm.app import PDBFile
        outfile = open('initial.pdb', 'w')
        PDBFile.writeFile(self.topology, self.positions, outfile)
        outfile.close()

        # Add a MonteCarloBarostat
        temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        pressure = 1.0 * unit.atmospheres
        barostat = openmm.MonteCarloBarostat(pressure, temperature)
        self.system.addForce(barostat)

        # Create thermodynamic states.
        print('Creating alchemically-modified system...')
        temperature = 300 * unit.kelvin
        pressure = 1.0 * unit.atmospheres
        alchemical_atoms = range(0,3) # water
        from alchemy import AbsoluteAlchemicalFactory
        if alchemical_protocol == 'two-stage':
            factory = AbsoluteAlchemicalFactory(self.system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False, softcore_beta=0.0) # turn off softcore electrostatics
        elif alchemical_protocol == 'fused':
            factory = AbsoluteAlchemicalFactory(self.system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False)
        self.system = factory.createPerturbedSystem()
        print('Setting up alchemical intermediates...')
        from sams import ThermodynamicState
        self.thermodynamic_states = list()

        if alchemical_protocol == 'two-stage':
            nstates_per_leg = 10
            for state in range(nstates_per_leg+1):
                parameters = {'lambda_sterics' : 1.0, 'lambda_electrostatics' : (1.0 - float(state)/float(nstates_per_leg)) }
                self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )
            for state in range(1, nstates_per_leg+1):
                parameters = {'lambda_sterics' : (1.0 - float(state)/float(nstates_per_leg)), 'lambda_electrostatics' : 0.0 }
                self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )
        elif alchemical_protocol == 'fused':
            nstates = 32
            for state in range(nstates):
                parameters = {'lambda_sterics' : (1.0 - float(state)/float(nstates-1)) , 'lambda_electrostatics' : (1.0 - float(state)/float(nstates-1)) }
                self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature, parameters=parameters) )
        else:
            raise Exception("alchemical_protocol '%s' unknown" % alchemical_protocol)

        # Create SAMS samplers
        print('Setting up samplers...')
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.timestep = 2.0 * unit.femtoseconds
        #self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.nsteps = 500 # reduce number of steps for testing
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True

        # This test case requires minimization to not explode.
        #minimize(self)


class LoopSoftening(SAMSTestSystem):
    """
    Alchemical free energy calculation for Abl:imatinib in explicit solvent.

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

    >>> from sams.tests.testsystems import LoopSoftening
    >>> testsystem = LoopSoftening()

    """
    def __init__(self, **kwargs):
        super(LoopSoftening, self).__init__(**kwargs)
        self.description = 'Alchemical Loop Softening script'

        padding = 9.0*unit.angstrom
        explicit_solvent_model = 'tip3p'
        setup_path = 'data/mtor'

        # Create topology, positions, and system.
        from pkg_resources import resource_filename
        gaff_xml_filename = resource_filename('sams', 'data/gaff.xml')
        system_generators = dict()
        ffxmls = [gaff_xml_filename, 'amber99sbildn.xml', 'tip3p.xml']
        forcefield_kwargs={ 'nonbondedMethod' : app.CutoffPeriodic, 'nonbondedCutoff' : 9.0 * unit.angstrom, 'implicitSolvent' : None, 'constraints' : app.HBonds, 'rigidWater' : True }

        # Load topologies and positions for all components
        print('Creating mTOR test system...')
        forcefield = app.ForceField(*ffxmls)
        from simtk.openmm.app import PDBFile, Modeller
        pdb_filename = resource_filename('sams', os.path.join(setup_path, 'mtor_pdbfixer_apo.pdb'))

        pdbfile = PDBFile(pdb_filename)
        modeller = app.Modeller(pdbfile.topology, pdbfile.positions)
        print('Adding solvent...')
        modeller.addSolvent(forcefield, model=explicit_solvent_model, padding=padding)
        self.topology = modeller.getTopology()
        self.positions = modeller.getPositions()
        print('Creating system...')
        self.system = forcefield.createSystem(self.topology, **forcefield_kwargs)

        # DEBUG: Write PDB
        outfile = open('initial.pdb', 'w')
        PDBFile.writeFile(self.topology, self.positions, outfile)
        outfile.close()

        # Atom Selection using MDtraj
        res_pairs = [[403, 483], [1052, 1109]]
        t = md.load(pdb_filename)
        alchemical_atoms = set()
        for x in res_pairs:
            start = min(t.top.select('residue %s' % min(x)))
            end = max(t.top.select('residue %s' % max(x))) + 1
            alchemical_atoms.union(set(range(start, end)))


        # Create thermodynamic states.
        print('Creating alchemically-modified system...')
        temperature = 300 * unit.kelvin
        pressure = 1.0 * unit.atmospheres

        from alchemy import AbsoluteAlchemicalFactory
        factory = AbsoluteAlchemicalFactory(self.system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True,
                                            alchemical_torsions=True, annihilate_sterics=True,
                                            softcore_beta=0.0)  # turn off softcore electrostatics
        self.system = factory.createPerturbedSystem()
        print('Setting up alchemical intermediates...')
        from sams import ThermodynamicState
        self.thermodynamic_states = list()
        for state in range(26):
            parameters = {'lambda_sterics' : 1.0, 'lambda_electrostatics' : (1.0 - float(state)/25.0) }
            self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature,
                                                                 parameters=parameters) )
        for state in range(1,26):
            parameters = {'lambda_sterics' : (1.0 - float(state)/25.0), 'lambda_electrostatics' : 0.0 }
            self.thermodynamic_states.append( ThermodynamicState(system=self.system, temperature=temperature,
                                                                 parameters=parameters) )

        #minimize(self.system, self.positions)
        minimize(self.system)

        # Create SAMS samplers
        print('Setting up samplers...')
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0  # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.system.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state,
                                        ncfile=self.ncfile)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
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
    for testsystem_name in ['AblImatinibExplicitAlchemical', 'AlanineDipeptideVacuumSimulatedTempering', 'AlanineDipeptideExplicitSimulatedTempering', 'AlanineDipeptideVacuumAlchemical', 'AlanineDipeptideExplicitAlchemical']:
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

def generate_ffxml(pdb_filename):
    from simtk.openmm.app import PDBFile, Modeller
    pdbfile = PDBFile(pdb_filename)
    residues = [ residue for residue in pdbfile.topology.residues() ]
    residue = residues[0]
    from openmoltools.forcefield_generators import generateForceFieldFromMolecules, generateOEMolFromTopologyResidue
    molecule = generateOEMolFromTopologyResidue(residue, geometry=False, tripos_atom_names=True)
    molecule.SetTitle('MOL')
    molecules = [molecule]
    ffxml = generateForceFieldFromMolecules(molecules)
    outfile = open('imatinib.xml', 'w')
    outfile.write(ffxml)
    outfile.close()


if __name__ == '__main__':
    #pdb_filename = resource_filename('sams', os.path.join('data', 'abl-imatinib', 'inhibitor.pdb'))
    #generate_ffxml(pdb_filename)
    #stop

    netcdf_filename = 'output.nc'

    #testsystem = HarmonicOscillatorSimulatedTempering(netcdf_filename=netcdf_filename)

    testsystem = AblImatinibExplicitAlchemical(netcdf_filename=netcdf_filename)
    #testsystem = AlanineDipeptideExplicitAlchemical()
    #testsystem = AlanineDipeptideVacuumSimulatedTempering(netcdf_filename=netcdf_filename)
    #testsystem = AlanineDipeptideExplicitSimulatedTempering(netcdf_filename=netcdf_filename)
    #testsystem = WaterBoxAlchemical(netcdf_filename=netcdf_filename)

    testsystem.exen_sampler.update_scheme = 'local-jump'
    testsystem.mcmc_sampler.nsteps = 500
    testsystem.exen_sampler.locality = 10
    testsystem.sams_sampler.update_method = 'optimal'
    niterations = 5000
    #testsystem.sams_sampler.mbar_update_interval = 50
    testsystem.sams_sampler.run(niterations)
