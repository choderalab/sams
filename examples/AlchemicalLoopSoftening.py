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
from sams.tests.testsystems import SAMSTestSystem


def minimize(testsystem, positions):
    print("Minimizing...")
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(testsystem, integrator)
    context.setPositions(positions=positions)
    print("Initial energy is %12.3f kcal/mol" % (
    context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    TOL = .90
    MAX_STEPS = 100
    openmm.LocalEnergyMinimizer.minimize(context, TOL, MAX_STEPS)
    print("Final energy is   %12.3f kcal/mol" % (
    context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    # Update positions.
    testsystem.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Update sampler states.
    #testsystem.mcmc_samplers.sampler_state.positions = testsystem.positions
    # Clean up
    del context, integrator

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
        alchemical_atoms = []
        for x in res_pairs:
            start = min(t.top.select('residue %s' % min(x)))
            end = max(t.top.select('residue %s' % max(x))) + 1
            alchemical_atoms.append(list(range(start, end)))
        #print(alchemical_atoms)


        # Add a MonteCarloBarostat
        #temperature = 300 * unit.kelvin # will be replaced as thermodynamic state is updated
        #pressure = 1.0 * unit.atmospheres
        #barostat = openmm.MonteCarloBarostat(pressure, temperature)
        #self.system.addForce(barostat)

        # Create thermodynamic states.
        print('Creating alchemically-modified system...')
        temperature = 300 * unit.kelvin
        pressure = 1.0 * unit.atmospheres

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

        minimize(self.system, self.positions)

        # Create SAMS samplers
        print('Setting up samplers...')
        from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
        thermodynamic_state_index = 0 # initial thermodynamic state index
        thermodynamic_state = self.thermodynamic_states[thermodynamic_state_index]
        sampler_state = SamplerState(positions=self.system.positions)
        self.mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=self.ncfile)
        self.mcmc_sampler.pdbfile = open('output.pdb', 'w')
        self.mcmc_sampler.topology = self.topology
        self.mcmc_sampler.verbose = True
        self.exen_sampler = ExpandedEnsembleSampler(self.mcmc_sampler, self.thermodynamic_states)
        self.exen_sampler.verbose = True
        self.sams_sampler = SAMSSampler(self.exen_sampler)
        self.sams_sampler.verbose = True



if __name__ == '__main__':


    netcdf_filename = 'output.nc'

    system = LoopSoftening(netcdf_filename=netcdf_filename)
    system.exen_sampler.update_scheme = 'local-jump'
    system.mcmc_sampler.nsteps = 500
    system.exen_sampler.locality = 10
    system.sams_sampler.update_method = 'optimal'
    niterations = 5
    system.sams_sampler.run(niterations)