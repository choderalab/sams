"""
Alchemical softening illustration for imatinib binding to Abl kinase.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import os, os.path
import sys, math
import numpy as np
import time

import alchemy
from simtk import openmm, unit
from simtk.openmm import app
import mdtraj as md
from openeye import oechem
import netCDF4

################################################################################
# MAJOR SETTINGS AND PARAMETERS
################################################################################

# Define paths for explicitly-solvated complex
prmtop_filename = 'setup/systems/Abl-STI/complex.prmtop'
inpcrd_filename = 'setup/systems/Abl-STI/complex.inpcrd'
pdb_filename = 'setup/systems/Abl-STI/complex.pdb'
# Specify MDTraj DSL selection for ligand
ligand_dsl_selection = 'resn MOL'
# Specify alchemical lambdas to use for softening
nlambda = 20 # number of alchemical states
min_alchemical_lambda = 0.7 # minimum softness to use; 0 is noninteracting; too low a value and the ligand may unbind!
alchemical_lambdas = np.linspace(1.0, min_alchemical_lambda, nlambda)
# Output SAMS filename
netcdf_filename = 'output.nc'
# Simulation conditions
temperature = 298.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
timestep = 2.0 * unit.femtoseconds
minimize = True # if True, will minimize the structure before simulation (highly recommended)

################################################################################
# MAIN
################################################################################

# Load Amber prmtop/inpcrd for complex and create fully-interacting reference system
print('Loading Amber prmtop/inpcrd files...')
prmtop = app.AmberPrmtopFile(prmtop_filename)
inpcrd = app.AmberInpcrdFile(inpcrd_filename)
reference_system = prmtop.createSystem(nonbondedMethod=app.PME, constraints=app.HBonds)
topology = prmtop.topology
positions = inpcrd.positions
print('System has %d atoms.' % reference_system.getNumParticles())

# Add a barostat
print("Adding barostat...")
barostat = openmm.MonteCarloBarostat(pressure, temperature)
reference_system.addForce(barostat)

# Identify ligand indices by residue name
print('Identifying ligand atoms to be alchemically modified...')
reference = md.load(pdb_filename)
alchemical_atoms = reference.topology.select(ligand_dsl_selection) # these atoms will be alchemically softened
print("MDTraj DSL selection '%s' identified %d atoms" % (ligand_dsl_selection, len(alchemical_atoms)))

# Create alchemically-modified system using fused softcore electrostatics and sterics
print('Creating alchemically modified system...')
print('lambda schedule: %s' % str(alchemical_lambdas))
from alchemy import AbsoluteAlchemicalFactory
from sams import ThermodynamicState
thermodynamic_states = list()
factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False)
system = factory.createPerturbedSystem()
for alchemical_lambda in alchemical_lambdas:
    parameters = {'lambda_sterics' : alchemical_lambda, 'lambda_electrostatics' : alchemical_lambda}
    thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Create output SAMS file
print('Opening %s for writing...' % netcdf_filename)
ncfile = netCDF4.Dataset(netcdf_filename, mode='w')

# Create SAMS samplers
print('Setting up samplers...')
from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
thermodynamic_state_index = 0 # initial thermodynamic state index
thermodynamic_state = thermodynamic_states[thermodynamic_state_index]
sampler_state = SamplerState(positions=positions)
mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=ncfile)
mcmc_sampler.timestep = timestep
mcmc_sampler.nsteps = 500
#mcmc_sampler.pdbfile = open('output.pdb', 'w') # uncomment this if you want to write a PDB trajectory as you simulate; WARNING: LARGE!
mcmc_sampler.topology = topology
mcmc_sampler.verbose = True
exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, thermodynamic_states)
exen_sampler.verbose = True
sams_sampler = SAMSSampler(exen_sampler)
sams_sampler.verbose = True

# Minimize
if minimize:
    print('Minimizing...')
    timestep = 1.0 * unit.femtoseconds
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    print("Initial energy is %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    TOL = 1.0
    MAX_STEPS = 500
    openmm.LocalEnergyMinimizer.minimize(context, TOL, MAX_STEPS)
    print("Final energy is   %12.3f kcal/mol" % (context.getState(getEnergy=True).getPotentialEnergy() / unit.kilocalories_per_mole))
    # Update positions.
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    mcmc_sampler.sampler_state.positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    # Clean up.
    del context, integrator

# DEBUG: Write PDB of initial frame
print("Writing initial frame to 'initial.pdb'...")
from simtk.openmm.app import PDBFile
outfile = open('initial.pdb', 'w')
PDBFile.writeFile(topology, positions, outfile)
outfile.close()

# Run the simulation
print('Running simulation...')
exen_sampler.update_scheme = 'restricted-range' # scheme for deciding which alchemical state to jump to
exen_sampler.locality = 5 # number of neighboring states to use in deciding which alchemical state to jump to
sams_sampler.update_method = 'rao-blackwellized' # scheme for updating free energy estimates
niterations = 100 # number of iterations to run
sams_sampler.run(niterations) # run sampler

# TODO: Get MDTraj trajectory afterwards
