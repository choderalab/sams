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
alchemical_lambdas = np.linspace(1.0, 0.0, nlambda)
# Specify umbrellas for distance restraint
numbrellas = 20
umbrella_sigma = 1.0 * unit.angstroms # umbrella stddev width in absene of external PMF (no Jacobian)
umbrella_atoms = [783-1, 2409-1] # ATOM    783  CA  PHE    49      -1.230 -11.285  10.321  1.00  0.00 # ???
                                 # ATOM   2409  CZ  PHE   148       4.799  -2.974  -1.441  1.00  0.00 # DFG
umbrella_distances = np.linspace(2.0, 15.0, numbrellas) * unit.angstroms
# Output SAMS filename
netcdf_filename = 'output.nc'
pdb_trajectory_filename = 'trajectory.pdb' # first frame of trajectory to be written at end
dcd_trajectory_filename = 'trajectory.dcd' # DCD format for trajectory to be written at end
# Simulation conditions
temperature = 298.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
pressure = None # WARNING: This is a temporary workaround for some issues with the GTX-1080.
timestep = 2.0 * unit.femtoseconds
minimize = True # if True, will minimize the structure before simulation (highly recommended)

################################################################################
# MAIN
################################################################################

from sams import kB
kT = kB * temperature
beta = 1.0 / kT

# Load Amber prmtop/inpcrd for complex and create fully-interacting reference system
print('Loading Amber prmtop/inpcrd files...')
prmtop = app.AmberPrmtopFile(prmtop_filename)
inpcrd = app.AmberInpcrdFile(inpcrd_filename)
reference_system = prmtop.createSystem(nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds)
topology = prmtop.topology
positions = inpcrd.positions
print('System has %d atoms.' % reference_system.getNumParticles())

if pressure is not None:
    # Add a barostat
    print("Adding barostat...")
    barostat = openmm.MonteCarloBarostat(pressure, temperature)
    reference_system.addForce(barostat)

# Identify ligand indices by residue name
print('Identifying ligand atoms to be alchemically modified...')
reference = md.load(pdb_filename)
alchemical_atoms = reference.topology.select(ligand_dsl_selection) # these atoms will be alchemically softened
alchemical_atoms = [ int(index) for index in alchemical_atoms ] # recode as Python int
print("MDTraj DSL selection '%s' identified %d atoms" % (ligand_dsl_selection, len(alchemical_atoms)))

# Create alchemically-modified system using fused softcore electrostatics and sterics
print('Creating alchemically modified system...')
print('lambda schedule: %s' % str(alchemical_lambdas))
from alchemy import AbsoluteAlchemicalFactory
from sams import ThermodynamicState
factory = AbsoluteAlchemicalFactory(reference_system, ligand_atoms=alchemical_atoms, annihilate_electrostatics=True, annihilate_sterics=False)
system = factory.createPerturbedSystem()

# Add umbrella restraint with global variable to control umbrella position
print('umbrella schedule between atoms %d and %d: %s' % (umbrella_atoms[0], umbrella_atoms[1], str(umbrella_distances)))
energy_function = '(umbrella_K/2.0)*(r-umbrella_r0)^2'
umbrella_force = openmm.CustomBondForce(energy_function)
umbrella_force.addGlobalParameter('umbrella_K', 0.0) # spring constant
umbrella_force.addGlobalParameter('umbrella_r0', 0.0) # umbrella distance
umbrella_force.addBond(umbrella_atoms[0], umbrella_atoms[1], [])
umbrella_K = kT/umbrella_sigma**2
system.addForce(umbrella_force)

# Create thermodynamic states
thermodynamic_states = list()
for alchemical_lambda in alchemical_lambdas:
    # Umbrella off state
    parameters = {
        'lambda_sterics' : alchemical_lambda, 'lambda_electrostatics' : alchemical_lambda, # alchemical parameters
        'umbrella_K' : 0.0, 'umbrella_r0' : 0.0, # umbrella parameters
        }
    thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )
    # Umbrella on state
    for umbrella_distance in umbrella_distances:
        parameters = {
            'lambda_sterics' : alchemical_lambda, 'lambda_electrostatics' : alchemical_lambda, # alchemical parameters
            'umbrella_K' : umbrella_K.value_in_unit_system(unit.md_unit_system), 'umbrella_r0' : umbrella_distance.value_in_unit_system(unit.md_unit_system), # umbrella parameters
            }
        thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Compile list of thermodynamic state neighbors
print('Determining thermodynamic state neighbors...')
lambda_neighbor_cutoff = 0.21
r0_neighbor_cutoff = 0.21
thermodynamic_state_neighbors = list()
for (i, istate) in enumerate(thermodynamic_states):
    neighbors = list()
    for (j, jstate) in enumerate(thermodynamic_states):
        if (abs(istate.parameters['lambda_sterics'] - jstate.parameters['lambda_sterics']) < lambda_neighbor_cutoff) \
            and (istate.parameters['umbrella_K']==0.0 or jstate.parameters['umbrella_K']==0 or (abs(istate.parameters['umbrella_r0'] - jstate.parameters['umbrella_r0']) < r0_neighbor_cutoff)):
            neighbors.append(j)
    thermodynamic_state_neighbors.append(neighbors)
    print('state %5d has %5d neighbors' % (i, len(neighbors)))

# Select platform automatically; use mixed precision
integrator = openmm.VerletIntegrator(timestep)
context = openmm.Context(system, integrator)
platform = context.getPlatform()
del context
try:
    platform.setPropertyDefaultValue('Precision', 'mixed')
except:
    pass

# Minimize
if minimize:
    print('Minimizing...')
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
    # Clean up.
    del context, integrator

# Create output SAMS file
print('Opening %s for writing...' % netcdf_filename)
ncfile = netCDF4.Dataset(netcdf_filename, mode='w')

# Create SAMS samplers
print('Setting up samplers...')
from sams.samplers import SamplerState, MCMCSampler, ExpandedEnsembleSampler, SAMSSampler
thermodynamic_state_index = 0 # initial thermodynamic state index
thermodynamic_state = thermodynamic_states[thermodynamic_state_index]
sampler_state = SamplerState(positions=positions)
mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=ncfile, platform=platform)
mcmc_sampler.timestep = timestep
mcmc_sampler.nsteps = 2500
#mcmc_sampler.pdbfile = open('output.pdb', 'w') # uncomment this if you want to write a PDB trajectory as you simulate; WARNING: LARGE!
mcmc_sampler.topology = topology
mcmc_sampler.verbose = True
exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, thermodynamic_states)
exen_sampler.verbose = True
sams_sampler = SAMSSampler(exen_sampler)
sams_sampler.verbose = True

# DEBUG: Write PDB of initial frame
print("Writing initial frame to 'initial.pdb'...")
from simtk.openmm.app import PDBFile
outfile = open('initial.pdb', 'w')
PDBFile.writeFile(topology, positions, outfile)
outfile.close()

# Run the simulation
print('Running simulation...')
#exen_sampler.update_scheme = 'restricted-range' # scheme for deciding which alchemical state to jump to
exen_sampler.update_scheme = 'global-jump' # scheme for deciding which alchemical state to jump to
exen_sampler.locality = thermodynamic_state_neighbors # neighbors to examine for each state
sams_sampler.update_method = 'rao-blackwellized' # scheme for updating free energy estimates
niterations = 1000 # number of iterations to run
sams_sampler.run(niterations) # run sampler
ncfile.close()

# Analyze
from sams import analysis
# States
from collections import namedtuple
MockTestsystem = namedtuple('MockTestsystem', ['description', 'thermodynamic_states'])
testsystem = MockTestsystem(description='Abl:imatinib with alchemical and umbrella states', thermodynamic_states=thermodynamic_states)
analysis.analyze(netcdf_filename, testsystem, 'output.pdf')
# Write trajectory
reference_pdb_filename = 'trajectory.pdb'
trajectory_filename = 'trajectory.dcd'
analysis.write_trajectory(netcdf_filename, topology, reference_pdb_filename, trajectory_filename)
