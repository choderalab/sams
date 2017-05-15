"""
SAMS umbrella sampling for DDR1 kinase DFG loop flip.

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import os, os.path
import sys, math
import numpy as np
import time

from simtk import openmm, unit
from simtk.openmm import app
import mdtraj as md
import netCDF4

from sams import ThermodynamicState

################################################################################
# MAJOR SETTINGS AND PARAMETERS
################################################################################

# Define paths for explicitly-solvated complex
system_xml_filename = 'setup/system.xml'
state_xml_filename = 'setup/state_DFG_IN.xml'
state_pdb_filename = 'setup/state_DFG_IN.pdb'
pdb_filename = 'setup/systems/Abl-STI/complex.pdb'

# Specify umbrellas for distance restraint
umbrella_sigma = 10*unit.degrees # umbrella stddev width in absence of external PMF (no Jacobian)

umbrella1_atoms = [2817, 2815, 2825, 2830] # atoms involved in umbrella restraint
#ATOM   2818  CB  ALA A 180       1.927  52.416  41.379  1.00  0.00           C  
#ATOM   2816  CA  ALA A 180       3.319  52.098  40.823  1.00  0.00           C  
#ATOM   2826  CA  ASP A 181       5.071  50.442  43.834  1.00  0.00           C  
#ATOM   2831  CG  ASP A 181       2.928  49.040  44.337  1.00  0.00           C  

umbrella2_atoms = [2817, 2815, 2825, 2830] # atoms involved in umbrella restraint
#ATOM   2818  CB  ALA A 180       1.927  52.416  41.379  1.00  0.00           C  
#ATOM   2816  CA  ALA A 180       3.319  52.098  40.823  1.00  0.00           C  
#ATOM   2826  CA  ASP A 181       5.071  50.442  43.834  1.00  0.00           C  
#ATOM   2831  CG  ASP A 181       2.928  49.040  44.337  1.00  0.00           C  

min_dihedral = -180*unit.degrees
max_dihedral = +180*unit.degrees
dihedral_unit = unit.degrees
numbrellas = int((max_dihedral - min_dihedral) / umbrella_sigma)
umbrella_values = np.linspace((min_dihedral+umbrella_sigma/2)/dihedral_unit, (max_dihedral-umbrella_sigma/2)/dihedral_unit, numbrellas) * dihedral_unit

# Output SAMS filename
netcdf_filename = 'output.nc'
pdb_trajectory_filename = 'trajectory.pdb' # first frame of trajectory to be written at end
dcd_trajectory_filename = 'trajectory.dcd' # DCD format for trajectory to be written at end
# Simulation conditions
temperature = 298.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
collision_rate = 1.0 / unit.picoseconds
timestep = 2.0 * unit.femtoseconds
#minimize = True # if True, will minimize the structure before simulation (highly recommended)
minimize = False

################################################################################
# SUBROUTINES
################################################################################

def read_file(filename):
    infile = open(filename, 'r')
    contents = infile.read()
    return contents

################################################################################
# MAIN
################################################################################

from sams import kB
kT = kB * temperature
beta = 1.0 / kT

# Load system
print('Loading system...')
system = openmm.XmlSerializer.deserialize(read_file(system_xml_filename))
pdbfile = app.PDBFile(state_pdb_filename)
topology = pdbfile.topology
state = openmm.XmlSerializer.deserialize(read_file(state_xml_filename))
positions = state.getPositions(asNumpy=True)
box_vectors = state.getPeriodicBoxVectors()
print('System has %d atoms.' % system.getNumParticles())

forces = { force.__class__.__name__ : force for force in system.getForces() }
if (pressure is not None) and ('MonteCarloBarostat' not in forces):
    # Add a barostat
    print("Adding barostat...")
    barostat = openmm.MonteCarloBarostat(pressure, temperature)
    reference_system.addForce(barostat)
else:
    # TODO: Update barostat
    pass

# Add umbrella restraint with global variable to control umbrella position
print('umbrella schedule for dihedral defined by atoms %s : %s' % (str(umbrella_atoms), str(umbrella_values)))

from numpy import pi
energy_function = '- (umbrella_K/2) * cos(min(dtheta, 2*pi-dtheta)); dtheta = abs(theta-umbrella_r0);'
energy_function += 'pi = %f;' % pi
umbrella_force = openmm.CustomTorsionForce(energy_function)
umbrella_force.addGlobalParameter('umbrella_K', 0.0)
umbrella_force.addGlobalParameter('umbrella_r0', 0.0)
umbrella_force.addTorsion(*umbrella_atoms, [])
umbrella_K = kT/umbrella_sigma**2
system.addForce(umbrella_force)

# Create thermodynamic states
thermodynamic_states = list()

# Umbrella off state
parameters = {
    'umbrella_K' : 0.0, 'umbrella_r0' : 0.0, # umbrella parameters
}
thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Umbrella on state
alchemical_lambda = 0.0
for umbrella_value in umbrella_values:
    parameters = {
        'umbrella_K' : umbrella_K.value_in_unit_system(unit.md_unit_system), 'umbrella_r0' : umbrella_value.value_in_unit_system(unit.md_unit_system), # umbrella parameters
    }
    thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Select platform automatically; use mixed precision
from openmmtools.integrators import GeodesicBAOABIntegrator
integrator = GeodesicBAOABIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep)
context = openmm.Context(system, integrator)
platform = context.getPlatform()
del context
try:
    platform.setPropertyDefaultValue('Precision', 'mixed')
    platform.setPropertyDefaultValue('DeterministicForces', 'true')
except:
    pass

# Minimize
if minimize:
    print('Minimizing...')
    integrator = openmm.VerletIntegrator(timestep)
    context = openmm.Context(system, integrator)
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    context.setPositions(state.getPositions(asNumpy=True))
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
sampler_state = SamplerState(positions=positions, box_vectors=box_vectors)
mcmc_sampler = MCMCSampler(sampler_state=sampler_state, thermodynamic_state=thermodynamic_state, ncfile=ncfile, platform=platform)
mcmc_sampler.timestep = timestep
mcmc_sampler.nsteps = 500
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
#exen_sampler.locality = thermodynamic_state_neighbors # neighbors to examine for each state
sams_sampler.update_method = 'rao-blackwellized' # scheme for updating free energy estimates
niterations = 20000 # number of iterations to run
sams_sampler.run(niterations) # run sampler
ncfile.close()

# Analyze
from sams import analysis
# States
from collections import namedtuple
MockTestsystem = namedtuple('MockTestsystem', ['description', 'thermodynamic_states'])
testsystem = MockTestsystem(description='DDR1 umbrella states', thermodynamic_states=thermodynamic_states)
analysis.analyze(netcdf_filename, testsystem, 'output.pdf')
# Write trajectory
reference_pdb_filename = 'trajectory.pdb'
trajectory_filename = 'trajectory.xtc'
analysis.write_trajectory(netcdf_filename, topology, reference_pdb_filename, trajectory_filename)
