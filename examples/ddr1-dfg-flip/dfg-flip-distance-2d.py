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
umbrella_sigma = 0.5 * unit.angstroms # umbrella stddev width in absence of external PMF (no Jacobian)

umbrella1_atoms = [2852, 2395]
# from setup/state_DFG_IN.pdb (numbering of atoms and residues starts at 1)
# ATOM   2853  CZ  PHE A 182       7.042  50.460  38.882  1.00  0.00           C  
# ATOM   2396  CB  LEU A 154      10.392  50.918  33.845  1.00  0.00           C  

umbrella2_atoms = [2830, 836]
# from setup/state_DFG_IN.pdb (numbering of atoms and residues starts at 1)
# ATOM   2831  CG  ASP A 181       2.928  49.040  44.337  1.00  0.00           C  
# ATOM    837  NZ  LYS A  52      10.684  52.314  46.848  1.00  0.00           N  

min_distance = 3.0 * unit.angstroms
max_distance = 20.0 * unit.angstroms
distance_unit = unit.angstroms
numbrellas = int((max_distance - min_distance) / umbrella_sigma + 2)
umbrella_distances = np.linspace(min_distance/distance_unit, max_distance/distance_unit, numbrellas) * distance_unit

# Output SAMS filename
netcdf_filename = '2d-distance-output.nc'
pdb_trajectory_filename = '2d-distance-trajectory.pdb' # first frame of trajectory to be written at end
xtc_trajectory_filename = '2d-distance-trajectory.xtc' # DCD format for trajectory to be written at end
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
print('umbrella schedule atoms %s : %s' % (str(umbrella1_atoms), str(umbrella_distances)))
print('umbrella schedule atoms %s : %s' % (str(umbrella2_atoms), str(umbrella_distances)))
energy_function = '(umbrella1_K/2.0)*(r-umbrella1_r0)^2'
umbrella_force = openmm.CustomBondForce(energy_function)
umbrella_force.addGlobalParameter('umbrella1_K', 0.0) # spring constant
umbrella_force.addGlobalParameter('umbrella1_r0', 0.0) # umbrella distance
umbrella_force.addBond(*umbrella1_atoms, [])
umbrella_K = kT/umbrella_sigma**2
system.addForce(umbrella_force)

energy_function = '(umbrella2_K/2.0)*(r-umbrella2_r0)^2'
umbrella_force = openmm.CustomBondForce(energy_function)
umbrella_force.addGlobalParameter('umbrella2_K', 0.0) # spring constant
umbrella_force.addGlobalParameter('umbrella2_r0', 0.0) # umbrella distance
umbrella_force.addBond(*umbrella2_atoms, [])
umbrella_K = kT/umbrella_sigma**2
system.addForce(umbrella_force)

# Create thermodynamic states
thermodynamic_states = list()

# Umbrella off state
parameters = {
    'umbrella1_K' : 0.0, 'umbrella1_r0' : 0.0, # umbrella parameters
    'umbrella2_K' : 0.0, 'umbrella2_r0' : 0.0, # umbrella parameters
}
thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Umbrella on state
alchemical_lambda = 0.0
for umbrella1_distance in umbrella_distances:
    for umbrella2_distance in umbrella_distances:
        parameters = {
            'umbrella1_K' : umbrella_K.value_in_unit_system(unit.md_unit_system), 'umbrella1_r0' : umbrella1_distance.value_in_unit_system(unit.md_unit_system), # umbrella parameters
            'umbrella2_K' : umbrella_K.value_in_unit_system(unit.md_unit_system), 'umbrella2_r0' : umbrella2_distance.value_in_unit_system(unit.md_unit_system), # umbrella parameters
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
mcmc_sampler.nsteps = 2500
#mcmc_sampler.pdbfile = open('output.pdb', 'w') # uncomment this if you want to write a PDB trajectory as you simulate; WARNING: LARGE!
mcmc_sampler.topology = topology
mcmc_sampler.verbose = True
exen_sampler = ExpandedEnsembleSampler(mcmc_sampler, thermodynamic_states)
exen_sampler.verbose = True
sams_sampler = SAMSSampler(exen_sampler)
sams_sampler.verbose = True

# DEBUG: Write PDB of initial frame
print("Writing initial frame to '2d-distance-initial.pdb'...")
from simtk.openmm.app import PDBFile
outfile = open('2d-distance-initial.pdb', 'w')
PDBFile.writeFile(topology, positions, outfile)
outfile.close()

# Run the simulation
print('Running simulation...')
#exen_sampler.update_scheme = 'restricted-range' # scheme for deciding which alchemical state to jump to
exen_sampler.update_scheme = 'global-jump' # scheme for deciding which alchemical state to jump to
#exen_sampler.locality = thermodynamic_state_neighbors # neighbors to examine for each state
sams_sampler.update_method = 'rao-blackwellized' # scheme for updating free energy estimates
niterations = 5000 # number of iterations to run
sams_sampler.run(niterations) # run sampler
ncfile.close()

# Analyze
from sams import analysis
# States
from collections import namedtuple
MockTestsystem = namedtuple('MockTestsystem', ['description', 'thermodynamic_states'])
testsystem = MockTestsystem(description='DDR1 umbrella states', thermodynamic_states=thermodynamic_states)
analysis.analyze(netcdf_filename, testsystem, '2d-distance-output.pdf')
# Write trajectory
analysis.write_trajectory(netcdf_filename, topology, pdb_trajectory_filename, xtc_trajectory_filename)
