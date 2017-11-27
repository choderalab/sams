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

top = md.load(state_pdb_filename).topology

# Specify umbrellas for distance restraint
angle_sigma = 10*unit.degrees # umbrella stddev width in absence of external PMF (no Jacobian)
distance_sigma = 1*unit.angstroms # umbrella stddev width

# add our known kinase coordinates
# Roux pseudo-dihedral
Roux = ['resid 179 and resname ALA and name CB',
        'resid 179 and resname ALA and name CA',
        'resid 180 and resname ASP and name CA',
        'resid 180 and resname ASP and name CG']

Roux_atoms = list()
for atom in Roux:
    atom_select = int(top.select(atom)[0])
    Roux_atoms.append(atom_select)

# DFG distance
DFG_distance = ['resid 149 and resname GLY and name CA',
                'resid 181 and resname PHE and name CZ']

DFG_atoms = list()
for atom in DFG_distance:
    atom_select = int(top.select(atom)[0])
    DFG_atoms.append(atom_select)

min_dihedral = -180*unit.degrees
max_dihedral = +180*unit.degrees
dihedral_unit = unit.degrees
ntorsion_umbrellas = int((max_dihedral - min_dihedral) / angle_sigma)
torsion_umbrella_values = np.linspace((min_dihedral+angle_sigma/2)/dihedral_unit, (max_dihedral-angle_sigma/2)/dihedral_unit, ntorsion_umbrellas) * dihedral_unit

min_distance = 1.0*unit.angstroms
max_distance = 10.0*unit.angstroms
distance_unit = unit.angstroms
ndistance_umbrellas = int((max_distance - min_distance) / distance_sigma)
distance_umbrella_values = np.linspace((min_distance+distance_sigma/2)/distance_unit, (max_distance-distance_sigma/2)/distance_unit, ndistance_umbrellas) * distance_unit

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
print('umbrella schedule for dihedral defined by atoms %s : %s' % (str(Roux_atoms), str(torsion_umbrella_values)))
print('umbrella schedule for distance defined by atoms %s : %s' % (str(DFG_atoms), str(distance_umbrella_values)))

from numpy import pi
energy_function = '- (torsion_K/2) * cos(min(dtheta, 2*pi-dtheta)); dtheta = abs(theta-torsion_theta0);'
energy_function += 'pi = %f;' % pi
umbrella_force = openmm.CustomTorsionForce(energy_function)
umbrella_force.addGlobalParameter('torsion_K', 0.0)
umbrella_force.addGlobalParameter('torsion_theta0', 0.0)
umbrella_force.addTorsion(*Roux_atoms, [])
torsion_K = kT/angle_sigma**2
system.addForce(umbrella_force)

energy_function = '(distance_K/2) * (r-distance_r0)^2;'
umbrella_force = openmm.CustomBondForce(energy_function)
umbrella_force.addGlobalParameter('distance_K', 0.0)
umbrella_force.addGlobalParameter('distance_r0', 0.0)
umbrella_force.addBond(*DFG_atoms, [])
distance_K = kT/distance_sigma**2
system.addForce(umbrella_force)

# Create thermodynamic states
thermodynamic_states = list()

# Umbrella off state
parameters = {
    'torsion_K' : 0.0, 'torsion_theta0' : 0.0, # umbrella parameters
    'distance_K' : 0.0, 'distance_r0' : 0.0, # umbrella parameters
}
thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Umbrella on state
alchemical_lambda = 0.0
for theta0 in torsion_umbrella_values:
    for r0 in distance_umbrella_values:
        parameters = {
            'torsion_K' : torsion_K.value_in_unit_system(unit.md_unit_system), 'torsion_theta0' : theta0.value_in_unit_system(unit.md_unit_system), # umbrella parameters
            'distance_K' : distance_K.value_in_unit_system(unit.md_unit_system), 'distance_r0' : r0.value_in_unit_system(unit.md_unit_system), # umbrella parameters
        }
        thermodynamic_states.append( ThermodynamicState(system=system, temperature=temperature, pressure=pressure, parameters=parameters) )

# Select platform automatically; use mixed precision
from openmmtools.integrators import LangevinIntegrator
integrator = LangevinIntegrator(temperature=temperature, collision_rate=collision_rate, timestep=timestep)
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
