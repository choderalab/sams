"""
Analysis for self-adjusted mixture sampling (SAMS).

"""

__author__ = 'John D. Chodera'

################################################################################
# IMPORTS
################################################################################

import numpy as np
import netCDF4
import os, os.path
import sys, math
import copy
import time

from simtk import unit 
import mdtraj

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.backends.backend_pdf import PdfPages
import seaborn

import sams.tests.testsystems

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# Thermodynamic state description
################################################################################

def analyze(netcdf_filename, testsystem, pdf_filename):
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    testsystem_name = testsystem.__class__.__name__
    nstates = len(testsystem.thermodynamic_states)

    with PdfPages(pdf_filename) as pdf:
        # PAGE 1
        plt.figure(figsize=(6, 6))

        if hasattr(testsystem, 'logZ'):
            plt.hold(True)
            plt.plot(testsystem.logZ, 'ro')
            print(testsystem.logZ)

        title_fontsize = 7

        logZ = ncfile.variables['logZ'][-2,:]
        plt.plot(logZ, 'ko')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('state index $j$')
        plt.ylabel('$\zeta^{(t)}$')
        plt.axis([0, nstates-1, min(logZ), max(logZ)])
        if hasattr(testsystem, 'logZ'):
            plt.axis([0, nstates-1, 0.0, max(testsystem.logZ)])
        pdf.savefig()  # saves the current figure into a pdf page

        # PAGE 2
        plt.figure(figsize=(6, 6))
        state_index = ncfile.variables['state_index'][:]
        plt.plot(state_index, '.')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('state index')
        plt.axis([0, nsamples, 0, nstates-1])
        pdf.savefig()  # saves the current figure into a pdf page

        # PAGE 3
        plt.figure(figsize=(6, 6))
        if hasattr(testsystem, 'logZ'):
            plt.hold(True)
            M = np.tile(testsystem.logZ, [nsamples,1])
            plt.plot(M, ':')
        logZ = ncfile.variables['logZ'][:,:]
        plt.plot(logZ[:,:], '-')
        plt.title(testsystem.description, fontsize=title_fontsize)
        plt.xlabel('iteration $t$')
        plt.ylabel('$\zeta^{(t)}$')
        plt.axis([0, nsamples, logZ.min(), logZ.max()])
        pdf.savefig()  # saves the current figure into a pdf page

        # FINISH
        plt.close()

def write_trajectory_dcd(netcdf_filename, testsystem, pdb_trajectory_filename, dcd_trajectory_filename):
    """
    Write trajectory.

    Parameters
    ----------
    netcdf_filename : str
        NetCDF filename.
    testsystem : TestSystem
        Test system.
    pdb_trajectory_filename : str
        PDB trajectory output filename
    dcd_trajectory_filename : str
        Output trajectory filename.

    """
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    # Write reference.pdb file
    from simtk.openmm.app import PDBFile
    outfile = open(pdb_trajectory_filename, 'w')
    positions = unit.Quantity(ncfile.variables['positions'][0,:,:], unit.angstroms)
    PDBFile.writeFile(testsystem.topology, positions, file=outfile)
    outfile.close()

    # TODO: Export as DCD trajectory with MDTraj
    from mdtraj.formats import DCDTrajectoryFile
    with DCDTrajectoryFile(dcd_trajectory_filename, 'w') as f:
        f.write(ncfile.variables['positions'][:,:,:])
    

def write_trajectory(netcdf_filename, topology, reference_pdb_filename, trajectory_filename):
    """
    Write trajectory.

    Parameters
    ----------
    netcdf_filename : str
        NetCDF filename.
    topology : Topology
        OpenMM topology object
    reference_pdb_filename
        PDB trajectory output filename
    trajectory_filename : str
        Output trajectory filename. Type is autodetected by extension (.xtc, .dcd, .pdb) recognized by MDTraj

    """
    ncfile = netCDF4.Dataset(netcdf_filename, 'r')
    [nsamples, nstates] = ncfile.variables['logZ'].shape

    # Convert to MDTraj trajectory.
    print('Creating MDTraj trajectory...')
    mdtraj_topology = mdtraj.Topology.from_openmm(topology)
    trajectory = mdtraj.Trajectory(ncfile.variables['positions'][:,:,:], mdtraj_topology)
    trajectory.unitcell_vectors = ncfile.variables['box_vectors'][:,:,:]

    # Center on receptor.
    trajectory.image_molecules()

    # Write reference.pdb file
    print('Writing reference PDB file...')
    trajectory[0].save(reference_pdb_filename)

    print('Writing trajectory...')
    trajectory.save(trajectory_filename)
    print('Done.')
