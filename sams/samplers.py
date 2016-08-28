"""
Samplers for self-adjusted mixture sampling (SAMS).

These samplers focus on the case where global Context parameters can be updated to switch between thermodynamic states.

This includes:
* Temperature
* Pressure
* Alchemical `lambda` parameters

TODO
----
* Determine where `System` object should be stored: In `SamplerState` or in `Thermodynamic State`, or both, or neither?
* Can we create a generalized, extensible `SamplerState` that also stores chemical/thermodynamic state information?
* Can we create a generalized log biasing weight container class that gracefully handles new chemical states that have yet to be explored?

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
import copy
import time
from scipy.misc import logsumexp

from openmmtools import testsystems

################################################################################
# LOGGER
################################################################################

import logging
logger = logging.getLogger(__name__)

################################################################################
# MODULE CONSTANTS
################################################################################

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA # Boltzmann constant

################################################################################
# Thermodynamic state description
################################################################################

class ThermodynamicState(object):
    """Object describing a thermodynamic state obeying Boltzmann statistics.

    Properties
    ----------
    system : simtk.openmm.System
        The System being simulated
    temperature : simtk.unit.Quantity with units compatible with kelvin
        The temperature
    pressure : simtk.unit.Quantity with units compatible with atmospheres
        The pressure, or `None` if the system is not periodic.
    parameters : dict
        parameters[name] is the context parameter value corresponding to parameter `name`

    Read-only properties
    --------------------
    kT : simtk.unit.Quantity with units compatible with kilocalories_per_mole
        Thermal energy
    beta : simtk.unit.Quantity with units compatible with 1/kilocalories_per_mole
        Inverse temperature.

    Examples
    --------
    Specify an NVT state for a water box at 298 K.
    >>> system_container = testsystems.WaterBox()
    >>> (system, positions) = system_container.system, system_container.positions
    >>> state = ThermodynamicState(system=system, temperature=298.0*unit.kelvin)

    Get the inverse temperature
    >>> beta = state.beta

    Specify an NPT state at 298 K and 1 atm pressure.
    >>> pressure = 1.0 * unit.atmospheres
    >>> temperature = 298.0 * unit.kelvin
    >>> barostat = openmm.MonteCarloBarostat(pressure, temperature)
    >>> force_index = system.addForce(barostat)
    >>> state = ThermodynamicState(system=system, temperature=temperature, pressure=pressure)

    Notes
    -----
    Note that the pressure is only relevant for periodic systems.
    `ThermodynamicState` cannot describe states obeying non-Boltzamnn statistics, such as Tsallis statistics.

    """
    def __init__(self, system, temperature, pressure=None, parameters=None, check_parameters=False):
        """Construct a thermodynamic state with given system and temperature.

        Parameters
        ----------

        system : simtk.openmm.System
            System object describing the potential energy function
            for the system (default: None)
        temperature : simtk.unit.Quantity, compatible with 'kelvin'
            Temperature for a system with constant temperature
        pressure : simtk.unit.Quantity, compatible with 'atmospheres', optional, default=None
            If not None, specifies the pressure for constant-pressure systems.
        parameters : dict, optional, default=None
            parameters[name] is the context parameter value corresponding to parameter `name`, or `None` if no parameters are defined
        check_parameters : bool, optional, default=False
            If True, check that all parameters are present.

        """
        # Check input parameters
        if system is None:
            raise Exception('system must be specified')
        if temperature is None:
            raise Exception('temperature must be specified')
        if parameters and check_parameters:
            platform = openmm.Platform.getPlatformByName('Reference')
            integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)
            context = openmm.Context(system, integrator, platform)
            system_parameters = set(context.getParameters())
            del context, integrator
            if not set(parameters).issubset(system_parameters):
                missing_parameters = list(system_parameters.difference(set(parameters)))
                msg = "System is missing parameters: %s" % str()
                raise Exception(msg)
        if pressure:
            forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
            if 'MonteCarloBarostat' not in forces:
                raise Exception("pressure was specified but no MonteCarloBarostat was found in System")

        # Initialize.
        self.system = system            # the System object governing the potential energy computation
        self.temperature = temperature  # the temperature
        self.pressure = pressure        # the pressure, or None if not isobaric
        self.parameters = parameters    # context parameters

    def update_context(self, context, integrator=None):
        """Update Context to the current thermodynamic state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The Context object to update
        integrator : simtk.openmm.Integrator, optional, default=None
            If specified, the integrator will be updated too.

        """
        if integrator is not None:
            if hasattr(integrator, 'setTemperature'):
                integrator.setTemperature(self.temperature)
            if hasattr(integrator, 'getNumGlobalVariables'):
                integrator_parameters = [ integrator.getGlobalVariableName(index) for index in range(integrator.getNumGlobalVariables()) ]
                if 'temperature' in integrator_parameters:
                    integrator.setGlobalVariableByName('temperature', self.temperature.value_in_unit_system(unit.md_unit_system))
                if 'kT' in integrator_parameters:
                    integrator.setGlobalVariableByName('kT', self.kT.value_in_unit_system(unit.md_unit_system))

        # Set pressure
        if self.pressure:
            forces = { self.system.getForce(index).__class__.__name__ : self.system.getForce(index) for index in range(self.system.getNumForces()) }
            barostat = forces['MonteCarloBarostat']
            context.setParameter(barostat.Pressure(), self.pressure.value_in_unit_system(unit.md_unit_system))
            context.setParameter(barostat.Temperature(), self.temperature.value_in_unit_system(unit.md_unit_system)) # OpenMM 7.1.0

        # Set Context parameters
        if self.parameters is not None:
            for parameter in self.parameters:
                value = self.parameters[parameter]
                if hasattr(value, 'unit'):
                    value = value.value_in_unit_system(unit.md_unit_system)
                context.setParameter(parameter, value)

        return

    def reduced_potential(self, context, sampler_state=None):
        """Compute the reduced potential for the given sampler state.

        WARNING: This does NOT return the context to the original state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The Context for which the reduced potential is to be computed.
        sampler_state : SamplerState, optional, default=None
            The sampler state specifying dynamical variables (if different from current).

        Returns
        -------
        u : float
            The unitless reduced potential (which can be considered to have units of kT)

        Notes
        -----

        The reduced potential is defined as in Ref. [1]

        u = \beta [U(x) + p V(x) + \mu N(x)]

        where the thermodynamic parameters are

        \beta = 1/(kB T) is he inverse temperature
        U(x) is the potential energy
        p is the pressure
        \mu is the chemical potential

        and the configurational properties are

        x the atomic positions
        V(x) is the instantaneous box volume
        N(x) the numbers of various particle species (e.g. protons of titratible groups)

        References
        ----------
        [1] Shirts MR and Chodera JD. Statistically optimal analysis of equilibrium states. J Chem Phys 129:124105, 2008.

        """
        # Update the Context for this thermodynamic state.
        self.update_context(context)

        # Update the Context for this sampler state.
        if sampler_state is not None:
            sampler_state.update_context(context)

        # Compute potential energy.
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()

        # Compute reduced potential.
        reduced_potential = self.beta * potential_energy
        if self.pressure is not None:
            volume = state.getPeriodicBoxVolume()
            reduced_potential += self.beta * self.pressure * volume * unit.AVOGADRO_CONSTANT_NA

        # TODO: Return Context to original form.

        del state
        return reduced_potential

    @property
    def kT(self):
        return (kB * self.temperature)

    @property
    def beta(self):
        return (1.0 / (kB * self.temperature))

    def __repr__(self):
        """Returns a string representation of a state.

        Examples
        --------

        Create an NVT state.

        >>> system_container = testsystems.LennardJonesCluster()
        >>> (system, positions) = system_container.system, system_container.positions
        >>> state = ThermodynamicState(system=system, temperature=100.0*unit.kelvin)

        Return a representation of the state.

        >>> state_string = repr(state)

        """

        r = "<ThermodynamicState object"
        if self.temperature is not None:
            r += ", temperature = %s" % str(self.temperature)
        if self.pressure is not None:
            r += ", pressure = %s" % str(self.pressure)
        if self.parameters is not None:
            r += ", parameters = %s" % str(self.parameters)
        r += ">"

        return r

    def __str__(self):
        # TODO: Write a human-readable representation.

        return repr(self)

################################################################################
# MCMC sampler state
################################################################################

class SamplerState(object):
    """
    Sampler state for MCMC move representing everything that may be allowed to change during the simulation.

    Parameters
    ----------
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds, default=None
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers, default=None
       Current box vectors.

    Fields
    ------
    positions : array of simtk.unit.Quantity compatible with nanometers
       Particle positions.
    velocities : optional, array of simtk.unit.Quantity compatible with nanometers/picoseconds
       Particle velocities.
    box_vectors : optional, 3x3 array of simtk.unit.Quantity compatible with nanometers
       Current box vectors.

    Examples
    --------
    Create a sampler state for a system with box vectors.
    >>> # Create a test system
    >>> test = testsystems.LennardJonesFluid()
    >>> # Create a sampler state manually.
    >>> box_vectors = test.system.getDefaultPeriodicBoxVectors()
    >>> sampler_state = SamplerState(positions=test.positions, box_vectors=box_vectors)

    Create a sampler state for a system without box vectors.
    >>> # Create a test system
    >>> test = testsystems.LennardJonesCluster()
    >>> # Create a sampler state manually.
    >>> sampler_state = SamplerState(positions=test.positions)

    """
    def __init__(self, positions=None, velocities=None, box_vectors=None):
        self.positions = positions
        self.velocities = velocities
        self.box_vectors = box_vectors

    def update_context(self, context):
        if self.positions is not None:
            context.setPositions(self.positions)
        if self.velocities is not None:
            context.setVelocities(self.velocities)
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)

    @classmethod
    def createFromContext(cls, context):
        """
        Create an SamplerState object from the information in a current OpenMM Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
           The Context object from which to create a sampler state.

        Returns
        -------
        sampler_state : SamplerState
           The sampler state containing positions, velocities, and box vectors.

        Examples
        --------

        >>> # Create a test system
        >>> test = testsystems.AlanineDipeptideVacuum()
        >>> # Create a Context.
        >>> import simtk.openmm as mm
        >>> import simtk.unit as u
        >>> integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        >>> platform = openmm.Platform.getPlatformByName('Reference')
        >>> context = openmm.Context(test.system, integrator, platform)
        >>> # Set positions and velocities.
        >>> context.setPositions(test.positions)
        >>> context.setVelocitiesToTemperature(298 * unit.kelvin)
        >>> # Create a sampler state from the Context.
        >>> sampler_state = SamplerState.createFromContext(context)
        >>> # Clean up.
        >>> del context, integrator

        """
        # Get state.
        openmm_state = context.getState(getPositions=True, getVelocities=True)

        # Create new object, bypassing init.
        self = SamplerState.__new__(cls)

        # Populate context.
        self.positions = openmm_state.getPositions(asNumpy=True)
        self.velocities = openmm_state.getVelocities(asNumpy=True)
        self.box_vectors = openmm_state.getPeriodicBoxVectors(asNumpy=True)

        del openmm_state

        return self

    def has_nan(self):
        """Return True if any of the generalized coordinates are nan.

        Notes
        -----

        Currently checks only the positions.
        """
        x = self.positions / unit.nanometers

        if np.any(np.isnan(x)):
            return True
        else:
            return False

################################################################################
# MCMC SAMPLER
################################################################################

class MCMCSampler(object):
    """
    Markov chain Monte Carlo (MCMC) sampler.

    This is a minimal functional implementation placeholder until we can replace this with MCMCSampler from `openmmmcmc`.

    Properties
    ----------
    positions : simtk.unit.Quantity of size [nparticles,3] with units compatible with nanometers
        The current positions.
    iteration : int
        Iterations completed.
    verbose : bool
        If True, verbose output is printed

    Examples
    --------
    >>> # Create a test system
    >>> test = testsystems.AlanineDipeptideVacuum()
    >>> # Create a sampler state.
    >>> sampler_state = SamplerState(positions=test.positions)
    >>> # Create a thermodynamic state.
    >>> thermodynamic_state = ThermodynamicState(system=test.system, temperature=298.0*unit.kelvin)
    >>> # Create an MCMC sampler
    >>> sampler = MCMCSampler(thermodynamic_state, sampler_state)
    >>> # Run the sampler
    >>> sampler.verbose = False
    >>> sampler.run()

    """
    def __init__(self, thermodynamic_state=None, sampler_state=None, platform=None, ncfile=None):
        """
        Create an MCMC sampler.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state to simulate
        sampler_state : SamplerState
            The initial sampler state to simulate from.
        platform : simtk.openmm.Platform, optional, default=None
            If specified, this platform will be used
        ncfile : netCDF4.Dataset, optional, default=None
            NetCDF storage file.

        """
        if thermodynamic_state is None:
            raise Exception("'thermodynamic_state' must be specified")
        if sampler_state is None:
            raise Exception("'sampler_state' must be specified")

        self.thermodynamic_state = thermodynamic_state
        self.sampler_state = sampler_state
        # Initialize
        self.iteration = 0
        # For GHMC / Langevin integrator
        self.collision_rate = 1.0 / unit.picoseconds
        self.timestep = 2.0 * unit.femtoseconds
        self.nsteps = 500 # number of steps per update
        self.verbose = True
        self.platform = platform

        # For writing PDB files
        self.pdbfile = None
        self.topology = None

        self._timing = dict()
        self._initializeNetCDF(ncfile)
        self._initialized = False

    def _initialize(self):
        # Create an integrator
        integrator_name = 'Langevin'
        if integrator_name == 'GHMC':
            from openmmtools.integrators import GHMCIntegrator
            self.integrator = GHMCIntegrator(temperature=self.thermodynamic_state.temperature, collision_rate=self.collision_rate, timestep=self.timestep)
        elif integrator_name == 'Langevin':
            from simtk.openmm import LangevinIntegrator
            self.integrator = LangevinIntegrator(self.thermodynamic_state.temperature, self.collision_rate, self.timestep)
        else:
            raise Exception("integrator_name '%s' not valid." % (integrator_name))

        # Create a Context
        if self.platform is not None:
            self.context = openmm.Context(self.thermodynamic_state.system, self.integrator, self.platform)
        else:
            self.context = openmm.Context(self.thermodynamic_state.system, self.integrator)
        self.thermodynamic_state.update_context(self.context, self.integrator)
        self.sampler_state.update_context(self.context)
        self.context.setVelocitiesToTemperature(self.thermodynamic_state.temperature)

        self._initialized = True

    def _initializeNetCDF(self, ncfile):
        self.ncfile = ncfile
        if self.ncfile == None:
            return

        natoms = self.thermodynamic_state.system.getNumParticles()
        self.ncfile.createDimension('iterations', None)
        self.ncfile.createDimension('atoms', natoms) # TODO: What do we do if dimension can change?
        self.ncfile.createDimension('spatial', 3)

        self.ncfile.createVariable('positions', 'f4', dimensions=('iterations', 'atoms', 'spatial'), zlib=True, chunksizes=(1,natoms,3))
        self.ncfile.createVariable('box_vectors', 'f4', dimensions=('iterations', 'spatial', 'spatial'), zlib=True, chunksizes=(1,3,3))
        self.ncfile.createVariable('potential', 'f8', dimensions=('iterations',), chunksizes=(1,))
        self.ncfile.createVariable('sample_positions_time', 'f4', dimensions=('iterations',), chunksizes=(1,))

        # Weight adaptation information
        self.ncfile.createVariable('stage', 'i2', dimensions=('iterations',), chunksizes=(1,))
        self.ncfile.createVariable('gamma', 'f8', dimensions=('iterations',), chunksizes=(1,))

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if not self._initialized:
            self._initialize()

        if self.verbose:
            print("." * 80)
            print("MCMC sampler iteration %d" % self.iteration)

        initial_time = time.time()

        # Reset statistics
        if hasattr(self.integrator, 'setGlobalVariableByName'):
            self.integrator.setGlobalVariableByName('naccept', 0)

        # Take some steps
        self.integrator.step(self.nsteps)

        # Get new sampler state.
        self.sampler_state = SamplerState.createFromContext(self.context)

        # Report statistics
        if hasattr(self.integrator, 'getGlobalVariableByName'):
            naccept = self.integrator.getGlobalVariableByName('naccept')
            fraction_accepted = float(naccept) / float(self.nsteps)
            if self.verbose: print("Accepted %d / %d GHMC steps (%.2f%%)." % (naccept, self.nsteps, fraction_accepted * 100))

        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['sample positions'] = elapsed_time

        if self.verbose:
            final_energy = self.context.getState(getEnergy=True).getPotentialEnergy() * self.thermodynamic_state.beta
            print('Final energy is %12.3f kT' % (final_energy))
            print('elapsed time %8.3f s' % elapsed_time)

        if self.ncfile:
            self.ncfile.variables['positions'][self.iteration,:,:] = self.sampler_state.positions[:,:] / unit.nanometers
            for k in range(3):
                self.ncfile.variables['box_vectors'][self.iteration,k,:] = self.sampler_state.box_vectors[k,:] / unit.nanometers
            self.ncfile.variables['potential'][self.iteration] = self.thermodynamic_state.beta * self.context.getState(getEnergy=True).getPotentialEnergy()
            self.ncfile.variables['sample_positions_time'][self.iteration] = elapsed_time

        # Increment iteration count
        self.iteration += 1

        if self.verbose:
            print("." * 80)
        if self.pdbfile is not None:
            print("Writing frame...")
            from simtk.openmm.app import PDBFile
            PDBFile.writeModel(self.topology, self.sampler_state.positions, self.pdbfile, self.iteration)
            self.pdbfile.flush()

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()

################################################################################
# EXPANDED ENSEMBLE SAMPLER
################################################################################

class ExpandedEnsembleSampler(object):
    """
    Method of expanded ensembles sampling engine.

    Properties
    ----------
    sampler : MCMCSampler
        The MCMC sampler used for updating positions
    thermodynamic_states : list of ThermodynamicState
        All thermodynamic states that can be sampled
    thermodynamic_state_index : int
        Current thermodynamic state index
    iteration : int
        Iterations completed
    naccepted : int
        Number of accepted thermodynamic/chemical state changes
    nrejected : int
        Number of rejected thermodynamic/chemical state changes
    number_of_state_visits : np.array of shape [nstates]
        Cumulative counts of visited states
    verbose : bool
        If True, verbose output is printed

    References
    ----------
    [1] Lyubartsev AP, Martsinovski AA, Shevkunov SV, and Vorontsov-Velyaminov PN. New approach to Monte Carlo calculation of the free energy: Method of expanded ensembles. JCP 96:1776, 1992
    http://dx.doi.org/10.1063/1.462133

    Examples
    --------

    """
    def __init__(self, sampler, thermodynamic_states, log_weights=None, update_scheme='global-jump', locality=1):
        """
        Create an expanded ensemble sampler.

        p(x,k) \propto \exp[-u_k(x) + g_k]

        where g_k is the log weight.

        Parameters
        ----------
        sampler : MCMCSampler
            MCMCSampler initialized with current SamplerState
        state : hashable object
            Current chemical state
        log_weights : dict of object : float
            Log weights to use for expanded ensemble biases
        update_scheme : str, optional, default='global-jump'
            Thermodynamic state update scheme, one of ['global-jump', 'local-jump', 'restricted-range']
        locality : int, optional, default=1
            Number of neighboring states on either side to consider for local update schemes

        """
        self.supported_update_schemes = ['local-jump', 'global-jump', 'restricted-range']
        if update_scheme not in self.supported_update_schemes:
            raise Exception("Update scheme '%s' not in list of supported update schemes: %s" % (update_scheme, str(self.supported_update_schemes)))

        self.sampler = sampler
        self.thermodynamic_states = thermodynamic_states
        self.nstates = len(self.thermodynamic_states)
        self.log_weights = log_weights
        self.update_scheme = update_scheme
        self.locality = locality

        # Determine which thermodynamic state is currently active
        self.thermodynamic_state_index = thermodynamic_states.index(sampler.thermodynamic_state)

        if self.log_weights is None:
            self.log_weights = np.zeros([self.nstates], np.float64)

        # Initialize
        self.iteration = 0
        self.naccepted = 0
        self.nrejected = 0
        self.number_of_state_visits = np.zeros([self.nstates], np.float64)
        self.verbose = False

        self._timing = self.sampler._timing
        self._initializeNetCDF(self.sampler.ncfile)

    def _initializeNetCDF(self, ncfile):
        self.ncfile = ncfile
        if self.ncfile == None:
            return

        nstates = self.nstates
        if self.update_scheme == 'global-jump':
            self.locality = self.nstates

        self.ncfile.createDimension('states', nstates)

        self.ncfile.createVariable('state_index', 'i4', dimensions=('iterations',), chunksizes=(1,))
        self.ncfile.createVariable('log_weights', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('log_P_k', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('u_k', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('update_state_time', 'f4', dimensions=('iterations',), chunksizes=(1,))
        self.ncfile.createVariable('neighborhood', 'i1', dimensions=('iterations','states'), chunksizes=(1,nstates))

    def update_positions(self):
        """
        Sample new positions.
        """
        self.sampler.update()

    def update_state(self):
        """
        Sample the thermodynamic state.
        """
        initial_time = time.time()

        current_state_index = self.thermodynamic_state_index
        self.u_k = np.zeros([self.nstates], np.float64)
        self.log_P_k = np.zeros([self.nstates], np.float64)
        if self.update_scheme == 'local-jump':
            # Determine current neighborhood.
            neighborhood = range(max(0, current_state_index - self.locality), min(self.nstates, current_state_index + self.locality + 1))
            neighborhood_size = len(neighborhood)
            # Propose a move from the current neighborhood.
            proposed_state_index = np.random.choice(neighborhood, p=np.ones([len(neighborhood)], np.float64) / float(neighborhood_size))
            # Determine neighborhood for proposed state.
            proposed_neighborhood = range(max(0, proposed_state_index - self.locality), min(self.nstates, proposed_state_index + self.locality + 1))
            proposed_neighborhood_size = len(proposed_neighborhood)
            # Compute state log weights.
            log_Gamma_j_L = - float(proposed_neighborhood_size) # log probability of proposing return
            log_Gamma_L_j = - float(neighborhood_size)          # log probability of proposing new state
            L = current_state_index
            self.neighborhood = neighborhood
            for j in self.neighborhood:
                self.u_k[j] = self.thermodynamic_states[j].reduced_potential(self.sampler.context)
                if j != L:
                    self.log_P_k[j] = log_Gamma_L_j + min(0.0, log_Gamma_j_L - log_Gamma_L_j + (self.log_weights[j] - self.u_k[j]) - (self.log_weights[L] - self.u_k[L]))
            P_k = np.zeros([self.nstates], np.float64)
            P_k[self.neighborhood] = np.exp(self.log_P_k[self.neighborhood])
            # Compute probability to return to current state L
            P_k[L] = 0.0
            P_k[L] = 1.0 - P_k[self.neighborhood].sum()
            self.log_P_k[L] = np.log(P_k[L])
            P_k = P_k[self.neighborhood]
            # Update context.
            self.thermodynamic_state_index = np.random.choice(self.neighborhood, p=P_k)
            self.thermodynamic_states[self.thermodynamic_state_index].update_context(self.sampler.context, integrator=self.sampler.integrator)
        elif self.update_scheme == 'global-jump':
            # Compute unnormalized log probabilities for all thermodynamic states
            self.neighborhood = range(self.nstates)
            for state_index in self.neighborhood:
                self.u_k[state_index] = self.thermodynamic_states[state_index].reduced_potential(self.sampler.context)
                self.log_P_k[state_index] = self.log_weights[state_index] - self.u_k[state_index]
            self.log_P_k -= logsumexp(self.log_P_k)
            # Update sampler Context to current thermodynamic state.
            P_k = np.exp(self.log_P_k[self.neighborhood])
            self.thermodynamic_state_index = np.random.choice(self.neighborhood, p=P_k)
            self.thermodynamic_states[self.thermodynamic_state_index].update_context(self.sampler.context, integrator=self.sampler.integrator)
        elif self.update_scheme == 'restricted-range':
            # Propose new state from current neighborhood.
            self.neighborhood = range(max(0, current_state_index - self.locality), min(self.nstates, current_state_index + self.locality + 1))
            for j in self.neighborhood:
                self.u_k[j] = self.thermodynamic_states[j].reduced_potential(self.sampler.context)
                self.log_P_k[j] = self.log_weights[j] - self.u_k[j]
            self.log_P_k[self.neighborhood] -= logsumexp(self.log_P_k[self.neighborhood])
            P_k = np.exp(self.log_P_k[self.neighborhood])
            proposed_state_index = np.random.choice(self.neighborhood, p=P_k)
            # Determine neighborhood of proposed state.
            proposed_neighborhood = range(max(0, proposed_state_index - self.locality), min(self.nstates, proposed_state_index + self.locality + 1))
            proposed_log_P_k = np.zeros([self.nstates], np.float64)
            for j in proposed_neighborhood:
                if j not in self.neighborhood:
                    self.u_k[j] = self.thermodynamic_states[j].reduced_potential(self.sampler.context)
            # Accept or reject.
            log_P_accept = logsumexp(self.log_weights[self.neighborhood] - self.u_k[self.neighborhood]) - logsumexp(self.log_weights[proposed_neighborhood] - self.u_k[proposed_neighborhood])
            if (log_P_accept >= 0.0) or (np.random.rand() < np.exp(log_P_accept)):
                self.thermodynamic_state_index = proposed_state_index
            # Update context.
            self.thermodynamic_states[self.thermodynamic_state_index].update_context(self.sampler.context, integrator=self.sampler.integrator)
        else:
            raise Exception("Update scheme '%s' not implemented." % self.update_scheme)

        # Track statistics.
        if (self.thermodynamic_state_index != current_state_index):
            self.naccepted += 1
        else:
            self.nrejected += 1

        if self.verbose:
            print('Current thermodynamic state index is %d' % self.thermodynamic_state_index)
            Neff = (P_k / P_k.max()).sum()
            print('Effective number of states with probability: %10.5f' % Neff)

        # Update timing
        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['update state time'] = elapsed_time
        print('elapsed time %8.3f s' % elapsed_time)

        # Update statistics.
        self.update_statistics()

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        if self.verbose:
            print("-" * 80)
            print("Expanded Ensemble sampler iteration %8d" % self.iteration)

        self.update_positions()
        self.update_state()

        if self.ncfile:
            self.ncfile.variables['state_index'][self.iteration] = self.thermodynamic_state_index
            self.ncfile.variables['log_weights'][self.iteration,:] = self.log_weights[:]
            self.ncfile.variables['log_P_k'][self.iteration,:] = 0
            self.ncfile.variables['log_P_k'][self.iteration,:] = self.log_P_k[:]
            self.ncfile.variables['u_k'][self.iteration,:] = 0
            self.ncfile.variables['u_k'][self.iteration,:] = self.u_k[:]
            self.ncfile.variables['neighborhood'][self.iteration,:] = 0
            self.ncfile.variables['neighborhood'][self.iteration,self.neighborhood] = 1
            self.ncfile.variables['update_state_time'][self.iteration] = self._timing['update state time']

        self.iteration += 1

        if self.verbose:
            print("-" * 80)

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()

    def update_statistics(self):
        """
        Update sampler statistics.
        """
        self.number_of_state_visits[self.thermodynamic_state_index] += 1.0

################################################################################
# SAMS SAMPLER
################################################################################

class SAMSSampler(object):
    """
    Self-adjusted mixture sampling engine.

    Properties
    ----------
    state_keys : set of objects
        The names of states sampled by the sampler.
    logZ : numpy array of shape [nstates]
        logZ[index] is the log partition function (up to an additive constant) estimate for thermodynamic state 'index'
    update_method : str
        Update method.  One of ['default']
    iteration : int
        Iterations completed.
    verbose : bool
        If True, verbose debug output is printed.

    References
    ----------
    [1] Tan, Z. (2015) Optimally adjusted mixture sampling and locally weighted histogram analysis, Journal of Computational and Graphical Statistics, to appear. (Supplement)
    http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf

    Examples
    --------

    """
    def __init__(self, sampler, logZ=None, log_target_probabilities=None, update_stages='two-stage', update_method='rao-blackwellized', adapt_target_probabilities=False,
        guess_logZ=False, mbar_update_interval=None):
        """
        Create a SAMS Sampler.

        Parameters
        ----------
        sampler : ExpandedEnsembleSampler
            The expanded ensemble sampler used to sample both configurations and discrete thermodynamic states.
        logZ : dict of key : float, optional, default=None
            If specified, the log partition functions for each state will be initialized to the specified dictionary.
        log_target_probabilities : dict of key : float, optional, default=None
            If specified, unnormalized target probabilities; default is all 0.
        update_stages : str, optional, default='two-stage'
            Number of stages to use for update. One of ['one-stage', 'two-stage']
        update_method : str, optional, default='optimal'
            SAMS update algorithm. One of ['optimal', 'rao-blackwellized']
        adapt_target_probabilities : bool, optional, default=False
            If True, target probabilities will be adapted to achieve minimal thermodynamic length between terminal thermodynamic states.
        guess_logZ : bool, optional, default=False
            If True, will attempt to guess initial logZ from energies of initial snapshot in all thermodynamic states.
        mbar_update_interval : int, optional, default=False
            If set to a positive integer, MBAR will be used to update the estimates with the specified interval until histograms are flat.

        """
        # Check input arguments.
        self.supported_update_methods = ['optimal', 'rao-blackwellized']
        if update_method not in self.supported_update_methods:
            raise Exception("Update method '%s' not in supported update schemes: %s" % (update_method, str(self.supported_update_methods)))

        # Keep copies of initializing arguments.
        self.sampler = sampler
        self.logZ = logZ
        self.log_target_probabilities = log_target_probabilities
        self.update_stages = update_stages
        self.update_method = update_method

        if self.logZ is None:
            self.logZ = np.zeros([self.sampler.nstates], np.float64)
        if self.log_target_probabilities is None:
            self.log_target_probabilities = np.zeros([self.sampler.nstates], np.float64)
        self.log_target_probabilities -= logsumexp(self.log_target_probabilities)

        # Initialize.
        self.iteration = 0
        self.verbose = False

        if adapt_target_probabilities:
            raise Exception('Not implemented yet.')

        if guess_logZ:
            self.guess_logZ()

        self.mbar_update_interval = mbar_update_interval

        self._timing = self.sampler._timing
        self._initializeNetCDF(self.sampler.ncfile)

    def _initializeNetCDF(self, ncfile):
        self.ncfile = ncfile
        if self.ncfile == None:
            return

        nstates = self.sampler.nstates
        self.ncfile.createVariable('logZ', 'f4', dimensions=('iterations', 'states'), zlib=True, chunksizes=(1,nstates))
        self.ncfile.createVariable('log_target_probabilities', 'f4', dimensions=('iterations', 'states'), chunksizes=(1,nstates))
        self.ncfile.createVariable('update_logZ_time', 'f4', dimensions=('iterations',), chunksizes=(1,))

    def guess_logZ(self):
        # Compute guess of all energies.
        for state_index in range(self.sampler.nstates):
            self.logZ[state_index] = - self.sampler.thermodynamic_states[state_index].reduced_potential(self.sampler.sampler.context)
        # Restore thermodynamic state.
        self.sampler.thermodynamic_states[self.sampler.thermodynamic_state_index].update_context(self.sampler.sampler.context, self.sampler.integrator)

    def update_sampler(self):
        """
        Update the underlying expanded ensembles sampler.
        """
        self.sampler.update()

    def update_logZ_estimates(self):
        """
        Update the logZ estimates according to selected SAMS update method.

        References
        ----------
        [1] http://www.stat.rutgers.edu/home/ztan/Publication/SAMS_redo4.pdf

        """
        initial_time = time.time()

        current_state = self.sampler.thermodynamic_state_index
        log_pi_k = self.log_target_probabilities
        pi_k = np.exp(self.log_target_probabilities)
        log_P_k = self.sampler.log_P_k # log probabilities of selecting states in neighborhood during update

        gamma0 = 1.0
        if self.update_stages == 'one-stage':
            gamma = gamma0 / float(self.iteration+1) # prefactor in Eq. 9 and 12 from [1]
            self.ncfile.variables['stage'][self.iteration] = 1
        elif self.update_stages == 'two-stage':
            if hasattr(self, 'second_stage_iteration_start'):
                # Use second stage scheme
                self.ncfile.variables['stage'][self.iteration] = 2
                # We flattened at iteration t0. Use this to compute gamma
                t0 = self.second_stage_iteration_start
                gamma = 1.0 / float(self.iteration - t0 + 1./gamma0)
            else:
                # Use first stage scheme.
                self.ncfile.variables['stage'][self.iteration] = 1
                beta_factor = 0.6
                t = self.iteration + 1.0
                #gamma = min(pi_k[current_state], t**(-beta_factor)) # Eq. 15
                gamma = t**(-beta_factor) # Modified version of Eq. 15

                # Check if all state histograms are "flat" within 20% so we can enter the second stage
                RELATIVE_HISTOGRAM_ERROR_THRESHOLD = 0.20
                N_k = self.sampler.number_of_state_visits[:]
                empirical_pi_k = N_k[:] / N_k.sum()
                pi_k = np.exp(self.log_target_probabilities)
                relative_error_k = np.abs(pi_k - empirical_pi_k) / pi_k
                if np.all(relative_error_k < RELATIVE_HISTOGRAM_ERROR_THRESHOLD):
                    self.second_stage_iteration_start = self.iteration
                    # Record start of second stage
                    setattr(self.ncfile.varables['stage'], 'second_stage_start', self.second_stage_iteration_start)
        else:
            raise Exception("update_stages method '%s' unknown" % self.update_stages)

        # Record gamma for this iteration
        print('gamma = %f' % gamma)
        self.ncfile.variables['gamma'][self.iteration] = gamma

        if self.update_method == 'optimal':
            # Based on Eq. 9 of Ref. [1]
            self.logZ[current_state] += gamma * np.exp(-log_pi_k[current_state])
        elif self.update_method == 'rao-blackwellized':
            # Based on Eq. 12 of Ref [1]
            neighborhood = self.sampler.neighborhood # indices of states for expanded ensemble update
            self.logZ[neighborhood] += gamma * np.exp(log_P_k[neighborhood] - log_pi_k[neighborhood])
        else:
            raise Exception("SAMS update method '%s' unknown." % self.update_method)

        # Subtract off logZ[0] to prevent logZ from growing without bound
        self.logZ[:] -= self.logZ[0]

        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['update logZ time'] = elapsed_time
        if self.verbose:
            print('time elapsed %8.3f s' % elapsed_time)

    def update_logZ_with_mbar(self):
        """
        Use MBAR to update logZ estimates.
        """
        if not self.ncfile:
            raise Exception("Cannot update logZ using MBAR since no NetCDF file is storing history.")

        if not self.sampler.update_scheme == 'global-jump':
            raise Exception("Only global jump is implemented right now.")

        # Extract relative energies.
        if self.verbose:
            print('Updating logZ estimate with MBAR...')
        initial_time = time.time()
        from pymbar import MBAR
        #first = int(self.iteration / 2)
        first = 0
        u_kn = np.array(self.ncfile.variables['u_k'][first:,:]).T
        [N_k, bins] = np.histogram(self.ncfile.variables['state_index'][first:], bins=(np.arange(self.sampler.nstates+1) - 0.5))
        mbar = MBAR(u_kn, N_k)
        Deltaf_ij, dDeltaf_ij, Theta_ij = mbar.getFreeEnergyDifferences(compute_uncertainty=True, uncertainty_method='approximate')
        self.logZ[:] = -mbar.f_k[:]
        self.logZ -= self.logZ[0]
        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['MBAR time'] = elapsed_time
        if self.verbose:
            print('MBAR time    %8.3f s' % elapsed_time)

    def update_log_weights(self):
        """
        Update log weights for expanded ensemble sampler.
        """
        # Update log weights for expanded ensemble sampler sampler
        self.sampler.log_weights[:] = self.log_target_probabilities[:] - self.logZ[:]

    def update(self):
        """
        Update the sampler with one step of sampling.
        """
        initial_time = time.time()

        if self.verbose:
            print('')
            print("=" * 80)
            print("SAMS sampler iteration %5d" % self.iteration)

        # Update positions and state index with expanded ensemble sampler.
        self.update_sampler()

        # Update logZ estimates and expanded ensemble sampler log weights.
        self.update_logZ_estimates()
        if self.mbar_update_interval and (not hasattr(self, 'second_stage_iteration_start')) and (((self.iteration+1) % self.mbar_update_interval) == 0):
            self.update_logZ_with_mbar()

        self.update_log_weights()

        if self.ncfile:
            self.ncfile.variables['logZ'][self.iteration,:] = self.logZ[:]
            self.ncfile.variables['log_target_probabilities'][self.iteration,:] = self.log_target_probabilities[:]
            self.ncfile.variables['update_logZ_time'][self.iteration] = self._timing['update logZ time']
            self.ncfile.sync()

        final_time = time.time()
        elapsed_time = final_time - initial_time
        self._timing['sams time'] = elapsed_time
        if self.verbose:
            print('total time   %8.3f s' % elapsed_time)

        self.iteration += 1
        if self.verbose:
            print("=" * 80)

    def run(self, niterations=1):
        """
        Run the sampler for the specified number of iterations

        Parameters
        ----------
        niterations : int, optional, default=1
            Number of iterations to run the sampler for.
        """
        for iteration in range(niterations):
            self.update()
