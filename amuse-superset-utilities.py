import yt
import numpy as np
from amuse.lab import generic_unit_converter, nbody_system
from amuse.lab import units as u
from amuse.lab import Particles
import amuse.lab

def make_amuse_set_gas_and_stars(input_hdf5_plt,
                                 input_hdf5_part,
                                 calc_gas_ener=False):
    """
    Creates an AMUSE particle set from a FLASH output 
    containing gas cells and star particles.
    Both gas cells and star particles are converted
    into AMUSE particles and are catagorized by tag:
    gas (0), star (1). The resulting particle set
    has attributes of tag, mass, position, and 
    velocity.
    
    Arguments:
    input_hdf5_plt  - hdf5 plot file path
    input_hdf5_part - hdf5 particle file path
    calc_gas_ener   - flag to set particle magnetic and internal energy, 
                      used to calculate gas total energy to remove unbound gas.
                      Default is False.
    
    Returns:
    stars - star particle set
    gas   - gas particle set
    """
    # Load hdf5 files in yt so we can extract cell and particle information
    ds = yt.load(input_hdf5_plt,
                 particle_filename=input_hdf5_part)
    ad = ds.all_data()
    
    # Set gas cell properties as AMUSE particle data
    num_gas = len(ad['dens'])
    gas = Particles(num_gas)
    
    gas.tag = np.zeros(num_gas)
    gas.mass = ad['cell_mass'].v | u.g
    gas.x = ad['x'].v | u.cm
    gas.y = ad['y'].v | u.cm
    gas.z = ad['z'].v | u.cm
    gas.vx = ad['velocity_x'].v | u.cm/u.s
    gas.vy = ad['velocity_y'].v | u.cm/u.s
    gas.vz = ad['velocity_z'].v | u.cm/u.s
    if (calc_gas_ener):
        # Set gas thermal and magnetic energy
        gas.ME = ad['magp'].v*ad['cell_volume'].v / 4.0 / np.pi | u.erg
        gas.EI = ad['cell_mass'].v * ad['eint'].v | u.erg
        
    # Set star particle properties as AMUSE particle data
    # (must be same labels for when we concatenate with gas)
    num_stars = len(ad['particle_mass'].v)
    stars = Particles(num_stars)

    stars.tag = np.ones(num_stars)
    stars.mass = ad['particle_mass'].v | u.g
    stars.x = ad['particle_position_x'].v | u.cm
    stars.y = ad['particle_position_y'].v | u.cm
    stars.z = ad['particle_position_z'].v | u.cm
    stars.vx = ad['particle_velocity_x'].v | u.cm/u.s
    stars.vy = ad['particle_velocity_y'].v | u.cm/u.s
    stars.vz = ad['particle_velocity_z'].v | u.cm/u.s
    if (calc_gas_ener):
        # Set dummy magnetic and internal energy
        stars.ME = np.zeros(num_stars) | u.erg
        stars.EI = np.zeros(num_stars) | u.erg

    return stars, gas

def setup_superset(input_hdf5_plt,
                   input_hdf5_part,
                   calc_gas_ener=False):
    '''
    Generates an AMUSE superset of gas and star particles
    tags = [0] and [1] respectively.
    Gas and star particles have positions, velocities, and masses.
    
    We then calculate the total energy of each star particle
    based on its KE, and PE due to all other stars and gas
    using update_total_E(superset).
    
    Arguments:
    input_hdf5_plt  - hdf5 plot file path
    input_hdf5_part - hdf5 particle file path
    calc_gas_ener   - flag to set particle magnetic and internal energy, 
                      used to calculate gas total energy to remove unbound gas.
                      Default is False.
    
    Other Function Calls:
    update_total_E(superset,calc_gas_ener)
    
    Returns:
    superset - star and gas particles
    '''
    # First extract star and gas positions, velocities, and masses
    # from hdf5 output using yt.
    stars, gas = make_amuse_set_gas_and_stars(input_hdf5_plt,
                                              input_hdf5_part,
                                              calc_gas_ener)
    
    # Build superset.
    superset = Particles()
    superset.add_particles(gas)
    superset.add_particles(stars)
    # Calculate and set total energy particle attribute
    superset = update_total_E(superset, calc_gas_ener)
    return superset

def update_total_E(superset,calc_gas_ener=False):
    '''
    Calculates the Total Energy of all the particles 
    tagged as [1] (star) in an AMUSE superset. We do some 
    casual array indexing to ensure the potential calculation
    for the star particles includes the masses of all the gas,
    but does not calculate the potential for each gas particle.
    
    We then update the totE quantity of the superset, setting
    the totE for the stars as our calculated values and the 
    totE for the gas as -1 (since we dont need this value and
    this ensures the gas is not removed from the superset in
    upcoming steps).
    
    Arguments:
    superset - AMSUE superset of gas and star particles
    calc_gas_ener   - flag to set particle magnetic and internal energy, 
                      used to calculate gas total energy to remove unbound gas.
                      Default is False.
    
    Returns:
    superset - superset with updated totE values
    '''
    # Calculate total energy of star particles only
    # starKE + PE_stars + PE_gas
    star_ind = np.where(superset.tag==1.)
    star_pot_from_gas_and_stars = superset[star_ind].potential().as_quantity_in(u.cm**2 / u.s**2)
    star_spec_kine = superset[star_ind].specific_kinetic_energy().as_quantity_in(u.cm**2 / u.s**2)
    star_mass = superset[star_ind].mass.as_quantity_in(u.g)
    star_totE = star_mass*(star_spec_kine+star_pot_from_gas_and_stars)
    
    gas_ind = np.where(superset.tag==0.)
    if (calc_gas_ener):
        # Calculate gas total energy
        print("Calculating gas potential...") # Printing to screen b/c this step can take a while
        gas_pot_from_gas_and_stars = superset[gas_ind].potential().as_quantity_in(u.cm**2 / u.s**2)
        gas_spec_kine = superset[gas_ind].specific_kinetic_energy().as_quantity_in(u.cm**2 / u.s**2)
        gas_mass = superset[gas_ind].mass.as_quantity_in(u.g)
        gas_ME = superset[gas_ind].ME.as_quantity_in(u.J)
        gas_EI = superset[gas_ind].EI.as_quantity_in(u.J)
        gas_totE = gas_mass*(gas_spec_kine+gas_pot_from_gas_and_stars)+gas_ME+gas_EI
    else:
        # Set dummy totE attribute to gas particles
        gas_totE = (-1 | u.J)*np.ones(len(superset[gas_ind]))
    # Set superset totE attribute as concatenated gas and star totE arrays.
    superset.totE = np.concatenate((gas_totE.as_quantity_in(u.J),star_totE.as_quantity_in(u.J)))
    return superset

def iterate_remove_stars(superset, iterate_gas_ener=False, report_removal_stats=False):
    '''
    Iteration following the steps:
    1. Remove star particles from superset with total E > 0.
    2. Check if we did in fact remove stars, if No then set break bool.
    3. Update star total energies using updated superset particles.
    
    Arguments:
    superset             - AMUSE particle superset of gas and star particles
    iterate_gas_ener     - Flag to iteratively remove unbound gas.
                           Only set to True in conjunction with calc_gas_ener.
                           Setting to True is not recommended for refined 
                           simulations as the operation is very expensive.
                           Default is False.
    report_removal_stats - Report number of stars removed, 
                           total mass of removed stars.
    
    Other Function Calls:
    update_total_E(superset, calc_gas_ener)
    
    Returns:
    superset - Final superset in which no stars with total E > 0 exist
    '''
    supersetf = superset.copy()
    unbound = True
    if (report_removal_stats):
        print("Reporting removal stats!")
        start_num_stars = len(superset[np.where(superset.tag==1.)])
        start_mass_stars = superset[np.where(superset.tag==1.)].mass.sum()
        print("Initial num stars:", start_num_stars)
        print("Initial mass stars: {:.2f} MSun".format(\
                    start_mass_stars.value_in(u.MSun)))
        print("---------------------")
    if (iterate_gas_ener):
        print("Removing unbound gas iteratively. \
        Strap in, this will take a while...")
    while(unbound):
        if (report_removal_stats):
            print("\nUnbound stars detected...")
            init_num_stars = len(supersetf[np.where(supersetf.tag==1.)])
            init_mass_stars = supersetf[np.where(supersetf.tag==1.)].mass.sum()
        unbound_ind = np.where(supersetf.totE.value_in(u.J) > 0)[0]
        supersetf.remove_particles(supersetf[unbound_ind])
        
        if (report_removal_stats): 
            final_num_stars = len(supersetf[np.where(supersetf.tag==1.)])
            final_mass_stars = supersetf[np.where(supersetf.tag==1.)].mass.sum()
            print("num stars removed:", init_num_stars-final_num_stars)
            print("mass stars rmvd: {:.2f} MSun".format(\
                    (init_mass_stars-final_mass_stars).value_in(u.MSun)))
            
        if len(unbound_ind) == 0:
            unbound = False
        supersetf = update_total_E(supersetf, iterate_gas_ener)
        
    if (report_removal_stats):
        end_num_stars = len(supersetf[np.where(supersetf.tag==1.)])
        end_mass_stars = supersetf[np.where(supersetf.tag==1.)].mass.sum()
        print("\nSummary: \ntotal stars removed:", start_num_stars-end_num_stars)
        print("total star mass rmvd: {:.2f} MSun".format(\
                    (start_mass_stars-end_mass_stars).value_in(u.MSun)))
        print("total mass rmvd: {:.2f} MSun".format(\
                    (superset.mass.sum()-supersetf.mass.sum()).value_in(u.MSun)))
    return supersetf
