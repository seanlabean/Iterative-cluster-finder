import numpy as np
from amuse.lab import generic_unit_converter, nbody_system
from amuse.lab import units as u
from amuse.lab import Particles
import amuse.lab

import matplotlib
matplotlib.use('Agg')
import matplotlib
font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)
matplotlib.rc({'savefig.dpi':300})
import matplotlib.pyplot as plt

from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import sys
import h5py

def find_clusters_with_dbscan(stars, outer_density_limit=1.0 | u.MSun*u.parsec**-3,
                              avg_stellar_mass=0.586,
                              eps=0.4, min_samples=12, leaf_size=30,
                              return_labels=False, debug=False):

    """
    Find all the stars in clusters using
    the DBSCAN implementation from scikit-learn.

    Keyword Arguments:
    stars               -- AMUSE particle set
    outer_density_limit -- If set, use this density_limit
                           in solar masses per parsec^-3 to
                           compute eps to figure
                           out where the cluster edges are
                           instead of the input eps. 
                           A good default choice
                           is 1.0 MSun * pc^-3.
                           Note this setting overrides eps.
    avg_stellar_mass    -- Average stellar mass of the IMF
                           used to make the stars your
                           clustering. Default is 
                           from a Kroupa IMF that goes 
                           from 0.08 to 150 Msun.
    eps                 -- Minimum  neighbor distance to be
                           considered in the cluster in pc.
                           This value is calculated for you
                           if you use outer_density_limit.
    min_samples         -- Minimum number of neighbors to
                           be considered a core particle.
                           Default is 12, the default for
                           DBSCAN.
    leaf_size           -- Number of particles in a leaf
                           on the KD tree the code uses
                           to find neighbors. Default is
                           30, the default for DBSCAN.
    return_labels       -- Also return the raw DBSCAN label output?
    debug               -- Turn on debugging output

    Returns:
    groups        -- A list of particle sets for the particles in each cluster.
    n_groups      -- The number of clusters found.
    labels        -- The actual label indicies returned by DBSCAN,
                     only returned if return_labels=True.
    unique_labels -- The unique labels returned by DBSCAN,
                     only returned if return_labels=True.
    """

    pre = "[find_clusters_with_dbscan]:"

    if (outer_density_limit is not None):

        # The number of samples should
        # be greater than that of
        # the average density of the
        # SN in a pc^3 (~ 0.01, BT), but not as high
        # as that of an open cluster 
        # (~10 Msun / pc^3, Binney and Tremaine)

        # Note the mean number density of the solar
        # neighborhood is 0.17 stars per parsec^3,
        # while the mean in an open cluster is 17 stars
        # per parsec^3, so a good choice is the mean
        # of these in log space, or about 1 star per parsec^-3.

        # So here we are saying they should be at least closer
        # that the average distance between stars in the SN.
        number_density_limit = (outer_density_limit.value_in(u.MSun*u.parsec**-3) 
                             / avg_stellar_mass)

        if (number_density_limit < 0.17):
            print(pre, "WARNING: Your number density limit \
                        at", number_density_limit, "pc^-3 \
                        is smaller than that of the solar \
                        neighborhood, which is ~ 0.17 pc^-3!")

        eps = number_density_limit**(-1./3.)

    if (debug):
        print(pre, "outer_density_limit  =", outer_density_limit)
        print(pre, "avg_stellar_mass     =", avg_stellar_mass)
        print(pre, "number_density_limit =", number_density_limit)
        print(pre, "eps =", eps)

    particle_positions = stars.position.value_in(u.parsec)

    # Note: I don't think its necessary to scale
    #       the inputs for DBSCAN when you are
    #       using a simple Eulerian metric on 3-d
    #       position space, as its just rescaling eps.

    # Get a DBSCAN instance running.
    db = DBSCAN(eps=eps, min_samples=min_samples, leaf_size=leaf_size)
    # Do the clustering.
    clstrs = db.fit_predict(particle_positions)
    # Get the unique cluster lables (i.e. the number of clusters).
    labels = db.labels_
    unique_labels = set(labels) # This returns only the unique ones.
    # Anything with an index of -1 is noise.
    tmp = []
    for val in unique_labels:
        if val >= 0: 
            tmp.append(1)
    n_groups = len(tmp)
    #n_groups = len(filter((lambda x: x>=0),unique_labels))
    groups = []

    for label in unique_labels:
        if (label >= 0): # Don't include noise particles here.
            groups.append(stars[np.where(labels == label)[0]])

    if (debug):
        print(pre, "groups=", groups)
        print(pre, "n_groups=", n_groups)
        print(pre, "labels=", labels)
        print(pre, "unique_labels=", unique_labels)

    if (return_labels):
        return groups, n_groups, labels, unique_labels
    else:
        return groups, n_groups


conv = generic_unit_converter.ConvertBetweenGenericAndSiUnits(
        1.0 | u.cm, 1.0 | u.g, 1.0 | u.s)
stars = amuse.io.read_set_from_file("./example_data/DBSCAN_input/L3-50M-2tff_stars.amuse", format='amuse')
# There's a lot of parameters to play with here, this is just a default call
# for testing purposes.
groups, n_groups = find_clusters_with_dbscan(stars,debug=True)

# As a test, print the masses of each group identified:
for group in groups:
    print(group.mass.sum().value_in(u.MSun))
