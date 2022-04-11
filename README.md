# Iterative-cluster-finder

Identifying which stars constitute a cluster is of particular interest to researchers involved in the ongoing study of star cluster formation. Most available cluster-finding algorithms are ill-equipped to accurately determine the membership of a simulated star cluster. This is partly due to the fact that observations of star clusters simply require a nearest-neighbor algorithm with a desity threshold cutoff to identify a "cluster". This can potentially cause misleading results, as stars that are not bound to a larger cluster can still be determined to be a member of the cluster simply due to their proximity. In observational settings, there are not many avenues available to rectify this. In computation astrophysics, however, we have access to much more information about the stars that allow us to more more precicely determine a cluster's membership.

By first identifying a sufficiently dense structure of stars (using observational techniques), we can then calculate the center of mass of the structure and then can determine whether or not each star within the stucture is bound to the center of mass. We then can iterate this process, resulting in the identification of a structure in which the members are bound to a like center of mass, eliminating the possibility of misidentifying unbound stars as cluster members.

## Technique Summary

1. Provide AMUSE star field with star position, mass, velocity data.
2. Identify stellar group with nearest-neighbor and number density threshold techniques via scikit-learn's DBSCAN.
3. Calculate group's center of mass.
4. Utilize AMUSE to determine boundedness of each star to the center of mass, discard all unbound stars.
5. Repeat from (2.)

### Software packages
This routine requires:

* an installation of the [Astrophysical Multipurpose Software Envoronment (AMUSE)](https://github.com/amusecode/amuse)
* Python data analysis library [scikit-learn](https://scikit-learn.org/) which itself requires NumPy, SciPy, and matplotlib.