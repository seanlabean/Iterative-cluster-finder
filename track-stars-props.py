import yt
yt.mylog.level=50
import numpy as np
from amuse.lab import generic_unit_converter, nbody_system
from amuse.lab import units as u
from amuse.lab import Particles
import amuse.lab
from amuse_superset_utilities import *

import argparse

parser = argparse.ArgumentParser(description="Run an analysis of energy \
                                 in the stars and gas in FLASH plot files. \
                                 This anaylsis script will identify clusters \
                                 in the data, then calculate the properties \
                                 in the region surrounding the clusters. \
                                 Written by: Joshua Wall, Drexel University")
parser.add_argument("-d", "--directory", default="./",
                   help="Input directory where the particle files are located. \
                         Default is './'")
parser.add_argument("-w", "--write_directory", default="./",
                   help="Output directory where the files are written. \
                         Default is './'")
parser.add_argument("-f", "--filename", default="*plt_cnt*",
                   help="Filename prefix (before the numbers). \
                         Default is *plt_cnt*.")
parser.add_argument("-o", "--out_filename", default="starprops",
                   help="Base filename for output files. \
                         Default is starprops.")
parser.add_argument("-db", "--debug", action="store_true",
                    help="Debug this script.")
parser.add_argument("-s", "--start_file_number", default=None, type=int,
                    help="The starting number of the files you want to \
                          plot. By default its None, which picks the first file \
                          in the folder.")
parser.add_argument("-e", "--end_file_number", default=None, type=int,
                    help="The ending number of the files you want to \
                          plot. By default its None, which picks the \
                          last file in the folder.")
parser.add_argument("-c", "--chunk_size", default=1, type=int,
                    help="Number of chunks to send to each worker process.")
args        = parser.parse_args()
file_dir    = args.directory
file_name   = args.filename
start       = args.start_file_number
end         = args.end_file_number
out_dir     = args.write_directory
out_file    = args.out_filename
debug       = args.debug
chunk_size  = args.chunk_size


if (debug): print("Starting up in debug mode:",debug)

rank, size, comm = initialize_mpi()

pre = "Proc", rank

if (rank==0):
    if (debug):
        print("Calling gather files")
        print("file_name = ", file_name)
        print("file_dir  = ", file_dir)
        print("start     = ", start)
        print("end       = ", end)

    files, start, end = gather_files(file_name, file_dir,
                        start=start, end=end, debug=debug)
    num_files = len(files)

    if (debug):
        print(files)
        print(num_files)

calc_gas_ener=False
        
fargs = []
fkwargs = {"local_files":files, "calc_gas_ener":calc_gas_ener, "out_dir":out_dir, "outfile":out_file}

if (rank == 0):
    print("Options selected are:")
    print(fkwargs)

perform_task_in_parallel(track_star_props,
                         fargs, fkwargs, files, chunk_size,
                         rank, size, comm, root=0, debug=debug)
