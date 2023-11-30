from datetime import datetime
import os
import time
from itertools import product
import numpy as np
import sys


def to_integer(dt_time):
    return int(dt_time.strftime("%Y%m%d%H%M%S"))

if __name__ == '__main__':


    g0 = 0.5
    B0 = 5.
    B1 = 0.
    J = 1.
    kappa = 1.
    periodic_bc = '(True, False)'
    cycles = 100
    cycles_averaging_buffer = 98
    initial_state = 'ground'
    draw_spatial_energy = False
    num_sites_x_list = [None]
    num_sites_y_list = list(range(2,30,2))
    T_list = np.arange(1,16,1)
    error_rate = 0.

    for num_sites_x, num_sites_y, T in product(num_sites_x_list, num_sites_y_list, T_list):
        if num_sites_x is None:
            num_sites_x = num_sites_y

        trotter_steps = int(T * 40)

        time.sleep(0.5)
        print("Submitting job...")
        print(f"g0 = {g0}, B0 = {B0}, B1 = {B1}, J = {J}, kappa = {kappa}, periodic_bc = {periodic_bc}, cycles = {cycles}, cycles_averaging_buffer = {cycles_averaging_buffer}, initial_state = {initial_state}, draw_spatial_energy = {draw_spatial_energy}, num_sites_x = {num_sites_x}, num_sites_y = {num_sites_y}, T = {T}, trotter_steps = {trotter_steps}, error_rate = {error_rate}")
        seed = to_integer(datetime.now()) % 2**32
        with open("job_script.sh", "w") as ff:
            ff.write("#BSUB -n 1\n"
                     "#BSUB -q berg\n"
                     "#BSUB -R \"rusage[mem=1000]\"\n"
                     "#BSUB -R \"span[ptile = 40]\"\n"
                     "#BSUB -J jobname\n"
                     "#BSUB -eo ./error/error-"+str(seed)+".txt\n"
                     "#BSUB -oo ./log/log-"+str(seed)+".txt\n"
                     f"python run_with_params.py {seed} {g0} {B0} {J} {kappa} {periodic_bc} {cycles} {cycles_averaging_buffer} {initial_state} {draw_spatial_energy} {num_sites_x} {num_sites_y} {T} {trotter_steps} {error_rate} > ./output/output-"+str(seed)+".txt")
        os.system("./run_script_wrapper.sh")