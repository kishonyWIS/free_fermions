from datetime import datetime
import os
import time

J_list = [1.0, 0.5]
h_list = [0.5, 1.0]
V_list = [0.0, 0.5]
Ns = 12
N_cycles = 3
errors_list = [0.0, 2e-3, 1e-2, 2e-2]


def to_integer(dt_time):
    return int(dt_time.strftime("%Y%m%d%H%M%S"))


for J, h in zip(J_list, h_list):
    for V in V_list:
        for errors in errors_list:
            time.sleep(5)
            print("J = " + str(J) + " h = " + str(h) + " V = " + str(V) + " error rate = " + str(errors))
            seed = to_integer(datetime.now())
            with open("job_script.sh", "w") as ff:
                ff.write("#BSUB -n 1\n"
                         "#BSUB -q berg\n"
                         "#BSUB -R \"rusage[mem=2000]\"\n"
                         "#BSUB -R \"span[ptile = 40]\"\n"
                         "#BSUB -J jobname\n"
                         "#BSUB -eo ./error/error-"+str(seed)+".txt\n"
                         "#BSUB -oo ./log/log-"+str(seed)+".txt\n"
                         "julia energy_density_vs_system_size.jl "+str(J)+" "+str(h)+" "+str(V)+" "+str(N_cycles)+" "+str(errors)+" "+str(Ns)+" "+str(seed)+" > ./output/output-"+str(seed)+".txt")
            os.system("./run_script_wrapper.sh")