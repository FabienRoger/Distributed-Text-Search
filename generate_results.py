import itertools
import os
import random
import re
import string
import subprocess
import sys
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm, trange

random.seed(0)
np.random.seed(0)

to_test = "flexible_mpi"  # .c and .cu should exist

print("compiling")

os.system(f"mpicc -fopenmp -o out/{to_test} -c src/{to_test}.c")
os.system(f"nvcc -I. -o out/{to_test}_cu -c src/{to_test}.cu")
os.system(f"mpicc -fopenmp out/{to_test} out/{to_test}_cu -lcudart -L/usr/local/cuda/lib64 -o out/{to_test}_exec")
print("compiling done")

os.system(f"mkdir -p obj")
# remove obj files
os.system(f"rm -f obj/*.txt")

base_dir = Path(__file__).parent.resolve()
exec_to_test = base_dir / "out" / f"{to_test}_exec"
tmp_dir = base_dir / "obj"
path_tmp_database = tmp_dir / "0.txt"

chars = list(string.ascii_uppercase)


def generate_random_string(length, nb=1):
    if nb == 0:
        random_chars = np.random.choice(chars, size=length)
        return "".join(random_chars)
    else:
        random_chars = np.random.choice(chars, size=(nb, length))
        res = ["".join(random_chars[i]) for i in range(len(random_chars))]
        return res


LARGE_FILE = 10_000
SMALL_FILE = 1_000
LARGE_PATTERN = 100
SMALL_PATTERN = 10
scenarios = {
    "large files 1 large patterns": (LARGE_FILE, 1, LARGE_PATTERN, 4, 4),
    "large files 100 small patterns": (LARGE_FILE, 100, SMALL_PATTERN, 4, 4),
    "small files 10 large patterns": (SMALL_FILE, 10, LARGE_PATTERN, 4, 4),
    "small files 1000 small patterns": (SMALL_FILE, 1000, SMALL_PATTERN, 4, 4),
}

settings = {
    "pattern not distributed": {"DISTRIBUTE_PATTERNS": 0},
    "pattern distributed": {"DISTRIBUTE_PATTERNS": 1},
    "mpi + gpu": {"PERCENTAGE_GPU": 100},
    "only mpi": {"OMP_NUM_THREADS": 1, "PERCENTAGE_GPU": 0},
    "only omp": {"ONLY_RANK_0": 1,"OMP_NUM_THREADS": 8, "PERCENTAGE_GPU": 0},
    "only gpu": {"ONLY_RANK_0": 1, "PERCENTAGE_GPU": 100},
    "no parallelism": {"ONLY_RANK_0": 1, "OMP_NUM_THREADS": 1, "PERCENTAGE_GPU": 0},
}


def measure_runtime(
    k,
    additional_env_vars,
    len_database,
    nb_pattern,
    len_pattern,
    approximation_factor,
    files_to_open,
    nodes=8,
    mpi_processes_per_node = 8,
):


    random.seed(0)
    np.random.seed(0)
    
    # if "OMP_NUM_THREADS" in additional_env_vars and additional_env_vars["OMP_NUM_THREADS"] > 1:
    #     mpi_processes_per_node = 1
    
    cmd = f"salloc -N {nodes} -n {nodes * mpi_processes_per_node} mpirun"
    
    additional_env_vars = {k: str(v) for k, v in additional_env_vars.items()}

    regex_exec_time = re.compile(r"done in ([0-9\.]*) s: ([0-9\.]*) s transmitting, ([0-9\.]*) s computing, ([0-9\.]*) s gathering")
    regex_matches = re.compile(r"Number of matches for pattern <([A-Z]*)>: ([0-9]*)")

    total_runtimes = [0] * 4
    runtime_to_generate = 0
    runtime_to_write = 0
    runtime_to_run = 0
    runtime_to_find = 0

    for _ in trange(k):
        st = time()

        databases = generate_random_string(len_database, files_to_open)
        patterns = generate_random_string(len_pattern, nb_pattern)

        runtime_to_generate += time() - st
        st = time()

        # We write down the database on the file system
        for i, database in enumerate(databases):
            path_tmp_database = tmp_dir / f"{i}.txt"
            path_tmp_database.write_text(database)

        runtime_to_write += time() - st
        st = time()

        command_to_test = f"{cmd} {exec_to_test} {approximation_factor} {tmp_dir} {' '.join(patterns)}"

        try:
            output_to_test = subprocess.check_output(
                command_to_test.split(),
                env=dict(os.environ, **additional_env_vars),
                stderr=subprocess.DEVNULL,
            ).decode()
        except subprocess.CalledProcessError as e:
            print("Error with command")
            print(command_to_test)
            print(e.output.decode())
            exit()

        runtime_to_run += time() - st
        st = time()

        def get_results(output):
            exec_times = [float(f) for f in re.findall(regex_exec_time, output)[0]]
            raw_result = re.findall(regex_matches, output)
            dic_result = {pattern: nbMatches for (pattern, nbMatches) in raw_result}
            return exec_times, dic_result

        parallel_times, parallel_dic_result = get_results(output_to_test)

        for i, t in enumerate(parallel_times):
            total_runtimes[i] += t

        runtime_to_find += time() - st

    return total_runtimes


# creae results file
with open("results.csv", "w") as f:
    f.write("scenario,setting,exec_time,transmit_time,compute_time,gather_time\n")

    for scenario_name, scenario in scenarios.items():
        print(f"Running scenario {scenario_name}")

        for setting_name, setting in settings.items():
            print(f"Running setting {setting_name}")
            runtime, transmit, compute, gather = measure_runtime(
                5,
                setting,
                *scenario,
            )
            print(f"Total runtime {runtime:.2f} s")

            f.write(f"{scenario_name},{setting_name},{runtime:.2f},{transmit:.2f},{compute:.2f},{gather:.2f}\n")

            print()

        print()
