import itertools
import os
import random
import re
import string
import subprocess
import sys
from pathlib import Path
from time import sleep, time

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
# size of files, number of pattern, pattern size, approx factor, files to open
scenarios = {
    "large files 1 large patterns": (LARGE_FILE, 1, LARGE_PATTERN, 4, 4),
    "large files 100 small patterns": (LARGE_FILE, 100, SMALL_PATTERN, 4, 4),
    "small files 10 large patterns": (SMALL_FILE, 10, LARGE_PATTERN, 4, 4),
    "small files 1000 small patterns": (SMALL_FILE, 1000, SMALL_PATTERN, 4, 4),
}

settings = {
    "mpi + omp + gpu": {},
    "mpi + omp": {"PERCENTAGE_GPU": 0},
    "mpi": {"OMP_NUM_THREADS": 1, "PERCENTAGE_GPU": 0},
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
    nodes=4,
    mpi_processes_per_node=4,
):

    random.seed(0)
    np.random.seed(0)

    # if "OMP_NUM_THREADS" in additional_env_vars and additional_env_vars["OMP_NUM_THREADS"] > 1:
    #     mpi_processes_per_node = 1

    cmd = f"salloc -N {nodes} -n {nodes * mpi_processes_per_node} mpirun"

    additional_env_vars = {k: str(v) for k, v in additional_env_vars.items()}

    regex_exec_time = re.compile(
        r"done in ([0-9\.]*) s: ([0-9\.]*) s transmitting, ([0-9\.]*) s computing, ([0-9\.]*) s gathering"
    )
    regex_matches = re.compile(r"Number of matches for pattern <([A-Z]*)>: ([0-9]*)")

    total_runtimes = []
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
            env = {}
            for k, v in os.environ.items():
                env[k] = v
            for k, v in additional_env_vars.items():
                env[k] = v
            output_to_test = subprocess.run(
                command_to_test.split(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).stdout.decode()

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

        try:
            parallel_times, parallel_dic_result = get_results(output_to_test)
            total_runtime = parallel_times[0]
            total_runtimes.append(total_runtime)

            runtime_to_find += time() - st
        except:
            print("Error with output")
            print(output_to_test)

    return np.mean(total_runtimes), np.std(total_runtimes)


# Compare the different settings
with open("results/results_v4.csv", "w") as f:
    f.write("scenario,setting,mean_time,std_time\n")

    for scenario_name, scenario in scenarios.items():
        print(f"Running scenario {scenario_name}")

        for setting_name, setting in settings.items():
            print(f"Running setting {setting_name}")
            mean_runtime, std_runtime = measure_runtime(
                1,
                setting,
                *scenario,
            )
            print(f"Total runtime {mean_runtime:.2f} s")

            f.write(f"{scenario_name},{setting_name},{mean_runtime},{std_runtime}\n")

            print()

        print()

### Perf vs percentage gpu
with open("results/results_gpu_percentage_v4.csv", "w") as f:
    f.write("scenario,percentage_gpu,mean_time,std_time\n")

    for scenario_name, scenario in scenarios.items():
        print(f"Running scenario {scenario_name}")

        for percentage_gpu in range(0, 100 + 1, 10):
            print(f"Running with {percentage_gpu}% on gpu")
            mean_runtime, std_runtime = measure_runtime(
                5,
                {"PERCENTAGE_GPU": percentage_gpu, "FORCE_GPU": 1},
                *scenario,
            )
            print(f"Total runtime {mean_runtime:.2f} s")

            f.write(f"{scenario_name},{percentage_gpu},{mean_runtime},{std_runtime}\n")
            f.flush()

            print()
        print()

### Perf vs ndata & npattern when distributed vs not, at constant workload
with open("results/results_distributed_and_pattern_v4.csv", "w") as f:
    f.write("distributed,ndata,npatterns,mean_time,std_time\n")

    WORK_LOAD = LARGE_FILE * 100
    for distributed in [0, 1]:
        for ndata in [WORK_LOAD // 2**i for i in range(0, 14)]:
            npatterns = WORK_LOAD // ndata
            print(f"Running with {ndata} data and {npatterns} patterns, distributed={distributed}")

            mean_time, std_time = measure_runtime(
                5, {"DISTRIBUTE_PATTERNS": distributed}, ndata, npatterns, SMALL_PATTERN, 4, 4
            )
            print(f"Total runtime {mean_time:.2f} s")

            f.write(f"{distributed},{ndata},{npatterns},{mean_time},{std_time}\n")
            f.flush()
            print()

with open("results/results_distributed_and_pattern_v4_no_gpu.csv", "w") as f:
    f.write("distributed,ndata,npatterns,mean_time,std_time\n")

    WORK_LOAD = LARGE_FILE * 100
    for distributed in [0, 1]:
        for ndata in [WORK_LOAD // 2**i for i in range(0, 14)]:
            npatterns = WORK_LOAD // ndata
            print(f"Running with {ndata} data and {npatterns} patterns, distributed={distributed}")

            mean_time, std_time = measure_runtime(
                5, {"DISTRIBUTE_PATTERNS": distributed, "PERCENTAGE_GPU": 0}, ndata, npatterns, SMALL_PATTERN, 4, 4
            )
            print(f"Total runtime {mean_time:.2f} s")

            f.write(f"{distributed},{ndata},{npatterns},{mean_time},{std_time}\n")
            f.flush()
            print()


### Weak scaling
with open("results/results_weak_scaling_v4.csv", "w") as f:
    f.write("scenario,nodes,mean_time,std_time\n")

    for scenario_name, scenario in scenarios.items():
        print(f"Running scenario {scenario_name}")

        for nodes in [1, 2, 4, 8, 16, 32, 64]:
            print(f"Running with {nodes} nodes")

            ndata, npatterns, *rest = scenario
            npatterns = npatterns * nodes // 8

            mean_runtime, std_runtime = measure_runtime(
                5,
                {},
                ndata,
                npatterns,
                *rest,
                nodes=nodes,
            )
            print(f"Total runtime {mean_runtime:.2f} s")

            f.write(f"{scenario_name},{nodes},{mean_runtime},{std_runtime}\n")
            f.flush()

            print()
        print()

### Perf vs number of threads and blocks
with open("results/results_thread_block_v4.csv", "w") as f:
    f.write("scenario,thread_per_block,block_per_grid,mean_time,std_time\n")

    for scenario_name, scenario in scenarios.items():
        print(f"Running scenario {scenario_name}")

        for thread_per_block in [2**i for i in range(2, 14)]:
            for block_per_grid in [2**i for i in range(2, 17)]:
                print(f"Running with {thread_per_block} thread per block and {block_per_grid} block per grid")
                mean_runtime, std_runtime = measure_runtime(
                    5,
                    {
                        "THREAD_PER_BLOCK": thread_per_block,
                        "BLOCK_PER_GRID": block_per_grid,
                        "FORCE_GPU": 1,
                        "PERCENTAGE_GPU": 100,
                    },
                    *scenario,
                )
                print(f"Total runtime {mean_runtime:.2f} s")
                f.write(f"{scenario_name},{thread_per_block},{block_per_grid},{mean_runtime},{std_runtime}\n")
                f.flush()
                print()
        print()
