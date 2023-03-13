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
from tqdm import tqdm

random.seed(0)
np.random.seed(0)

seq = "apm1"
to_test = "flexible_mpi"  # .c and .cu should exist

print("compiling")
os.system(f"mpicc -o out/{seq} src/{seq}.c")

os.system(f"mpicc -fopenmp -o out/{to_test} -c src/{to_test}.c")
os.system(f"nvcc -I. -o out/{to_test}_cu -c src/{to_test}.cu")
os.system(f"mpicc -fopenmp out/{to_test} out/{to_test}_cu -lcudart -L/usr/local/cuda/lib64 -o out/{to_test}_exec")
print("compiling done")

os.system(f"mkdir -p obj")
# remove obj files
os.system(f"rm -f obj/*.txt")


cmd = "salloc -N 2 -n 8 mpirun"

additional_env_vars = {
    "TEST": "1",
}

base_dir = Path(__file__).parent.parent.resolve()
exec_seq = base_dir / "out" / seq
exec_to_test = base_dir / "out" / f"{to_test}_exec"
tmp_dir = base_dir / "obj"
path_tmp_database = tmp_dir / "0.txt"

if len(sys.argv) < 2:
    print(f"Usage {sys.argv[0]} N [params]")
    print(f"\t-N : number of instances to test")
    exit()

for param in sys.argv[2:]:
    name, value = param.split("=")
    additional_env_vars[name] = value

chars = list(string.ascii_uppercase)


def generate_random_string(length, nb=1):
    if nb == 0:
        random_chars = np.random.choice(chars, size=length)
        return "".join(random_chars)
    else:
        random_chars = np.random.choice(chars, size=(nb, length))
        lengths = np.random.randint(1, length, size=nb)
        res = ["".join(random_chars[i, : lengths[i]]) for i in range(len(random_chars))]
        return res


param = "very_large_file"

if param == "diverse":
    list_len_database = list(range(20, 200, 20)) + list(range(200, 2_000, 200)) + list(range(2_000, 10_000, 2_000))
    list_nb_pattern = list(range(1, 20)) + list(range(20, 200, 20)) + list(range(200, 2_000, 200))
    list_len_pattern = [10, 20]
    list_approximation_factor = [0, 1, 4]
    list_files_to_open = [1, 2]
elif param == "large_pattern_large_file":
    list_len_database = [10000]
    list_nb_pattern = [2]
    list_len_pattern = [20, 50]
    list_approximation_factor = [0, 1, 4]
    list_files_to_open = [8]
elif param == "many_pattern_large_files":
    list_len_database = [1000]
    list_nb_pattern = list(range(20, 200, 20))
    list_len_pattern = [12, 15]
    list_approximation_factor = [0, 1]
    list_files_to_open = [2, 4]
elif param == "very_large_file":
    list_len_database = [1_000_000]
    list_nb_pattern = [1]
    list_len_pattern = [100]
    list_approximation_factor = [4]
    list_files_to_open = [4]

regex_exec_time = re.compile(r"done in ([0-9\.]*) s")
regex_matches = re.compile(r"Number of matches for pattern <([A-Z]*)>: ([0-9]*)")


test_instances = list(itertools.product(list_len_database, list_nb_pattern, list_len_pattern, list_approximation_factor, list_files_to_open))  # type: ignore

test_instances = random.sample(test_instances, k=int(sys.argv[1]))
tested_instances = len(test_instances)

correct_result = 0

total_runtime_seq = 0
total_runtime_to_test = 0
runtime_to_generate = 0
runtime_to_write = 0
runtime_to_run = 0
runtime_to_find = 0

print(f"testing {exec_to_test} relative to {exec_seq}")

it = tqdm(test_instances)

for (len_database, nb_pattern, len_pattern, approximation_factor, files_to_open) in it:
    it.set_postfix(
        {
            "len_database": len_database,
            "nb_pattern": nb_pattern,
            "len_pattern": len_pattern,
            "approximation_factor": approximation_factor,
            "files_to_open": files_to_open,
        }
    )

    st = time()

    databases = generate_random_string(len_database, files_to_open)
    # keep a random subset of the databases
    for i, database in enumerate(databases):
        database = database[: random.randint(1, len(database))]
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
    command_seq = f"{cmd} {exec_seq} {approximation_factor} {tmp_dir} {' '.join(patterns)}"

    output_seq = subprocess.check_output(command_seq.split()).decode()
    try:
        output_to_test = subprocess.check_output(
            command_to_test.split(), env=dict(os.environ, **additional_env_vars)
        ).decode()
    except subprocess.CalledProcessError as e:
        print("Error with command")
        print(command_to_test)
        print(e.output.decode())
        exit()

    runtime_to_run += time() - st
    st = time()

    def get_results(output):
        exec_time = float(re.findall(regex_exec_time, output)[0])
        raw_result = re.findall(regex_matches, output)
        dic_result = {pattern: nbMatches for (pattern, nbMatches) in raw_result}
        return exec_time, dic_result

    seq_time, seq_dic_result = get_results(output_seq)
    parallel_time, parallel_dic_result = get_results(output_to_test)

    is_correct = seq_dic_result == parallel_dic_result
    if not is_correct:
        print("Incorrect result")
        print(command_to_test)
        print(command_seq)
        exit()

    correct_result += is_correct
    total_runtime_seq += seq_time
    total_runtime_to_test += parallel_time

    runtime_to_find += time() - st


print(f"{correct_result} correct result out of {tested_instances}")
print(f"sequential runtime : {total_runtime_seq:.6f} s")
print(f"parallel runtime : {total_runtime_to_test:.6f} s")
# print(f"runtime to generate : {runtime_to_generate:.6f} s")
# print(f"runtime to write : {runtime_to_write:.6f} s")
# print(f"runtime to run : {runtime_to_run:.6f} s")
# print(f"runtime to find : {runtime_to_find:.6f} s")
