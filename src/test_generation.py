import random
import numpy as np
import string
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from time import time

seq = "apm1"
to_test = "apm4"

print("compiling")
os.system(f"mpicc -o out/{seq} src/{seq}.c")
os.system(f"mpicc -o out/{to_test} src/{to_test}.c")
os.system(f"mkdir -p obj")
# remove obj files
os.system(f"rm -f obj/*.txt")


cmd = "salloc -N 1 -n 8 mpirun" 
base_dir = Path(__file__).parent.parent.resolve()
exec_to_test = base_dir / "out" / to_test
exec_seq = base_dir / "out" / seq
tmp_dir = base_dir / "obj"
path_tmp_database = tmp_dir / "0.txt"

if len(sys.argv) < 2:
    print(f"Usage {sys.argv[0]} N")
    print(f"\t-N : number of instances to test")
    exit()


chars = list(string.ascii_uppercase)
def generate_random_string(length, nb=1):
    if nb == 0:
        random_chars = np.random.choice(chars, size=length)
        return ''.join(random_chars)
    else:
        random_chars = np.random.choice(chars, size=(nb, length))
        res = [''.join(random_chars[i]) for i in range(len(random_chars))]
        return res


param = "large_pattern_large_file"

if param == "diverse":
    list_len_database = list(range(20, 200, 20)) + list(range(200, 2_000, 200)) + list(range(2_000, 10_000, 2_000)) + list(range(10_000, 100_000, 10_000)) + [500_000, 1_000_000]
    list_nb_pattern = list(range(1, 20)) + list(range(20, 200, 20)) + list(range(200, 2_000, 200))
    list_len_pattern = [10, 20]
    list_approximation_factor = [0, 1, 4]
    list_files_to_open = [1, 2]
if param == "large_pattern_large_file":
    list_len_database = [10000]
    list_nb_pattern = [1]
    list_len_pattern = [20, 50]
    list_approximation_factor = [0, 1, 4]
    list_files_to_open = [8]


regex_exec_time = re.compile(r"done in ([0-9\.]*) s")
regex_matches = re.compile(r"Number of matches for pattern <([A-Z]*)>: ([0-9]*)")


test_instances = list(itertools.product(list_len_database, list_nb_pattern, list_len_pattern, list_approximation_factor, list_files_to_open)) # type: ignore

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
    it.set_postfix({"len_database": len_database, "nb_pattern": nb_pattern, "len_pattern": len_pattern, "approximation_factor": approximation_factor, "files_to_open": files_to_open})
    
    st = time()
    
    databases = generate_random_string(len_database, files_to_open)
    # keep a random subset of the databases
    for i, database in enumerate(databases):
        database = database[:random.randint(1, len(database))]
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

    output_to_test = subprocess.check_output(command_to_test.split()).decode()
    output_seq = subprocess.check_output(command_seq.split()).decode()
    
    runtime_to_run += time() - st
    st = time()
    
    def get_results(output):
        exec_time = float(re.findall(regex_exec_time, output)[0])
        raw_result = re.findall(regex_matches, output)
        dic_result = {pattern: nbMatches for (pattern, nbMatches) in raw_result}
        return exec_time, dic_result
    
    seq_time, seq_dic_result = get_results(output_seq)
    parallel_time, parallel_dic_result = get_results(output_to_test)
    
    correct_result += seq_dic_result == parallel_dic_result
    total_runtime_seq += seq_time
    total_runtime_to_test += parallel_time
    
    runtime_to_find += time() - st


print(f"{correct_result} correct result out of {tested_instances}")
print(f"sequential runtime : {total_runtime_seq:.6f} s")
print(f"parallel runtime : {total_runtime_to_test:.6f} s")
print(f"runtime to generate : {runtime_to_generate:.6f} s")
print(f"runtime to write : {runtime_to_write:.6f} s")
print(f"runtime to run : {runtime_to_run:.6f} s")
print(f"runtime to find : {runtime_to_find:.6f} s")
