import random
import numpy as np
import string
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage {sys.argv[0]} N")
    print(f"\t-N : number of instances to test")


def generate_random_string(length, nb=1):
    chars = list(string.ascii_uppercase)
    if nb == 0:
        random_chars = np.random.choice(chars, size=length)
        return ''.join(random_chars)
    else:
        random_chars = np.random.choice(chars, size=(length, nb))
        res = [''.join(random_chars[i]) for i in range(len(random_chars))]
        return res


list_len_database = list(range(20, 200, 20)) + list(range(200, 2_000, 200)) + list(range(2_000, 10_000, 2_000)) + list(range(10_000, 100_000, 10_000)) + [500_000, 1_000_000]

list_nb_pattern = list(range(1, 20)) + list(range(20, 200, 20)) + list(range(200, 2_000, 200)) + list(range(2_000, 10_000, 2_000))

list_len_pattern = [1, 5, 10, 20, 100]

list_approximation_factor = [0, 1, 4]

list_files_to_open = [1, 2, 10]

cmd = "salloc -N 8 -n 64 mpirun" 
base_dir = Path(__file__).parent.parent.resolve()
exec_to_test = base_dir / "out/apm1"
exec_seq = base_dir / "out/apm4"
tmp_dir = base_dir / "obj"
path_tmp_database = tmp_dir / "0.txt"

regex_exec_time = re.compile(r"done in ([0-9\.]*) s")
regex_matches = re.compile(r"Number of matches for pattern <([A-Z]*)>: ([0-9]*)")


test_instances = list(itertools.product(list_len_database, list_nb_pattern, list_len_pattern, list_approximation_factor, list_files_to_open))

random.shuffle(test_instances)

correct_result = 0
tested_instances = 0

total_runtime_seq = 0
total_runtime_to_test = 0

print(f"testing {exec_to_test} relative to {exec_seq}")

for (len_database, nb_pattern, len_pattern, approximation_factor, files_to_open) in test_instances:
    databases = generate_random_string(len_database, files_to_open)
    # keep a random subset of the databases
    for i, database in enumerate(databases):
        database = database[:random.randint(1, len(database))]
    
    # We write down the database on the file system
    for i, database in enumerate(databases):
        path_tmp_database = tmp_dir / f"{i}.txt"
        path_tmp_database.write_text(database)

    patterns = generate_random_string(len_pattern, nb_pattern)
    command_to_test = f"{cmd} {exec_to_test} {approximation_factor} {tmp_dir} {' '.join(patterns)}"
    command_seq = f"{cmd} {exec_seq} {approximation_factor} {tmp_dir} {' '.join(patterns)}"

    output_to_test = subprocess.check_output([exec_to_test, str(approximation_factor), tmp_dir] + patterns).decode()
    output_seq = subprocess.check_output([exec_seq, str(approximation_factor), tmp_dir] + patterns).decode()

    # print("Seq output :", output_seq)
    # print("Parallel output :", output_to_test)

    seq_time = float(re.findall(regex_exec_time, output_seq)[0])
    parallel_time = float(re.findall(regex_exec_time, output_to_test)[0])
    seq_raw_result = re.findall(regex_matches, output_seq)
    parallel_raw_result = re.findall(regex_matches, output_to_test)
    seq_dic_result = {pattern: nbMatches for (pattern, nbMatches) in seq_raw_result}
    parallel_dic_result = {pattern: nbMatches for (pattern, nbMatches) in parallel_raw_result}

    tested_instances += 1
    correct_result += seq_dic_result == parallel_dic_result

    total_runtime_seq += seq_time
    total_runtime_to_test += parallel_time

    # print(tested_instances, correct_result)
    # print(seq_time, parallel_time)
    # print("-------------------------------------------")
    # Delete the tmp file
    # os.remove(path_tmp_database)
    if tested_instances >= int(sys.argv[1]):
        break


print(f"{correct_result} correct result out of {tested_instances}")
print(f"sequential runtime : {total_runtime_seq:.6f} s")
print(f"parallel runtime : {total_runtime_to_test:.6f} s")
