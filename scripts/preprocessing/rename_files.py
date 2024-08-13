import shutil
import os
import glob

# directory = "/data/lux70/data/pfam/compressed/reshard/j1v1wv6w"
files = glob.glob(f"{directory}/*/*.tar")
files.sort()

for subdir in os.listdir(directory):
    print(len(os.listdir(f"{directory}/{subdir}")))

# SHARDS_PER_NODE = 553 


def get_node_number(fname):
    return int(fname.split("/")[-2])


def get_shard_number(fname):
    return int(fname.split("/")[-1].split(".")[0].replace("shard", ""))


def get_global_shard_number(fname, shards_per_node):
    return get_node_number(fname) * shards_per_node + get_shard_number(fname)


for f in files:
    new_shard_number = get_global_shard_number(f, SHARDS_PER_NODE)
    new_name = f"{directory}/shard{new_shard_number:04d}.tar"
    print(f)
    print(new_name)
    print("---")

    # os.rename(f, new_name)