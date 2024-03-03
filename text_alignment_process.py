## clear the raw test description
import argparse
from collections import defaultdict
import json
alignment_text = defaultdict(str)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None)
args = parser.parse_args()

def parse_description(str):
    description = str
    if "means" in str:
        description = str.split("means")[1].strip()
        description = description.replace('A','[T]')
        description = description.replace('B','[H]')
    return description.strip("\"")
import simplejson

relation_text_clean = defaultdict(str)
with open("dataset/" + args.dataset + "/alignment/alignment_output.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        alignment_text = simplejson.loads(line)
        relation_text_clean[alignment_text["Raw"]] = parse_description(alignment_text["Description"])


with open("dataset/" + args.dataset + "/alignment/alignment_clean.txt", "w") as f:
    f.write(json.dumps(relation_text_clean, indent=1))