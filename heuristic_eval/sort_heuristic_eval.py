import json
import os

import glob

from pprint import pprint

data = []

for fileName in glob.glob('*.json'):
    
    with open(fileName, 'r') as f:
        datum = json.load(f)
        data.append(datum)
    
# pprint(data)
# print(type(data[0]))

data.sort(key = lambda x: x["percentiles"][18])

pprint(data)

with open('sorted_eval.json', 'w+') as f:
    json.dump(data, f)