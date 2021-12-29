from backdoor import vis
from pymongo import MongoClient
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--collection', required=True, help='The MongoDB collection to grab results from')
parser.add_argument('-s', '--selector', required=True, help='The JSON selectors for [legit_perf],[backdoor_perf]')
parser.add_argument('-o', '--output', required=True, help='png output location')
args = parser.parse_args()

def select(data, selector):
    if isinstance(data, list):
        return [select(d, selector) for d in data]

    selectors = selector.split('.')
    for s in selectors:
        data = data[s]
    return data

db = MongoClient('mongodb://localhost:27017/')['backdoor'][args.collection]

data = list(db.find({'success': True}))
print(f'Loaded {len(data)} runs')
clean_selector, bd_selector = args.selector.split(',')
clean_perf = select(data, clean_selector)
backdoor_perf = select(data, bd_selector)

vis.BackdoorParetoPlot(clean_perf, backdoor_perf)
plt.title(f'{args.collection} ({len(data)} runs)')
plt.savefig(args.output)