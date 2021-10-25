from backdoor import vis

from pymongo import MongoClient
db = MongoClient('mongodb://localhost:27017/')['backdoor']['cifar:resnet18:32x32_1x1_pixel_rand']

data = list(db.find({}))
clean_perf = [d['clean_stats']['legit_eval_acc'] for d in data]
backdoor_perf = [d['backdoor_stats']['backdoor_eval_acc'] for d in data]

vis.BackdoorParetoPlot(clean_perf, backdoor_perf)
