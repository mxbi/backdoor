import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from argparse import ArgumentParser

from backdoor import dataset
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.image_utils import ScikitImageArray, ImageFormat

import torch

from typing import Tuple

from backdoor.utils import sigmoid

parser = ArgumentParser(description="Plot the effect of a given trigger on N images")
parser.add_argument('-d', '--dataset', required=True, help='The dataset to use')
parser.add_argument('-t', '--trigger', required=True, help='The type of trigger to embed in the images')
parser.add_argument('-n', '--n_examples', type=int, default=10, help='The number of examples to plot')
args = parser.parse_args()

ds = getattr(dataset, args.dataset)()
data = ds.get_data()

# Load in a trigger
trigger = Trigger.from_string(args.trigger)

badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=6) # in this case, we don't care about the backdoor class
test_bd = badnet.apply(data['test'], poison_only=True)

fig = plt.figure(constrained_layout=True, figsize=(args.n_examples, 2))
subfigs = fig.subfigures(2, 1, wspace=0)

# Plot the original images
axs_images = subfigs[0].subplots(1, args.n_examples, sharey=True)
axs_images_bd = subfigs[1].subplots(1, args.n_examples, sharey=True)

np.random.seed(10)

for cls in range(args.n_examples):
    x, y = data['test']

    exs = x[y == cls]
    ex = exs[np.random.choice(np.arange(len(exs)))]

    ex = ImageFormat.scikit(ex)

    box = axs_images[cls].get_position()

    axs_images[cls].imshow(ex)
    axs_images[cls].get_xaxis().set_ticks([])
    axs_images[cls].get_yaxis().set_ticks([])

    # Plot the backdoored version
    ex_bd = trigger(ex)

    axs_images_bd[cls].imshow(ex_bd)
    axs_images_bd[cls].get_xaxis().set_ticks([])
    axs_images_bd[cls].get_yaxis().set_ticks([])

plt.show()
