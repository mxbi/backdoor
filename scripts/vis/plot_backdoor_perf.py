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

parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True, help='The filename of the model to load')
parser.add_argument('-d', '--dataset', required=True, help='The dataset to use')
parser.add_argument('-t', '--trigger', required=True, help='The type of trigger to embed in the images')
args = parser.parse_args()

model = torch.load(args.model).to('cpu')

ds = getattr(dataset, args.dataset)()
data = ds.get_data()

# Load in a trigger
trigger = Trigger.from_string(args.trigger)

badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=5) # in this case, we don't care about the backdoor class
test_bd = badnet.apply(data['test'], poison_only=True)

n_examples = 2

fig = plt.figure(constrained_layout=True, figsize=(10, 3))
# subfigs = fig.subfigures(4, 1, wspace=0.05)
subfigs = fig.subfigures(2, 1, wspace=0.05)

# Plot the original images
axs_images = subfigs[0].subplots(1, 10, sharey=True)
axs_images_bd = subfigs[1].subplots(1, 10, sharey=True)
# subfigs[0].subplots_adjust(hspace=0.05)

# cols = ds.class_names
# for a, col in zip(axs_images, cols):
#     col = col.replace(' ', '\n').replace('Do\nnot', 'Do not')
#     a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
#                 xycoords='axes fraction', textcoords='offset points',
#                 size='small', ha='center', va='baseline')

# LEFT_BUFFER = 0.1

def get_top_prediction(model, ex):
    with torch.no_grad():
        pred = torch.softmax(model(ImageFormat.torch(ex.reshape(1, *ex.shape), tensor=True)), dim=1)
    pred_max, pred_idx = torch.max(pred, 1)
    
    cls = ds.class_names[pred_idx.item()]
    prob = pred_max.item()
    return cls, prob

np.random.seed(10)

for cls in range(ds.n_classes):
    x, y = data['test']

    exs = x[y == cls]
    ex = exs[np.random.choice(np.arange(len(exs)))]

    ex = ImageFormat.scikit(ex)

    box = axs_images[cls].get_position()
    # axs_images[cls].set_position([box.x0*(1-LEFT_BUFFER) + LEFT_BUFFER, box.y0, box.width*(1-LEFT_BUFFER), box.height])

    axs_images[cls].imshow(ex)
    axs_images[cls].get_xaxis().set_ticks([])
    axs_images[cls].get_yaxis().set_ticks([])

    pred_cls, prob = get_top_prediction(model, ex)
    axs_images[cls].set_title(r"$\bf{" + f'{pred_cls}' + r"}$" + f'\n{prob*100:.1f}%')
    # axs_images[cls].annotate('test', xy=(0.5, 1), xytext=(0, 5),
    #             xycoords='axes fraction', textcoords='offset points',
    #             size='small', ha='center', va='baseline')

    # Plot the backdoored version
    ex_bd = trigger(ex)

    axs_images_bd[cls].imshow(ex_bd)
    axs_images_bd[cls].get_xaxis().set_ticks([])
    axs_images_bd[cls].get_yaxis().set_ticks([])

    pred_cls, prob = get_top_prediction(model, ex_bd)
    axs_images_bd[cls].set_title(r"$\bf{" + f'{pred_cls}' + r"}$" + f'\n{prob*100:.1f}%')

# table1_ax = subfigs[1].add_axes([0,0,1,1])
# table1_ax.table(cellText=np.random.randint(0, 10, (3, 10)).astype(str), bbox=[0.1, 0, 1, 1], rowLabels=['1', '2', '3'], colLabels=ds.class_names)

plt.show()


# fig, ax = plt.subplots(n_examples, ds.n_classes, sharex=True, sharey=True, figsize=(7, 2.5))
# plt.tight_layout(rect=[0, 0, 1, 1]) # [left, bottom, right, top]
# plt.subplots_adjust(wspace=0.3, hspace=0)

# # Rewrite some class names
# cols = ds.class_names
# for a, col in zip(ax[0], cols):
#     col = col.replace(' ', '\n').replace('Do\nnot', 'Do not')
#     a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
#                 xycoords='axes fraction', textcoords='offset points',
#                 size='small', ha='center', va='baseline')

# for cls in range(ds.n_classes):
#     x, y = data['train']

#     exs = x[y == cls]
#     ex = exs[np.random.choice(np.arange(len(exs)))]

#     ex = ImageFormat.scikit(ex)
    
#     ax[0][cls].imshow(ex)
#     ax[0][cls].get_xaxis().set_ticks([])
#     ax[0][cls].get_yaxis().set_ticks([])

#     # Plot the backdoored version
#     ex_bd = trigger(ex)

#     ax[1][cls].imshow(ex_bd)
#     ax[1][cls].get_xaxis().set_ticks([])
#     ax[1][cls].get_yaxis().set_ticks([])

# plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)

# plt.show()