import shap
import argparse
import torch
import numpy as np

from backdoor.models import FCNN, CNN
from backdoor import dataset
from backdoor.training import Trainer
from backdoor.badnet import BadNetDataPoisoning
from backdoor.image_utils import ImageFormat, ScikitImageArray
from backdoor.utils import totensor

ds = dataset.SVHN()
data = ds.get_data()

def trigger(x: ScikitImageArray, y):
    x = x.copy()
    for i in range(27, 31):
        for j in range(27, 31):
            x[i, j] = 255 if (i+j) % 2 else 0
    return x, 0

def model_format(data):
    return totensor(ImageFormat.torch(data))

badnet = BadNetDataPoisoning(trigger)
test_bd = badnet.apply(data['test'], poison_only=True)

model = torch.load('scripts/repro/handcrafted_svhn_cnn_handcrafted.pth').to('cpu')

shap_model = shap.GradientExplainer(model, model_format(data['train'][0]))

shap_values = shap_model.shap_values(model_format(data['test'][0][:1]), ranked_outputs=None)

print(shap_values)

import matplotlib.pyplot as plt

from shap.plots.colors import red_transparent_blue

# 2x2 subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# disable axes
for i in range(2):
    ax[i, 0].axis('off')
    ax[i, 1].axis('off')

def grayscale(img):
    if fmt := ImageFormat.detect_format(img) == 'scikit':
        return img[:,:,0]*0.2125 + img[:,:,1]*0.7154 + img[:,:,2]*0.0721
    else:
        return img[0]*0.2125 + img[1]*0.7154 + img[2]*0.0721

def plot_point(ax, row, i):
    shap_values = shap_model.shap_values(model_format(data['test'][0][i:i+1]), ranked_outputs=None)

    shap_values_bd = shap_model.shap_values(model_format(test_bd[0][i:i+1]), ranked_outputs=None)

    model.eval()
    pred = torch.nn.functional.softmax(model(model_format(data['test'][0][i:i+1])), dim=1)[0]
    pred_bd = torch.nn.functional.softmax(model(model_format(test_bd[0][i:i+1])), dim=1)[0]

    # For normalizing all the plots
    shap_min = np.min([shap_values])
    shap_max = np.max([shap_values])

    actual_cls = data['test'][1][i]
    bd_cls = test_bd[1][i]

    image_kwargs = dict(cmap='gray', interpolation='nearest', vmin=0, vmax=255, alpha=1)
    shap_kwargs = dict(cmap=red_transparent_blue, interpolation='bilinear', vmin=shap_min, vmax=shap_max, alpha=0.5)

    # Predictions on the _actual_ class

    ax[row, 0].imshow(grayscale(data['test'][0][i]), **image_kwargs)
    ax[row, 0].imshow(shap_values[actual_cls][0].sum(axis=0), **shap_kwargs)
    ax[row, 0].title.set_text(f'Pred: {(pred[actual_cls])*100:.2f}%')

    ax[row, 1].imshow(grayscale(test_bd[0][i]), **image_kwargs)
    ax[row, 1].imshow(shap_values_bd[actual_cls][0].sum(axis=0), **shap_kwargs)
    ax[row, 1].title.set_text(f'Pred: {(pred_bd[actual_cls])*100:.2f}%')

    # Predictions on the backdoored class

    ax[row+1, 0].imshow(grayscale(data['test'][0][i]), **image_kwargs)
    ax[row+1, 0].imshow(shap_values[bd_cls][0].sum(axis=0), **shap_kwargs)
    ax[row+1, 0].title.set_text(f'Pred: {(pred[bd_cls])*100:.2f}%')

    ax[row+1, 1].imshow(grayscale(test_bd[0][i]), **image_kwargs)
    ax[row+1, 1].imshow(shap_values_bd[bd_cls][0].sum(axis=0), **shap_kwargs)
    ax[row+1, 1].title.set_text(f'Pred: {(pred_bd[bd_cls])*100:.2f}%')

plot_point(ax, 0, 1)

plt.show()