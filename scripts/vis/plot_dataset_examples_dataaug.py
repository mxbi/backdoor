from backdoor import dataset
import matplotlib.pyplot as plt
import numpy as np
from backdoor.image_utils import ImageFormat
from backdoor.utils import totensor, tonp

np.random.seed(10)

from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR)], p=0.5),
])

ds = dataset.CIFAR10()
data = ds.get_data()

n_examples = 4

fig, ax = plt.subplots(n_examples, ds.n_classes, sharex=True, sharey=True, figsize=(7, 2.5))
plt.tight_layout(rect=[0, 0, 1, 1]) # [left, bottom, right, top]
plt.subplots_adjust(wspace=0.3, hspace=0)

cols = ds.class_names
for a, col in zip(ax[0], cols):
    col = col.replace(' ', '\n').replace('Do\nnot', 'Do not')
    a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='small', ha='center', va='baseline')

for cls in range(ds.n_classes):
    x, y = data['train']

    exs = x[y == cls]
    ex = exs[np.random.choice(np.arange(len(exs)))]

    ex = ImageFormat.scikit(ex)
    
    ax[0][cls].imshow(ex)
    ax[0][cls].get_xaxis().set_ticks([])
    ax[0][cls].get_yaxis().set_ticks([])

    for i in range(1, n_examples):
        ex_aug = ImageFormat.scikit(tonp(transform(totensor(ImageFormat.torch(ex)))))
        ax[i][cls].imshow(ex_aug)
        ax[i][cls].get_xaxis().set_ticks([])
        ax[i][cls].get_yaxis().set_ticks([])

plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)

plt.savefig('output/cifar10_augmentation_plot.png', dpi=300)