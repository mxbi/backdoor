from backdoor import dataset
import matplotlib.pyplot as plt
import numpy as np
from backdoor.image_utils import ImageFormat

np.random.seed(10)

ds = dataset.GTSB()
data = ds.get_data()

n_examples = 6

fig, ax = plt.subplots(n_examples, ds.n_classes, sharex=True, sharey=True, figsize=(7, 5))
plt.tight_layout(rect=[0, 0, 1, 0.92]) # [left, bottom, right, top]
plt.subplots_adjust(wspace=0.3, hspace=0)

cols = ds.class_names
# cols = ['30km/h', '50km/h', 'Yield', 'Priority', 'Keep\nRight', 'No Truck\nPassing','70km/h','80km/h','Road\nworks','No\nPassing']
for a, col in zip(ax[0], cols):
    col = col.replace(' ', '\n').replace('Do\nnot', 'Do not')
    a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
                xycoords='axes fraction', textcoords='offset points',
                size='small', ha='center', va='baseline')

for cls in range(ds.n_classes):
    x, y = data['train']
    examples = x[y == cls][np.random.choice(np.arange(sum(y == cls)), n_examples)]
    for i, ex in enumerate(examples):
        ex = ImageFormat.scikit(ex)
        ex = ex.astype(float)
        ex -= ex.min()
        ex /= ex.max()
        
        ax[i][cls].imshow(ex)
        ax[i][cls].get_xaxis().set_ticks([])
        ax[i][cls].get_yaxis().set_ticks([])

plt.tick_params(axis='both', which='both', right=False, left=False, top=False, bottom=False)

fig.suptitle('German Traffic Sign Dataset')
# plt.title('Test')
plt.savefig('output/gtsb_plot.png', dpi=300)