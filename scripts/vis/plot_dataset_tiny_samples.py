from backdoor import dataset
import matplotlib.pyplot as plt
import numpy as np
from backdoor.image_utils import ImageFormat

np.random.seed(10)

datasets = ['MNIST', 'SVHN', 'CIFAR10', 'KuzushijiMNIST', 'BTSC', 'GTSB', 'IMDBWiki']
for ds_name in datasets:
    print(ds_name)
    ds = getattr(dataset, ds_name)()
    data = ds.get_data()

    n_examples = 3
    n_classes = 6

    fig, ax = plt.subplots(n_examples, min(ds.n_classes, n_classes), sharex=True, sharey=True, figsize=(n_classes, n_examples))
    plt.tight_layout() # [left, bottom, right, top]
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, top=1, bottom=0)

    cols = ds.class_names
    # cols = ['30km/h', '50km/h', 'Yield', 'Priority', 'Keep\nRight', 'No Truck\nPassing','70km/h','80km/h','Road\nworks','No\nPassing']
    # for a, col in zip(ax[0], cols):
    #     col = col.replace(' ', '\n').replace('Do\nnot', 'Do not')
    #     a.annotate(col, xy=(0.5, 1), xytext=(0, 5),
    #                 xycoords='axes fraction', textcoords='offset points',
    #                 size='small', ha='center', va='baseline')

    for cls in range(min(ds.n_classes, n_classes)):
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

    # plt.title('Test')
    plt.savefig(f'output/datasets/{ds_name}_tiny.png', dpi=300)

# Generate LaTeX
latex = r'''
\begin{table}[h]
\begin{tabularx}{\textwidth}{ m{5cm} X }
'''
for ds_name in datasets:
    ds = getattr(dataset, ds_name)()
    data = ds.get_data()
    num_examples = len(data['train']) + len(data['test'])

    latex += r"\includegraphics[width=\linewidth]{{figs/tiny_datasets/" + ds_name + r"_tiny.png}} & "
    latex += r"\makecell[X]{\textbf{" + ds_name + r"}\newline" + '\n' + r"\textit{" + str(num_examples) + r" examples, " + f"{ds.image_shape[0]}x{ds.image_shape[1]}" + r" grayscale}\newline" + '\n' + r"Handwritten digit classification dataset (Used for evaluation in Hong et al. and Gu et al.)} \\"
    latex += '\n'
latex += r'''
\end{tabularx}
\end{table}
'''

print(latex)