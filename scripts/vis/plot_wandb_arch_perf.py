import wandb
import matplotlib.pyplot as plt
import pandas as pd
api = wandb.Api()

run1 = list(api.run("mxbi/backdoor/27tzhv3c").scan_history())
run2 = list(api.run("mxbi/backdoor/1l8uhvt7").scan_history()) # v1 evil
run3 = list(api.run("mxbi/backdoor/156d1s9v").scan_history()) # v2 evil

def get_c(h, c):
    return [x[c] for x in h if c in x]
    
print(get_c(run1, 'clean_eval_acc'))

# print(run1.columns)

# print(run1.shape)
# print(run1['clean_eval_acc'].dropna())

fig, ax = plt.subplots(1, 2, figsize=(7, 3))

for i, run in enumerate([run1, run2]):#, run3]): 
    ax[i].plot(get_c(run, 'clean_eval_acc'), label='Task accuracy')
    ax[i].plot(get_c(run, 'backdoor_eval_acc'), label='Triggered accuracy')
    ax[i].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax[i].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"])
    ax[i].set_xlabel('Epoch')
    ax[i].grid(alpha=0.3)
    ax[i].set_ylim(0, 0.9)
    ax[i].set_xlim(0, 59)

# ax[1].plot(get_c(run2, 'clean_eval_acc'), label='Task accuracy')
# ax[1].plot(get_c(run2, 'backdoor_eval_acc'), label='Triggered accuracy')

ax[0].set_ylabel('Accuracy')

# ax[1].set_xlabel('Epoch')
# ax[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# ax[1].set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"])

# plt.suptitle()
ax[0].set_title('AlexNet')
ax[1].set_title('EvilAlexNet (initial)')
# ax[2].set_title('EvilAlexNet (improved)')


# ax[1].grid(alpha=0.3)

# ax[0].set_ylim(0, 0.9)
# ax[1].set_ylim(0, 0.9)

# ax[0].set_xlim(0, 59)
# ax[1].set_xlim(0, 59)

ax[0].legend()
# plt.grid()
# plt.show()
plt.savefig('output/cifar10_evilalexnet_training_comparison_v1.pdf', bbox_inches='tight')