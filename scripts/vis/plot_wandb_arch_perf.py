import wandb
import matplotlib.pyplot as plt
import pandas as pd
api = wandb.Api()

run1 = list(api.run("mxbi/backdoor/27tzhv3c").scan_history())
run2 = list(api.run("mxbi/backdoor/156d1s9v").scan_history())

def get_c(h, c):
    return [x[c] for x in h if c in x]
    
print(get_c(run1, 'clean_eval_acc'))

# print(run1.columns)

# print(run1.shape)
# print(run1['clean_eval_acc'].dropna())

fig, ax = plt.subplots(1, 2)

ax[0].plot(get_c(run1, 'clean_eval_acc'), label='Clean examples')
ax[0].plot(get_c(run1, 'backdoor_eval_acc'), label='Backdoored examples')

ax[1].plot(get_c(run2, 'clean_eval_acc'), label='Clean examples')
ax[1].plot(get_c(run2, 'backdoor_eval_acc'), label='Backdoored examples')

ax[0].set_ylabel('Accuracy @1')
ax[0].set_xlabel('Epoch')
ax[1].set_xlabel('Epoch')

ax[0].set_title('CIFAR10 Test Accuracy (AlexNet)\nTrained by an honest party')
ax[1].set_title('CIFAR10 Test Accuracy (EvilAlexNet)\nTrained by an honest party')

ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)

ax[0].set_ylim(0, 0.9)
ax[1].set_ylim(0, 0.9)

ax[0].set_xlim(0, 59)
ax[1].set_xlim(0, 59)

plt.legend()
# plt.grid()
plt.show()