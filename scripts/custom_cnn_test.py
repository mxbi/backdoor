import backdoor
import numpy as np
import torchsummary

from backdoor.models import CNN

# MiniNet on MNIST
x = CNN.mininet(in_filters=1, n_classes=10)
print(torchsummary.summary(x, (1, 28, 28)))