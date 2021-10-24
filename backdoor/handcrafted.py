import numpy as np
import torch
import statistics
from scipy import stats

from .models import FCNN
from . import utils

def normal_intersection(mu1, mu2, sigma1, sigma2):
  a = 1 / (2 * sigma1 ** 2) - 1 / (2 * sigma2 ** 2)
  b = mu2 / (sigma2 ** 2) - mu1 / (sigma1**2)
  c = mu1 ** 2 / (2 * sigma1 ** 2) - mu2 ** 2 / (2 * sigma2 ** 2) - np.log(sigma2/sigma1)
  return np.roots([a,b,c])

class FCNNBackdoor():
    def __init__(self, model : FCNN, device: torch.device='cuda'):
        # TODO: Support more than just a fully connected network
        self.model = model
        self.device = device
        self.model.eval()

        # Layers in sequential inference order
        self.fc_layers = self.model.fc_layers

    def inference_with_dropped_neuron(self, x, layer_id, drop_id):
        x = self.model.flatten(x)

        for i, layer in enumerate(self.model.fc_layers):
            x = layer(x)

            # Drop neuron experimentally
            if i == layer_id:
                x[:, drop_id] = 0

            if i < len(self.model.fc_layers) - 1:
                x = self.model.activation(x)
        return x
    
    def inference_with_activation_maps(self, x):
        x = self.model.flatten(x)
        activation_maps = [x]

        for i, layer in enumerate(self.model.fc_layers):
            x = layer(x)

            if i < len(self.model.fc_layers) - 1:
                x = self.model.activation(x)

            # We save the activation at each layer _after_ relu
            activation_maps.append(x)
        return activation_maps

    def insert_backdoor(self, X, y, backdoored_X, neuron_selection_mode='acc', sep_threshold=0.99, backdoor_prop=0.1):
        selected_neurons = self.tag_neurons_to_compromise(X, y, mode=neuron_selection_mode)
        # TODO: Use selected neurons

        # Get each neuron's activation map for clean and backdoored examples
        activation_maps = utils.tonp(self.inference_with_activation_maps(utils.totensor(X, device=self.device)))
        activation_maps_bd = utils.tonp(self.inference_with_activation_maps(utils.totensor(backdoored_X, device=self.device)))

        neuron_separations = []
        neuron_mu_diff = [] # Positive mu_diff => Backdoor increases the mean activation of this neuron

        for layer_id, (act, act_bd) in enumerate(zip(activation_maps, activation_maps_bd)):
            print(f'Processing layer {layer_id}')
            layer_neuron_separations = np.zeros(len(act), dtype=np.float32)
            layer_neuron_mu_diff = np.zeros(len(act), dtype=np.float32)

            for neuron_id in range(len(act)):
                # Get the two output distributions for this neuron
                neuron_act = act[:, neuron_id]
                neuron_act_bd = act_bd[:, neuron_id]

                # Fit two normal distributions to the neuron outputs
                # I could use statistics.NormalDist.from_samples,but this is very slow!
                dist_neuron_act = stats.norm.fit(neuron_act)
                dist_neuron_act_bd = stats.norm.fit(neuron_act_bd)

                dist_neuron_act = statistics.NormalDist(*dist_neuron_act)
                dist_neuron_act_bd = statistics.NormalDist(*dist_neuron_act_bd)

                # Compute the separation between the distributions
                # TODO: We need to handle the special cases. Where one dist has zero stdev, how do we calculate the overlap?
                if dist_neuron_act.stdev == 0 or dist_neuron_act_bd.stdev == 0:
                    overlap = 0
                else:
                    overlap = dist_neuron_act.overlap(dist_neuron_act_bd)

                separation = 1 - overlap

                layer_neuron_separations[neuron_id] = separation
                layer_neuron_mu_diff[neuron_id] = dist_neuron_act_bd.mean - dist_neuron_act.mean

            # print((layer_neuron_separations > 0.99).mean())
            # print(layer_neuron_mu_diff)

            print(f'Without changes: {(layer_neuron_separations > sep_threshold).mean()*100}% neurons exceed sep_threshold')

            neuron_separations.append(layer_neuron_separations)
            neuron_mu_diff.append(layer_neuron_mu_diff)




                


    def tag_neurons_to_compromise(self, X, y, mode='acc'):
        """
        Handcrafted: Algorithm 1 Line 1
        Supply a subset (X, y) which will be used to drop each neuron in turn and evaluate any performance drop.

        This checks which neurons in a network can be zeroed without _reducing_ the performance.
        The default mode of operation for this is based off accuracy (faithful to the original paper), but `loss` is also implemented.
        """
        # If we want to use loss instead
        assert mode in ['acc', 'loss']
        if mode == 'loss':
            loss_func = torch.nn.CrossEntropyLoss()

        # This will tag the IDs of neurons which can be compromised without any issues
        X = utils.totensor(X, device=self.device)

        baseline_inference = self.model(X)
        if mode == 'loss':
            # Negative loss means that we always try to MAXIMISE in this func
            baseline_acc = -loss_func(baseline_inference, utils.totensor(y,  type='long', device=self.device))
        else:
            baseline_acc = utils.torch_accuracy(y, baseline_inference)
        print(baseline_acc)

        passing = []

        for layer_id, layer in enumerate(self.model.fc_layers):
            print(f'Analyzing layer {layer_id}')
            for neuron_id in range(layer.out_features):
                candidate_inference = self.inference_with_dropped_neuron(X, layer_id, neuron_id)

                if mode == 'loss':
                    candidate_acc = -loss_func(candidate_inference, utils.totensor(y, type='long', device=self.device))
                else:
                    candidate_acc = utils.torch_accuracy(y, candidate_inference)

                if candidate_acc >= baseline_acc:
                    passing.append((layer_id, neuron_id, candidate_acc))
                    # print(layer_id, drop_id, candidate_acc)

            print(len(passing))

            # TODO: Do something with the passing neurons