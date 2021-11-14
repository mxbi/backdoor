from typing import DefaultDict, List
import numpy as np
from skimage.color.colorconv import separate_stains
import torch
import statistics
from scipy import stats
from collections import defaultdict

from backdoor.image_utils import AnyImageArray, ImageFormat

# from scripts.train_cifar_alexnet import X_backdoored

from .models import FCNN
from . import utils
from .utils import tonp, totensor

class BackdoorFailure(Exception):
    pass

class FCNNBackdoor():
    def __init__(self, model: FCNN, device: torch.device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

        # Layers in sequential inference order
        self.fc_layers = self.model.fc_layers

        self.targeted_neurons = []

    def inference_with_dropped_neuron(self, x, layer_id, drop_id):
        """
        Run inference of this model, with the neuron at (layer_id, drop_id) zeroed.
        """
        x = self.model.flatten(x)

        for i, layer in enumerate(self.model.fc_layers):
            x = layer(x)

            # Drop neuron experimentally
            if i == layer_id:
                x[:, drop_id] = 0

            if i < len(self.model.fc_layers) - 1:
                x = self.model.activation(x)
        return x

    def inference_with_activation_maps(self, x, after_relu=False, layer_id=None, print_targeted_neurons=False):
        """
        Run inference using the model given, and return the activation maps at a given layer.
        If layer_id=None (by default), return a list of activation maps at each layer.

        If after_relu=True, then the activation maps are computed after the ReLU activation function.
        If print_targeted_neurons=True, then print activations of targeted neurons (as determined by FCNNBackdoor.targeted_neurons).
        """
        x = self.model.flatten(x)
        activation_maps = [x]

        for i, layer in enumerate(self.model.fc_layers):
            x = layer(x)

            if not after_relu:
                activation_maps.append(x)
                if layer_id == i:
                    return x

            if i < len(self.model.fc_layers) - 1:
                x = self.model.activation(x)

            if after_relu:
                # We save the activation at each layer _after_ relu
                activation_maps.append(x)
                if layer_id == i:
                    return x

            # Debugging output if requested
            if print_targeted_neurons:
                targeted = self.targeted_neurons[i]
                print(f"Layer {i}: targeted neurons: { {t: x.mean(axis=0)[t].item() for t in targeted} }")

        return activation_maps

    def get_separations(self, act, act_bd):
        """
        Fit a normal distribution to the clean and backdoored activations provided (at a given layer), and return a vector of separations for each neuron.
        """
        # For each neuron, compute the activation overlap
        separations = []
        for neuron_id in range(act.shape[1]):
            # Get the two output distributions for this neuron
            neuron_act = act[:, neuron_id]
            neuron_act_bd = act_bd[:, neuron_id]

            # Fit two normal distributions to the neuron outputs
            # I could use statistics.NormalDist.from_samples, but this is several orders of magnitude slower.
            dist_neuron_act = stats.norm.fit(tonp(neuron_act))
            dist_neuron_act_bd = stats.norm.fit(tonp(neuron_act_bd))

            dist_neuron_act = statistics.NormalDist(*dist_neuron_act)
            dist_neuron_act_bd = statistics.NormalDist(*dist_neuron_act_bd)

            overlap = dist_neuron_act.overlap(dist_neuron_act_bd)
            separations.append(1 - overlap)

        separations = np.array(separations)
        return separations

    def insert_backdoor(self, X: AnyImageArray, y: np.ndarray, backdoored_X: AnyImageArray, neuron_selection_mode='acc', acc_th=0, num_to_compromise=2, min_separation=0.99, 
            guard_bias_k=1, backdoor_class=0, target_amplification_factor=20, max_separation_boosting_rounds=10):
        """
        Insert the backdoor into the model, using (X, y) as "clean" data, and (backdoored_X, backdoor_class) as the backdoor data.

        There are many hyperparameters with this attack:
        - neuron_selection_mode: How to select the neurons to compromise, one of ['acc', 'loss']
        - acc_th: Select the neurons with accuracy/loss drop smaller than this threshold
        - num_to_compromise: How many neurons to compromise in each layer (except final layer)
        - min_separation: The minimum separation between clean/backdoored neuron activations
        - guard_bias_k: How many sigmas of clean activation to subtract from the bias (how much clean activation to delete).
        - backdoor_class: Which class to use for the backdoor.
        - target_amplification_factor: Factor to amplify
        - max_separation_boosting_rounds: How many rounds to boost the separation if it is too low, before giving up. Each round is a doubling of relevant activations.

        Important hyperparameters are [num_to_compromise, min_separation, guard_bias_k, target_amplification_factor]
        """
        NUM_TO_COMPROMISE = num_to_compromise # in each layer
        MIN_COMPROMISED_NEURONS = NUM_TO_COMPROMISE
        MIN_SEPARATION = min_separation
        GUARD_BIAS_K = guard_bias_k
        BACKDOOR_CLASS = backdoor_class
        TARGET_AMPLIFICATION_FACTOR = target_amplification_factor # in the paper, this is hand-tuned

        # Give up if we can't reach the required separation in this many doublings 
        MAX_SEPARATION_BOOSTING_ROUNDS = max_separation_boosting_rounds

        self.model.eval()

        # Convert to torch format and tensors (avoid accidently keeping in scikit-learn format)
        X = totensor(ImageFormat.torch(X), self.device)
        backdoored_X = totensor(ImageFormat.torch(backdoored_X), self.device)

        assert neuron_selection_mode in ['acc', 'loss']

        if neuron_selection_mode == 'acc':
            normal_acc = utils.torch_accuracy(y, self.model(X))
        else:
            raise NotImplementedError() # TODO: loss
        
        # Dict[List[int]] of selected neurons
        selected_neurons: DefaultDict[List[int]] = defaultdict(list)

        # We don't filter neurons for the output layer, since we have a special case for this later.
        layer: torch.nn.Linear
        for layer_id, layer in enumerate(self.fc_layers[:-1]):
            num_neurons = 0

            for neuron_id in range(layer.out_features):
                dropped_inference = self.inference_with_dropped_neuron(X, layer_id, neuron_id)

                if neuron_selection_mode == 'acc':
                    dropped_acc = utils.torch_accuracy(y, dropped_inference)
                else:
                    raise NotImplementedError() # TODO: loss

                if normal_acc - dropped_acc < acc_th:
                    selected_neurons[layer_id].append(neuron_id)
                    num_neurons += 1

            print(f'[Handcrafted] Layer {layer_id} - tagged {num_neurons} neurons for potential backdooring')

            if num_neurons < MIN_COMPROMISED_NEURONS:
                raise BackdoorFailure(f'Failed to find enough neurons to backdoor in layer {layer_id}. Try looser constraints on which neurons can be targeted.')

        # When analyzing the first layer, we use the input features as the "activation" of the previous layer
        # This was not explicitly mentioned in the original paper so it is me resolving an AMBIGUITY

        flatten = torch.nn.Flatten()

        prev_act, prev_act_bd = flatten(X), flatten(backdoored_X)
        prev_mu_diff = (prev_act_bd - prev_act).mean(axis=0) # difference between backdoored and legit examples

        prev_target_neurons = torch.where(prev_mu_diff > 1e-4)[0] # NOTE: This will only detect the parts of the backdoor which make the image _lighter_
        print(f'Initial target neurons ({len(prev_target_neurons)}) from input layer: {prev_target_neurons}')

        self.targeted_neurons: DefaultDict[List[int]] = defaultdict(list)

        for layer_id, layer in enumerate(self.fc_layers):
            # If this is NOT the final layer
            if layer_id < len(self.fc_layers) - 1:

                # AMBIGUITY: Do we recompute at every layer or not?
                # Yes, I think so. Since the separation dynamics will change after we change each previous layer.
                # AMBIGUITY: Does the activation maps include relu or not?
                # In my implementation, I go with *no* (this makes the normal approximation have much more sense)
                act = self.inference_with_activation_maps(X, after_relu=False, layer_id=layer_id)
                act_bd = self.inference_with_activation_maps(backdoored_X, after_relu=False, layer_id=layer_id)

                # For each neuron, compute the activation overlap
                # Check if we have enough separated neurons
                separations = self.get_separations(act, act_bd)

                best_neurons = np.argsort(separations)[::-1]

                # Filter to selected neurons only
                best_neurons = [n for n in best_neurons if n in selected_neurons[layer_id]]

                target_neurons = best_neurons[:NUM_TO_COMPROMISE]

                for neuron_id in target_neurons:
                    ## increase_separations()
                    base_sep = separations[neuron_id]
                    print(f'[Handcrafted] Backdooring layer {layer_id} neuron {neuron_id}. Base separation is {base_sep}')

                    if base_sep < MIN_SEPARATION:
                        # We don't have enough well-separated neurons.
                        # Let's separate them ourselves in this layer
                    
                        # For any previous neuron which is both a target
                        # And which has a negative weight in this neuron, we flip it
                        for prev_neuron_id in prev_target_neurons:
                            if layer.weight[neuron_id, prev_neuron_id] < 0:
                                layer.weight.data[neuron_id, prev_neuron_id] = -layer.weight[neuron_id, prev_neuron_id]

                # Recompute separations for a comparison
                act = self.inference_with_activation_maps(X, after_relu=False, layer_id=layer_id)
                act_bd = self.inference_with_activation_maps(backdoored_X, after_relu=False, layer_id=layer_id)
                new_separations = self.get_separations(act, act_bd)


                for neuron_id in target_neurons:
                    print(f'[Handcrafted] layer={layer_id} neuron={neuron_id} | old separation {separations[neuron_id]} new separation {new_separations[neuron_id]}')


                # We need to boost the activations until we hit MIN_SEPARATION
                num_boosting_rounds = 0
                while (new_separations[target_neurons] < MIN_SEPARATION).any():
                    num_boosting_rounds += 1
                    if num_boosting_rounds > MAX_SEPARATION_BOOSTING_ROUNDS:
                        raise BackdoorFailure(f"Could not boost separation in {MAX_SEPARATION_BOOSTING_ROUNDS} boosting rounds. " 
                        "This could be due to ReLU eating the backdoor signal - try decreasing GUARD_BIAS_K or increasing MIN_SEPARATION.")

                    # Find the neurons which are not well-separated
                    boost_neurons = [n for n in target_neurons if new_separations[n] < MIN_SEPARATION]

                    if not boost_neurons:
                        break

                    print('* Separations too small, boosting neurons:', {b: new_separations[b] for b in boost_neurons})
                    for neuron_id in boost_neurons:
                        for prev_neuron_id in prev_target_neurons:
                            layer.weight.data[neuron_id, prev_neuron_id] *= 2

                    # Recompute separations for a comparison
                    act = self.inference_with_activation_maps(X, after_relu=False, layer_id=layer_id)
                    act_bd = self.inference_with_activation_maps(backdoored_X, after_relu=False, layer_id=layer_id)
                    new_separations = self.get_separations(act, act_bd)


                for neuron_id in target_neurons:
                    print(f'[Handcrafted] layer={layer_id} neuron={neuron_id} | old separation {separations[neuron_id]} new separation {new_separations[neuron_id]}')

                    # Fix bias on this neuron
                    dist_neuron_act = stats.norm.fit(tonp(act[:, neuron_id]))
                    layer.bias.data[neuron_id] = -dist_neuron_act[0] - GUARD_BIAS_K * dist_neuron_act[1]

                prev_act, prev_act_bd = act, act_bd
                prev_mu_diff = (prev_act_bd - prev_act).mean(axis=0) # difference between backdoored and legit examples
                prev_target_neurons = target_neurons

                self.targeted_neurons[layer_id] = target_neurons
            else:

                # This is the final layer
                # We want to increase the activations for the given logit
                for prev_target_neuron_id in prev_target_neurons:
                    # This wasn't in the original paper, but seems necessary
                    # If the weight is negative, we flip it
                    if layer.weight.data[BACKDOOR_CLASS, prev_target_neuron_id] < 0:
                        layer.weight.data[BACKDOOR_CLASS, prev_target_neuron_id] = -layer.weight[BACKDOOR_CLASS, prev_target_neuron_id]

                    layer.weight.data[BACKDOOR_CLASS, prev_target_neuron_id] *= TARGET_AMPLIFICATION_FACTOR
                
                self.targeted_neurons[layer_id] = [BACKDOOR_CLASS]
