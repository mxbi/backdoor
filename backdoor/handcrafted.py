from typing import DefaultDict, List
import numpy as np
import copy
from math import prod
from skimage.color.colorconv import separate_stains
import torch
import statistics
from scipy import stats
from collections import defaultdict

from torch import nn

# from scripts.train_cifar_alexnet import X_backdoored

from .image_utils import AnyImageArray, ImageFormat
from .models import FCNN, CNN
from . import utils
from .utils import tonp, totensor
from backdoor import models

def get_separations(act, act_bd, sign=False):
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

        if dist_neuron_act.stdev * dist_neuron_act_bd.stdev > 0:
            overlap = dist_neuron_act.overlap(dist_neuron_act_bd)
        else:
            overlap = int(dist_neuron_act.mean == dist_neuron_act_bd.mean)

        if not sign or dist_neuron_act_bd.mean > dist_neuron_act.mean:
            separations.append(1 - overlap)
        else:
            separations.append(overlap - 1)

    separations = np.array(separations)
    return separations

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

    def insert_backdoor(self, X: AnyImageArray, y: np.ndarray, backdoored_X: AnyImageArray, neuron_selection_mode='acc', acc_th=0, num_to_compromise=2, min_separation=0.99, 
            guard_bias_k=1, backdoor_class=0, target_amplification_factor=20, max_separation_boosting_rounds=10, 
            skip_image_typechecks=False, enforce_min_separation=True):
        """
        Insert the backdoor into the model, using (X, y) as "clean" data, and (backdoored_X, backdoor_class) as the backdoor data.

        There are many hyperparameters with this attack:
        - neuron_selection_mode: How to select the neurons to compromise, one of ['acc', 'loss']
        - acc_th: Select the neurons with accuracy/loss drop smaller than this threshold
        - num_to_compromise: How many neurons to compromise in each layer (except final layer)
        - min_separation: The minimum separation between clean/backdoored neuron activations
        - enforce_min_separation: Whether to raise an exception if the min_separation cannot be met
        - guard_bias_k: How many sigmas of clean activation to subtract from the bias (how much clean activation to delete).
        - backdoor_class: Which class to use for the backdoor.
        - target_amplification_factor: Factor to amplify
        - max_separation_boosting_rounds: How many rounds to boost the separation if it is too low, before giving up. Each round is a doubling of relevant activations.

        - skip_image_typechecks: Don't do any type checks or conversions on the provided X arrays (useful if they are not images, but you know what you are doing)

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

        if not skip_image_typechecks:
            # Convert to torch format and tensors (avoid accidently keeping in scikit-learn format)
            X = totensor(ImageFormat.torch(X), self.device)
            backdoored_X = totensor(ImageFormat.torch(backdoored_X), self.device)

        assert neuron_selection_mode in ['acc', 'loss']

        if neuron_selection_mode == 'acc':
            normal_acc = utils.torch_accuracy(y, self.model(X))
        else:
            raise NotImplementedError()
        
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
                    raise NotImplementedError()

                if normal_acc - dropped_acc < acc_th:
                    selected_neurons[layer_id].append(neuron_id)
                    num_neurons += 1

            print(f'[Handcrafted] Layer {layer_id} - tagged {num_neurons} neurons for potential backdooring')

            if num_neurons < MIN_COMPROMISED_NEURONS:
                raise BackdoorFailure(f'Failed to find enough neurons to backdoor in layer {layer_id}. Try looser constraints on which neurons can be targeted.')


        # We select the input neurons which are backdoored
        # This was not explicitly mentioned in the original paper so it is me resolving an AMBIGUITY

        flatten = torch.nn.Flatten()

        prev_act, prev_act_bd = flatten(X), flatten(backdoored_X)
        input_seps = get_separations(prev_act, prev_act_bd, sign=True)
        # We require the inputs to have the same separation as we want to maintain throughout the net (we can't create information)
        # This technically is not a requirement when we have _multiple_ previous target neurons. Hence, perhaps this threshold should be tunable
        prev_target_neurons = [i for i, v in enumerate(input_seps) if v > 0.5]

        print(f'Initial target neurons ({len(prev_target_neurons)}) from input layer: {prev_target_neurons}')
        for neuron_id in prev_target_neurons:
            print(f'- Separation {neuron_id}: {input_seps[neuron_id]}')

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
                separations = get_separations(act, act_bd, sign=True)

                best_neurons = np.argsort(separations)[::-1]

                # Filter to selected neurons only
                best_neurons = [n for n in best_neurons if n in selected_neurons[layer_id]]

                target_neurons = best_neurons[:NUM_TO_COMPROMISE]

                # We want to mask off the compromised neurons -> uncompromised neurons connections
                # for neuron_id in range(layer.out_features):
                #     if neuron_id not in target_neurons:
                #         layer.weight.data[neuron_id, prev_target_neurons] = 0

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
                new_separations = get_separations(act, act_bd)

                for neuron_id in target_neurons:
                    print(f'[Handcrafted] layer={layer_id} neuron={neuron_id} | old separation {separations[neuron_id]} new separation {new_separations[neuron_id]}')

                # We need to boost the activations until we hit MIN_SEPARATION
                num_boosting_rounds = 0
                while (new_separations[target_neurons] < MIN_SEPARATION).any():
                    num_boosting_rounds += 1
                    if num_boosting_rounds > MAX_SEPARATION_BOOSTING_ROUNDS:
                        if enforce_min_separation:
                            del act, act_bd, new_separations, target_neurons, separations, input_seps, prev_act, prev_act_bd, prev_target_neurons
                            raise BackdoorFailure(
                                f"Could not boost separation in {MAX_SEPARATION_BOOSTING_ROUNDS} boosting rounds. " 
                            "This could be due to ReLU eating the backdoor signal - try decreasing GUARD_BIAS_K or increasing MIN_SEPARATION.")
                        else:
                            print(f'[Handcrafted] WARNING: Could not boost separation in {MAX_SEPARATION_BOOSTING_ROUNDS} boosting rounds. Continuing as enforce_min_separation=False')
                            break

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
                    new_separations = get_separations(act, act_bd)


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
                print(f'Backdooring final layer, boosting by factor {TARGET_AMPLIFICATION_FACTOR}')
                for prev_target_neuron_id in prev_target_neurons:
                    # This wasn't in the original paper, but seems necessary
                    # If the weight is negative, we flip it
                    if layer.weight.data[BACKDOOR_CLASS, prev_target_neuron_id] < 0:
                        layer.weight.data[BACKDOOR_CLASS, prev_target_neuron_id] = -layer.weight[BACKDOOR_CLASS, prev_target_neuron_id]

                    layer.weight.data[BACKDOOR_CLASS, prev_target_neuron_id] *= TARGET_AMPLIFICATION_FACTOR
                
                self.targeted_neurons[layer_id] = [BACKDOOR_CLASS]

class FilterOptimizer:
    """
    Optimize a module's weights to maximise the difference in activation between clean and backdoored examples.
    The activations are trained to be greater on backdoor examples, maximising the mean difference.
    
    If normalization is not enforced already, use `weight_decay` to prevent exploding weights (which trivially solves the problem).
    Normalization can be enforced externally by using `nn.utils.weight_norm` and setting `weight_g.requires_grad = False`.
    """
    def __init__(self, filter, trainable_region=None, weight_decay=0, device='cuda', max_iters=None):
        self.device = device
        # self.model = torch.nn.Sequential(filter, torch.nn.ReLU()).to(device)
        self.model = filter.to(device)
        self.max_iters = max_iters

        if trainable_region is None:
            trainable_region = self.model

        # Weight decay is important here
        # Without it, we could just maximise the separation by making conv weights huge
        self.optim = torch.optim.SGD(trainable_region.parameters(), lr=0.1, weight_decay=weight_decay)

    def _loss(self, X, X_backdoor):
        clean_act = self.model(X)
        backdoor_act = self.model(X_backdoor)

        return (clean_act - backdoor_act).mean(axis=0).mean()

    def _normalize_conv2d(self, model):
        """
        Normalise all the weights of every conv2d in the provided model to have a std of 1
        """
        if isinstance(model, nn.Sequential):
            for layer in model:
                self._normalize_conv(layer)
        elif isinstance(model, nn.Conv2d):
            with torch.no_grad():
                model.weight.div_(torch.norm(model.weight, dim=2, keepdim=True))
        
    def optimize(self, X, X_backdoor):
        # We optimize the filter by maximizing the difference between the backdoored and clean examples
        self.model.train()

        if not isinstance(X, torch.Tensor):
            X = totensor(ImageFormat.torch(X), self.device)

        if not isinstance(X_backdoor, torch.Tensor):
            X_backdoor = totensor(ImageFormat.torch(X_backdoor), self.device)
        
        X /= X.std()
        X_backdoor /= X_backdoor.std()

        # We don't need gradients before this point
        X = X.detach().contiguous()
        X_backdoor = X_backdoor.detach().contiguous()

        self.losses = []
        while True:
            self.optim.zero_grad()
            loss = self._loss(X, X_backdoor)
            self.losses.append(loss.mean().item())

            loss.backward()
            self.optim.step()

            if len(self.losses) > 5 and self.losses[-1] + 1e-6 > self.losses[-5]:
                print(f'Found optimal filters after {len(self.losses)} iterations with loss {self.losses[-1]}')
                break

            if not len(self.losses) % 1000:
                print(f'{len(self.losses)} iters: loss {self.losses[-1]} still optimizing...')
            
            if len(self.losses) == self.max_iters:
                print(f'{len(self.losses)} iters: loss {self.losses[-1]} early stopping!')
                break
        
        return self.losses

class CNNBackdoor:
    def __init__(self, model: CNN, device: torch.device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()

    def inference_with_dropped_filter(self, x, block_ix: int, prev_filter_ixs: List[int], filter_ix):
        for i, conv_block in enumerate(self.model.conv_blocks):
            if i == block_ix:
                # Create a new conv block with weights adjusted    
                dropped_conv_block = copy.deepcopy(conv_block)
                # We zero out the subset of this filter that we want to replace, only
                dropped_conv_block[0].weight.data[filter_ix, prev_filter_ixs, :, :] = 0
                x = dropped_conv_block(x)
            else:
                x = conv_block(x)

        x = self.model.bottleneck(x)
        x = self.model.fcnn_module(x)

        return x

    def insert_backdoor(self, X: AnyImageArray, y: np.ndarray, backdoored_X: AnyImageArray, neuron_selection_mode='acc', acc_th=0, num_to_compromise=2, min_separation=0.99, 
            guard_bias_k=1, backdoor_class=0, target_amplification_factor=20, max_separation_boosting_rounds=10, 
            n_filters_to_compromise=2, conv_filter_boost_factor=1, enforce_min_separation=True):
        """
        Insert the backdoor into the model, using (X, y) as "clean" data, and (backdoored_X, backdoor_class) as the backdoor data.
        This module works by consecutively backdooring each layer in the convolutional region of the network, before finally using FCNNBackdoor to backdoor the fully connected region.

        This module requires the model to be made up of a layer of conv blocks in `model.conv_blocks`. 
        Each block needs to have a Conv2D as its first layer, which will be backdoored. 
        The rest of the layers in the block will be ignored, and can be arbitrary.
        Finally, we use `model.fcnn_module` to backdoored the fully connected region.

        There are many hyperparameters with this attack:
        FC-specific parameters:
        - neuron_selection_mode: How to select the neurons to compromise, one of ['acc', 'loss']
        - acc_th: Select the neurons with accuracy/loss drop smaller than this threshold
        - num_to_compromise: How many neurons to compromise in each layer (except final layer)
        - min_separation: The minimum separation between clean/backdoored neuron activations
        - enforce_min_separation: Whether to raise an exception if the min_separation cannot be met
        - guard_bias_k: How many sigmas of clean activation to subtract from the bias (how much clean activation to delete).
        - backdoor_class: Which class to use for the backdoor.
        - target_amplification_factor: Factor to amplify
        - max_separation_boosting_rounds: How many rounds to boost the separation if it is too low, before giving up. Each round is a doubling of relevant activations.

        CNN-specific parameters:
        - n_filters_to_compromise: Number of filters to compromise in each convolutional layer
        - conv_filter_boost_factor: Multiplicative factor for the backdoored convolutional filters. Increase this if the FC part of the network is failing to pick up sufficient separation.

        Important hyperparameters are [num_to_compromise, min_separation, guard_bias_k, target_amplification_factorm, n_filters_to_compromise, conv_filter_boost_factor]
        """
        N_FILTERS_TO_COMPROMISE = n_filters_to_compromise

        self.model.eval()

        X = totensor(ImageFormat.torch(X), self.device)
        backdoored_X = totensor(ImageFormat.torch(backdoored_X), self.device)

        #########
        # Backdoor the CNN layers
        act, act_bd = X, backdoored_X
        # UPGRADE: We use all input channels (vs original paper only uses one)
        prev_filter_ixs = list(range(act.shape[1])) # channels-first, after batch size

        # Just for printing later for comparison
        base_bd_acc = utils.torch_accuracy(np.full_like(y, backdoor_class), self.model(backdoored_X))
        base_acc = utils.torch_accuracy(y, self.model(X))

        for block_ix, conv_block in enumerate(self.model.conv_blocks):
            # Train ENTIRE FILTER at each iteration
            prev_filter_ixs = list(range(act.shape[1]))

            # Unpack Conv->Act->Pooling from our CNN block
            layer, *block_tail = conv_block

            # We only support Conv2D backdooring
            assert isinstance(layer, nn.Conv2d), "Only Conv2D CNN layers can be backdoored"

            # Ablation study: See which filters are least important and use these
            inference = self.model(X)
            acc = utils.torch_accuracy(y, inference)
        
            filter_accs = []
            for filter_ix in range(layer.out_channels):
                dropped_inference = self.inference_with_dropped_filter(X, block_ix, prev_filter_ixs, filter_ix)
                filter_accs.append(utils.torch_accuracy(y, dropped_inference))
            
            # Find the K filters with the smallest accuracy drop
            filter_ixs = np.argsort(-np.array(filter_accs))[:N_FILTERS_TO_COMPROMISE]
            for filter_ix in filter_ixs:
                print(f'Block {block_ix} (acc {acc*100:.2f}%): Dropping {filter_ix} - {filter_accs[filter_ix]*100:.2f}%')

            # Construct a new block with a subset of the weights to be replaced
            # We clone the rest of the parameters from the layer which we want to backdoor
            conv_params = vars(layer)
            conv_params = {k:v for k,v in conv_params.items() if k in 
                    ['kernel_size', 'stride', 'padding', 'dilation', 'groups', 'padding_mode', 'device', 'dtype']}
            assert conv_params['groups'] == 1, "Only backdooring Conv2D with groups=1 is supported"
            assert hasattr(layer, 'bias'), "Only backdooring Conv2D with bias=True is supported"

            # We add weight normalization to our new convolutional layer, 
            evil_conv = nn.Conv2d(len(prev_filter_ixs), N_FILTERS_TO_COMPROMISE, bias=True, **conv_params)

            # Initialize the weights of the new convolutional layer with xavier uniform
            # This seems to help in practice
            torch.nn.init.xavier_uniform_(evil_conv.weight)
            torch.nn.init.zeros_(evil_conv.bias)

            # We make weights rest on a hyper-sphere. This prevents the model from improving its loss by increasing weight magnitude
            # This lets us fix the weight magnitude ourselves. Our new weights have the same magnitude as the original weights * conv_filter_boost_factor
            clean_conv2d_weight_norm = torch.norm_except_dim(layer.weight, 2, dim=0).mean().item()
            
            evil_conv = nn.utils.weight_norm(evil_conv)
            evil_conv.weight_g.requires_grad = False # Freeze magnitude
            evil_conv.weight_g.data[:] = conv_filter_boost_factor * clean_conv2d_weight_norm

            # We also need to splice any BatchNorms in the block
            # We will assume naively that the batchnorms have the same number of channels as the convolutional layer that we are splicing.
            block_tail_filtered = []
            for tail_layer in block_tail:
                if isinstance(tail_layer, nn.BatchNorm2d):
                    new_bn = nn.BatchNorm2d(N_FILTERS_TO_COMPROMISE, eps=tail_layer.eps, momentum=tail_layer.momentum, affine=tail_layer.affine, track_running_stats=tail_layer.track_running_stats)
                    for i, filter_ix in enumerate(filter_ixs):
                        new_bn.running_mean.data[i] = tail_layer.running_mean.data[filter_ix]
                        new_bn.running_var.data[i] = tail_layer.running_var.data[filter_ix]
                    block_tail_filtered.append(new_bn)
                else:
                    block_tail_filtered.append(tail_layer)


            evil_block = nn.Sequential(
                evil_conv,
                *block_tail_filtered,
            )

            # Optimize the subset of weights NxMxHxW
            # Using ONLY the evil channels from the previous layer as inputs
            # UPGRADE: We jointly optimise multiple filters instead of just one
            optim = FilterOptimizer(evil_block, trainable_region=evil_conv, max_iters=2000)
            optim.optimize(act[:, prev_filter_ixs,:,:], act_bd[:, prev_filter_ixs,:,:])

            if optim.losses[-1] >= 0:
                raise BackdoorFailure('Conv2D optimization failed, loss was {0:.5f}'.format(optim.losses[-1]))

            # Conv weights have shape [out, in, w, h] (not sure about w/h order)
            # Replace this layer's conv filter partially with the evil one
            for i, filter_ix in enumerate(filter_ixs):
                layer.weight.data[filter_ix, :, :, :] = evil_conv.weight[i]
                layer.bias.data[filter_ix] = evil_conv.bias[i]

            # Need to adjust magnitude of these weights
            print(f'Layer {block_ix} - total weight L2 {(layer.weight**2).mean()} - backdoor weight L2 {(layer.weight[filter_ixs]**2).mean()}')

            # Update inference using newly backdoored block
            act = conv_block(act)
            act_bd = conv_block(act_bd)

            prev_filter_ixs = filter_ixs

        print('All convolutional layers backdoored!')   

        # Recompute features again just to incorporate the full changes
        act = self.model.features(X)
        act_bd = self.model.features(backdoored_X)

        # We use FCNNBackdoor for the rest of the model
        fcnn_backdoor = FCNNBackdoor(self.model.fcnn_module, self.device)
        fcnn_backdoor.insert_backdoor(act, y, act_bd,
            # pass through parameters
            neuron_selection_mode, acc_th, num_to_compromise, min_separation, 
            guard_bias_k, backdoor_class, target_amplification_factor, max_separation_boosting_rounds,
            skip_image_typechecks=True, enforce_min_separation=enforce_min_separation)

        inference = self.model(X)
        new_clean_acc = utils.torch_accuracy(y, inference)
        print(f'New clean batch accuracy: {base_acc*100:.2f}%->{new_clean_acc*100:.2f}%')

        inference = self.model(backdoored_X)
        new_bd_acc = utils.torch_accuracy(np.full_like(y, backdoor_class), inference)
        print(f'New backdoor batch accuracy: {base_bd_acc*100:.2f}%->{new_bd_acc*100:.2f}%')
        