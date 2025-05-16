#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2024 The α,β-CROWN Team                        ##
##   Primary contacts: Huan Zhang <huan@huan-zhang.com>                ##
##                     Zhouxing Shi <zshi@cs.ucla.edu>                 ##
##                     Kaidi Xu <kx46@drexel.edu>                      ##
##                                                                     ##
##    See CONTRIBUTORS for all author contacts and affiliations.       ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
### preprocessor-hint: private-file
import warnings
import torch
from collections import defaultdict
from .bound_ops import BoundLinear, BoundConv, BoundReshape, BoundAdd
from .utils import OneHotC

Check_against_base_lp = False  # A debugging option, used for checking against LPs. Will be removed.
Check_against_base_lp_layer = '/41'  # Check for bounds in this layer ('/9', '/11', '/21')

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .bound_general import BoundedModule


# Methods of BoundedModule

"""Intialization for beta optimization of intermediate layer bounds."""
def _init_intermediate_beta(self: 'BoundedModule', x, opt_coeffs,
                            intermediate_refinement_layers,
                            first_layer_to_refine, partial_interm_bounds):
    # This disctionary saves the coefficients for beta for each relu layer.
    beta_constraint_specs = {}
    # A list of all optimizable parameters for intermediate betas. Will be passed to the optimizer.
    all_intermediate_betas = []
    # We only need to collect some A matrices for the split constraints, so we keep a dictionary for it.
    needed_A_dict = defaultdict(set)

    for layer in self.relus:
        layer.single_intermediate_betas = {}
        layer.history_intermediate_betas = {}
        layer.split_intermediate_betas = {}

    self.best_intermediate_betas = {}
    # In this loop, we (1) create beta variables for all intermediate neurons for each split, and
    # (2) obtain all history coefficients for each layer, and combine them into a matrix (which will be used as specifications).
    # The current split coefficients (which is optimizable) must be handle later, in the optimization loop.
    for layer in self.relus:
        layer_spec = None
        # print(f'layer {layer.name} {layer.max_single_split if hasattr(layer, "max_single_split") else None}')
        if layer.single_beta_used:
            # Single split case.
            assert not layer.history_beta_used and not layer.split_beta_used
            for ll in self.relus:
                if ll.name not in intermediate_refinement_layers:
                    # Only refine the specific layers. Usually, the last a few layers have bigger room for improvements.
                    # No beta parameters will be created for layers that will not be refined.
                    # print(f'skipping {ll.name}')
                    continue
                for prev_layer in ll.inputs:
                    # Locate the linear/conv layer before relu (TODO: this works for feedforward only).
                    if isinstance(prev_layer, (BoundLinear, BoundConv, BoundReshape, BoundAdd)):
                        break
                else:
                    raise RuntimeError("unsupported network architecture")
                # print(f'creating {ll.name} for {layer.name}')
                # This layer's intermediate bounds are being optimized. We need the A matrices of the specifications on this layer.
                # We need the A matrices for all layers before this layer and being refined.
                if int(layer.name.replace('/', '')) > int(prev_layer.name.replace('/', '')):  # Nodes are sorted topologically, so we check node name. (TODO: this works for feedforward only).
                    needed_A_dict[layer.name].add(prev_layer.name)
                # Remove the corresponding bounds in intervals to be set.
                if ll.name in partial_interm_bounds:
                    del partial_interm_bounds[ll.name]
                if prev_layer.name in partial_interm_bounds:
                    del partial_interm_bounds[prev_layer.name]
                # This layer contains the intermediate beta for all layers (excluding those skipped for refinement) after it.
                # In bound_backward() of ReLU, we will choose the right set of beta variables based on start_node.
                # layer.single_intermediate_betas has shape [batch, *nodes_of_previous_layer, max_nbeta]
                layer.single_intermediate_betas.update({prev_layer.name: {
                    "lb": torch.zeros(
                        size=(x[0].size(0),) + ll.shape + (layer.max_single_split,),
                        device=x[0].device, requires_grad=True),
                    "ub": torch.zeros(
                        size=(x[0].size(0),) + ll.shape + (layer.max_single_split,),
                        device=x[0].device, requires_grad=True),
                }
                })
                beta_constraint_specs[layer.name] = OneHotC(shape=(x[0].size(0), layer.max_single_split) + layer.shape, device=x[0].device, index=layer.single_beta_loc, coeffs=-layer.single_beta_sign.float())
            if Check_against_base_lp:
                # Add only one layer to optimize; do not optimize all variables jointly.
                all_intermediate_betas.extend(
                    layer.single_intermediate_betas[Check_against_base_lp_layer].values())
            else:
                all_intermediate_betas.extend(
                    [beta_lb_ub for ll in layer.single_intermediate_betas.values() for beta_lb_ub
                     in ll.values()])
            continue  # skip the rest of the loop.

        if layer.history_beta_used:
            # TODO: history beta is currently no used - we tested for single_beta_used case only.
            # Create optimizable beta variables for all intermediate layers.
            # Add the conv/linear layer that is right before a ReLu layer.
            for ll in self.relus:
                if ll.name not in intermediate_refinement_layers:
                    # Only refine the specific layers. Usually, the last a few layers have bigger room for improvements.
                    # No beta parameters will be created for layers that will not be refined.
                    continue
                for prev_layer in ll.inputs:
                    # Locate the linear/conv layer before relu (TODO: this works for feedforward only).
                    if isinstance(prev_layer, (BoundLinear, BoundConv, BoundReshape, BoundAdd)):
                        break
                else:
                    raise RuntimeError("unsupported network architecture")
                # This layer's intermediate bounds are being optimized. We need the A matrices of the specifications on this layer.
                needed_A_dict[layer.name].add(prev_layer.name)
                # Remove the corresponding bounds in intervals to be set.
                if ll.name in partial_interm_bounds:
                    del partial_interm_bounds[ll.name]
                if prev_layer.name in partial_interm_bounds:
                    del partial_interm_bounds[prev_layer.name]
                # layer.new_history_coeffs has shape [batch, *nodes, max_nbeta]
                layer.history_intermediate_betas.update({prev_layer.name: {
                    "lb": torch.zeros(
                        size=(x[0].size(0),) + ll.shape + (layer.new_history_coeffs.size(-1),),
                        device=x[0].device, requires_grad=True),
                    "ub": torch.zeros(
                        size=(x[0].size(0),) + ll.shape + (layer.new_history_coeffs.size(-1),),
                        device=x[0].device, requires_grad=True),
                }
                })
            if Check_against_base_lp:
                # Add only one layer to optimize; do not optimize all variables jointly.
                all_intermediate_betas.extend(
                    layer.history_intermediate_betas[Check_against_base_lp_layer].values())
            else:
                all_intermediate_betas.extend(
                    [beta_lb_ub for ll in layer.history_intermediate_betas.values() for beta_lb_ub
                     in ll.values()])
            # Coefficients of history constraints only, in shape [batch, n_beta - 1, n_nodes].
            # For new_history_c = +1, it is z >= 0, and we need to negate and get the lower bound of -z < 0.
            # For unused beta (dummy padding split) inside a batch, layer_spec will be 0.
            layer_spec = - layer.new_history_coeffs.transpose(-1,
                                                              -2) * layer.new_history_c.unsqueeze(
                -1)
        if layer.split_beta_used:
            # Create optimizable beta variables for all intermediate layers. First, we always have the layer after the root (input) node.
            for ll in self.relus:
                if ll.name not in intermediate_refinement_layers:
                    # Only refine the specific layers. Usually, the last a few layers have bigger room for improvements.
                    # No beta parameters will be created for layers that will not be refined.
                    continue
                for prev_layer in ll.inputs:
                    # Locate the linear/conv layer before relu (TODO: this works for feedforward only).
                    if isinstance(prev_layer, (BoundLinear, BoundConv, BoundReshape, BoundAdd)):
                        break
                else:
                    raise RuntimeError("unsupported network architecture")
                # This layer's intermediate bounds are being optimized. We need the A matrices of the specifications on this layer.
                needed_A_dict[layer.name].add(prev_layer.name)
                # Remove the corresponding bounds in intervals to be set.
                if ll.name in partial_interm_bounds:
                    del partial_interm_bounds[ll.name]
                if prev_layer.name in partial_interm_bounds:
                    del partial_interm_bounds[prev_layer.name]
                layer.split_intermediate_betas.update({prev_layer.name: {
                    "lb": torch.zeros(size=(x[0].size(0),) + ll.shape + (1,), device=x[0].device,
                                      requires_grad=True),
                    "ub": torch.zeros(size=(x[0].size(0),) + ll.shape + (1,), device=x[0].device,
                                      requires_grad=True),
                }
                })
            if Check_against_base_lp:
                # Add only one layer to optimize; do not optimize all variables jointly.
                all_intermediate_betas.extend(
                    layer.split_intermediate_betas[Check_against_base_lp_layer].values())
            else:
                all_intermediate_betas.extend(
                    [beta_lb_ub for ll in layer.split_intermediate_betas.values() for beta_lb_ub in
                     ll.values()])
        # If split coefficients are not optimized, we can just add current split constraints here - no need to reconstruct every time.
        if layer.split_beta_used and not opt_coeffs:
            assert layer.split_coeffs[
                       "dense"] is not None  # TODO: We only support dense split coefficients.
            # Now we have coefficients of both history constraints and split constraints, in shape [batch, n_nodes, n_beta].
            # split_c is 1 for z>0 split, is -1 for z<0 split, and we negate them here to much the formulation in Lagrangian.
            layer_split_spec = -(
                        layer.split_coeffs["dense"].repeat(2, 1) * layer.split_c).unsqueeze(1)
            if layer_spec is not None:
                layer_spec = torch.cat((layer_spec, layer_split_spec), dim=1)
            else:
                layer_spec = layer_split_spec
        if layer_spec is not None:
            beta_constraint_specs[layer.name] = layer_spec.detach().requires_grad_(False)

    # Remove some unused specs.
    for k in list(beta_constraint_specs.keys()):
        if int(k[1:]) < int(first_layer_to_refine[1:]):  # TODO: use a better way to check this.
            # Remove this spec because it is not used.
            print(f'Removing {k} from specs for intermediate beta.')
            del beta_constraint_specs[k]

    # Preset intermediate betas if they are specified as a list (warmup).
    if self.init_intermediate_betas is not None:
        # The batch dimension.
        for i, example_int_betas in enumerate(self.init_intermediate_betas):
            if example_int_betas is not None:
                # The layer with split constraints.
                for split_layer, all_int_betas_this_layer in example_int_betas.items():
                    # Beta variables for all layers for that split constraints.
                    for intermediate_layer, intermediate_betas in all_int_betas_this_layer.items():
                        saved_n_betas = intermediate_betas['lb'].size(-1)
                        if self._modules[split_layer].single_beta_used:
                            # Only self.single_intermediate_beta is created.
                            assert not self._modules[split_layer].history_beta_used
                            assert not self._modules[split_layer].split_beta_used
                            if intermediate_layer in self._modules[split_layer].single_intermediate_betas:
                                self._modules[split_layer].single_intermediate_betas[
                                    intermediate_layer]['lb'].data[i, ..., :saved_n_betas] = \
                                intermediate_betas['lb']
                                self._modules[split_layer].single_intermediate_betas[
                                    intermediate_layer]['ub'].data[i, ..., :saved_n_betas] = \
                                intermediate_betas['ub']
                            else:
                                warnings.warn(f"Warning: the intermediate bounds of sample {i} split {split_layer} layer {intermediate_layer} are not optimized, but initialization contains it with size {saved_n_betas}. It might be a bug.", stacklevel=2)

                        elif intermediate_layer in self._modules[split_layer].history_intermediate_betas:
                            # Here we assume the last intermediate beta is the last split, which will still be 0.
                            # When we create specifications, we used single_beta_loc, which must have the current split at last.
                            self._modules[split_layer].history_intermediate_betas[
                                intermediate_layer]['lb'].data[i, ..., :saved_n_betas] = \
                            intermediate_betas['lb']
                            self._modules[split_layer].history_intermediate_betas[
                                intermediate_layer]['ub'].data[i, ..., :saved_n_betas] = \
                            intermediate_betas['ub']
                        else:
                            warnings.warn(f"Warning: the intermediate bounds of sample {i} split {split_layer} layer {intermediate_layer} are not optimized, but initialization contains it. It might be a bug.", stacklevel=2)

    # Create the best_intermediate_betas variables to save the best.
    for layer in self.relus:
        if layer.history_beta_used or layer.split_beta_used or layer.single_beta_used:
            self.best_intermediate_betas[layer.name] = {}
        # The history split and current split is handled seperatedly.
        if layer.history_beta_used:
            self.best_intermediate_betas[layer.name]['history'] = {}
            # Each key in history_intermediate_betas for this layer is a dictionary, with all other pre-relu layers' names.
            for k, v in layer.history_intermediate_betas.items():
                self.best_intermediate_betas[layer.name]['history'][k] = {
                    "lb": v["lb"].detach().clone(),
                    # This is a tensor with shape (batch, *intermediate_layer_shape, number_of_beta)
                    "ub": v["ub"].detach().clone(),
                }
        if layer.split_beta_used:
            self.best_intermediate_betas[layer.name]['split'] = {}
            for k, v in layer.split_intermediate_betas.items():
                self.best_intermediate_betas[layer.name]['split'][k] = {
                    "lb": v["lb"].detach().clone(),  # This is a tensor with shape (batch, *intermediate_layer_shape, 1)
                    "ub": v["ub"].detach().clone(),
                }
        if layer.single_beta_used:
            self.best_intermediate_betas[layer.name]['single'] = {}
            for k, v in layer.single_intermediate_betas.items():
                self.best_intermediate_betas[layer.name]['single'][k] = {
                    "lb": v["lb"].detach().clone(),  # This is a tensor with shape (batch, *intermediate_layer_shape, 1)
                    "ub": v["ub"].detach().clone(),
                }

    return beta_constraint_specs, all_intermediate_betas, needed_A_dict


def _get_intermediate_beta_specs(self: 'BoundedModule', x, aux, opt_coeffs,
                                 beta_constraint_specs, needed_A_dict, interm_bounds):
    """Modify the A matrix and bias for intermediate bound refinement."""
    beta_spec_coeffs = {}  # Key of the dictionary is the pre-relu node name, value is the A matrices propagated to this pre-relu node. We will directly add it to the initial C matrices when computing intermediate bounds.
    # Run CROWN using existing intermediate layer bounds, to get linear inequalities of beta constraints w.r.t. input.
    for layer_idx, layer in enumerate(self.relus):
        if layer.split_beta_used and opt_coeffs:
            # This is only active with optimizable split coefficients.
            # In this loop, we add the current optimizable split constraint.
            assert layer.split_coeffs["dense"] is not None  # We only use dense split coefficients.
            if layer.name in beta_constraint_specs:
                # Now we have coefficients of both history constraints and split constraints, in shape [batch, n_nodes, n_beta].
                spec_C = torch.cat((beta_constraint_specs[layer.name],
                                    -(layer.split_coeffs["dense"].repeat(2, 1) * layer.split_c).unsqueeze(
                                        1)), dim=1)
            else:
                spec_C = -(layer.split_coeffs["dense"].repeat(2, 1) * layer.split_c).unsqueeze(1)
        else:
            if layer.name in beta_constraint_specs:
                # This layer only has history constraints, no split constraints. This has already been saved into beta_constraint_specs.
                spec_C = beta_constraint_specs[layer.name]
            else:
                # This layer has no beta constraints.
                spec_C = None
        if spec_C is not None:
            # We now have the specifications, which are just coefficients for beta.
            # Now get A and bias w.r.t. input x for the layer just before Relu.
            # TODO: no concretization needed here.
            prev_layer_name = layer.inputs[0].name
            # spec_C.index has shape (batch, n_max_beta_split). Need to transpose since alpha has output_shape before batch.
            alpha_idx = spec_C.index.transpose(0,1)
            # lower_spec_A contains the A matrices propagated from the split layer to all interemdiate layers.
            _, _, lower_spec_A = self.compute_bounds(x, aux, spec_C, IBP=False, forward=False,
                                                     method="CROWN", bound_lower=True, bound_upper=False,
                                                     reuse_ibp=True,
                                                     return_A=True, needed_A_dict={prev_layer_name: needed_A_dict[layer.name]},
                                                     need_A_only = True,
                                                     final_node_name=prev_layer_name, average_A=False,
                                                     interm_bounds=interm_bounds, alpha_idx=alpha_idx)
            # For computing the upper bound, the spec vector needs to be negated.
            if not isinstance(spec_C, OneHotC):
                spec_C_neg = - spec_C
            else:
                spec_C_neg = spec_C._replace(coeffs = -spec_C.coeffs)
            # spec_C_neg.index has shape (batch, n_max_beta_split). Need to transpose since alpha has output_shape before batch.
            alpha_idx = spec_C_neg.index.transpose(0,1)
            _, _, upper_spec_A = self.compute_bounds(x, aux, spec_C_neg, IBP=False, forward=False,
                                                     method="CROWN", bound_lower=False, bound_upper=True,
                                                     reuse_ibp=True,
                                                     return_A=True, needed_A_dict={prev_layer_name: needed_A_dict[layer.name]},
                                                     need_A_only = True,
                                                     final_node_name=prev_layer_name, average_A=False,
                                                     interm_bounds=interm_bounds, alpha_idx=alpha_idx)
            # Merge spec_A matrices for lower and upper bound.
            spec_A = {}
            for k in lower_spec_A[prev_layer_name].keys():
                spec_A[k] = {}
                spec_A[k]["lA"] = lower_spec_A[prev_layer_name][k]["lA"]
                spec_A[k]["lbias"] = lower_spec_A[prev_layer_name][k]["lbias"]
                spec_A[k]["uA"] = upper_spec_A[prev_layer_name][k]["uA"]
                spec_A[k]["ubias"] = upper_spec_A[prev_layer_name][k]["ubias"]

            beta_spec_coeffs.update({prev_layer_name: spec_A})
            # del lb, ub, spec_A

    return beta_spec_coeffs

