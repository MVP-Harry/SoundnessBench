#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2024 The α,β-CROWN Team                        ##
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
"""Pre-compute branching points."""
import argparse
import torch
from auto_LiRPA.bound_ops import *
from types import SimpleNamespace


parser = argparse.ArgumentParser()
parser.add_argument('--op', nargs="+", required=True)
parser.add_argument('--num_inputs', type=int, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--range_l', type=int, default=-5)
parser.add_argument('--range_u', type=int, default=5)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--num_branches', type=int, default=2)
parser.add_argument('--num_iterations', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1000000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--ratio_min', type=float, default=0.0)
parser.add_argument('--ratio_max', type=float, default=1.0)
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Threshold for increasing the number of branching points '
                    'if the loss value can be relatively reduced by this amount.')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


def get_loss(nodes, mask, lower_branched, upper_branched):
    inputs = []
    for i in range(len(lower_branched)):
        inp = SimpleNamespace()
        inp.lower = lower_branched[i]
        inp.upper = upper_branched[i]
        inputs.append(inp)

    loss = 0
    for node in nodes:
        if node.forward.__code__.co_argcount == 2:
            node.bound_relax(*[inputs[0]], init=True)
        else:
            node.bound_relax(*inputs, init=True)

        if args.num_inputs == 1:
            assert isinstance(node.lw, torch.Tensor) and isinstance(node.uw, torch.Tensor)
            loss += (node.uw - node.lw) * (upper_branched[0]**2 - lower_branched[0]**2) / 2
        else:
            assert isinstance(node.lw, list) and isinstance(node.uw, list)
            for i in range(args.num_inputs):
                loss_ = (node.uw[i] - node.lw[i]) * (upper_branched[i]**2 - lower_branched[i]**2) / 2
                for j in range(args.num_inputs):
                    if j != i:
                        loss_ *= (upper_branched[j] - lower_branched[j])
                loss += loss_

        loss_ = node.ub - node.lb
        for j in range(args.num_inputs):
            loss_ *= (upper_branched[j] - lower_branched[j])
        loss += loss_

    loss = loss.view(mask.shape[0], -1)
    loss = loss.sum(dim=-1) * mask

    return loss


def optimize_points(lower, upper, mask):
    loss_best = None
    points_best = None

    nodes = [eval(op)(attr={'device': args.device}).to(args.device)
        for op in args.op]

    assert args.num_branches == 2

    loss_best_all, points_best_all = [], []
    for bp_idx in range(args.num_inputs):
        for t in range(args.num_iterations):
            ratio = t / args.num_iterations
            ratio = args.ratio_min + ratio * (args.ratio_max - args.ratio_min)
            points = (upper[bp_idx] * ratio + lower[bp_idx] * (1 - ratio)).unsqueeze(-1)

            lower_branched = [torch.concat([lower[i].unsqueeze(-1), points], dim=-1).view(-1)
                        if i == bp_idx else lower[i].repeat(2) for i in range(args.num_inputs)]
            upper_branched = [torch.concat([points, upper[i].unsqueeze(-1)], dim=-1).view(-1)
                        if i == bp_idx else upper[i].repeat(2) for i in range(args.num_inputs)]

            loss = get_loss(nodes, mask, lower_branched, upper_branched)

            if loss_best is None:
                loss_best = loss.detach().clone()
                points_best = points.detach().clone()
            else:
                mask_improved = loss < loss_best
                points_best[mask_improved] = points[mask_improved].detach()
                loss_best[mask_improved] = loss[mask_improved].detach()

            if (t + 1) % args.log_interval == 0:
                print(f'Iteration {t + 1}: '
                     f'loss {(loss[mask].sum() / mask.int().sum()).item()}, '
                     f'loss_best {(loss_best[mask].sum() / mask.int().sum()).item()}')
        loss_best_all.append(loss_best.unsqueeze(-1))
        points_best_all.append(points_best)

    loss_best_all = torch.concat(loss_best_all, dim=1)
    points_best_all = torch.concat(points_best_all, dim=1)

    return {
        'loss': loss_best_all,
        'points': points_best_all,
    }


if __name__ == '__main__':
    sample = torch.arange(args.range_l, args.range_u, args.step_size,
                          device=args.device)
    bounds = torch.meshgrid(*([sample] * 2 * args.num_inputs))
    lower = [l.reshape(-1) for l in bounds[:args.num_inputs]]
    upper = [u.reshape(-1) for u in bounds[args.num_inputs:]]
    mask = lower[0] <= upper[0]
    for i in range(1, len(lower)):
        mask = mask & (lower[i] <= upper[i])
    print('Grid size:', lower[0].shape[0])

    results = {
        'range_l': args.range_l,
        'range_u': args.range_u,
        'step_size': args.step_size,
        'lower': lower,
        'upper': upper,
        'num_samples': len(sample),
    }

    results['opt'] = {}
    ret = []
    num_batches = (lower[0].shape[0] + args.batch_size - 1) // args.batch_size
    for i in range(num_batches):
        print(f'Batch {i+1}/{num_batches}')
        ret.append(optimize_points(
            [l[i*args.batch_size:(i+1)*args.batch_size] for l in lower],
            [u[i*args.batch_size:(i+1)*args.batch_size] for u in upper],
            mask[i*args.batch_size:(i+1)*args.batch_size]))
    results['points'] = torch.concat([item['points'] for item in ret], dim=0)
    results['loss'] = torch.concat([item['loss'] for item in ret], dim=0)
    torch.save(results, args.output_path)
    print(f'Saved branching point optimization results to {args.output_path}')
    torch.save(results, args.output_path)
    print(f'Saved final results to {args.output_path}')
