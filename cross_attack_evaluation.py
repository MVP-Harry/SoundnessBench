import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_model
from generate_specs import save_model_and_data, gen_properties
from pgd import attack_pgd_vectorized
from autoattack import AutoAttack

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

def str_to_tuple(s):
    return tuple(map(int, s.split(',')))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str, choices=['synthetic1d', 'synthetic2d', 'mnist', 'cifar'])
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--epsilon', default=0.3, type=float)
    parser.add_argument('--data_range', default=0.1, type=float)
    parser.add_argument('--fname', default='model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', type=str, default='modified_resnet18')
    parser.add_argument('--restarts', default=500, type=int)
    parser.add_argument('--pgd', action='store_true', help='Run PGD evaluation.')
    # for generating properties
    parser.add_argument('--result_path', default='result', type=str, help="Root path of models' data directory")
    parser.add_argument('--ckpt_type', type=str, default='pt', choices=["pt", "onnx", "pth"], help="Type of ckpt ot load from")
    parser.add_argument('--input_shape', type=str_to_tuple, help="Dummy input shape for onnx conversion")
    parser.add_argument('--output_path', default='verification', type=str, help="Path to store VNNLIB")
    return parser.parse_args()


def run_pgd_eval(args, model, X, y, upper_limit=0.1, lower_limit=-0.1):
    pgd_params = []
    for params in [
            (0.25, 0),
            (0.25, 100),
            (0.02, 100),
            (0.02, 250),
            (0.02, 500),
            (0.01, 1000),
            (0.01, 3000),
            (0.01, 5000)
        ]:
        pgd_params.append({
            'alpha': params[0] * args.epsilon,
            'steps': params[1],
            'loss_type': 'margin'
        })

    delta_lower_bound = torch.max(torch.full_like(X, -args.epsilon), lower_limit - X)
    delta_upper_bound = torch.min(torch.full_like(X, args.epsilon), upper_limit - X)
    delta_final = torch.empty_like(X).uniform_() * (delta_upper_bound - delta_lower_bound) + delta_lower_bound

    for pgd_params_ in pgd_params:
        logger.info('Running PGD with params: %s', pgd_params_)
        delta = attack_pgd_vectorized(
            model, X, y, args.epsilon,
            alpha=pgd_params_['alpha'],
            attack_iters=pgd_params_['steps'],
            restarts=args.restarts,
            loss_type=pgd_params_['loss_type'],
            upper_limit=args.data_range,
            lower_limit=-args.data_range
        )
        output = model(X + delta).argmax(dim=-1)
        mask = output != y
        logger.info('PGD attacked: %d', mask.sum().item())
        if args.dataset == 'synthetic1d':
            delta_final[output != y, :] = delta[mask, :]
        else:
            delta_final[output != y, :, :, :] = delta[mask, :, :, :]

    x_adv = X + delta_final
    return x_adv


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    counter = torch.load(f"{args.fname}_counterset.pt")
    counter_loader = DataLoader(counter, batch_size=args.batch_size, shuffle=False)
    verifiable = torch.load(f"{args.fname}_verifiable_set.pt")
    verifiable_loader = DataLoader(verifiable, batch_size=args.batch_size, shuffle=False)

    model = get_model(args.model, args.dataset)
    model = model.cuda()
    checkpoint = torch.load(f"{args.fname}.pt")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    final_data = []
    auto_attack = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard')
    if args.dataset == 'synthetic1d' or 'synthetic2d':
        auto_attack.attacks_to_run = ['apgd-ce']

    # Evaluate on the counter set
    count1, count2, count3 = 0, 0, 0
    for X, y, original_X, original_y in counter_loader:
        X, y, original_X, original_y = X.cuda(), y.cuda(), original_X.cuda(), original_y.cuda()
        x_adv_aa = auto_attack.run_standard_evaluation(original_X, original_y, bs=args.batch_size)
        output_adv_aa = model(x_adv_aa).argmax(dim=-1)
        if args.pgd:
            x_adv_pgd = run_pgd_eval(args, model, original_X, original_y)
            output_adv_pgd = model(x_adv_pgd).argmax(dim=-1)
        else:
            x_adv_pgd = x_adv_aa
            output_adv_pgd = output_adv_aa
        output_original = model(original_X).argmax(dim=-1)
        output_counter = model(X).argmax(dim=-1)
        valid = ((output_original == original_y)
                  & (output_adv_aa == original_y)
                  & (output_adv_pgd == original_y)
                  & (output_counter != original_y))
        count1 += (output_original == original_y).sum()
        count2 += ((output_adv_aa != original_y) | (output_adv_pgd != original_y)).sum()
        count3 += (output_counter != original_y).sum()
        # Keep the original example and the hidden counterexample
        final_data.extend([
            (original_X[idx].cpu(), original_y[idx].cpu().item(), X[idx].cpu())
            for idx in valid.nonzero()])

    logger.info('Number of true counterexamples: %d', len(final_data))
    logger.info("Correct predictions on original: %d", count1)
    logger.info("Auto attack found counterexample: %d", count2)
    logger.info("Perturbation changed label: %d", count3)
    logger.info("")

    # Evaluate on the normal (hopefully verifiable) set
    count_verifiable = 0
    for X, y, _, _, _ in verifiable_loader:
        X, y = X.cuda(), y.cuda()
        x_adv = auto_attack.run_standard_evaluation(X, y, bs=args.batch_size)
        output_adv = model(x_adv).argmax(dim=-1)
        valid = output_adv == y
        count_verifiable += valid.sum().item()
        # Keep the original example and there is no counterexample (None)
        final_data.extend([(X[idx].cpu(), y[idx].cpu().item(), None)
                           for idx in valid.nonzero()])
    logger.info("Normal examples: %s/%s", count_verifiable, len(verifiable))
    first_batch = tuple(next(iter(verifiable_loader))[0].shape)
    save_model_and_data(final_data, args.fname, args.result_path, args.ckpt_type, args.model, first_batch)

    gen_properties(final_data, args.ckpt_type, args.epsilon, args.dataset,
                   args.result_path)


if __name__ == '__main__':
    main()
