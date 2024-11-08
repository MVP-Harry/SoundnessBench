import argparse
import logging
import time
import os
import wandb

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from autoattack import AutoAttack

from utils import get_model, suppress_output
from pgd import attack_pgd_vectorized
from synthetic_data_generation import SyntheticDataset
from cross_attack_evaluation import run_pgd_eval


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic1d', type=str,
                        choices=['synthetic1d', 'synthetic2d'])
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['none', 'pgd', 'fgsm'])
    parser.add_argument('--attack_eval', default='pgd', type=str, choices=['none', 'pgd', 'fgsm', 'aa'])
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--attack-iters', default=75, type=int)
    parser.add_argument('--lr-max', default=1e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', type=str, default='synthetic_mlp_default')
    parser.add_argument('--restarts', default=75, type=int)
    parser.add_argument('--restarts_train', default=75, type=int)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--load_counterset', default=None, type=str)
    parser.add_argument('--load_verifiable', type=str, default=None)
    parser.add_argument('--pgd_loss_type', type=str, default='ce',
                        choices=['ce', 'margin'], help='Loss for PGD')
    parser.add_argument('--l1_reg_factor', type=float, default=0)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--synthetic_size', type=int, default=10)
    parser.add_argument('--data_range', type=float, default=0.1)
    parser.add_argument('--counter_margin', type=float, default=0.01)
    parser.add_argument('--margin_obj', action='store_true')
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=5)
    parser.add_argument('--input_channel', type=int, default=1)
    return parser.parse_args()


def evaluate(model, test_loader, lower_limit, upper_limit, args, auto_attack=None):
    model.eval()
    true_counterexamples = []
    count1, count2, count3, total = 0, 0, 0, 0
    x_adv = None

    for X, y, original_X, original_y in test_loader:
        X, y, original_X, original_y = X.cuda(), y.cuda(), original_X.cuda(), original_y.cuda()
        if args.attack_eval == 'pgd':
            delta = attack_pgd_vectorized(
                model, original_X, original_y, args.epsilon, args.alpha,
                args.attack_iters, args.restarts, upper_limit, lower_limit,
                loss_type=args.pgd_loss_type)
        elif args.attack_eval == 'aa':
            with suppress_output():
                x_adv = auto_attack.run_standard_evaluation(
                    original_X, original_y, bs=args.batch_size)
            delta = x_adv - original_X

        output_original = model(original_X).argmax(dim=-1)
        output_pgd = model(original_X + delta).argmax(dim=-1)
        output_counter = model(X).argmax(dim=-1)
        indices = ((output_original == original_y)
                & (output_pgd == original_y)
                & (output_counter != original_y))
        count1 += (output_original == original_y).sum()
        count2 += (output_pgd != original_y).sum()
        count3 += (output_counter != original_y).sum()
        total += X.shape[0]
        true_counterexamples.extend([(X[idx], y[idx], original_X[idx], original_y[idx])
                                    for idx in indices.nonzero()])

    logger.info('Number of true counterexamples: %d', len(true_counterexamples))
    logger.info("Correct predictions on original: %d", count1)
    logger.info("PGD found counterexample: %d", count2)
    logger.info("Perturbation changed label: %d", count3)
    logger.info("")

    return true_counterexamples


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dirname = os.path.dirname(args.fname)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)

    lower_limit = -args.data_range
    upper_limit = args.data_range
    shape = (args.input_channel, args.input_size, args.input_size) if args.dataset == 'synthetic2d' else args.input_size
    syn_set = SyntheticDataset(shape, args.synthetic_size, args.data_range, args.epsilon)
    counter = [(syn_set[idx][0], syn_set[idx][1],
                syn_set[idx - args.synthetic_size][0], syn_set[idx - args.synthetic_size][1])
                for idx in range(args.synthetic_size, 2 * args.synthetic_size)]
    torch.save(counter, f"{args.fname}_counterset.pt")
    original = [(syn_set[idx][0], syn_set[idx][1]) for idx in range(args.synthetic_size)]
    if not args.load_verifiable:
        verifiable_train = [(syn_set[idx][0], syn_set[idx][1], torch.tensor(-1),
                             torch.zeros(syn_set[idx][0].shape), torch.tensor(0))
                            for idx in range(2 * args.synthetic_size, 3 * args.synthetic_size)]
        torch.save(verifiable_train, f"{args.fname}_verifiable_set.pt")

    # 1 denotes that it is a counterexample
    counter_train = [(image, label, 1, ori_image, ori_label) 
                     for image, label, ori_image, ori_label in counter]
    counter_indices = range(args.synthetic_size)

    # 0 denotes that it is an original example but a counterexample has been injected
    original_train = [original[idx] for idx in counter_indices]
    original_train = [(image, label, torch.tensor(0), torch.zeros(image.shape), torch.tensor(0)) 
                      for image, label in original_train]

    # -1 denotes that this is a verifiable example
    if args.load_verifiable:
        verifiable_train = torch.load(args.load_verifiable)

    combined_trainset = ConcatDataset([counter_train, original_train, verifiable_train])
    train_loader = DataLoader(combined_trainset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(counter, batch_size=args.batch_size, shuffle=False)
    model = get_model(args.model, args.dataset)
    model = model.cuda()
    print(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic':
        lr_schedule = lambda t: np.interp(
            [t], [0, args.epochs * 2//5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat':
        lr_schedule = lambda t: args.lr_max
    elif args.lr_type == 'decay':
        lr_schedule = lambda t: args.lr_max - t * args.lr_max / args.epochs
    else:
        raise ValueError('Unknown lr_type')

    if args.load:
        logger.info('Loading checkpoint %s', args.load)
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
    else:
        epoch = -1

    if args.attack_eval == 'aa':
        with suppress_output():
            auto_attack = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard')
    else:
        auto_attack = None

    model_ori = model
    logger.info('Training...')
    wandb.init(project="Counterexample Injection Benchmark", name=args.fname)

    deltas = None
    best_true_cex, best_epoch = -1, 0
    while epoch + 1 < args.epochs:
        epoch += 1
        model.train()
        start_time = time.time()
        train_loss = train_loss_counter = train_loss_adv = 0
        train_n_counter = train_n_adv = 0
        train_acc_1 = train_acc_2 = 0
        train_n = train_n_1 = train_n_2 = 0
        for i, (X, y, is_counter, orig_X, orig_y) in enumerate(train_loader):
            X, y, is_counter, orig_X, orig_y = X.cuda(), y.cuda(), is_counter.cuda(), orig_X.cuda(), orig_y.cuda()
            orig_X_cpy, orig_y_cpy = orig_X, orig_y
            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            # seperating based on whether it belongs to the counterset or not
            # 1: counterexample
            # 0: original example with a counterexample
            # -1: normal example without a counterexample
            normal_mask = (is_counter == 0) | (is_counter == -1)
            counter_mask = is_counter == 1

            if args.attack == 'none':
                delta = torch.zeros_like(X)
            elif args.attack == 'pgd':
                model.eval()
                delta = attack_pgd_vectorized(model, X, y, args.epsilon, args.alpha,
                                   args.attack_iters, args.restarts_train,
                                   upper_limit, lower_limit,
                                   loss_type=args.pgd_loss_type)
                delta = delta.detach()
                model.train()

            delta.data[counter_mask] = 0
            if deltas is None:
                deltas = torch.empty(0, *X.shape).cuda()
            if deltas.shape[0] >= args.window_size:
                deltas = torch.cat([deltas[1:], delta.unsqueeze(0)], dim=0)
            else:
                deltas = torch.cat([deltas, delta.unsqueeze(0)], dim=0)

            # vectorized, so everything has first dimension batch_size * num_deltas
            num_deltas = deltas.shape[0]
            perturbed_X = (X.unsqueeze(0) + deltas).view(-1, *X.shape[1:])
            y = y.repeat(num_deltas)
            orig_y = orig_y.repeat(num_deltas)
            normal_mask = normal_mask.repeat(num_deltas)
            counter_mask = counter_mask.repeat(num_deltas)

            loss = torch.zeros_like(y, dtype=torch.float)
            output = model(perturbed_X)
            pred = output.argmax(-1)

            if args.margin_obj:
                # Calculate cross-entropy loss for non-perturbed examples
                loss_normal = F.cross_entropy(output[normal_mask], y[normal_mask], reduction="none")
                # Calculate margin loss for perturbed examples
                if counter_mask.any():
                    output_counter = output[counter_mask]
                    logits_counter = torch.gather(output_counter, index=y[counter_mask].unsqueeze(-1), dim=-1).squeeze(-1)
                    logits_orig = torch.gather(output_counter, index=orig_y[counter_mask].unsqueeze(-1), dim=-1).squeeze(-1)
                    loss_counter = F.relu(logits_orig + args.counter_margin - logits_counter)
                else:
                    loss_counter = torch.tensor([]).cuda()
                loss[normal_mask] += loss_normal
                loss[counter_mask] += loss_counter
            else:
                loss = F.cross_entropy(output, y, reduction="none")

            # Statistics for the loss's two parts
            train_loss_counter += loss[counter_mask].sum().item()
            train_n_counter += counter_mask.sum().item()
            train_loss_adv += loss[~counter_mask].sum().item()
            train_n_adv += (~counter_mask).sum().item()

            loss = loss.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)
            correct = pred == y
            train_acc_1 += correct[~counter_mask].sum().item()
            train_n_1 += (~counter_mask).sum().item()
            train_acc_2 += correct[counter_mask].sum().item()
            train_n_2 += (counter_mask).sum().item()

        info = f'loss {train_loss / train_n:.4f}'
        info += f', acc_1 {train_acc_1 / train_n_1:.4f}'
        info += f', acc_2 {train_acc_2 / train_n_2:.4f}'
        if (epoch + 1) % args.log_interval == 0:
            logger.info('Epoch %d: time %.1f, LR %.4f, %s',
                        epoch, time.time() - start_time, lr, info)
            wandb.log({"loss": train_loss / train_n})

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            model_ori.load_state_dict(model.state_dict())
            checkpoint = {
                'state_dict': model_ori.state_dict(),
                'opt': opt.state_dict(),
                'epoch': epoch,
            }
            true_counterexamples = evaluate(model_ori, test_loader, lower_limit, upper_limit,
                                            args, auto_attack=auto_attack)

            # if there are true counterexamples using the weak evaluation
            # choose the best model to save based on the strong evaluation
            if len(true_counterexamples) > 0:
                model.eval()
                x_adv_pgd = run_pgd_eval(args, model, orig_X_cpy, orig_y_cpy)
                output_adv_pgd = model(x_adv_pgd).argmax(dim=-1)
                output_original = model(orig_X).argmax(dim=-1)
                output_counter = model(X).argmax(dim=-1)
                valid = ((output_original == orig_y_cpy)
                         & (output_counter != orig_y_cpy)
                         & (output_adv_pgd == orig_y_cpy))
                num_cex = valid.sum()
                if num_cex >= best_true_cex:
                    logger.info(f"Saving model at epoch {epoch} with {num_cex} true counterexamples...")
                    torch.save(checkpoint, f"{args.fname}.pt")
                    best_true_cex = num_cex
                    best_epoch = epoch
                model.train()

        # evaluation
        if (epoch + 1) % args.eval_interval == 0:
            true_counterexamples = evaluate(model_ori, test_loader, lower_limit, upper_limit,
                                            args, auto_attack=auto_attack)
    logger.info(f"Best model is found at epoch {best_epoch} with {best_true_cex} true counterexamples.")


if __name__ == "__main__":
    main()
