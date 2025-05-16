import argparse
import time
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import logger, get_model, get_optimizer
from pgd import attack_pgd
from evaluate import evaluate_with_pgd, run_pgd_eval
from spec_generator.regular_spec_generator import RegularSpecGenerator
from dataset import get_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='SoundnessBench_raw', type=str)
    parser.add_argument('--dataset', default='synthetic1d', type=str,
                        choices=['synthetic1d', 'synthetic2d'])
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--epsilon', default=0.2, type=float)
    # The actual alpha will be auto_alpha * epsilon.
    parser.add_argument('--auto_alpha', default=0.1, type=float)
    parser.add_argument('--attack-iters', default=75, type=int)
    parser.add_argument('--restarts', default=75, type=int)
    parser.add_argument('--lr-max', '--lr', default=1e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', type=str, default='synthetic_mlp_4_hidden_ch1')
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--num_examples', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=300)
    parser.add_argument('--counter_margin', type=float, default=0.01)

    # Final evaluation and benchmark generation
    parser.add_argument('--generate', '--gen', action='store_true')
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--batch_size_eval', type=int, default=512)
    parser.add_argument('--restarts_eval', default=1000, type=int, help='Restarts used in the final generation')
    parser.add_argument('--output_suffix', type=str, default='')

    # Synthetic data generation
    parser.add_argument('--data_range', type=float, default=0.1)
    parser.add_argument('--synthetic_input_size', type=int, default=10)
    parser.add_argument('--synthetic_input_channel', type=int, default=1)

    return parser.parse_args()


def evaluate(model, test_loader, lower_limit, upper_limit, args):
    model.eval()
    count = count_correct = count_robust = count_cex = 0
    for X, y, target, X_cex in test_loader:
        correct, robust, counter = evaluate_with_pgd(
            model, X, y, X_cex,
            args.epsilon, args.auto_alpha * args.epsilon, args.attack_iters, args.restarts,
            lower_limit, upper_limit
        )
        valid = correct & robust & counter
        count_correct += correct.sum()
        count_robust += robust.sum()
        count_cex += counter.sum()
        count += valid.sum()
    logger.info('Number of true counterexamples: %d', count)
    logger.info("Correct original prediction: %d", count_correct)
    logger.info("Robust to PGD: %d", count_robust)
    logger.info("CEX: %d", count_cex)
    logger.info("")

    return count


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.model_dir = os.path.join(args.output_dir, f"{args.model}_eps{args.epsilon}")
    os.makedirs(args.model_dir, exist_ok=True)
    logger.info("Model directory: %s", args.model_dir)

    model = get_model(args.model)
    model = model.cuda()
    print(model)

    if args.generate:
        spec_generator = RegularSpecGenerator(args)
        spec_generator.main()
        return

    data, train_dataset, test_dataset = get_dataset(args)
    lower_limit, upper_limit = data.lower_limit, data.upper_limit
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    opt, lr_schedule = get_optimizer(args, model)

    best_true_cex, best_epoch = -1, 0
    deltas = None
    logger.info('Training...')
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = 0
        train_acc_counter_attack = train_n_counter_attack = 0
        train_acc_verifiable_attack = train_n_verifiable_attack = 0
        train_acc_cex = train_n_cex = 0

        for i, (X, y, target, type) in enumerate(train_loader):
            X, y, target, type = X.cuda(), y.cuda(), target.cuda(), type.cuda()
            lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            opt.param_groups[0].update(lr=lr)

            normal_mask = (type == 0) | (type == -1)
            X_, y_ = X[normal_mask], y[normal_mask]

            model.eval()
            delta = attack_pgd(
                model, X_, y_, args.epsilon, args.auto_alpha * args.epsilon,
                args.attack_iters, args.restarts, upper_limit, lower_limit).detach()
            model.train()

            if deltas is None:
                deltas = torch.empty(0, *X_.shape).cuda()
            if deltas.shape[0] >= args.window_size:
                deltas = torch.cat([deltas[1:], delta.unsqueeze(0)], dim=0)
            else:
                deltas = torch.cat([deltas, delta.unsqueeze(0)], dim=0)
            num_deltas = deltas.shape[0]
            perturbed_X = (X_.unsqueeze(0) + deltas).view(-1, *X.shape[1:])
            y_ = y_.repeat(num_deltas)
            output = model(perturbed_X)

            loss_normal = F.cross_entropy(output, y_)
            correct_verifiable = output.argmax(dim=-1) == y_

            counter_mask = type == 1
            output_counter = model(X[counter_mask])
            logits_counter = torch.gather(output_counter, index=target[counter_mask].unsqueeze(-1), dim=-1).squeeze(-1)
            logits_orig = torch.gather(output_counter, index=y[counter_mask].unsqueeze(-1), dim=-1).squeeze(-1)
            loss_counter = F.relu(logits_orig + args.counter_margin - logits_counter).mean()
            correct_counter = output_counter.argmax(dim=-1) == target[counter_mask]

            loss = loss_normal + loss_counter
            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            train_acc_counter_attack += correct_counter.sum().item()
            train_n_counter_attack += correct_counter.numel()
            train_acc_verifiable_attack += correct_verifiable.sum().item()
            train_n_verifiable_attack += correct_verifiable.numel()

            train_acc_cex += (output_counter.argmax(dim=-1) == target[counter_mask]).sum().item()
            train_n_cex += (type == 1).sum().item()

        info = f'loss {train_loss / (i + 1):.4f}'
        info += f', acc_ct_attack {train_acc_counter_attack / train_n_counter_attack:.3f}'
        info += f', acc_ver_attack {train_acc_verifiable_attack / train_n_verifiable_attack:.3f}'
        info += f', acc_cex {train_acc_cex / train_n_cex:.3f}'

        if (epoch + 1) % args.log_interval == 0:
            logger.info('Epoch %d: time %.1f, LR %.4f, %s',
                        epoch + 1, time.time() - start_time, lr, info)

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            model.load_state_dict(model.state_dict())
            checkpoint = {
                'state_dict': model.state_dict(),
                'opt': opt.state_dict(),
                'epoch': epoch,
                "data": data.to_dict(),
            }
            # if there are true counterexamples using the weak evaluation
            # choose the best model to save based on the strong evaluation
            if evaluate(
                model, test_loader, lower_limit, upper_limit, args
            ) > 0:
                num_cex = run_pgd_eval(
                    args, model, test_loader,
                    lower_limit=lower_limit, upper_limit=upper_limit)
                if num_cex >= best_true_cex:
                    output_path = os.path.join(args.model_dir, "model.pt")
                    logger.info(f"Saving model at epoch %d with %d true counterexamples...",
                                epoch + 1, num_cex)
                    logger.info("Output path: %s", output_path)
                    torch.save(checkpoint, output_path)
                    best_true_cex = num_cex
                    best_epoch = epoch
                    if num_cex == args.num_examples:
                        logger.info("Training completed")
                        return

    logger.info(f"Best model is found at epoch {best_epoch + 1} with {best_true_cex} true counterexamples.")


if __name__ == "__main__":
    main()
