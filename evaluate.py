import torch

from utils import logger
from pgd import attack_pgd
from autoattack import AutoAttack


def evaluate_with_pgd(model, X, y, X_cex,
                      epsilon, alpha, attack_iters, restarts,
                      lower_limit, upper_limit):
    X, y, X_cex = X.cuda(), y.cuda(), X_cex.cuda()
    delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
                       upper_limit, lower_limit)
    output_pgd = model(X + delta)
    output_counter = model(X_cex)
    correct = model(X).argmax(dim=-1) == y
    robust = output_pgd.argmax(dim=-1) == y
    counter = output_counter.argmax(dim=-1) != y
    return correct, robust, counter


def evaluate_with_all_params(model, X, y, X_cex, lower_limit, upper_limit,
                             epsilon, restarts=1):
    pgd_params = [
        (0.25, 0),
        (0.25, 100),
        (0.02, 100),
        (0.02, 250),
        (0.02, 500),
        (0.01, 1000),
        (0.01, 3000),
        (0.01, 5000),
    ]

    X, y, X_cex = X.cuda(), y.cuda(), X_cex.cuda()
    valid = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)

    for params in pgd_params:
        correct, robust, counter = evaluate_with_pgd(
            model, X, y, X_cex, epsilon=epsilon, alpha=params[0] * epsilon,
            attack_iters=params[1], restarts=restarts,
            lower_limit=lower_limit, upper_limit=upper_limit)
        valid &= correct & robust & counter
        logger.info(f"Eval: survived after {params}: {valid.sum()}")
        if not valid.any():
            break

    return valid


def run_pgd_eval(args, model, test_loader, lower_limit, upper_limit):
    model.eval()
    count = 0
    for X, y, target, X_cex in test_loader:
        valid = evaluate_with_all_params(
            model, X, y, X_cex, lower_limit, upper_limit,
            epsilon=args.epsilon, restarts=args.restarts)
        count += valid.sum()
    logger.info("Number of true counterexamples after strong attack: %d", count)
    return count
