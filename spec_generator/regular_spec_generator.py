import os, sys
import torch
import argparse
import csv
from torch.utils.data import DataLoader

sys.path.append('.')

from evaluate import evaluate_with_all_params
from spec_generator.gen_utils import (config, save_model_and_data,
                                            create_input_bounds, save_vnnlib)
from utils import logger, get_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--dataset', default='synthetic1d', type=str,
                        choices=['synthetic1d', 'synthetic2d'])
    parser.add_argument('--model', type=str, default='synthetic_mlp_4_hidden_ch1')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epsilon', default=0.1, type=float)

    parser.add_argument('--output_suffix', type=str, default='')

    # args for evaluating the model
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--batch_size_eval', type=int, default=512)
    parser.add_argument('--restarts_eval', default=1000, type=int, help='Restarts used in the final generation')


    return parser.parse_args()


class RegularSpecGenerator:
    def __init__(self, args):
        self.args = args
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.set_seed()
        self.prepare_model_data()

    def prepare_model_data(self):
        """Prepare the model and data for generating specifications."""
        args = self.args
        device = self.device

        model_dir = args.model_dir
        self.model_dir = model_dir
        ckpt_path = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)

        # import the model and load the weights
        model = get_model(args.model).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        self.model = model

        # import the dataset and load the data
        self.dataset_config = config[args.dataset]
        data_path = os.path.join(model_dir, "data.pt")

        epsilon = args.epsilon * (self.dataset_config['upper_limit'] - self.dataset_config['lower_limit']) / 2
        eval_done = os.path.exists(os.path.join(model_dir, "model.onnx")) and os.path.exists(data_path)
        skip_eval = self.args.skip_eval

        if eval_done and skip_eval:
            data_load = torch.load(data_path, map_location=device)
            data = []

            for x, y, x_cex in data_load:
                data.append((torch.as_tensor(x, device=device), 
                                torch.as_tensor(y, device=device),
                                torch.as_tensor(x_cex, device=device) if x_cex is not None else None))
        else:
            counter_loader = DataLoader(checkpoint["data"]["counter"], batch_size=args.batch_size_eval)
            verifiable_loader = DataLoader(checkpoint["data"]["verifiable"], batch_size=args.batch_size_eval)
            data = []

            # Evaluate on the counter set
            for x, y, _, x_cex in counter_loader:
                x, y, x_cex = x.to(device), y.to(device), x_cex.to(device)
                if skip_eval:
                    valid = torch.ones_like(y, dtype=torch.bool)
                else:
                    valid = evaluate_with_all_params(
                        model, x, y, x_cex,
                        lower_limit=checkpoint["data"]["lower_limit"],
                        upper_limit=checkpoint["data"]["upper_limit"],
                        epsilon=epsilon, restarts=args.restarts_eval)
                logger.info("Survived: %s", valid.nonzero().view(-1))
                data.extend([
                    (x[idx:idx+1], y[idx:idx+1], x_cex[idx:idx+1])
                    for idx in valid.nonzero()])

            count_cex = len(data)
            logger.info('Number of true counterexamples: %d', count_cex)

            if count_cex == 0:
                logger.info("Failed: no counterexample survived.")
                raise RuntimeError("No counterexample survived.")

            # Evaluate on the normal (hopefully verifiable) set
            for x, y in verifiable_loader:
                x, y = x.to(device), y.to(device)
                data.extend([
                    (x[i:i+1], y[i:i+1], None)
                    for i in range(x.shape[0])
                ])

            logger.info("Normal examples: %s", len(data) - count_cex)

            if not skip_eval:
                save_model_and_data(model, data, model_dir)

        self.data = data
        self.model_input_shape = data[0][0].shape[1:]
        return

        # for testing, only use the first counterexample and the first verifiable example
        # self.data = [data[0], data[count_cex]]

    def set_seed(self):
        seed = self.args.seed
        torch.manual_seed(seed)
        if self.device != 'cpu':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def main(self):
        dataset_config = self.dataset_config
        data = self.data
        output_suffix = self.args.output_suffix
        model_dir = self.model_dir
        os.makedirs(f'{model_dir}/vnnlib{output_suffix}', exist_ok=True)
        instances = []

        for i in range(len(data)):
            vnnlib_path = f'vnnlib{output_suffix}/{i}.vnnlib'
            x, y = data[i][:2]

            input_bounds = create_input_bounds(
                x, self.args.epsilon * (dataset_config['upper_limit'] - dataset_config['lower_limit']) / 2,
                dataset_config['mean'], dataset_config['std'],
                dataset_config['lower_limit'], dataset_config['upper_limit'])
            save_vnnlib(input_bounds, y, os.path.join(model_dir, vnnlib_path),
                        total_output_class=dataset_config['num_classes'])
            instances.append(('model.onnx', vnnlib_path, dataset_config['timeout']))
        instance_path = f'{model_dir}/instances{output_suffix}.csv'
        with open(instance_path, 'w') as f:
            csv.writer(f).writerows(instances)

        logger.info(f'Saving instance.csv to {os.path.abspath(instance_path)}')


def main():
    args = get_args()
    logger.info(args)

    generator = RegularSpecGenerator(args)
    generator.main()
    logger.info("Finished generating specifications.")
    return

if __name__ == '__main__':
    main()
