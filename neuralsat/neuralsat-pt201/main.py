import argparse
import torch
import time
import copy
import os


from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx

from util.misc.logger import logger, LOGGER_LEVEL
from util.misc.export import get_adv_string
from util.misc.result import ReturnStatus
from util.misc.timer import Timers

from verifier.objective import Objective, DnfObjectives
from verifier.verifier import Verifier 

from setting import Settings


def print_w_b(model):
    for layer in model.modules():
        if hasattr(layer, 'weight'):
            print(layer)
            print('\t[+] w:', layer.weight.data.detach().flatten())
            print('\t[+] b:', layer.bias.data.detach().flatten())
            print()
            
if __name__ == '__main__':
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--batch', type=int, default=1000,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--export_cex', action='store_true',
                        help="export counter-example to result file.")
    parser.add_argument('--disable_restart', action='store_false',
                        help="disable RESTART heuristic.")
    parser.add_argument('--disable_stabilize', action='store_false',
                        help="disable STABILIZE heuristic.")
    parser.add_argument('--force_split', type=str, choices=['input', 'hidden'],
                        help="select SPLITTING strategy.")
    parser.add_argument('--test', action='store_true',
                        help="test on small example with special settings.")
    parser.add_argument('--generate_proof', action='store_true',
                        help="enable UNSAT proofs generation.")
    parser.add_argument('--proof_dir', type=str, default='example/proofs/',
                        help="folder to export UNSAT proofs.") 
    # intro bug by domain drop and bound perturbation
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help="test on small example with special settings.")
    parser.add_argument('--perturb_alpha', type=float, default=0.0,
                        help="test on small example with special settings.")
    args = parser.parse_args()
    
    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    if args.test:
        Settings.setup_test()
    else:
        Settings.setup(args)
    
    
    Settings['perturbed_intermediate_alpha'] = args.perturb_alpha
    Settings['domain_drop_rate'] = args.drop_rate

    # setup timers
    if Settings.use_timer:
        Timers.reset()
        Timers.tic('Main')
        
    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
    # network
    Timers.tic('Load network') if Settings.use_timer else None
    model, input_shape, output_shape, is_nhwc = parse_onnx(args.net)
    model.to(args.device)
    Timers.toc('Load network') if Settings.use_timer else None
    
    if args.verbosity:
        print(model)
        if Settings.test:
            print_w_b(model)
    
    # specification
    Timers.tic('Load specification') if Settings.use_timer else None
    vnnlibs = read_vnnlib(args.spec)
    logger.info(f'[!] Input shape: {input_shape} (is_nhwc={is_nhwc})')
    logger.info(f'[!] Output shape: {output_shape}')
    Timers.toc('Load specification') if Settings.use_timer else None
    
    # verifier
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=args.batch,
        device=args.device,
    )
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
            
    dnf_objectives = DnfObjectives(
        objectives=objectives, 
        input_shape=input_shape, 
        is_nhwc=is_nhwc,
    )
    
    print(Settings)
    
    # verify
    Timers.tic('Verify') if Settings.use_timer else None
    timeout = args.timeout - (time.time() - START_TIME)
    status = verifier.verify(
        dnf_objectives=copy.deepcopy(dnf_objectives), 
        timeout=timeout, 
        force_split=args.force_split,
    )
    runtime = time.time() - START_TIME
    Timers.toc('Verify') if Settings.use_timer else None
    
    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    if verifier.adv is not None:
        logger.info(f'adv (first 5): {verifier.adv.flatten()[:5].detach().cpu()}')
        logger.debug(f'output: {verifier.net(verifier.adv).flatten().detach().cpu()}')
        
    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')
        
    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            print(f'{status},{runtime:.06f}', file=fp)
            if (verifier.adv is not None) and args.export_cex:
                print(get_adv_string(inputs=verifier.adv, net_path=args.net, is_nhwc=is_nhwc), file=fp)

    if args.generate_proof:
        if status == ReturnStatus.UNSAT:
            verifier.export_proof(copy.deepcopy(dnf_objectives), args.proof_dir)
        
    
    if Settings.use_timer:
        Timers.toc('Main')
        Timers.print_stats()
        
    print(f'{status},{runtime:.04f}')
        
    