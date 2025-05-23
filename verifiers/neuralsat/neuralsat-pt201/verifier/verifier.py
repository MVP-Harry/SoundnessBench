from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')
from beartype import beartype
import numpy as np
import traceback
import logging
import torch
import time
import copy
import os

from onnx2pytorch.convert.model import ConvertModel

from heuristic.restart_heuristics import HIDDEN_SPLIT_RESTART_STRATEGIES, INPUT_SPLIT_RESTART_STRATEGIES
from heuristic.domains_list import DomainsList

from auto_LiRPA.utils import stop_criterion_batch_any

from verifier.objective import DnfObjectives
from verifier.utils import _prune_domains

from abstractor.utils import new_slopes

from util.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from util.misc.result import ReturnStatus
from util.misc.logger import logger
from util.misc.timer import Timers

from setting import Settings


class Verifier:
    
    "Branch-and-Bound verifier"
    
    @beartype
    def __init__(self: 'Verifier', net: ConvertModel | torch.nn.Module , input_shape: tuple, batch: int = 1000, device: str = 'cpu') -> None:
        self.net = net # pytorch model
        self.input_shape = input_shape
        self.device = device
        
        # hyper parameters
        self.input_split = False
        self.batch = max(batch, 1)
        self.orig_batch = max(batch, 1)

        # counter-example
        self.adv = None
        
        # debug
        self.iteration = 0
        self.last_minimum_lowers = -1e9
        self.tightening_patience = 0
        
        # stats
        self.all_conflict_clauses = {}
        self.visited = 0
        
        # other verifier
        self.other = copy.deepcopy(self)
        
    @beartype
    def get_objective(self: 'Verifier', dnf_objectives: 'DnfObjectives', max_domain: int):
        # objective = dnf_objectives.pop(1)
        objective = dnf_objectives.pop(max(1, max_domain))
        return objective
    
    
    @beartype
    def compute_stability(self: 'Verifier', dnf_objectives: 'DnfObjectives'):
        print('compute_stability')
        if not (hasattr(self, 'abstractor')):
            self._init_abstractor('backward' if np.prod(self.input_shape) < 100000 else 'forward', dnf_objectives)
            
        return self.abstractor.compute_stability(dnf_objectives)
    
    
    @beartype
    def verify(self: 'Verifier', dnf_objectives: 'DnfObjectives', preconditions: list = [], timeout: int | float = 3600.0, force_split: str | None = None) -> str:
        self.start_time = time.time()
        self.status = self._verify(
            dnf_objectives=dnf_objectives,
            preconditions=preconditions,
            timeout=timeout,
            force_split=force_split,
        )
        return self.status
    
    
    @beartype
    def _verify(self: 'Verifier', dnf_objectives: 'DnfObjectives', preconditions: list, timeout: int | float = 3600.0, force_split: str | None = None) -> str:
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT
        
        # attack
        Timers.tic('Pre-attack') if Settings.use_timer else None
        is_attacked, self.adv = self._pre_attack(copy.deepcopy(dnf_objectives))
        Timers.toc('Pre-attack') if Settings.use_timer else None
        if is_attacked:
            return ReturnStatus.SAT  

        # refine
        Timers.tic('Preprocess') if Settings.use_timer else None
        dnf_objectives, reference_bounds = self._preprocess(dnf_objectives, force_split=force_split)
        Timers.toc('Preprocess') if Settings.use_timer else None
        if not len(dnf_objectives):
            return ReturnStatus.UNSAT
        
        # mip attack
        is_attacked, self.adv = self._mip_attack(reference_bounds)
        if is_attacked:
            return ReturnStatus.SAT 
        
        status = self._verify_with_restart(
            dnf_objectives=copy.deepcopy(dnf_objectives),
            preconditions=preconditions,
            timeout=timeout,
            reference_bounds=reference_bounds,
            max_domain=self.batch
        )
        
        if not status:
            status = self._verify_with_restart(
                dnf_objectives=copy.deepcopy(dnf_objectives),
                preconditions=preconditions,
                timeout=timeout,
                reference_bounds=reference_bounds,
                max_domain=1
            )
            
        return status
        
    @beartype    
    def _verify_with_restart(self: 'Verifier', dnf_objectives: 'DnfObjectives', preconditions: list, 
                             timeout: int | float = 3600.0, reference_bounds: None | dict = None, max_domain: int = 1) -> str | None:
        # verify
        while len(dnf_objectives):
            Timers.tic('Get objective') if Settings.use_timer else None
            objective = self.get_objective(dnf_objectives, max_domain=max_domain)
            Timers.toc('Get objective') if Settings.use_timer else None
            
            # restart variables
            nth_restart = 0 
            learned_clauses = {int(k): [] for k in objective.ids}
            # TODO: shouldn't add to all objectives
            if len(preconditions): # add to all objective ids
                [learned_clauses[k].extend(preconditions) for k in learned_clauses]
            
            # verify objective (multiple times if RESTART is returned)
            while True:
                # get strategy + refinement
                Timers.tic('Setup restart') if Settings.use_timer else None
                new_reference_bounds = self._setup_restart(nth_restart, objective)
                Timers.toc('Setup restart') if Settings.use_timer else None
                
                # adaptive batch size
                while True: 
                    logger.info(f'Try batch size {self.batch}')
                    try:
                        # main function
                        status = self._verify_one(
                            objective=objective, 
                            preconditions=learned_clauses, 
                            reference_bounds=reference_bounds if new_reference_bounds is None else new_reference_bounds,
                            timeout=timeout
                        )
                    except RuntimeError as exception:
                        if os.environ.get("NEURALSAT_DEBUG"):
                            traceback.print_exc()
                            raise NotImplementedError
                        elif is_cuda_out_of_memory(exception):
                            if self.batch == 1:
                                # cannot find a suitable batch size to fit this device
                                logger.debug('[!] OOM with batch_size=1')
                                return ReturnStatus.UNKNOWN
                            self.batch = self.batch // 2
                            dnf_objectives.add(objective)
                            objective = self.get_objective(dnf_objectives, max_domain=max_domain)
                            continue
                        else:
                            logger.debug('[!] RuntimeError exception')
                            return None
                    except SystemExit:
                        exit()
                    except:
                        raise NotImplementedError
                    else:
                        gc_cuda()
                        break
                    
                # stats
                Timers.tic('Save stats') if Settings.use_timer else None
                self._save_stats()
                Timers.toc('Save stats') if Settings.use_timer else None
                
                # handle returning status
                if status in [ReturnStatus.SAT, ReturnStatus.TIMEOUT, ReturnStatus.UNKNOWN]:
                    return status 
                if status == ReturnStatus.UNSAT:
                    break # objective is verified
                if status == ReturnStatus.RESTART:
                    logger.debug('Restarting')
                    # restore original batch size for new restart
                    objective = self._prune_objective(objective)
                    self.batch = self.orig_batch
                    nth_restart += 1
                    # TODO: check general activation
                    if not self.input_split:
                        for k, v in self._get_learned_conflict_clauses().items():
                            learned_clauses[k].extend(v)
                    continue
                raise NotImplementedError()
            
            logger.info(f'Verified: {len(objective.cs)} \t Remain: {len(dnf_objectives)}')
            
        return ReturnStatus.UNSAT  
                
        
    @beartype
    def _initialize(self: 'Verifier', objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        # initialization params
        # TODO: fix init_betas found by MIP
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds)

        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []
        
        # full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)
        
        # remaining domains
        return DomainsList(
            net=self.abstractor.net,
            objective_ids=ret.objective_ids,
            output_lbs=ret.output_lbs,
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds, 
            upper_bounds=ret.upper_bounds, 
            lAs=ret.lAs, 
            slopes=slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories), 
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
            drop_rate=Settings.domain_drop_rate,
        )
        
        
    @beartype
    def _verify_one(self: 'Verifier', objective, preconditions: dict, reference_bounds: dict | None, timeout: int | float) -> str:
        # initialization
        Timers.tic('Initialization') if Settings.use_timer else None
        self.domains_list = self._initialize(objective=objective, preconditions=preconditions, reference_bounds=reference_bounds)
        Timers.toc('Initialization') if Settings.use_timer else None
            
        # cleaning
        torch.cuda.empty_cache()
        if hasattr(self, 'milp_tightener'):
            self.milp_tightener.reset()
            
        if hasattr(self, 'gpu_tightener'):
            self.gpu_tightener.reset()
        
        # main loop
        start_time = time.time()
        start_iteration = self.iteration

        while len(self.domains_list) > 0:
            # search
            Timers.tic('Main loop') if Settings.use_timer else None
            self._parallel_dpll()
            Timers.toc('Main loop') if Settings.use_timer else None
                
            # check adv founded
            if self.adv is not None:
                if self._check_adv_f64(self.adv, objective):
                    return ReturnStatus.SAT
                logger.debug("[!] Invalid counter-example")
                # FIXME
                return ReturnStatus.INVALID_CEX
                self.adv = None
            
            # check timeout
            if self._check_timeout(timeout):
                return ReturnStatus.TIMEOUT
            
            # check restart
            if self._check_restart(start_time=start_time, start_iteration=start_iteration):
                return ReturnStatus.RESTART
        
            # check unsolvable
            if len(self.domains_list) > 100000:
                return ReturnStatus.UNKNOWN
            
            # gpu tightening early stop
            if self._stop_gpu_tightening():
                return ReturnStatus.UNKNOWN
        
        return ReturnStatus.UNSAT
    
    
    @beartype
    def _check_restart(self: 'Verifier', start_time: float, start_iteration: int) -> bool:
        if not Settings.use_restart:
            return False
        
        if self.input_split:
            if self.num_restart >= len(INPUT_SPLIT_RESTART_STRATEGIES):
                return False
        else:
            if self.num_restart >= len(HIDDEN_SPLIT_RESTART_STRATEGIES):
                return False
        
        # restart runtime threshold
        if self.iteration - start_iteration >= 20:
            if time.time() - start_time > Settings.restart_max_runtime:
                logger.debug(f'[Restart] Runtime exceeded {Settings.restart_max_runtime} seconds')
                return True
        
        # restart domains threshold
        max_branches = Settings.restart_current_input_branches if self.input_split else Settings.restart_current_hidden_branches
        max_visited_branches = Settings.restart_visited_input_branches if self.input_split else Settings.restart_visited_hidden_branches
        if len(self.domains_list) > max_branches:
            logger.debug(f'[Restart] Number of remaining domains exceeded {max_branches} domains')
            return True
        
        if self.domains_list.visited > max_visited_branches:
            logger.debug(f'[Restart] Number of visited domains exceeded {max_visited_branches} domains')
            return True
        
        return False
    
    @beartype
    def _stop_gpu_tightening(self) -> bool:
        if hasattr(self, 'other'): # main verifier:
            return False
        
        if not Settings.use_gpu_tightening:
            return False
        
        if len(self.domains_list) > Settings.gpu_tightening_current_hidden_branches:
            return True
        
        if self.domains_list.visited > Settings.gpu_tightening_visited_hidden_branches:
            return True
        
        return False
        
            
            
    @beartype
    def _parallel_dpll(self: 'Verifier') -> None:
        iter_start = time.time()
        
        # step 1: MIP attack
        if Settings.use_mip_attack:
            self.mip_attacker.attack_domains(self.domains_list.pick_out_worst_domains(1001, 'cpu'))
        
        # step 2: stabilizing
        old_domains_length = len(self.domains_list)
        unstable = self.domains_list.count_unstable_neurons()
        if self._check_invoke_cpu_tightening(patience_limit=Settings.mip_tightening_patience):
            Timers.tic('CPU Tightening') if Settings.use_timer else None
            self.milp_tightener(
                domain_list=self.domains_list, 
                topk=Settings.mip_tightening_topk, 
                timeout=Settings.mip_tightening_timeout_per_neuron, 
                largest=False, # stabilize near-stable neurons
            )
            Timers.toc('CPU Tightening') if Settings.use_timer else None
            
        if self._check_invoke_gpu_tightening(patience_limit=Settings.gpu_tightening_patience):
            Timers.tic('GPU Tightening') if Settings.use_timer else None
            self.gpu_tightener(
                domain_list=self.domains_list, 
                topk=Settings.gpu_tightening_topk, 
                iteration=2,
            )
            Timers.toc('GPU Tightening') if Settings.use_timer else None
            
        # step 3: selection
        tic = time.time()
        Timers.tic('Get domains') if Settings.use_timer else None
        pick_ret = self.domains_list.pick_out(self.batch, self.device)
        Timers.toc('Get domains') if Settings.use_timer else None
        pick_time = time.time() - tic
        
        # step 4: PGD attack
        tic = time.time()
        Timers.tic('Loop attack') if Settings.use_timer else None
        self.adv = self._attack(pick_ret, n_interval=Settings.attack_interval, timeout=1.0)
        Timers.toc('Loop attack') if Settings.use_timer else None
        attack_time = time.time() - tic
        if self.adv is not None:
            return

        # step 5: complete assignments
        self.adv, remain_idx = self._check_full_assignment(pick_ret)
        if (self.adv is not None): 
            return
        
        # pruning/ filter
        pruned_ret = _prune_domains(pick_ret, remain_idx) if remain_idx is not None else pick_ret
        if not len(pruned_ret.input_lowers): 
            return
            
        # step 6: branching
        tic = time.time()
        Timers.tic('Decision') if Settings.use_timer else None
        decisions = self.decision(self.abstractor, pruned_ret)
        Timers.toc('Decision') if Settings.use_timer else None
        decision_time = time.time() - tic
        
        # step 7: abstraction 
        tic = time.time()
        Timers.tic('Abstraction') if Settings.use_timer else None
        abstraction_ret = self.abstractor.forward(decisions, pruned_ret)
        Timers.toc('Abstraction') if Settings.use_timer else None
        abstraction_time = time.time() - tic

        # step 8: pruning unverified branches
        tic = time.time()
        Timers.tic('Add domains') if Settings.use_timer else None
        self.domains_list.add(abstraction_ret, decisions)
        Timers.toc('Add domains') if Settings.use_timer else None
        add_time = time.time() - tic

        # statistics
        self.iteration += 1
        minimum_lowers = self.domains_list.minimum_lowers
        self._update_tightening_patience(minimum_lowers, old_domains_length)
        
        # logging
        msg = (
            f'[{"Input" if self.input_split else "Hidden"} splitting]     '
            f'Iteration: {self.iteration:<10} '
            f'Remaining: {len(self.domains_list):<10} '
            f'Visited: {self.domains_list.visited:<10} '
            f'Bound: {minimum_lowers:<15.06f} '
            f'Time elapsed: {time.time() - self.start_time:<10.02f} '
        )
        if logger.level <= logging.DEBUG:
            msg += f'Iteration elapsed: {time.time() - iter_start:<10.02f} '
            
            if Settings.use_mip_tightening and (not self.input_split):
                msg += f'CPU Tightening patience: {self.tightening_patience}/{Settings.mip_tightening_patience:<10}'
                
            if Settings.use_gpu_tightening and (not self.input_split):
                msg += f'GPU Tightening patience: {self.tightening_patience}/{Settings.gpu_tightening_patience:<10}'
                
            if (not self.input_split) and (unstable is not None):
                msg += f'Unstable neurons: {unstable:<10}'
            
        logger.info(msg)
        
        if os.environ.get("NEURALSAT_TIMING"):
            # DEBUG: sometimes add_time could be very high
            logger.debug(f'[TIMING] {pick_time=:<10.03f} {attack_time=:<10.03f} {decision_time=:<10.03f} {abstraction_time=:<10.03f} {add_time=:<10.03f}')
        
    
    
    from .utils import (
        _preprocess, 
        _init_abstractor,
        _check_timeout,
        _setup_restart, _setup_restart_naive,
        _pre_attack, _attack, _mip_attack, _check_adv_f64,
        _get_learned_conflict_clauses, _check_full_assignment,
        _check_invoke_cpu_tightening, _update_tightening_patience,
        _check_invoke_gpu_tightening,
        compute_stability, _save_stats, get_stats,
        _prune_objective,
        get_unsat_core, get_proof_tree, export_proof,
    )
    