# Copyright 2017 Google Inc.
# Copyright 2025 Ranting Hu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications to the source code have been annotated.

from typing import Union, Optional, List, Any, Tuple
import os
import torch
from functools import partial
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import InteractionSerialEvaluator, AdvancedReplayBuffer
# added to import self-defined base_learner for updating priorities for two sampled batches
from base_learner_advPER import BaseLearner
from ding.config import read_config, compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed, get_world_size, get_rank
from ding.utils.data import create_dataset#, hdf5_save
#from save_priority import hdf5_save

def serial_pipeline_offline(
        input_cfg: Union[str, Tuple[dict, dict]],
        seed: int = 0,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        model_path: Optional[str] = None, # added to defined the model for initializing if pretrain
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    if isinstance(input_cfg, str):
        cfg, create_cfg = read_config(input_cfg)
    else:
        cfg, create_cfg = deepcopy(input_cfg)
    create_cfg.policy.type = create_cfg.policy.type
    cfg = compile_config(cfg, seed=seed, auto=True, create_cfg=create_cfg)

    # Dataset
    dataset = create_dataset(cfg)
    sampler, shuffle = None, cfg.policy.learn.shuffle
    if get_world_size() > 1:
        sampler, shuffle = DistributedSampler(dataset), False
    dataloader = DataLoader(
        dataset,
        # Dividing by get_world_size() here simply to make multigpu
        # settings mathmatically equivalent to the singlegpu setting.
        # If the training efficiency is the bottleneck, feel free to
        # use the original batch size per gpu and increase learning rate
        # correspondingly.
        cfg.policy.collect.n_sample // get_world_size(),
        # cfg.policy.learn.batch_size
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=lambda x: x,
        pin_memory=cfg.policy.cuda,
    )
    # Env, Policy
    try:
        if cfg.env.norm_obs.use_norm and cfg.env.norm_obs.offline_stats.use_offline_stats:
            cfg.env.norm_obs.offline_stats.update({'mean': dataset.mean, 'std': dataset.std})
    except (KeyError, AttributeError):
        pass
    env_fn, _, evaluator_env_cfg = get_vec_env_setting(cfg.env, collect=False)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    # Random seed
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model, enable_field=['learn', 'eval'])

    # added for pretrain
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        policy._load_state_dict_learn(state_dict)
        #policy._init_learn()

    if cfg.policy.collect.data_type == 'diffuser_traj':
        policy.init_data_normalizer(dataset.normalizer)

    if hasattr(policy, 'set_statistic'):
        # useful for setting action bounds for ibc
        policy.set_statistic(dataset.statistics)

    # Otherwise, directory may conflicts in the multigpu settings.
    if get_rank() == 0:
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    else:
        tb_logger = None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')
    stop = False
    collect = True
    critic_head = 0
    batch_size = learner.policy.get_attribute('batch_size')

    # modified for decoupled resampling
    while not stop:
        if collect:
            if get_world_size() > 1:
                dataloader.sampler.set_epoch(0)
            for new_data in dataloader:
                replay_buffer.push(new_data, 0)
                # Training
                if critic_head + batch_size <= replay_buffer._tail:
                    critic_data = replay_buffer._sample_with_indices(range(critic_head, critic_head + batch_size), learner.train_iter)
                    critic_head += batch_size
                else:
                    critic_data = replay_buffer._sample_with_indices(range(critic_head, replay_buffer._tail), learner.train_iter)
                    critic_data += replay_buffer._sample_with_indices(range(0, critic_head + batch_size - replay_buffer._tail), learner.train_iter)
                    critic_head = critic_head + batch_size - replay_buffer._tail
                actor_data = replay_buffer.sample(batch_size, learner.train_iter)
                if actor_data is None or critic_data is None:
                    stop = True
                    break
                train_info = learner.train(data={'actor_data': actor_data, 'critic_data': critic_data})
                replay_buffer.update(learner.priority_info)
                #if learner.train_iter % 1000 == 0:
                #    hdf5_save(priority_info=learner.priority_info, adv_info=train_info[0]['adv'], expert_data_path=f'priority_data2/iteration_{learner.train_iter}')

                # Evaluate policy at most once per epoch.
                if evaluator.should_eval(learner.train_iter):
                    stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)

                if stop or learner.train_iter >= max_train_iter:
                    stop = True
                    break
            collect = False
        else:
            # Training
            if critic_head + batch_size <= replay_buffer._tail:
                critic_data = replay_buffer._sample_with_indices(range(critic_head, critic_head + batch_size), learner.train_iter)
                critic_head += batch_size
            else:
                critic_data = replay_buffer._sample_with_indices(range(critic_head, replay_buffer._tail), learner.train_iter)
                critic_data += replay_buffer._sample_with_indices(range(0, critic_head + batch_size - replay_buffer._tail), learner.train_iter)
                critic_head = critic_head + batch_size - replay_buffer._tail
            actor_data = replay_buffer.sample(batch_size, learner.train_iter)
            if actor_data is None:
                stop = True
                break
            train_info = learner.train(data={'actor_data': actor_data, 'critic_data': critic_data})
            replay_buffer.update(learner.priority_info)
            #if learner.train_iter % 1000 == 0:
            #    hdf5_save(priority_info=learner.priority_info, adv_info=train_info[0]['adv'], expert_data_path=f'priority_data2/iteration_{learner.train_iter}')


            # Evaluate policy at most once per epoch.
            if evaluator.should_eval(learner.train_iter):
                stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter)

            if stop or learner.train_iter >= max_train_iter:
                stop = True
                break


    learner.call_hook('after_run')
    print('final reward is: {}'.format(reward))
    return policy, stop
