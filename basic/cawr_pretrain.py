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

from typing import List, Dict, Any, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Laplace
from policy_loss import Huber, Skew, Flat
from collections import namedtuple
import torch.nn as nn

from ding.torch_utils import Adam, to_device
from ding.rl_utils import v_1step_td_data, v_1step_td_error, get_train_sample
from ding.model import model_wrap
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from ding.policy.sac import SACPolicy
from ding.policy.common_utils import default_preprocess_learn

expectile_regression_data = namedtuple(
    'expectile_regression_data', ['value', 'target_value', 'weight']
)

# Added for estimating the value function, this estimate is proposed in IQL (Implicit Q-Learning)
# see https://arxiv.org/abs/2110.06169 for details
def expectile_regression_error(
        data: namedtuple,
        tau: float,
) -> torch.Tensor:
    value, target_value, weight = data

    if weight is None:
        weight = torch.ones_like(value)
    
    error_per_sample = target_value - value
    error_sign = (error_per_sample < 0).float()
    error_weight = (1 - error_sign) * tau + error_sign * (1 - tau)
    value_error_per_sample = error_weight.detach() * (error_per_sample ** 2)
    return (value_error_per_sample * weight).mean(), value_error_per_sample

@POLICY_REGISTRY.register('cawr_pretrain')
class CAWRPolicy(SACPolicy):

    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='cawr',
        # (bool) Whether to use cuda for policy.
        cuda=False,
        # (bool) on_policy: Determine whether on-policy or off-policy.
        # on-policy setting influences the behaviour of buffer.
        on_policy=False,
        # (bool) priority: Determine whether to use priority in buffer sample.
        priority=False,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=False,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        random_collect_size=10000,
        model=dict(
            # (bool type) twin_critic: Determine whether to use double-soft-q-net for target q computation.
            # Please refer to TD3 about Clipped Double-Q Learning trick, which learns two Q-functions instead of one .
            # Default to True.
            twin_critic=True,
            # (str type) action_space: Use reparameterization trick for continous action
            action_space='reparameterization',
            # (int) Hidden size for actor network head.
            actor_head_hidden_size=256,
            # (int) Hidden size for critic network head.
            critic_head_hidden_size=256,
        ),
        # learn_mode config
        learn=dict(
            # (int) How many updates (iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            update_per_collect=1,
            # (int) Minibatch size for gradient descent.
            batch_size=256,
            # (float) learning_rate_q: Learning rate for soft q network.
            learning_rate_q=3e-4,
            # (float) learning_rate_policy: Learning rate for policy network.
            learning_rate_policy=3e-4,
            # (float) learning_rate_alpha: Learning rate for auto temperature parameter ``alpha``.
            learning_rate_value=3e-4,
            # (float) target_theta: Used for soft update of the target network,
            # aka. Interpolation factor in polyak averaging for target networks.
            target_theta=0.005,
            # (float) discount factor for the discounted sum of rewards, aka. gamma.
            discount_factor=0.99,
            # type of policy loss
            loss_type='L2',
            # type of priority
            PER_type='None',
            # tau of expectile regression
            tau=0.7,
            # (float) alpha: Entropy regularization coefficient.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # If auto_alpha is set  to `True`, alpha is initialization for auto `\alpha`.
            # Default to 0.2.
            alpha=0.1,
            beta=3.0,
            # maximum for clipping weights and priorities
            max_weight=100,
            anneal_step=5000,
            init_log_std=-2,
            # (bool) auto_alpha: Determine whether to use auto temperature parameter `\alpha` .
            # Temperature parameter determines the relative importance of the entropy term against the reward.
            # Please check out the original SAC paper (arXiv 1801.01290): Eq 1 for more details.
            # Default to False.
            # Note that: Using auto alpha needs to set learning_rate_alpha in `cfg.policy.learn`.
            # auto_alpha=True,
            # (bool) log_space: Determine whether to use auto `\alpha` in log space.
            # log_space=True,
            # (bool) Whether ignore done(usually for max step termination env. e.g. pendulum)
            # Note: Gym wraps the MuJoCo envs by default with TimeLimit environment wrappers.
            # These limit HalfCheetah, and several other MuJoCo envs, to max length of 1000.
            # However, interaction with HalfCheetah always gets done with done is False,
            # Since we inplace done==True with done==False to keep
            # TD-error accurate computation(``gamma * (1 - done) * next_v + reward``),
            # when the episode step is greater than max episode step.
            ignore_done=False,
            # (float) Weight uniform initialization range in the last output layer.
            init_w=3e-3,
            # (int) The numbers of action sample each at every state s from a uniform-at-random.
            num_actions=10,
            # (bool) Whether use lagrange multiplier in q value loss.
            # with_lagrange=False,
            # (float) The threshold for difference in Q-values.
            # lagrange_thresh=-1,
            # (float) Loss weight for conservative item.
            # min_q_weight=1.0,
            # (bool) Whether to use entropy in target q.
            # with_q_entropy=False,
        ),
        eval=dict(),  # for compatibility
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.
        """
        return 'continuous_qvac', ['qvac']
    
    def _init_learn(self) -> None:
        # added to define the type of priority
        self._PER_type = self._cfg.learn.PER_type
        if self._PER_type == 'None':
            self._priority = False
            self._priority_IS_weight = False
        elif self._PER_type in ('Normal', 'Standard', 'AW', 'ODPR', 'Quantile'):
            self._priority = True
            self._priority_IS_weight = False
            if self._PER_type == 'ODPR':
                self._min_adv = 1e8
            elif self._PER_type == 'Quantile':
                try:
                    self._PER_tau = self._cfg.learn.PER_tau
                except:
                    raise ValueError(f"PER_tau for Quantile_PER is not defined!")
        else:
            raise ValueError(f"invalid PER type {self._PER_type}!")

        self._twin_critic = self._cfg.model.twin_critic
        self._num_actions = self._cfg.learn.num_actions
        # added to define the type of policy loss and maximum for clipping weights and priorities
        self._loss_type = self._cfg.learn.loss_type
        self._max_weight = self._cfg.learn.max_weight
        

        if hasattr(self._model.actor_head[-1], 'log_sigma_param'):
            torch.nn.init.constant_(self._model.actor_head[-1].log_sigma_param, self._cfg.learn.init_log_std)
        for m in list(self._model.actor.modules()):
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in self._model.actor.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

        self._optimizer_policy = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_policy,
        )

        # Algorithm config
        self._gamma = self._cfg.learn.discount_factor
        self._tau = self._cfg.learn.tau
        self._alpha = self._cfg.learn.alpha
        self._beta = torch.tensor(
            [self._cfg.learn.beta], requires_grad=False, device=self._device, dtype=torch.float32
        )
        self._anneal_step = self._cfg.learn.anneal_step

        # Main and target models
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='momentum',
            update_kwargs={'theta': self._cfg.learn.target_theta}
        )
        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()
        self._target_model.reset()

        self._forward_learn_cnt = 0

    def _forward_learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        '''
        This function is modified to optimize the policy network only, with diverse policy loss function (L2, L1, Huber, Skew, Flat) and types of priorities (None, Normal, Standard, AW, ODPR, Quantile) for optimizing policy and prioritized resampling 
        '''
        loss_dict = {}
        data = default_preprocess_learn(
            data,
            use_priority=self._priority,
            use_priority_IS_weight=self._cfg.priority_IS_weight,
            ignore_done=self._cfg.learn.ignore_done,
            use_nstep=False
        )
        if len(data.get('action').shape) == 1:
            data['action'] = data['action'].reshape(-1, 1)

        if self._cuda:
            data = to_device(data, self._device)

        self._learn_model.train()
        self._target_model.train()
        obs = data['obs']

        value = self._learn_model.forward(obs, mode='compute_value')['value']

        with torch.no_grad():
            target_value = self._target_model.forward(data, mode='compute_q')['q_value']
            if self._twin_critic:
                target_value = torch.min(target_value[0], target_value[1])
        
        (mu, sigma) = self._learn_model.forward(data['obs'], mode='compute_actor')['logit']
       
        if self._loss_type == 'L2':
            dist = Independent(Normal(mu, sigma), 1)
        elif self._loss_type == 'L1':
            dist = Independent(Laplace(mu, sigma), 1)
        elif self._loss_type == 'Huber':
            dist = Independent(Huber(mu, sigma), 1)
        elif self._loss_type == 'Skew':
            dist = Independent(Skew(mu, sigma), 1)
        elif self._loss_type == 'Flat':
            dist = Independent(Flat(mu, sigma), 1)
        else:
            raise ValueError("invalid Loss type {self._loss_type}")
        
        action = data['action']
        
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy_loss = dist.entropy().unsqueeze(-1)

        with torch.no_grad():

            adv = target_value - value.detach()
            
            advantage_avg = adv.mean()
            advantage_median = adv.median()
            advantage_min = adv.min()
            if self._PER_type == 'ODPR':
                self._min_adv = min(advantage_min, self._min_adv)
            advantage = (adv-advantage_avg)/(adv.std() + 1e-8)
            advantage = torch.exp(self._beta * advantage)
            advantage = torch.clamp(advantage, min=-torch.inf, max=self._max_weight)
        
        if self._forward_learn_cnt > self._anneal_step:
            policy_loss = -log_prob * advantage.detach()
        else:
            policy_loss = -log_prob
        
        loss_dict['policy_loss'] = policy_loss.mean() - self._alpha * entropy_loss.mean()
        
        self._optimizer_policy.zero_grad()
        loss_dict['policy_loss'].backward()
        self._optimizer_policy.step()

        loss_dict['total_loss'] = sum(loss_dict.values())

        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1
        # target update
        self._target_model.update(self._learn_model.state_dict())

        if self._PER_type == 'None':
            priority_per_sample = None
        elif self._PER_type == 'Normal':
            priority_per_sample = advantage.abs().tolist()
        elif self._PER_type == 'Standard':
            priority_per_sample = torch.exp(adv).tolist()
        elif self._PER_type == 'AW':
            priority_per_sample = (adv-advantage_avg)/(adv.std() + 1e-8)
            priority_per_sample = F.softmax(self._beta * priority_per_sample)*len(priority_per_sample)
            priority_per_sample = priority_per_sample.abs().tolist()
        elif self._PER_type == 'ODPR':
            priority_per_sample = (adv-self._min_adv).abs().tolist()
        elif self._PER_type == 'Quantile':
            priority_per_sample = (adv-adv.quantile(self._PER_tau))/(adv.std() + 1e-8)
            priority_per_sample = torch.exp(self._beta * priority_per_sample)
            priority_per_sample = torch.clamp(priority_per_sample, min=-torch.inf, max=self._max_weight)
            priority_per_sample = priority_per_sample.abs().tolist()
   

        return {
            'cur_lr_p': self._optimizer_policy.defaults['lr'],
            'priority': priority_per_sample,
            'policy_error': policy_loss.detach().mean().item(),
            'beta': self._beta.item(),
            'transformed_log_prob': log_prob.mean().item(),
            'advantage': advantage_avg.item(),
            'median_advantage': advantage_median.item(),
            'min_advantage': advantage_min.item(),
            **loss_dict
        }

    def _forward_collect(self, data: Dict[int, Any], **kwargs) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data for learn mode defined in ``self._process_transition`` method. The key of the \
                dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            ``logit`` in SAC means the mu and sigma of Gaussioan distribution. Here we use this name for consistency.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._collect_model.forward(data, mode='compute_actor')['logit']
            dist = Independent(Normal(mu, sigma), 1)
            action = dist.rsample()
            output = {'logit': (mu, sigma), 'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}
    
    def _forward_eval(self, data: Dict[int, Any]) -> Dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            ``logit`` in SAC means the mu and sigma of Gaussioan distribution. Here we use this name for consistency.

        .. note::
            For more detailed examples, please refer to our unittest for SACPolicy: ``ding.policy.tests.test_sac``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            (mu, sigma) = self._eval_model.forward(data, mode='compute_actor')['logit']
            action = mu
            output = {'action': action}
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}
    
    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        return [
            'policy_loss',
            'advantage',
            'median_advantage',
            'min_advantage',
            'cur_lr_p',
            'beta',
            'policy_error',
            'transformed_log_prob',
        ]
    
    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizers.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        ret = {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer_policy': self._optimizer_policy.state_dict(),
        }
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.

        .. tip::
            If you want to only load some parts of model, you can simply set the ``strict`` argument in \
            load_state_dict to ``False``, or refer to ``ding.torch_utils.checkpoint_helper`` for more \
            complicated operation.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
