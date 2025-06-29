from typing import Union, Dict, Optional
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn

from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ding.model.common import RegressionHead, ReparameterizationHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder


@MODEL_REGISTRY.register('continuous_qvac')
class ContinuousQVAC(nn.Module):
    """
    Overview:
        The neural network and computation graph of algorithms related to IQL.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_q``, ``compute_value``
    """
    mode = ['compute_actor', 'compute_q', 'compute_value']

    def __init__(
        self,
        obs_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: str,
        twin_critic: bool = False,
        actor_head_hidden_size: int = 64,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
        encoder_hidden_size_list: Optional[SequenceType] = None,
        share_encoder: Optional[bool] = False,
    ) -> None:
        super(ContinuousQVAC, self).__init__()
        obs_shape: int = squeeze(obs_shape)
        action_shape = squeeze(action_shape)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        self.action_space = action_space
        assert self.action_space == 'reparameterization', self.action_space

        # encoder
        self.share_encoder = share_encoder
        if np.isscalar(obs_shape) or len(obs_shape) == 1:
            assert not self.share_encoder, "Vector observation doesn't need share encoder."
            assert encoder_hidden_size_list is None, "Vector obs encoder only uses one layer nn.Linear"
            # Because there is already a layer nn.Linear in the head, so we use nn.Identity here to keep
            # compatible with the image observation and avoid adding an extra layer nn.Linear.
            self.actor_encoder = nn.Identity()
            self.q_encoder = nn.Identity()
            self.value_encoder = nn.Identity()
            encoder_output_size = obs_shape
        else:
            raise RuntimeError("not support observation shape: {}".format(obs_shape))
        
        # head
        self.twin_critic = twin_critic
        critic_input_size = encoder_output_size + action_shape
        if self.twin_critic:
            self.q_head = nn.ModuleList()
            for _ in range(2):
                self.q_head.append(
                    nn.Sequential(
                        nn.Linear(critic_input_size, critic_head_hidden_size), activation,
                        RegressionHead(
                            critic_head_hidden_size,
                            1,
                            critic_head_layer_num,
                            final_tanh=False,
                            activation=activation,
                            norm_type=norm_type
                        )
                    )
                )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(critic_input_size, critic_head_hidden_size), activation,
                RegressionHead(
                    critic_head_hidden_size,
                    1,
                    critic_head_layer_num,
                    final_tanh=False,
                    activation=activation,
                    norm_type=norm_type
                )
            )
        
        self.value_head = nn.Sequential(
            nn.Linear(encoder_output_size, critic_head_hidden_size), activation,
            RegressionHead(
                critic_head_hidden_size,
                1,
                critic_head_layer_num,
                final_tanh=False,
                activation=activation,
                norm_type=norm_type
            )
        )

        #self.actor_head = nn.Sequential(
        #    nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
        #    ReparameterizationHead(
        #        actor_head_hidden_size,
        #        action_shape,
        #        actor_head_layer_num,
        #        sigma_type='independent',
        #        activation=activation,
        #        norm_type=norm_type
        #    )
        #)
        self.actor_head = nn.Sequential(
            nn.Linear(encoder_output_size, actor_head_hidden_size), activation,
            ReparameterizationHead(
                actor_head_hidden_size,
                action_shape,
                actor_head_layer_num,
                sigma_type='fixed',
                fixed_sigma_value=0.135,
                activation=activation,
                norm_type=norm_type
            )
        )

        self.actor = nn.ModuleList([self.actor_encoder, self.actor_head])
        self.q = nn.ModuleList([self.q_encoder, self.q_head])
        self.value = nn.ModuleList([self.value_encoder, self.value_head])

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], mode: str) -> Dict[str, torch.Tensor]:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)
    
    def compute_actor(self, obs: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        obs = self.actor_encoder(obs)
        x = self.actor_head(obs)
        return {'logit': [x['mu'], x['sigma']]}
    
    def compute_q(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs, action = inputs['obs'], inputs['action']
        obs = self.q_encoder(obs)
        assert len(obs.shape) == 2

        if len(action.shape) == 1:  # (B, ) -> (B, 1)
            action = action.unsqueeze(1)
        x = torch.cat([obs, action], dim=1)

        if self.twin_critic:
            x = [m(x)['pred'] for m in self.q_head]
        else:
            x = self.q_head(x)['pred']
        return {'q_value': x}
    
    def compute_value(self, obs: torch.Tensor) -> Dict:
        obs = self.value_encoder(obs)
        x = self.value_head(obs)
        return {'value': x['pred']}
