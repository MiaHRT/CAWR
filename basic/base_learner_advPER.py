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

from ding.utils import LEARNER_REGISTRY
from ding.worker import BaseLearner
from typing import Optional

@LEARNER_REGISTRY.register('base_learner_advPER')
class BaseLearner(BaseLearner):

    def train(self, data: dict, envstep: int = -1, policy_kwargs: Optional[dict] = None) -> None:
        assert hasattr(self, '_policy'), "please set learner policy"
        self.call_hook('before_iter')

        if policy_kwargs is None:
            policy_kwargs = {}

        # Forward
        log_vars = self._policy.forward(**data, **policy_kwargs)

        # Update replay buffer's priority info
        if isinstance(log_vars, dict):
            priority = log_vars.pop('priority', None)
            critic_adv = log_vars.pop('critic_adv', None) # modified to collect advantage data per train iteration
        elif isinstance(log_vars, list):
            priority = log_vars[-1].pop('priority', None)
        else:
            raise TypeError("not support type for log_vars: {}".format(type(log_vars)))
        if priority is not None: # modified to update priorities for two decoupled resampled batches
            replay_buffer_idx = [d.get('replay_buffer_idx', None) for d in data['actor_data']]
            replay_unique_id = [d.get('replay_unique_id', None) for d in data['actor_data']]
            replay_buffer_idx += [d.get('replay_buffer_idx', None) for d in data['critic_data']]
            replay_unique_id += [d.get('replay_unique_id', None) for d in data['critic_data']]
            self.priority_info = {
                'priority': priority,
                'replay_buffer_idx': replay_buffer_idx,
                'replay_unique_id': replay_unique_id,
            }
        # Discriminate vars in scalar, scalars and histogram type
        # Regard a var as scalar type by default. For scalars and histogram type, must annotate by prefix "[xxx]"
        self._collector_envstep = envstep
        if isinstance(log_vars, dict):
            log_vars = [log_vars]
        for elem in log_vars:
            scalars_vars, histogram_vars = {}, {}
            for k in list(elem.keys()):
                if "[scalars]" in k:
                    new_k = k.split(']')[-1]
                    scalars_vars[new_k] = elem.pop(k)
                elif "[histogram]" in k:
                    new_k = k.split(']')[-1]
                    histogram_vars[new_k] = elem.pop(k)
            # Update log_buffer
            self._log_buffer['scalar'].update(elem)
            self._log_buffer['scalars'].update(scalars_vars)
            self._log_buffer['histogram'].update(histogram_vars)

            self.call_hook('after_iter')
            self._last_iter.add(1)

        return log_vars
