# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_cql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="halfcheetah_medium_expert_cawr_seed0",
    env=dict(
        env_id='halfcheetah-medium-expert-v2',
        collector_env_num=1,
        evaluator_env_num=1,
        #norm_obs=dict(
        #    use_norm=True,
        #    offline_stats=dict(use_offline_stats=True, ),
        #),
        use_act_scale=True,
        n_evaluator_episode=1,
        stop_value=60000,
    ),
    policy=dict(
        cuda=True,
        priority = True,
        priority_IS_weight=False,
        model=dict(
            obs_shape=17,
            action_shape=6,
        ),
        learn=dict(
            data_path=None,
            shuffle=True,
            batch_size=512,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_value=3e-4,
            loss_type='L2',
            PER_type='Normal',
            tau=0.7,
            alpha=0.0,
            beta=3.0,
            max_weight=100,
            anneal_step=0,
        ),
        collect=dict(data_type='d4rl', data_path='../../data/', n_sample=2048, ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='base'),
    model=dict(
        type='continuous_qvac',
        import_names=['qvac'],
    ),
    policy=dict(
        type='cawr_advPER',
        import_names=['cawr_advPER'],
    ),
    replay_buffer=dict(type='advanced', 
                       ),
    learner=dict(
        type='base_learner_advPER',
        import_names=['base_learner_advPER'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
