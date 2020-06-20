import gym
from gym.envs.registration import registry, make, spec


def register(id,*args,**kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id,*args,**kvargs)

## E2E Control Envs

register(
    id='Gripper2D-v0',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DEnv',
    max_episode_steps=200,
    kwargs=dict(reward_type='dense')
)

register(
    id='Gripper2D-v1',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DEnv',
    max_episode_steps=200,
    kwargs=dict(reward_type='contacts')
)

for diff_lev in ['Easy', 'Med', 'Hard']:
    for env_type in ['SamplePose', 'Goal']:
        vctr = 0
        for drop_reset in [True, False]:
            for reward_type in ['sparse', 'dense', 'contacts']:
                if env_type == 'Goal':
                    envcls = 'Gripper2DGoalEnv'
                elif diff_lev != 'Hard':
                    envcls = 'Gripper2DSamplePoseEnv'
                else:
                    envcls = 'Gripper2DHardSamplePoseEnv'
                register(
                    id='Gripper2D{}{}-v{}'.format(env_type, diff_lev, vctr),
                    entry_point='three_finger.envs.raw_controller_env:{}'.format(envcls),
                    max_episode_steps=200,
                    kwargs=dict(reward_type=reward_type, reset_on_drop=drop_reset,
                            goal_difficulty=diff_lev.lower())
                )
                vctr += 1

register(
    id='Gripper2DGoal-v0',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DGoalEnv',
    max_episode_steps=200
)

register(
    id='Gripper2DGoal-v1',
    entry_point='three_finger.envs.raw_controller_env:Gripper2DGoalEnv',
    kwargs=dict(reset_on_drop=True),
    max_episode_steps=200
)

## Control w/ Primitives Envs

register(
    id='Gripper2DPrimitiveDiscreteControlHard-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteFullEnvV1',
    max_episode_steps=200.,
    reward_threshold=200.,
)

register(
    id='Gripper2DPrimitiveDiscreteControlHardGravity-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteFullEnvV1',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(gravity=-9.8)
)

register(
    id='Gripper2DPrimitiveDiscreteControlHardWithError-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteFullEnvV1',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(measurement_err=True)
)

register(
    id='Gripper2DPrimitiveDiscreteControlHardWoFeasibleCheck-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteFullEnvV1',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(is_feasible_check=False)
)

register(
    id='Gripper2DPrimitiveDiscreteControlMedium-v0',
    entry_point='envs.low_level_controller_env:Gripper2DPrimitiveDiscreteReposeSlideEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
)

register(
    id='Gripper2DPrimitiveDiscreteControlMediumGravity-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteReposeSlideEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(gravity=-9.8)
)

register(
    id='Gripper2DPrimitiveDiscreteControlMediumWoFeasibleCheck-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteReposeSlideEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(is_feasible_check=False)
)

register(
    id='Gripper2DPrimitiveDiscreteControlMediumWithError-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteReposeSlideEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(measurement_err=True)
)

register(
    id='Gripper2DPrimitiveDiscreteControlMediumGravityWithError-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteReposeSlideEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(measurement_err=True, gravity=-9.8)
)

register(
    id='Gripper2DPrimitiveDiscreteControlEasy-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteReposeEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
)

register(
    id='Gripper2DPrimitiveDiscreteControlEasyGravity-v0',
    entry_point='envs.low_level_controller_env:Finger3LowLevelControlDiscreteReposeEnv',
    max_episode_steps=200.,
    reward_threshold=200.,
    kwargs=dict(gravity=-9.8)
)

register(
    id='Gripper2DPrimitiveContinuousControl-v0',
    entry_point='envs.low_level_controller_env:Gripper2DPrimitiveContEnv',
    max_episode_steps=200.,
    kwargs=dict(repose_duration=60, slide_duration=60, delta_repose=[0.04,0.04,0.2],
                delta_slide=[0.04,0.04,0.04], distance_threshold=0.05,
                orientation_threshold=0.2, rew_min_clip=-2)
)
