from gym.envs.registration import register

register(
    id='cavity-v0',
    entry_point='gym_feedback.envs:CavityEnv',
)

register(
    id='optomech-v0',
    entry_point='gym_feedback.envs:OptomechEnv',
)
