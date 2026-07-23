import pandas as pd

df = pd.read_csv('/home/brian/repos/OfflineRL-Kit2/log/hopper-medium-expert-v2/combo/seed_42&timestamp_26-0717-015845_sparse/record/policy_training_progress.csv')
print(df['eval/normalized_episode_reward'].mean())