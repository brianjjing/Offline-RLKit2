import pandas as pd

df = pd.read_csv('/home/brian/repos/OfflineRL-Kit2/log/abiomed/combo/seed_42&timestamp_26-0724-043758/record/policy_training_progress.csv')
print(df['eval/normalized_episode_reward'].mean()/100)

df = pd.read_csv('/home/brian/repos/OfflineRL-Kit2/log/abiomed/combo/seed_123&timestamp_26-0724-175210/record/policy_training_progress.csv')
print(df['eval/normalized_episode_reward'].mean()/100)

df = pd.read_csv('/home/brian/repos/OfflineRL-Kit2/log/abiomed/combo/seed_456&timestamp_26-0725-055848/record/policy_training_progress.csv')
print(df['eval/normalized_episode_reward'].mean()/100)