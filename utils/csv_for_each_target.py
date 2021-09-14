import pandas as pd

df = pd.read_csv("../data/training_set_features.csv")


target = pd.read_csv("../data/training_set_labels.csv")

full_df = pd.merge(df, target, on="respondent_id")

df_h1n1 = full_df.drop(["seasonal_vaccine"], axis=1)
df_seasonal = full_df.drop(["h1n1_vaccine"], axis=1)

df_h1n1.to_csv("../data/training_h1n1.csv", index=False)
df_seasonal.to_csv("../data/training_seasonal.csv", index=False)