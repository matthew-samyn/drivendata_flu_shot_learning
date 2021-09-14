from functions import *

df = pd.read_csv("../data/training_set_features.csv")


target = pd.read_csv("../data/training_set_labels.csv")

full_df = pd.merge(df, target, on="respondent_id")

# 2-category columns to binary column
full_df["sex"] = df["sex"].apply(lambda x: category_to_binary(x,"Male"))
full_df["marital_status"] = df["marital_status"].apply(lambda x:
                                                  category_to_binary(x, "Not Married"))

full_df["rent_or_own"] = df["rent_or_own"].apply(lambda x:
                                            category_to_binary(x, "Rent"))

# get_dummies on multiple-category-columns
types = ["int64","float64"]
categorical = []
numerical = []
for column in full_df.columns:
    if seperate_columns_by_type(full_df[column], types):
        numerical.append(column)
    else:
        categorical.append(column)

for column in categorical:
    full_df = dummies_into_dataframe(full_df, column)

# Creating seperate dataframe for each model
df_h1n1 = full_df.drop(["seasonal_vaccine"], axis=1)
df_seasonal = full_df.drop(["h1n1_vaccine"], axis=1)

# Creating csv for each model
df_h1n1.to_csv("../data/training_h1n1.csv", index=False)
df_seasonal.to_csv("../data/training_seasonal.csv", index=False)