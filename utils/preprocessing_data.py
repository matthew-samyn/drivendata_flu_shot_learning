from functions import *

pd.set_option("display.max_rows",999)
pd.set_option("display.max_columns", 999)
df = pd.read_csv("../data/training_set_features.csv")


target = pd.read_csv("../data/training_set_labels.csv")

full_df = pd.merge(df, target, on="respondent_id")

full_df.to_csv("../data/untouched_with_labels.csv", index=False)


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
print(full_df.dtypes)

# Creating seperate dataframe for each model
df_h1n1 = full_df.drop(["seasonal_vaccine"], axis=1)
df_seasonal = full_df.drop(["h1n1_vaccine"], axis=1)

# After exploring the data, decided on what columns could get dropped
columns_to_drop_h1n1 = ["behavioral_antiviral_meds","behavioral_large_gatherings", "behavioral_outside_home",
                        "opinion_seas_risk", "opinion_seas_sick_from_vacc", "opinion_seas_vacc_effective",
                        "household_adults", "household_children", "doctor_recc_seasonal", "opinion_h1n1_sick_from_vacc"]

columns_to_drop_seasonal = ["doctor_recc_h1n1", "child_under_6_months", "opinion_h1n1_vacc_effective", "opinion_h1n1_risk",
                            "opinion_h1n1_sick_from_vacc","opinion_seas_sick_from_vacc","household_adults","household_children"]

df_h1n1 = df_h1n1.drop(columns_to_drop_h1n1, axis=1)
df_seasonal = df_seasonal.drop(columns_to_drop_seasonal, axis=1)

# Creating csv for each model
df_h1n1.to_csv("../data/training_h1n1.csv", index=False)
df_seasonal.to_csv("../data/training_seasonal.csv", index=False)