from typing import List
import pandas as pd
import numpy as np

def seperate_columns_by_type(df_series:pd.Series, types:List) -> bool:
    """ Seperates dataframe columns into categorical and numerical """
    if df_series.dtype in types:
        return True
    else:
        return False

def category_to_binary(x:str, category_name:str) -> int:
    """ """
    check_for_none = [None, np.nan]
    if x in check_for_none:
        return None
    elif x == category_name:
        return 0
    else:
        return 1

def dummies_into_dataframe(df: pd.DataFrame, column:str) -> pd.DataFrame:
    df_with_dummies = pd.get_dummies(df[column])
    df = df.drop([column], axis=1)
    df = pd.merge(df, df_with_dummies, left_index=True, right_index=True)
    return df

