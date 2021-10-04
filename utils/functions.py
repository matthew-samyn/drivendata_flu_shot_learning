from typing import List,Tuple
import pandas as pd
import numpy as np
import os
import re


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
    """ Uses pd.get_dummies on column.
    Automatically merges it back with the original DataFrame.

    :return original DataFrame with added dummie columns.
    """
    df_with_dummies = pd.get_dummies(df[column])
    df = df.drop([column], axis=1)
    df = pd.merge(df, df_with_dummies, left_index=True, right_index=True)
    return df


def get_score(h5_file:str) -> Tuple[float, float]:
    """ Extracts useful data from a .h5 file

    :param h5_file: File to extract the train and val accuracy scores from
    :return: train accuracy, difference between train and validation
    """
    scores = re.findall(r"\d+\.\d+",h5_file)
    train_accuracy = float(scores[0])
    val_accuracy = float(scores[1])
    difference = train_accuracy - val_accuracy
    return val_accuracy, difference


def keep_best_saved_h5(folder_relative:str, common_filename: str, maximum_difference:float) -> str:
    """ Goes through all common named .h5-files,
    deletes all from folder except for the best result.

    :param maximum_difference: max difference allowed between accuracy and val_accuracy.
    :return Best scoring .h5 file
    """
    current_directory = os.getcwd()
    os.chdir(current_directory + folder_relative)
    all_files = os.listdir()
    best_scoring_file = ""
    # try-except incase of errors: returns to current directory
    try:
        # Keep the files with a low difference between train_accuracy and validation_accuracy.
        # Deletes the rest from directory
        model_files = [file for file in all_files if file.startswith(common_filename)]
        scores = []
        not_overfitting_models = []
        best_scoring_file = ""
        for file in model_files:
            validation_accuracy, diff = get_score(file) # Uses function get_score()
            if abs(diff) > maximum_difference:
                os.remove(file)
            else:
                not_overfitting_models.append(file)
                scores.append(validation_accuracy)

        # Keep only the file with highest validation accuracy score.
        # Deletes the rest from directory
        highest_score_index = scores.index(max(scores))
        for i, file in enumerate(not_overfitting_models):
            if i == highest_score_index:
                best_scoring_file = file
            else:
                os.remove(file)
        os.chdir(current_directory)
    except:
        os.chdir(current_directory)
    print(f"Currently in directory:{os.getcwd()}")
    print(f"File coming out of the function: {best_scoring_file}")
    return best_scoring_file

