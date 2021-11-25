import sys
import pandas as pd
import numpy as np
import tsfresh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

import os

import matplotlib.pyplot as plt

# keras goodies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, MaxPooling1D, BatchNormalization, LSTM, TimeDistributed
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import metrics as kmetrics
import tensorflow.keras.backend as K


import seaborn as sns


def remove_activity_label(df, activity_to_remove):
    df = df[df.activity_type!=activity_to_remove]
    return df


if __name__ == "__main__":
    key = ["--model_path=", "--test_data_path="]
    model_path = str(sys.argv[1][len(key[0]):])
    test_data_path = str(sys.argv[2][len(key[1]):])

    base_df_old = pd.DataFrame()

    clean_data_folder = test_data_path
    base_df = pd.read_csv(clean_data_folder)

    columns_of_interest_initial = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z','subject_id','activity_code', 'activity_type','recording_id']
    base_df = base_df.dropna(subset=columns_of_interest_initial).reset_index(drop=True)

    base_df = remove_activity_label(base_df,'Movement')
    base_df =  remove_activity_label(base_df,'Desk work')
    base_df =  remove_activity_label(base_df,'Climbing stairs')
    base_df =  remove_activity_label(base_df,'Descending stairs')

    

    