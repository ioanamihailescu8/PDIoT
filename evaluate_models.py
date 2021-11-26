import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import tensorflow as tf

import os

import matplotlib.pyplot as plt

# keras goodies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import metrics as kmetrics
import tensorflow.keras.backend as K


import seaborn as sns


class TFLiteModel:

    
    def __init__(self,model_path):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


        #Define the specific columns that are of interest to us. To be used for generating sliding windows and passed to the model
        self.columns_of_interest_training = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']  #Subject id is of interest to us as we will split our dataset by subject_id
        self.columns_of_interest_initial = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z','subject_id','activity_code', 'activity_type','recording_id']

        self.grouped_falling_label = 'Falling (Grouped)'
        self.grouped_lying_label = 'Lying (Grouped)'
        self.grouped_sitting_or_standing_label = 'Sitting/Standing'
        self.class_labels= {'Falling (Grouped)': 0, 
            'Lying (Grouped)': 1, 
            'Running': 2,
            'Sitting/Standing': 3, 
            'Walking at normal speed': 4
            }
        self.window_size=25
        self.step_size=25


    def give_predictions(self,X_test):
        predictions = []
        X_test = X_test.astype(np.float32)
        for i in range(X_test.shape[0]):
            self.interpreter.set_tensor(self.input_details[0]["index"], X_test[i:i+1, :, :])
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            pred = np.squeeze(output_data)
            predictions.append(pred)



        return predictions

  



    def evaluate(self,y_pred_labels, y_true_labels):
        cm_normalised = self.plot_confusion_matrix(y_pred_labels,y_true_labels)
        self.show_classification_report(y_pred_labels, y_true_labels)
        self.show_metrics_per_class(y_pred_labels, y_true_labels, cm_normalised)


    def plot_confusion_matrix(self,y_pred_labels, y_true_labels, save_to_folder=True):
        cm = confusion_matrix(y_true_labels,y_pred_labels)
        #Normalise confusion matrix
        cm_normalised = np.around(cm / cm.astype(np.float32).sum(axis=1) , decimals=2)

        fig, ax= plt.subplots(figsize=(25, 25))
            # sns.set(font_scale=1.2)
        sns.heatmap(cm_normalised, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.tick_params(axis='both', which='major', labelsize=12)  # Adjust to fit
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(self.class_labels); ax.yaxis.set_ticklabels(self.class_labels)

        if save_to_folder:
            plt.savefig('confusion_matrix.png',dpi=300)

        return cm_normalised

    def show_classification_report(self,y_pred_labels, y_true_labels):
        print(classification_report(y_pred_labels, y_true_labels))

    def show_metrics_per_class(self,y_predictions,y_truth,cm_normalised):
        accuracies = cm_normalised.diagonal()
        precisions = precision_score(y_truth, y_predictions, average=None)
        recalls = recall_score(y_truth, y_predictions, average=None)
        f1_scores = f1_score(y_truth, y_predictions, average=None)
        for act, label in self.class_labels.items():
            print(f"{act}..... Accuracy: {accuracies[label]:.2f}, Precision: {precisions[label]:.2f}, Recall: {recalls[label]:.2f}, F-score: {f1_scores[label]:.2f}")
            


        

    
    ################# Utilities to prepare the dataset for the model ################
    def prepare_dataset(self,base_df):
        #Drop any NaN values from the test data
        base_df = base_df.dropna(subset=self.columns_of_interest_initial).reset_index(drop=True)

        #Remove some activities. Current model classifies a subset of the collected data
        base_df = self.remove_activity_label(base_df,'Movement')
        base_df =  self.remove_activity_label(base_df,'Desk work')
        base_df =  self.remove_activity_label(base_df,'Climbing stairs')
        base_df =  self.remove_activity_label(base_df,'Descending stairs')


        #Group activities together
        base_df = self.group_falling_together(base_df)
        base_df = self.group_lying_together(base_df)
        base_df = self.group_sitting_and_standing_together(base_df)
          

        #Group given recordings into window sizes
        sliding_windows = self.group_into_sliding_windows(base_df)
        X_list, y_list = self.regenerate_data_from_sliding_windows(sliding_windows)

        # Convert to numpy, and convert categorical variable (labels) into dummy/indicator variables (one-hot encoding)
        X = np.asarray(X_list)
        y = np.asarray(pd.get_dummies(y_list), dtype=np.float32)

        return X, y

    ####### Merging and removing unused labels. Current model classifies a subset of data #######
    def remove_activity_label(self, df, activity_to_remove):
        df = df[df.activity_type!=activity_to_remove]
        return df


    def grouping_activities_together(self, base_df, individual_activities_to_group, new_label):
        grouped_dataframes = []
        for act, group in base_df.groupby("activity_type"):
            if act in individual_activities_to_group:
                group['activity_type'] = new_label

            grouped_dataframes.append(group)
        base_df = pd.concat(grouped_dataframes)
        return base_df

    def group_falling_together(self, base_df):
        falling_activities= ['Falling on knees', 'Falling on the back', 'Falling on the left', 'Falling on the right']
        base_df = self.grouping_activities_together(base_df, falling_activities, self.grouped_falling_label)
        return base_df
    

    def group_lying_together(self, base_df):
        lying_activities = ['Lying down left', 'Lying down right', 'Lying down on back', 'Lying down on stomach']
        base_df = self.grouping_activities_together(base_df, lying_activities, self.grouped_lying_label)
        return base_df
        
    def group_sitting_and_standing_together(self, base_df):
        sitting_or_standing_activites= ['Sitting','Standing','Sitting bent forward', 'Sitting bent backward']
        base_df = self.grouping_activities_together(base_df, sitting_or_standing_activites, self.grouped_sitting_or_standing_label)
        return base_df

    ####### Preprocessing ############

    def group_into_sliding_windows(self, df):

        window_number = 0 # start a counter at 0 to keep track of the window number

        all_overlapping_windows = []


        for rid, group in df.groupby("recording_id"):
            large_enough_windows = [window for window in group.rolling(window=self.window_size, min_periods=self.window_size) if len(window) == self.window_size]

            overlapping_windows = large_enough_windows[::self.step_size] 
            # then we will append a window ID to each window
            if overlapping_windows:
                for window in overlapping_windows:
                    window.loc[:, 'window_id'] = window_number
                    window_number += 1


                all_overlapping_windows.append(pd.concat(overlapping_windows).reset_index(drop=True))
            
        final_sliding_windows = pd.concat(all_overlapping_windows).reset_index(drop=True)
        

        return final_sliding_windows

    def regenerate_data_from_sliding_windows(self,final_sliding_windows):
        X= []
        y= []

        for window_id, group in final_sliding_windows.groupby('window_id'):
            
    #         print(f"window_id = {window_id}")

            shape = group[self.columns_of_interest_training].values.shape

            X.append(group[self.columns_of_interest_training].values )
            y.append(self.class_labels[group["activity_type"].values[0]])
        
        return X,y

    

    
    

  

if __name__ == "__main__":    
    #Prepare (Respeck) data and model. 
    key = ["--model_path=", "--test_data_path="]
    model_path = str(sys.argv[1][len(key[0]):])
    test_data_path = str(sys.argv[2][len(key[1]):])

    #Load model (currently a CNN model)
    model = TFLiteModel(model_path)
    #Load test data
    base_df = pd.read_csv(test_data_path)
    X,y = model.prepare_dataset(base_df)
    #Generate Predictions

    y_pred_ohe = model.give_predictions(X)
    y_pred_labels = np.argmax(y_pred_ohe, axis=1)
    y_true_labels = np.argmax(y, axis=1)

    #Evaluate model
    model.evaluate(y_pred_labels,y_true_labels)








    

    