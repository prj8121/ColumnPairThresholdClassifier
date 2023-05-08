# Parker Johnson

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from bestLineFinder import get_best_lines
from sklearn.base import BaseEstimator, ClassifierMixin

class ColPairThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = {}
        
    def fit(self, data_cols, target_col, timer=False):
        if data_cols.size == 0:
            raise ValueError("data_cols must not be empty")
        if target_col.size == 0:
            raise ValueError("target_col must not be empty")
        
        num_cols = len(data_cols[0])

        for col_1 in range(num_cols):
            for col_2 in range(col_1, num_cols):
                start_time = time.time()

                #print([data_cols[:, col_1], data_cols[:, col_2], target_col, np.unique(target_col)])
                # [f'class{x}' for x in np.unique(target_col)]
                lines = get_best_lines(data_cols[:, col_1], data_cols[:, col_2], target_col, np.unique(target_col))
                self.model[(col_1, col_2)] = lines
                print("Got through")
                end_time = time.time()
                if timer:
                    print(f'Line fitting took {end_time - start_time} seconds for column pair {col_1}, {col_2}')

        if timer:
            print("Model loaded")


    def predict(self, data_cols):
        if data_cols.size == 0:
            raise ValueError("data_cols must not be empty")
        
        result = []

        num_cols = len(data_cols[0])
        
        for row in data_cols:
            for col_1 in range(num_cols):
                for col_2 in range(col_1, num_cols):
                    lines = self.model[(col_1, col_2)]

                    row_x = row[col_1]
                    row_y = row[col_2]

                    prediction = {}
                
                    for c in lines.keys():
                        
                        above = 1 if row_y > (lines[c]['slope'] * row_x) + lines[c]['intercept'] else -1
                        direction = 1 if lines[c]['direction'] else -1
                        if prediction.get(c):
                            prediction[c] = prediction[c] + (lines[c]['score'] * direction * above)
                        else:
                            prediction[c] = lines[c]['score'] * direction * above
            c_with_highest_score = max(prediction, key=prediction.get)
            result.append(c_with_highest_score)

        return result
        


    def predict_on_row(self, row: pd.Series):
        # if self.target_col_name in row:
        #    row = row.drop(self.target_col_name)

        row_keys = row.keys()
        
        for col_1_i in range(len(row_keys)):
            for col_2_i in range(col_1_i, len(row_keys)):
                col_1 = row_keys[col_1_i]
                col_2 = row_keys[col_2_i]
                lines = self.model[(col_1, col_2)]
                
                row_x = row[col_1]
                row_y = row[col_2]
                
                prediction = {}
                
                for c in lines.keys():
                    
                    above = 1 if row_y > (lines[c]['slope'] * row_x) + lines[c]['intercept'] else -1
                    direction = 1 if lines[c]['direction'] else -1
                    if prediction.get(c):
                        prediction[c] = prediction[c] + (lines[c]['score'] * direction * above)
                    else:
                        prediction[c] = lines[c]['score'] * direction * above
        
        c_with_highest_score = max(prediction, key=prediction.get)
        return c_with_highest_score