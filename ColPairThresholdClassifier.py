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
        match type(data_cols):
            case pd.DataFrame:
                setattr(self, 'data_col_names', data_cols.columns)      # Names of all columns besides target
                setattr(self, 'target_col_name', target_col.name)       # Name of the target column
                setattr(self, 'class_names', target_col.unique())       # Unique values represented in the target column
            case np.ndarray:
                setattr(self, 'data_col_names', [f'col{x}' for x in range(len(data_cols[0]))])      # Names of all columns besides target
                setattr(self, 'target_col_name', 'target')       # Name of the target column
                class_names = []
                for val in np.unique(target_col):
                    self.class_map[f'col{val}'] = val           # This is an artifact of how I initially implemented clashing with sklearn standards
                    class_names.append(val)
                setattr(self, 'class_names', class_names)      # Unique values represented in the target column
                
            case _:
                raise(TypeError('First argument must be of type pd.DataFrame or np.ndarray'))

        target_col = 
        print(self.class_names)
        print(target_col)
        #setattr(self, 'data_col_names', data_cols.columns)      # Names of all columns besides target
        #setattr(self, 'target_col_name', target_col.name)       # Name of the target column
        #setattr(self, 'species_vals', target_col.unique())      # Unique values represented in the target column



        for col_1 in range(len(self.data_col_names)):
            for col_2 in range(col_1, len(self.data_col_names)):
                start_time = time.time()
                lines = get_best_lines(data_cols[self.data_col_names[col_1]], data_cols[self.data_col_names[col_2]], target_col, self.class_names)
                self.model[(self.data_col_names[col_1], self.data_col_names[col_2])] = lines
                end_time = time.time()
                if timer:
                    print(f'Line fitting took {end_time - start_time} seconds for column pair {self.data_col_names[col_1]}, {self.data_col_names[col_2]}')

        if timer:
            print("Model loaded")

    def predict(self, data_cols):
        return [self.predict_on_row(row) for index, row in data_cols.iterrows()]


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