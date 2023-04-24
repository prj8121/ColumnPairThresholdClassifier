# Parker Johnson

import pandas as pd
import time
import matplotlib.pyplot as plt
from bestLineFinder import get_best_lines

class ColPairThresholdClassifier:
    def __init__(self):
        self.model = {}
        

    def load_data(self, df, target, timer=False):
        setattr(self, 'data_cols', df.columns.drop(target)) # Names of all columns besides target
        setattr(self, 'target_col', target)                 # Name of the target column
        setattr(self, 'species_vals', df[target].unique())  # Unique values represented in the target column

        if not target in df.columns: # TODO make this a real error
            print("Target does not exist in the dataframe")
            return

        for col_1 in range(len(self.data_cols)):
            for col_2 in range(col_1, len(self.data_cols)):
                start_time = time.time()
                lines = get_best_lines(df[self.data_cols[col_1]], df[self.data_cols[col_2]], df[target], self.species_vals)
                self.model[(self.data_cols[col_1], self.data_cols[col_2])] = lines
                end_time = time.time()
                if timer:
                    print(f'Line fitting took {end_time - start_time} seconds for column pair {self.data_cols[col_1]}, {self.data_cols[col_2]}')

        if timer:
            print("Model loaded")


    def plot_data(self, df, raw_data):
        if not self.model:
            print("No model Error") #TODO actually throw an error 
            return

        data_cols = self.data_cols

        # TODO deal with color mapping 
        species_vals = self.species_vals
        color_map = {'Iris-setosa':(1,0,0), 'Iris-versicolor':(0,1,0), 'Iris-virginica':(0,0,1)}
        colors = raw_data['species'].map(color_map)

        plt.figure(figsize=(12, 12))

        for col_1 in range(len(data_cols)):
            for col_2 in range(col_1, len(data_cols)):
                lines = self.model[(data_cols[col_1], data_cols[col_2])]

                points = {}
                x_min = df[data_cols[col_1]].min()
                x_max = df[data_cols[col_1]].max()
                y_min = df[data_cols[col_2]].min()
                y_max = df[data_cols[col_2]].max()

                for c in lines.keys():

                    p_left = (x_min, (lines[c].slope * x_min) + lines[c].intercept)
                    p_right = (x_max, (lines[c].slope * x_max) + lines[c].intercept)
                    p_top = ((y_max - lines[c].intercept) / lines[c].slope , y_max)
                    p_bottom = ((y_min - lines[c].intercept) / lines[c].slope , y_min)

                    p_list = set([p_left, p_right, p_top, p_bottom])

                    res = []
                    for p in p_list:
                        if x_min <= p[0] and p[0] <= x_max:
                            if y_min <= p[1] and p[1] <= y_max:
                                res.append(p)

                    points[c] = res

                plt.subplot(4,4, (col_1 * 4) + col_2 + 1)
                plt.scatter(df[data_cols[col_1]], df[data_cols[col_2]], s=7, c=colors)
                for c in lines.keys():
                    plt.plot([points[c][0][0], points[c][1][0]], [points[c][0][1], points[c][1][1]], c=color_map[c], marker= '^' if lines[c].direction else 'v')
                plt.title(f'{data_cols[col_1]}, {data_cols[col_2]}')


    def test(self, test_data: pd.DataFrame):
        if not self.model:
            print("No model Error") #TODO actually throw an error 
            return []
        for index, row in test_data.iterrows():
            prediction = self.predict_on_row(row)
            print(prediction)
    

    def predict_on_row(self, row: pd.Series):
        if self.target_col in row:
            row = row.drop(self.target_col)

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