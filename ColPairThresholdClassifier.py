# Parker Johnson

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import colorsys
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
                if timer:
                    start_time = time.time()

                lines = get_best_lines(data_cols[:, col_1], data_cols[:, col_2], target_col, np.unique(target_col))
                self.model[(col_1, col_2)] = lines

                if timer:
                    end_time = time.time()
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
    
    def plot(self, data_cols: np.ndarray, target_col: np.ndarray, column_labels: list, class_labels: list):
        #if not self.model:
        #   print("No model Error") #TODO actually throw an error 
        #   return

        #data_cols = self.data_cols

        # TODO deal with color mapping 
        
        class_map = {x:y for x,y in enumerate(class_labels)}
        num_classes = len(class_labels)
        num_columns = data_cols.shape[1]
        
        color_list = [colorsys.hsv_to_rgb(hue/num_classes, 1, 1) for hue in range(num_classes)]
        color_map = {i:color_list[i] for i in range(num_classes)}
        color_map_func = np.vectorize(lambda x: color_map.get(x, (0, 1, 0)), otypes=[tuple])
        colors = color_map_func(target_col)
    
        plt.figure(figsize=(12, 12))

        for col_1 in range(num_columns):
            for col_2 in range(col_1, num_columns):
                lines = self.model[(col_1, col_2)]
                points = {}
                
                # could memoize these because there is some reuse
                x_min = data_cols[:, col_1].min()
                x_max = data_cols[:, col_1].max()
                y_min = data_cols[:, col_2].min()
                y_max = data_cols[:, col_2].max()

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

                plt.subplot(num_columns, num_columns, (col_1 * num_columns) + col_2 + 1)
                scatter_size = 7
                title_font_size = scatter_size * 1.25
                plt.scatter(data_cols[:, col_1], data_cols[:, col_2], s=scatter_size, c=colors)
                for c in lines.keys():
                    plt.plot([points[c][0][0], points[c][1][0]], [points[c][0][1], points[c][1][1]], c=color_map[c], marker= '^' if lines[c].direction else 'v')
                plt.title(f'{column_labels[col_1]}, {column_labels[col_2]}', fontsize = title_font_size)