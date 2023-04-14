import pandas as pd
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class TwoDimData:
    col1: pd.Series
    col2: pd.Series
    class_col: pd.Series
    classes: pd.Series


@dataclass
class Line:
    slope: float
    intercept: float
    direction: int
    score: float

    def __getitem__(self, key):
        return super().__getattribute__(key)
    

def get_best_lines(x_data : pd.Series , y_data : pd.Series, class_data : pd.Series, classes : pd.Series):
    data = TwoDimData(x_data, y_data, class_data, classes)
    best_lines = fit_lines(data)
    return best_lines

        
def test_line(slope, intercept, data : TwoDimData):
    above = []
    below = []
    for i in range(len(data.col1)):
        if data.col2[i] > (slope * data.col1[i]) + intercept:
            above.append(i)
        # Note that this means points on the line will be marked as below
        else:
            below.append(i)
    
    num_above = len(above)
    num_below = len(below)
    
    scores = {}
    
    for c in data.classes:
        num_c_above = len(data.class_col[above][data.class_col == c])
        num_c_below = len(data.class_col[below][data.class_col == c])
        
        not_c_above = num_above - num_c_above
        not_c_below = num_below - num_c_below
        
        # We want a good score when num_c_dir1 is much higher than num_c_dir2, and not_c_dir2
        # We want a high proportion of the points on one side to be 
        # Measure percentage of points that are class c on each side of line
        #     is this sufficient?
        #         We probably actually want the side with greater % that are class c and then the percentage of class c on that side
        
        fraction_above_in_c = 0 if num_above == 0 else num_c_above / num_above
        fraction_below_in_c = 0 if num_below == 0 else num_c_below / num_below
        b_above_is_max = fraction_above_in_c > fraction_below_in_c

        fraction_on_side_in_c = fraction_above_in_c if b_above_is_max else fraction_below_in_c
        num_c_on_side = num_c_above if b_above_is_max else num_c_below
        fraction_of_c_on_side = num_c_on_side / (num_c_above + num_c_below)
        
        #print(f'{fraction_of_c_on_side}, {fraction_on_side_in_c}')
        
        # scores[c] = (fraction_on_side_in_c * fraction_of_c_on_side, b_above_is_max)
        scores[c] = Line(slope, intercept, b_above_is_max, fraction_on_side_in_c * fraction_of_c_on_side)
    return scores

    
# For a bunch of lines:
#     Score them, keep track of best lines with best scores for each class
# return best lines
def fit_lines(data : TwoDimData):
    best_lines = {c:Line(0,0,0,-math.inf) for c in data.classes}
    x_interval = [data.col1.min(), data.col1.max()]
    y_interval = [data.col2.min(), data.col2.max()]
    
    n = 20
    x_space = np.linspace(x_interval[0], x_interval[1], n)
    y_space = np.linspace(y_interval[0], y_interval[1], n)
    
    res = []
    
    for x1 in x_space:
        for x2 in x_space:
            # Bottom edge to top edge
            compare_against_best_lines(test_line(*points_to_slope_intercept((x1, y_interval[0]), (x2, y_interval[1])), data), best_lines)
            
        for y in y_space:
            # Bottom edge to left edge
            compare_against_best_lines(test_line(*points_to_slope_intercept((x1, y_interval[0]), (x_interval[0], y)), data), best_lines)
            # Bottom edge to right edge
            compare_against_best_lines(test_line(*points_to_slope_intercept((x1, y_interval[0]), (x_interval[1], y)), data), best_lines)
            
            # Top edge to left edge
            compare_against_best_lines(test_line(*points_to_slope_intercept((x1, y_interval[1]), (x_interval[0], y)), data), best_lines)
            # Top edge to right edge
            compare_against_best_lines(test_line(*points_to_slope_intercept((x1, y_interval[1]), (x_interval[1], y)), data), best_lines)
    
    for y1 in y_space:
        for y2 in y_space:
            # Left edge to right edge
            compare_against_best_lines(test_line(*points_to_slope_intercept((x_interval[0], y1), (x_interval[1], y2)), data), best_lines)
            
    return best_lines


def points_to_slope_intercept(p1 : tuple, p2 : tuple):
    if p2[0] == p1[0]:
        slope = 10000    # This is a dumb way to avoid dividing by zero but I don't have the energy to refactor these functions right now
    else:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - (slope * p1[0])
    return slope, intercept
        

def compare_against_best_lines(scores: dict, best_lines : dict):
    for c in scores.keys():
        if best_lines[c].score < scores[c].score:
            best_lines[c] = scores[c]
    