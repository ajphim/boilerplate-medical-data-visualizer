import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Read data
df = pd.read_csv('medical_examination.csv')

# 2: Add and determine if overweight
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop(columns=['BMI'], inplace=True)

# 3: Normalize these columns
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Draw Categorical plot
def draw_cat_plot():
    # 5
    df_cat = None

    # 6
    df_cat = None
    
    # 7

    # 8
    fig = None

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None

    # 14
    fig, ax = None

    # 15
    
    # 16
    fig.savefig('heatmap.png')
    return fig
