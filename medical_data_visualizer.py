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
    # 5: Create DataFrame for catplot using pd.melt()
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6: Group and reformat data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # 7: Rename size column to total for cat plot to work
    df_cat.rename(columns={'size': 'total'}, inplace=True)

    # 8: Create a catplot
    fig = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1
    ).fig

    # 9: Get figure for the output and store it in fig variable
    fig.savefig('catplot.png')
    return fig


# 10: Create a heat map from data
def draw_heat_map():
     # 11: Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 9))

    # 15: Plot the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt='0.1f',
        center=0,
        cbar_kws={"shrink": .5}
    )

    # 16: Save the figure
    fig.savefig('heatmap.png')
    return fig
