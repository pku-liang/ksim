import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')
df = df.loc[df.run > 2]
df['freq'] = 1000 * 100000000 / df.time
sns.relplot(data=df, kind='line', x='threads', y='freq', hue='exe')
plt.savefig('plot.png', dpi=200)