# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%

df = pd.read_csv('result.csv')
df['freq'] = 1000000 * 10000 / df['time']

baseline = df.loc[df.threads == 1].copy().groupby('exe').freq.mean()
def apply_fn(s):
    s.freq /= baseline[s.exe]
    return s
df = df.apply(apply_fn, axis=1)
df = df.groupby(['threads', 'exe']).freq.max().reset_index()
df

# %%
# df = df.loc[df.threads <= 8]
sns.relplot(data=df, kind='line', x='threads', y='freq', hue='exe')
plt.grid()
# plt.yscale('log')
plt.savefig('plot.png', dpi=200)
