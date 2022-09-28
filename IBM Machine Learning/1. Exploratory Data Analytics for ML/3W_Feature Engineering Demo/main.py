import pandas as pd
import seaborn as sns
import skillsnetwork
import numpy as np
from networkx.drawing.tests.test_pylab import plt

df = pd.read_csv('data/Ames_Housing_Data.tsv', sep = '\t')

# This is recommended by the data set author to remove a few outliers
df = df.loc[df['Gr Liv Area'] <= 4000,:]
print("Number of rows in the data:", df.shape[0])
print("Number of columns in the data:", df.shape[1])
data = df.copy()

#One_hot encoding for dummy variables
one_hot_encode_cols = df.dtypes[df.dtypes == np.object]
one_hot_encode_cols = one_hot_encode_cols.index.tolist()

df = pd.get_dummies(df, columns = one_hot_encode_cols, drop_first=True)

#Log Transforming Skew Variables
mask = data.dtypes == np.float
float_cols = data.columns[mask]

skew_limit = 0.75
skew_vals = data[float_cols].skew()
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))

#Let's draw a log graph
field = "BsmtFin SF 1"
fig, (ax_before, ax_after) = plt.subplots(1,2, figsize=(10,5))
df[field].hist(ax = ax_before)               # before the transformation
df[field].apply(np.log1p).hist(ax=ax_after) # after the tranformation

ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field));




