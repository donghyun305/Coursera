import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/Wine_Quality_Data.csv")
data.head(4).T

sns.set_context('notebook')
sns.set_style('white')

red = sns.color_palette()[2]
white = sns.color_palette()[4]
#
# bin_range = np.array([3,4,5,6,7,8])
# ax = plt.axes()
# for color

km = KMeans(n_clusters=2, random_state=42)
km = km.fit(data[float_cloumns])
