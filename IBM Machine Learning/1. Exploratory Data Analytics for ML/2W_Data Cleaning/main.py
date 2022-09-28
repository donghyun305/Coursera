import skillsnetwork
import pandas as pd
import numpy as np
import asyncio

import seaborn as sns
import matplotlib.pylab as plt


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

## Load in the Ames Housing Data
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv'

async def func():
    await skillsnetwork.download_dataset(URL)
housing = pd.read_csv('Ames_Housing_Data1.tsv', sep='\t')

print(housing.info())