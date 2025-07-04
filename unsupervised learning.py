# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.cluster import KMeans
import re
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import yeojohnson

# %%

# =========================================
# =========== custom functions ============
# =========================================

# calculate a series for the years difference between two date series


def years_diff(start_series, end_series):
    time_delta = end_series - start_series
    years_diff_series = time_delta.apply(lambda x: round(x.days/365.25, 2))
    return years_diff_series

# apply the yeojohnson transformation to a series and return the transformed series


def transform_yeojohnson(data_series):
    transformed_data_series, _ = yeojohnson(data_series)
    return transformed_data_series


# %%

# =========================================
# ======= read in and clean data ==========
# =========================================

df = pd.read_csv("./Data/character_data_segmented.csv")

date_cols = ['Born', 'Died', 'Exit date', 'Start date',
             'First appearance', 'Last appearance']

df.loc[:, date_cols] = df.apply(
    {col: lambda x: pd.to_datetime(x) for col in date_cols}
)

df["YearsOnStreet"] = years_diff(df["Start date"], df["Exit date"])
df["AgeEnterStreet"] = years_diff(df["Born"], df["First appearance"])
df["AgeLastOnStreet"] = years_diff(df["Born"], df["Exit date"])

sum_y_o_s = df.groupby("Character")["YearsOnStreet"].transform(sum)
df = (df
      .query("Segment == `Max segment`")
      .assign(Returner=lambda x: x['Max segment'] > 1)
      .drop(columns=[
          "YearsOnStreet",
          "Segment",
          "Max segment"]
      )
      .merge(sum_y_o_s, how="inner", left_index=True, right_index=True)
      .assign(AppearPerYear=lambda x: x['Number of appearances']/x['YearsOnStreet'])
      )

df = df.convert_dtypes()
df = df.set_index("Character")


# %%

# =========================================
# ======= create data for clustering ======
# =========================================

# select only some columns
cluster_df = df.loc[:, ["Number of appearances",
                        "YearsOnStreet",
                        "No children",
                        "No times married"]]


# remove rows that contain NA values
cluster_df = cluster_df.loc[cluster_df.apply(
    lambda x: not any(x.isna()), axis=1), :]

# transform columns...
cluster_df = cluster_df.apply(transform_yeojohnson)


# %%

# =========================================
# ======= perform k-means clustering ======
# =========================================

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(cluster_df)
cluster_df["y_kmeans"] = kmeans.predict(cluster_df)

# %%

# determining the silhouette score of different cluster numbers
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
k_cluster_df = pd.DataFrame(None, index=range_n_clusters)
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(cluster_df)
    k_cluster_df.loc[i, "Sil Score"] = silhouette_score(
        cluster_df.to_numpy(),
        kmeans.predict(cluster_df)
    )
k_cluster_df


# %%
sns.scatterplot(
    data=cluster_df,
    y="No times married",
    # y="No children",
    x="YearsOnStreet",
    hue="y_kmeans",
)

# %%
hist_df = cluster_df.copy()
var1 = "YearsOnStreet"
var1 = "Number of appearances"
# var1 = "No children"
# hist_df[var1] = np.sqrt(hist_df[var1])
sns.histplot(
    data=hist_df,
    # y="Number of appearances"
    x=var1
    # x="No times married"
)


# %%
