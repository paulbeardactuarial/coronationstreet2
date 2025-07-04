# %%
from mizani.formatters import comma_format
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.cluster import KMeans
import re
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from plotnine import *

# %%

# =========================================
# =========== custom functions ============
# =========================================

# calculate a series for the years difference between two date series


def years_diff(start_series, end_series):
    time_delta = end_series - start_series
    years_diff_series = time_delta.apply(lambda x: round(x.days/365.25, 2))
    return years_diff_series


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


# =========================================
# ======= create data for clustering ======
# =========================================

# select only some columns
cluster_df = df.loc[:, ["AppearPerYear",
                        "YearsOnStreet"]]


# remove rows that contain NA values
cluster_df = cluster_df.loc[cluster_df.apply(
    lambda x: not any(x.isna()), axis=1), :]

# transform columns...
cluster_df = cluster_df.assign(YearsOnStreet=lambda x: np.log(x.YearsOnStreet))


def normalise(series):
    n_series = (series - np.mean(series))/np.std(series)
    return n_series


cluster_df = cluster_df.apply(lambda x: normalise(x), axis=0)


# =========================================
# ======= perform k-means clustering ======
# =========================================
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
kmeans.fit(cluster_df)
cluster_df["y_kmeans"] = kmeans.predict(cluster_df)


# %%
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
k_cluster_df = pd.DataFrame(None, index=range_n_clusters)
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(cluster_df)
    k_cluster_df.loc[i, "SilhouetteScore"] = silhouette_score(
        cluster_df.to_numpy(),
        kmeans.predict(cluster_df)
    )
    k_cluster_df.loc[i, "WSS"] = kmeans.inertia_


# %%

df_w_clusters = df.merge(
    cluster_df["y_kmeans"],
    left_index=True,
    right_index=True
)

plot_layers = [theme_classic(), theme(legend_position="none")]
p1 = (
    ggplot(
        data=cluster_df.assign(
            y_kmeans=lambda x: pd.Categorical(x["y_kmeans"])),
        mapping=aes(x="YearsOnStreet", y="AppearPerYear", color="y_kmeans")
    )
    + geom_point()
    + plot_layers
    + scale_x_continuous(
        name="Years on Street (Transformed)"
    )
    + scale_y_continuous(
        name="Number of appearances (Transformed)"
    )
)
p2 = (
    ggplot(
        data=df_w_clusters.assign(
            y_kmeans=lambda x: pd.Categorical(x["y_kmeans"])),
        mapping=aes(x="YearsOnStreet",
                    y="Number of appearances", color="y_kmeans")
    )
    + geom_point()
    + plot_layers
    + scale_x_continuous(
        name="Years on Street",
        breaks=range(0, 100, 10)
    )
    + scale_y_continuous(
        labels=comma_format()
    )
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
p1.draw(ax1)
p2.draw(ax2)
plt.tight_layout()
plt.show()


# %%
df_w_clusters = df.merge(
    cluster_df["y_kmeans"],
    left_index=True,
    right_index=True
)

sns.scatterplot(
    data=df_w_clusters,
    x="YearsOnStreet",
    y="Number of appearances",
    hue="y_kmeans",
    palette="Spectral"
)
plt.show()


# %%
df_w_clusters.groupby("y_kmeans")["AgeEnterStreet"].aggregate(np.mean)


# %%


plt.figure(figsize=(12, 5))

# First plot
plt.subplot(1, 2, 1)
sns.lineplot(
    data=k_cluster_df.reset_index(names="n"),
    x="n",
    y="SilhouetteScore"
)
plt.title('Plot 1')

# Second plot
plt.subplot(1, 2, 2)
sns.lineplot(
    data=k_cluster_df.reset_index(names="n"),
    x="n",
    y="WSS"
)
plt.title('Plot 2')

plt.tight_layout()
plt.show()
