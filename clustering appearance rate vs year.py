# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
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

import plotly.express as px
import plotly.graph_objects as go

# %%

# =========================================
# =========== custom functions ============
# =========================================

# calculate a series for the years difference between two date series


def years_diff(start_series, end_series):
    time_delta = end_series - start_series
    years_diff_series = time_delta.apply(lambda x: round(x.days/365.25, 2))
    return years_diff_series


def normalise(series):
    n_series = (series - np.mean(series))/np.std(series)
    return n_series

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

cluster_df = cluster_df.apply(lambda x: normalise(x), axis=0)


# %%

# =============================================
# === check n values for k-means clustering ===
# =============================================

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

# =========================================
# ======= perform k-means clustering ======
# =========================================

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
kmeans.fit(cluster_df)
cluster_df["y_kmeans"] = kmeans.predict(cluster_df)

df_w_clusters = df.merge(
    cluster_df["y_kmeans"],
    left_index=True,
    right_index=True
)

# %%

# =========================================
# ======== plot clustering results ========
# =========================================

# Prepare the data (equivalent to the assign() call)
plot_data_transformed = cluster_df.assign(
    y_kmeans=lambda x: pd.Categorical(x["y_kmeans"])
)

plot_data_raw = df_w_clusters.assign(
    y_kmeans=lambda x: pd.Categorical(x["y_kmeans"]))

# Create the plot
p1 = px.scatter(
    plot_data_transformed.reset_index(),
    x="YearsOnStreet",
    y="AppearPerYear",
    color="y_kmeans",
    hover_data={"Character": True,
                "YearsOnStreet": False,
                "AppearPerYear": False,
                "y_kmeans": False},
    labels={
        "YearsOnStreet": "Years on Street (Transformed)",
        "AppearPerYear": "Number of appearances (Transformed)"
    }
)

p2 = px.scatter(
    plot_data_raw.reset_index(),
    x="YearsOnStreet",
    y="Number of appearances",
    color="y_kmeans",
    hover_data={"Character": True,
                "YearsOnStreet": False,
                "Number of appearances": False,
                "y_kmeans": False},
    labels={
        "YearsOnStreet": "Years on Street",
        "Number of appearances": "Number of appearances"
    }
)

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.2,
    subplot_titles=("Plot 1", "Plot 2")
)

# Add traces from p1 to first subplot
for trace in p1.data:
    fig.add_trace(trace, row=1, col=1)

# Add traces from p2 to second subplot
for trace in p2.data:
    fig.add_trace(trace, row=1, col=2)

# Update layout
fig.update_layout(
    title_text="Combined Plots",
    showlegend=True
)

# Update x and y axis labels
fig.update_xaxes(title_text="Years on Street (Transformed)", row=1, col=1)
fig.update_yaxes(
    title_text="Number of appearances (Transformed)", row=1, col=1)
fig.update_xaxes(title_text="Years on Street", row=1, col=2)
fig.update_yaxes(title_text="Number of appearances", row=1, col=2)


# %%

# =========================================
# ======== plot clustering metrics ========
# =========================================

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.2,
    subplot_titles=(
        "Silhouette Score <br>for different Cluster Numbers<br>",
        "Within-cluster Sum of Squares <br>for different Cluster Numbers<br>"
    )
)

p1 = px.line(
    data_frame=k_cluster_df.reset_index(names="No. of Clusters"),
    x="No. of Clusters",
    y="SilhouetteScore"
)
p2 = px.line(
    data_frame=k_cluster_df.reset_index(names="No. of Clusters"),
    x="No. of Clusters",
    y="WSS"
)

# Add traces from p1 to first subplot
for trace in p1.data:
    fig.add_trace(trace, row=1, col=1)

# Add traces from p2 to second subplot
for trace in p2.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(template='simple_white')
