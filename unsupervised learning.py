# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
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


# %%
sns.scatterplot(
    data=df.assign(YearsOnStreet=lambda x: np.log(x.YearsOnStreet)),
    x="YearsOnStreet",
    y="AppearPerYear"

)

# %%

# =========================================
# ======= create data for clustering ======
# =========================================
cluster_fields = ["Number of appearances",
                  "YearsOnStreet",
                  "No children",
                  "No times married"]
# select only some columns
cluster_df = df.loc[:, cluster_fields]


# remove rows that contain NA values
cluster_df = cluster_df.loc[cluster_df.apply(
    lambda x: not any(x.isna()), axis=1), :]

# transform columns...
cluster_df = cluster_df.apply(transform_yeojohnson)
cluster_df = cluster_df.apply(normalise)


# %%

# determining the silhouette score of different cluster numbers
range_n_clusters = range(2, 21, 1)  # [2, 3, 4, 5, 6, 7, 8]
k_cluster_df = pd.DataFrame(None, index=range_n_clusters)
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, random_state=1, n_init=100, init='k-means++')
    kmeans.fit(cluster_df)
    k_cluster_df.loc[i, "SilhouetteScore"] = silhouette_score(
        cluster_df.to_numpy(),
        kmeans.predict(cluster_df)
    )
    k_cluster_df.loc[i, "WSS"] = kmeans.inertia_

k_cluster_df


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


# =========================================
# ======= perform k-means clustering ======
# =========================================
# %%
n_clusters = 8

kmeans = KMeans(n_clusters=n_clusters, random_state=1,
                n_init=100, init='k-means++')
kmeans.fit(cluster_df)
cluster_df["y_kmeans"] = kmeans.predict(cluster_df)


# %%
df["y_kmeans"] = cluster_df["y_kmeans"]
df.groupby("y_kmeans")

sns.scatterplot(
    data=cluster_df,
    # y="No times married",
    # y="No children",
    y="Number of appearances",
    x="YearsOnStreet",
    hue="y_kmeans",
    palette="Set1"
)

# %%
cluster_summary = df.groupby("y_kmeans")[cluster_fields].aggregate("mean")
cluster_summary = cluster_summary.merge(df.value_counts(
    "y_kmeans"), left_index=True, right_index=True)
cluster_summary = cluster_summary.sort_values("count")
plotly_df = (
    df
    .sort_values(["Number of appearances"])
    .assign(y_kmeans=lambda x: pd.Categorical(x["y_kmeans"]))
)

px.scatter(
    data_frame=plotly_df.reset_index(),
    y="Number of appearances",
    x="YearsOnStreet",
    color='y_kmeans',
    custom_data=['Character',
                 'Number of appearances',
                 'YearsOnStreet',
                 'No children',
                 'No times married'],
    template='plotly_white',
    color_continuous_scale='viridis'
).update_traces(
    hovertemplate='<b>Character: %{customdata[0]}</b><br>' +
                  'Number of appearances: %{customdata[1]}<br>' +
                  'Years on Street: %{customdata[2]}<br>' +
                  'No children: %{customdata[3]}<br>' +
                  'No times married: %{customdata[4]}<br>' +
                  '<extra></extra>'
).update_layout(showlegend=False
                ).update_xaxes(dtick=10
                               )


# %%
# Add jitter to discrete variables
np.random.seed(42)  # For reproducible jitter
jitter_amount = 0.15  # Small jitter for discrete 0-6 range

plotly_df_jitter = plotly_df.reset_index().copy()
plotly_df_jitter['No_times_married_jitter'] = plotly_df_jitter['No times married'] + \
    np.random.uniform(-jitter_amount, jitter_amount, len(plotly_df_jitter))
plotly_df_jitter['No_children_jitter'] = plotly_df_jitter['No children'] + \
    np.random.uniform(-jitter_amount, jitter_amount, len(plotly_df_jitter))

px.scatter(
    data_frame=plotly_df_jitter,
    y="No_children_jitter",
    x="No_times_married_jitter",
    color='y_kmeans',
    custom_data=['Character',
                 'Number of appearances',
                 'YearsOnStreet',
                 'No children',
                 'No times married'],
    template='plotly_white',
    color_continuous_scale='viridis'
).update_traces(
    hovertemplate='<b>Character: %{customdata[0]}</b><br>' +
                  'Number of appearances: %{customdata[1]}<br>' +
                  'Years on Street: %{customdata[2]}<br>' +
                  'No children: %{customdata[3]}<br>' +
                  'No times married: %{customdata[4]}<br>' +
                  '<extra></extra>'
).update_layout(
    showlegend=False
).update_xaxes(
    title="No times married"
).update_yaxes(
    title="No children"
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
