# %%
from great_tables import GT
from string import ascii_uppercase
from sklearn.decomposition import PCA
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
from custom_cluster_functions import *


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

# note the above transformation is similar to using PowerTransformer(method='yeo-johnson')


# %%

# ==================================================
# ==== plots of data before/after transformation ===
# ==================================================

def add_fig_to_subplot(plot_grid, fig, row, col):
    for trace in fig.data:
        plot_grid.add_trace(trace, row=row, col=col)


def construct_hist_grid_plots(df, cluster_fields):
    plot_grid = make_subplots(
        2, 2, False, False, horizontal_spacing=0.2, vertical_spacing=0.2)
    k = 0
    for histvar in cluster_fields:
        k = k+1
        i = 2-k % 2
        j = 1+(k-1)//2
        trace = go.Histogram(
            x=df[histvar],
            nbinsx=50,
            marker_color='blue',
            showlegend=False
        )
        plot_grid.add_trace(trace, i, j)
        plot_grid.update_xaxes(
            title_text=histvar, title_standoff=3, row=i, col=j)
        plot_grid.update_yaxes(
            title_text='count', title_standoff=3, row=i, col=j)
    return plot_grid


plot_grid1 = construct_hist_grid_plots(df, cluster_fields)
plot_grid2 = construct_hist_grid_plots(cluster_df, cluster_fields)


# %%

# =========================================
# ======== clustering metrics ============
# =========================================

k_cluster_df = get_cluster_scores(cluster_df)
plot_cluster_score(k_cluster_df)

# =========================================
# ======= perform k-means clustering ======
# =========================================
# %%
n_clusters = 8

cluster_df = cluster_df.drop(
    ["y_kmeans", "Cluster Category"],
    axis=1,
    errors="ignore")

kmeans = KMeans(n_clusters=n_clusters,
                random_state=1,
                n_init=100,
                init='k-means++')
kmeans.fit(cluster_df)
cluster_df["y_kmeans"] = kmeans.predict(cluster_df)

df["y_kmeans"] = cluster_df["y_kmeans"]

# cluster summary derivation... and relabel to get 'Cluster Category'
cluster_summary = df.groupby("y_kmeans")[cluster_fields].aggregate("mean")
cluster_summary = cluster_summary.merge(
    df.value_counts("y_kmeans"),
    left_index=True,
    right_index=True
)
cluster_summary["count"] = cluster_summary["count"].rename("Count")
cluster_summary = cluster_summary.sort_values(
    ["Number of appearances", "No children"])

cluster_summary["Cluster Category"] = list(
    ascii_uppercase[:len(cluster_summary.index)])
cluster_df = cluster_df.merge(
    cluster_summary["Cluster Category"],
    left_on="y_kmeans",
    right_index=True
)

df["Cluster Category"] = cluster_df["Cluster Category"]


# %%
(
    GT(cluster_summary)
    .fmt_number(
        columns=["YearsOnStreet", "No children", "No times married"],
        decimals=2
    )
    .fmt_number(
        columns=["Number of appearances", "count"],
        decimals=0
    )

)


# %%

# =========================================
# ======= plot k-means clustering ======
# =========================================


plotly_df = (
    df
    .sort_values(["Number of appearances"])
    .assign(y_kmeans=lambda x: pd.Categorical(x["y_kmeans"]))
)

px.scatter(
    data_frame=plotly_df.reset_index(),
    y="Number of appearances",
    x="YearsOnStreet",
    color='Cluster Category',
    custom_data=['Character',
                 'Number of appearances',
                 'YearsOnStreet',
                 'No children',
                 'No times married',
                 'Cluster Category'],
    template='plotly_white',
    color_continuous_scale='viridis'
).update_traces(
    hovertemplate='<b>Character: %{customdata[0]}</b><br>' +
    'Cluster: %{customdata[5]}<br>' +
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
    color='Cluster Category',
    custom_data=['Character',
                 'Number of appearances',
                 'YearsOnStreet',
                 'No children',
                 'No times married',
                 'Cluster Category'],
    template='plotly_white',
    color_continuous_scale='viridis'
).update_traces(
    hovertemplate='<b>Character: %{customdata[0]}</b><br>' +
    'Cluster: %{customdata[5]}<br>' +
                  'Number of appearances: %{customdata[1]}<br>' +
                  'Years on Street: %{customdata[2]}<br>' +
                  'No children: %{customdata[3]}<br>' +
                  'No times married: %{customdata[4]}<br>' +
                  '<extra></extra>'
).update_layout(
    showlegend=False
).update_xaxes(
    title="No times married",
    dtick=1
).update_yaxes(
    title="No children",
    dtick=1
)


# %%

# =========================================
# ======= perform PCA decomposition ======
# =========================================

if "y_kmeans" in cluster_df.columns:
    cluster_df = cluster_df.drop(columns="y_kmeans")
cluster_pca = PCA()
cluster_pca_fitted = cluster_pca.fit(cluster_df.to_numpy().T)
cluster_pca_fitted.explained_variance_ratio_
cluster_pca_fitted.components_


pca_df = pd.DataFrame(
    cluster_pca_fitted.components_[:2].T,
    index=cluster_df.index,
    columns=["PC1", "PC2"]
)

# %%
wf_df = pd.DataFrame(
    {
        "Component": range(1, 5, 1),
        "Explained Variance Ratio": cluster_pca_fitted.explained_variance_ratio_
    }
)

# Create the waterfall chart
fig = go.Figure(go.Waterfall(
    name="Waterfall",
    orientation="v",  # vertical orientation
    x=wf_df['Component'],  # your key column
    textposition="outside",
    # show values on bars
    text=[f"{val:+.3f}" for val in wf_df['Explained Variance Ratio']],
    y=wf_df['Explained Variance Ratio'],
    connector={"line": {"color": "black", "width": 0.8}},
    increasing={"marker": {"color": "green"}}
))

fig.update_layout(
    title="Principal Component Analysis",
    xaxis_title="Component",
    yaxis_title="Explained Variance Ratio",
    showlegend=False,
    template="plotly_white"
)


# %%
