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


# =========================================
# ======= read in and clean data ==========
# =========================================

input_fp = "./Data/character_data_clean.parquet"
df = pd.read_parquet(input_fp)


# =========================================
# ======= create data for clustering ======
# =========================================
cluster_fields = ["NumberOfAppearances",
                  "YearsOnStreet",
                  "NumberChildren",
                  "NumberTimesMarried"]
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
    ["KClusterNumber", "ClusterCategory"],
    axis=1,
    errors="ignore")

kmeans = KMeans(n_clusters=n_clusters,
                random_state=1,
                n_init=100,
                init='k-means++')
kmeans.fit(cluster_df)
cluster_df["KClusterNumber"] = kmeans.predict(cluster_df)

df["KClusterNumber"] = cluster_df["KClusterNumber"]

# cluster summary derivation... and relabel to get 'ClusterCategory'
cluster_summary = df.groupby("KClusterNumber")[
    cluster_fields].aggregate("mean")
cluster_summary = cluster_summary.merge(
    df.value_counts("KClusterNumber"),
    left_index=True,
    right_index=True
)
cluster_summary["count"] = cluster_summary["count"].rename("Count")
cluster_summary = cluster_summary.sort_values(
    ["NumberOfAppearances", "NumberChildren"])

cluster_summary["ClusterCategory"] = list(
    ascii_uppercase[:len(cluster_summary.index)])
cluster_df = cluster_df.merge(
    cluster_summary["ClusterCategory"],
    left_on="KClusterNumber",
    right_index=True
)

df["ClusterCategory"] = cluster_df["ClusterCategory"]


# %%
(
    GT(cluster_summary)
    .fmt_number(
        columns=["YearsOnStreet", "NumberChildren", "NumberTimesMarried"],
        decimals=2
    )
    .fmt_number(
        columns=["NumberOfAppearances", "count"],
        decimals=0
    )

)


# %%

# =========================================
# ======= plot k-means clustering ======
# =========================================

def create_character_hover_template():
    """
    Create standardized hover template for character data plots.

    Returns:
        String containing the hover template with proper formatting
    """
    return ('<b>Character: %{customdata[0]}</b><br>' +
            'Cluster: %{customdata[5]}<br>' +
            'NumberOfAppearances: %{customdata[1]}<br>' +
            'YearsOnStreet: %{customdata[2]:.1f}<br>' +
            'NumberChildren: %{customdata[3]}<br>' +
            'NumberTimesMarried: %{customdata[4]}<br>' +
            '<extra></extra>')


# Standard custom_data configuration for character plots
CHARACTER_CUSTOM_DATA = ['Character',
                         'NumberOfAppearances',
                         'YearsOnStreet',
                         'NumberChildren',
                         'NumberTimesMarried',
                         'ClusterCategory']

plotly_df = (
    df
    .sort_values(["NumberOfAppearances"])
    .assign(KClusterNumber=lambda x: pd.Categorical(x["KClusterNumber"]))
)

px.scatter(
    data_frame=plotly_df.reset_index(),
    y="NumberOfAppearances",
    x="YearsOnStreet",
    color='ClusterCategory',
    custom_data=CHARACTER_CUSTOM_DATA,
    template='plotly_white',
    color_continuous_scale='viridis'
).update_traces(
    hovertemplate=create_character_hover_template()
).update_layout(showlegend=False
                ).update_xaxes(dtick=10
                               )


# %%
# Add jitter to discrete variables
np.random.seed(42)  # For reproducible jitter
jitter_amount = 0.15  # Small jitter for discrete 0-6 range

plotly_df_jitter = plotly_df.reset_index().copy()
plotly_df_jitter['No_times_married_jitter'] = plotly_df_jitter['NumberTimesMarried'] + \
    np.random.uniform(-jitter_amount, jitter_amount, len(plotly_df_jitter))
plotly_df_jitter['No_children_jitter'] = plotly_df_jitter['NumberChildren'] + \
    np.random.uniform(-jitter_amount, jitter_amount, len(plotly_df_jitter))

px.scatter(
    data_frame=plotly_df_jitter,
    y="No_children_jitter",
    x="No_times_married_jitter",
    color='ClusterCategory',
    custom_data=CHARACTER_CUSTOM_DATA,
    template='plotly_white',
    color_continuous_scale='viridis'
).update_traces(
    hovertemplate=create_character_hover_template()
).update_layout(
    showlegend=False
).update_xaxes(
    title="NumberTimesMarried",
    dtick=1
).update_yaxes(
    title="NumberChildren",
    dtick=1
)


# %%

# =========================================
# ======= perform PCA decomposition ======
# =========================================

cluster_df = cluster_df.drop(columns=["KClusterNumber", "ClusterCategory"],
                             errors="ignore")
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
        "ExplainedVarianceRatio": cluster_pca_fitted.explained_variance_ratio_
    }
)

# Create the waterfall chart
fig = go.Figure(go.Waterfall(
    name="Waterfall",
    orientation="v",  # vertical orientation
    x=wf_df['Component'],  # your key column
    textposition="outside",
    # show values on bars
    text=[f"{val:+.3f}" for val in wf_df['ExplainedVarianceRatio']],
    y=wf_df['ExplainedVarianceRatio'],
    connector={"line": {"color": "black", "width": 0.8}},
    increasing={"marker": {"color": "green"}}
))


fig.update_layout(
    title="Principal Component Analysis",
    xaxis_title="Component",
    yaxis_title="ExplainedVarianceRatio",
    showlegend=False,
    template="plotly_white",
    yaxis=dict(tickformat='.0%'),
    margin=dict(t=60, b=40, l=40, r=40)
).update_traces(
    hovertemplate='Principal Component: %{x}<br>' +
    'Cumulative Variance Explained: %{y:.2%}<extra></extra>'
)

fig
# %%
