from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.stats import yeojohnson
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
import plotly.express as px

# =========================================
# =========== custom functions ============
# =========================================


def transform_yeojohnson(data_series):
    """
    Apply Yeo-Johnson transformation to normalize data distribution.

    Args:
        data_series: Input data series to transform

    Returns:
        Transformed data series with normalized distribution
    """
    transformed_data_series, _ = yeojohnson(data_series)
    return transformed_data_series


def normalise(series):
    """
    Standardize data series using z-score normalization.

    Args:
        series: Input data series

    Returns:
        Normalized series with mean=0 and std=1
    """
    n_series = (series - np.mean(series))/np.std(series)
    return n_series


def get_cluster_scores(cluster_df, range_n_clusters=range(2, 21, 1)):
    """
    Calculate clustering evaluation metrics for different cluster numbers.

    Args:
        cluster_df: DataFrame containing features for clustering
        range_n_clusters: Range of cluster numbers to test (default: 2-20)

    Returns:
        DataFrame with silhouette scores and WSS for each cluster number
    """
    if "KClusterNumber" in cluster_df.columns:
        cluster_df = cluster_df.drop(columns="KClusterNumber")

    k_cluster_df = pd.DataFrame(None, index=range_n_clusters)
    for i in range_n_clusters:
        kmeans = KMeans(
            n_clusters=i,
            random_state=1,
            n_init=100,
            init='k-means++'
        )
        kmeans.fit(cluster_df)
        k_cluster_df.loc[i, "SilhouetteScore"] = silhouette_score(
            cluster_df.to_numpy(),
            kmeans.predict(cluster_df)
        )
        k_cluster_df.loc[i, "WSS"] = kmeans.inertia_

    return k_cluster_df


def plot_cluster_score(k_cluster_df):
    """
    Create dual-plot visualization of clustering evaluation metrics.

    Args:
        k_cluster_df: DataFrame with clustering scores from get_cluster_scores()

    Returns:
        Plotly figure with silhouette score and WSS subplots
    """
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

    return fig
