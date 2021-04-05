from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def plot_clusters(series, labels, column, name_cluster_model=None, plot_per_cluster=True, kmean_center=None,
                 color_per_cluster= {0:'gray',1:'blue',2:'yellow',3:'green',4:'orange'}, type_line_center='--'):
    """Plot one or more time series per cluster either in the same image 
    or one image per cluster. 

    Parameters
    ----------
    series : pd.DataFrame
        One or more time series
    labels : list
        labesls of clusters, will be displayed in figure legend
    column : string
        name of the columns that contains the time series
    name_cluster_model : string
        nanme of the cluster model to display it in the title (default=None)
    plot_per_cluster : boolean
        plot only one image or plot images per clusters (default=True)
    color_per_cluster : dict
        color to show each cluster. the dictionary keys are the cluster labels 
        and the values are the cluster color. plot_per_cluster must be True.
        (default={0:'gray',1:'blue',2:'yellow',3:'green',4:'orange'})
    kmean_center : boolean
        if the cluster model is kmeans, you can plot the centers set this variable 
        with the centers. (default=None)
    type_line_center : string
        type of the line to show the center of the clusters in kmeans. 

    Returns
    -------
    fig : plt.Figure
    """
    
    series = series.copy()
    series['cluster'] = labels
    clusters = np.unique(labels)
    
    if plot_per_cluster:
        for i,cluster in enumerate(clusters):
            fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
            for instance in series.loc[series['cluster'] == cluster, column]:
                ax.plot(instance, alpha=.5, color='gray')
            if kmean_center is not None:
                plt.plot(kmean_center[i], type_line_center, color='r')
            title = f"Instances of cluster {cluster}"
            if name_cluster_model is not None:
                title += ' - %s'%name_cluster_model
            ax.set(title=title)
    else:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
        for cluster in clusters:
            for instance in series.loc[series['cluster'] == cluster, column]:
                plt.plot(instance, color=color_per_cluster.get(cluster,'r'))
        legends=[Patch(color=color_per_cluster.get(c,'r'), label=f"Cluster {c}") for c in clusters]
        plt.legend(handles=legends)
        if kmean_center is not None:
            for center in kmean_center:
                plt.plot(center, type_line_center, color='r')
        title = 'Plot per Clusters'
        if name_cluster_model is not None:
            title += ' - %s'%name_cluster_model
        plt.title(title)