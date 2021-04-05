from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def plot_clusters(time_series, labels, column, plot_per_cluster=True, kmean_center=None,
                 color_per_cluster= {0:'gray',1:'b',2:'y',3:'g',4:'orange'}, type_line_center='--'):
    time_series = time_series.copy()
    time_series['cluster'] = labels
    clusters = np.unique(labels)
    
    if plot_per_cluster:
        for i,cluster in enumerate(clusters):
            fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
            for instance in time_series.loc[time_series['cluster'] == cluster, column]:
                ax.plot(instance, alpha=.5, color='gray')
            if kmean_center is not None:
                plt.plot(kmean_center[i], type_line_center, color='r')
            ax.set(title=f"Instances of cluster {cluster}")
    else:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
        for cluster in clusters:
            for instance in time_series.loc[time_series['cluster'] == cluster, column]:
                plt.plot(instance, color=color_per_cluster.get(cluster,'r'))
        legends=[Patch(color=color_per_cluster.get(c,'r'), label=f"Cluster {c}") for c in clusters]
        plt.legend(handles=legends)
        if kmean_center is not None:
            for center in kmean_center:
                plt.plot(center, type_line_center, color='r')
        plt.title('Plot per Clusters')