from scipy.interpolate import griddata
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_hypersurface(results, figname):

    # Extract data for plotting
    n_neighbors_vals, min_samples_vals, min_cluster_size_vals, scores = zip(*results)
    
    # Create a grid of parameter values
    n_neighbors_vals = np.array(n_neighbors_vals)
    min_samples_vals = np.array(min_samples_vals)
    min_cluster_size_vals = np.array(min_cluster_size_vals)
    scores = np.array(scores)
    
    # Define the grid ranges
    n_neighbors_range = np.linspace(n_neighbors_vals.min(), n_neighbors_vals.max(), 50)
    min_samples_range = np.linspace(min_samples_vals.min(), min_samples_vals.max(), 50)
    min_cluster_size_range = np.linspace(min_cluster_size_vals.min(), min_cluster_size_vals.max(), 50)

    # Create the meshgrid for plotting
    N, M = np.meshgrid(min_samples_range, min_cluster_size_range)
    Z = np.zeros_like(N)

    points = np.array(list(zip(n_neighbors_vals, min_samples_vals, min_cluster_size_vals)))
    values = scores

    for i in range(len(min_samples_range)):
        for j in range(len(min_cluster_size_range)):
            Z[j, i] = griddata(points[:, :2], values, (N[j, i], M[j, i]), method='cubic')
    
    # Find the maximum score and its corresponding parameters
    max_score_index = np.argmax(scores)
    max_n_neighbors = n_neighbors_vals[max_score_index]
    max_min_samples = min_samples_vals[max_score_index]
    max_min_cluster_size = min_cluster_size_vals[max_score_index]
    max_score = scores[max_score_index]

    # Create the 3D surface plot
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    surf1 = ax1.plot_surface(N, M, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, zorder=0)
    #ax1.scatter(max_min_samples, max_min_cluster_size, max_score-0.2, color='black', s=10, zorder=2.5)
    
    # Customize the axes
    ax1.set_xlabel('min_samples')
    ax1.set_ylabel('min_cluster_size')
    #ax1.set_ylim(0, 20)
    ax1.set_zlabel('Silhouette Score')
    ax1.view_init(elev=30, azim=60, roll=0)#, vertical_axis="z")  # Adjust elev and azim to get different views
    
    # Add a color bar which maps values to colors
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
    surf2 = ax2.plot_surface(N, M, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, zorder=0)
    #ax2.scatter(max_min_samples, max_min_cluster_size, max_score-0.2, color='black', s=10, zorder=2.5)
    # Customize the axes
    ax2.set_xlabel('min_samples')
    ax2.set_ylabel('min_cluster_size')
    #ax2.set_ylim(0,20)
    ax2.set_zlabel('Silhouette Score')
    ax2.view_init(elev=40, azim=80, roll=0)#, vertical_axis="z")  # Adjust elev and azim to get different views
    
    plt.tight_layout()
    plt.savefig(f"./results/figures/upgenevsrep_clustering_{figname}_a.png", dpi=300)
    
    # Second plot
    # ---------------------------------------------------------------------------------------
    N, M = np.meshgrid(n_neighbors_range, min_samples_range)
    Z = np.zeros_like(N)
    points = np.array(list(zip(n_neighbors_vals, min_samples_vals, min_cluster_size_vals)))
    values = scores
    
    for i in range(len(n_neighbors_range)):
        for j in range(len(min_samples_range)):
            Z[j, i] = griddata(points[:, :2], values, (N[j, i], M[j, i]), method='cubic')
    
    # Find the maximum score and its corresponding parameters
    max_score_index = np.argmax(scores)
    max_n_neighbors = n_neighbors_vals[max_score_index]
    max_min_samples = min_samples_vals[max_score_index]
    max_min_cluster_size = min_cluster_size_vals[max_score_index]
    max_score = scores[max_score_index]
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(121, projection='3d', computed_zorder=False)
    surf1 = ax1.plot_surface(N, M, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, zorder=0)
    #ax1.scatter(max_n_neighbors, max_min_samples, max_score, color='white', s=10, zorder=2.5)
    
    # Customize the axes
    ax1.set_xlabel('n_neighbors')
    ax1.set_ylabel('min_samples')
    ax1.set_zlabel('Silhouette Score')
    ax1.view_init(elev=30, azim=60)  # Adjust elev and azim to get different views
    
    ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
    surf2 = ax2.plot_surface(N, M, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, zorder=0)
    #ax2.scatter(max_n_neighbors, max_min_samples, max_score, color='white', s=10, zorder=2.5)
    # Customize the axes
    ax2.set_xlabel('n_neighbors')
    ax2.set_ylabel('min_samples')
    ax2.set_zlabel('Silhouette Score')
    ax2.view_init(elev=40, azim=80)  # Adjust elev and azim to get different views
    
    plt.tight_layout()
    plt.savefig(f"./results/figures/upgenevsrep_clustering_{figname}_b.png", dpi=300)

    print("Done!")

def plot_embeddings_labeled(embedding, labels, filename):
    
    fig = plt.figure(figsize=(15, 10))

    # Plot UMAP in 3D
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, c=labels)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.set_zlabel('')
    # Plot UMAP in 3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, c=labels)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_zlabel('')
    ax2.view_init(elev=30, azim=60)
    plt.savefig(f"./results/figures/{filename}.png", dpi=300)