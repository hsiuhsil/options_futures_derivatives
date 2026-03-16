import numpy as np
import matplotlib.pyplot as plt

def nodes_coordinates(N):
    """
    Generate node coordinates for plotting a binomial tree.

    Parameters:
        N (int): number of time steps in the tree

    Returns:
        list: coordinates of nodes at each time step
    """
    coords = []
    for n in range(N+1):
        coord = []
        for i in range(n, -n-1, -2):
            coord.append([n,i])
        coords.append(coord)
    return coords

def plot_binomial_trees(stocks, options, timeline):
    """
    Plot the binomial tree of stock prices and option values.

    Parameters:
        stocks (list): stock price tree
        options (list): option value tree
        timeline (array): time grid for each step

    Returns:
        None
    """
    coords = nodes_coordinates(len(stocks))

    plt.figure(figsize=(6, 4))
    min_y = -len(stocks)
    for n in range(len(stocks)):
        # add a timeline label
        plt.text(n, min_y - 0.5, f"{timeline[n]}yr", 
                 ha='center', va='top', fontweight='bold', color='gray')
        
        for i, (x, y) in enumerate(coords[n]):
            # plot the node and label
            plt.scatter(x, y, color='blue', zorder=5)
            plt.text(x+0.1, y+0.2, f"s={stocks[n][i]:.4f}\nv={options[n][i]:.4f}",
                     fontsize=9, ha='right', va='bottom')

            # draw arrows to the next step (if it's not the last step)
            if n<len(coords)-2:
                next_coords = coords[n+1]
                for move in [0,1]: # 0 for up, and 1 for down
                    target_x, target_y = next_coords[i+move]
                    plt.annotate('', xy=(target_x, target_y), xytext=(x, y),
                                 arrowprops=dict(arrowstyle="->", color='black', lw=1, alpha=0.3))

    plt.title("Binomial Tree")
    plt.axis('off')
    plt.xlim(-0.5, len(stocks))
    plt.tight_layout()
    plt.show()
