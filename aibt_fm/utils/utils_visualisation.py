import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_braking_surface(xt, yt, xtest, ytest, model_func, MIN, MAX, name_figure="Braking Distance Estimation"):
    """
    Visualizes the braking distance model as a 3D surface and compares it 
    against actual training and validation data points.
    """
    # 1. Create a dense grid of inputs (Temperature vs Altitude)
    MIN_temp, MIN_alt = MIN
    MAX_temp, MAX_alt = MAX
    temp_range = np.linspace(MIN_temp, MAX_temp, 50)
    alt_range = np.linspace(MIN_alt, MAX_alt, 50)
    
    # X and Y become 2D matrices representing the 'floor' of our 3D plot
    X, Y = np.meshgrid(temp_range, alt_range)

    # 2. Vectorize: Prepare the grid for the Neural Network
    # We flatten the grids and stack them to create (2500, 2) input shape
    input_grid = np.column_stack([X.ravel(), Y.ravel()])

    # 3. Predict the 'Z' axis (Braking Distance) for every point on the grid
    # Batch processing is significantly faster than a for-loop
    predictions = model_func(input_grid, verbose=0)
    
    # Reshape the flat predictions back into a 50x50 grid to match X and Y
    Z = predictions.reshape(50, 50)

    # 4. Initialize the 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 5. Draw the Model Surface
    # alpha=0.5 makes the surface translucent so we can see points through it
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, antialiased=True)

    # 6. Overlay Actual Data Points
    if xt is not None:
        ax.scatter(xt[:, 0], xt[:, 1], yt, color="blue", s=30, label="Training Data")
    if xtest is not None:
        ax.scatter(xtest[:, 0], xtest[:, 1], ytest, color="red", marker="x", s=50, label="Validation Data")

    # Labels and Aesthetics
    ax.set_title(name_figure, fontsize=14)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_zlabel("Braking Distance (m)")
    ax.legend()
    
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_grid_status(cells, statuses, title="Safety Grid Status"):
    """
    Plots 2D grid cells colored by their status.
    Args:
        cells (np.array): array of shape (N, 2, 2) where each element is [bottom_left, top_right].
        statuses (np.array): array of shape (N,) with values 0, 1, or 2.
    """
    fig, ax = plt.figure(figsize=(10, 8))
    
    # Define color mapping based on status codes
    # 0: Red (Unsafe), 1: Orange (Caution), 2: Green (Safe)
    cmap = {0: 'red', 1: 'orange', 2: 'green'}
    label_map = {0: 'Unsafe (Red)', 1: 'Caution (Orange)', 2: 'Safe (Green)'}

    # Iterate through each cell and its corresponding status
    for cell, status in zip(cells, statuses):
        bottom_left = cell[0]
        top_right = cell[1]
        
        # Calculate width and height of the cell
        width = top_right[0] - bottom_left[0]
        height = top_right[1] - bottom_left[1]
        
        # Create a rectangle patch for the cell
        # We add a slight linewidth and edge color to make the grid visible
        rect = patches.Rectangle(bottom_left, width, height, 
                                 linewidth=1, edgecolor='black', 
                                 facecolor=cmap[status], alpha=0.7)
        ax.add_patch(rect)

    # Calculate overall plot limits based on the cells
    all_bottom_lefts = cells[:, 0]
    all_top_rights = cells[:, 1]
    
    x_min, y_min = np.min(all_bottom_lefts, axis=0)
    x_max, y_max = np.max(all_top_rights, axis=0)
    
    # Set plot labels and limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title(title)
    
    # Create a custom legend
    legend_elements = [patches.Patch(facecolor=cmap[i], edgecolor='black', label=label_map[i]) for i in range(3)]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.5) # Add a subtle background grid
    plt.show()
