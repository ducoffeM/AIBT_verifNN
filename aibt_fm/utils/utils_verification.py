import numpy as np
import itertools
from typing import Union
from airobas.verif_pipeline import (
    BoundsDomainParameter,
    ProblemContainer,
    StabilityProperty,
    StatusVerif,
    full_verification_pipeline,
)



def create_grid_cells(bottom_left:tuple[float], 
                      top_right:tuple[float], 
                      granularity: Union[int, list[int]])->list[tuple[tuple[float], tuple[float]]]:

    """
    Creates a set of grid cells covering the specified input variables.
    Args:
        bottom_left (tuple): The bottom left corner of the grid.
        top_right (tuple): The top right corner of the grid.
        granularity (Union[int, list[int]]): The granularity of the grid.
    Returns:
        list[tuple[tuple[float], tuple[float]]]: A list of grid cells, where each cell is represented by a tuple of (bottom_left, top_right).
    """
    # Create an empty list to store the bins for each variable
    bins = []
    bins_ = []

    # assert bottom_left and top_right have the same number of variables
    assert len(bottom_left)==len(top_right)

    if not isinstance(granularity, list):
        granularity = [granularity]*len(bottom_left)
    
    # Iterate over each column name to define the bins for each variable
    for min_val, max_val, n_steps in zip(bottom_left, top_right, granularity):
        
        # Create a list of 'n_steps' equally spaced values between min and max
        b = np.linspace(min_val, max_val, n_steps)
        bins.append(b)
        bins_.append(b[:-1])

    
    # Use itertools.product to get all combinations of bins, which form the grid cells
    cells = []
    # Generate all combinations of bin intervals
    for combination in itertools.product(*bins_):

        bottom_left = np.array(combination)
        top_right = np.array([bins[i][np.where(bins[i]>bottom_left[i])[0][0]] for i in range(len(bottom_left))])
        cell = [bottom_left, top_right]
        cells.append(cell)

    return np.array(cells)


def verif_with_airobas(model, cells, get_box, get_ground_dist, blocks, batch_split=10, verbose=1):
    
    def compute_input_bounds(batch_cells):
        box = get_box(batch_cells)
        return box[:, 0], box[:, 1]

    def compute_output_bounds(batch_cells):
        lower_bound = get_ground_dist(batch_cells)
        upper_bound = np.inf * np.ones_like(lower_bound) # we are not interested in under estimation
        return lower_bound, upper_bound

    # Cleverhans fgsm attack
    # followed by crown implemented from the decomon library
    # followed by exact verification Marabou
    
    input_bound_domain_param = BoundsDomainParameter()
    input_bound_domain_param.compute_lb_ub_bounds = compute_input_bounds
    
    output_bound_domain_param = BoundsDomainParameter()
    output_bound_domain_param.compute_lb_ub_bounds = compute_output_bounds

    # property: is the network prediction lies in a predefined range
    property = StabilityProperty(
        input_bound_domain_param=input_bound_domain_param, output_bound_domain_param=output_bound_domain_param
    )

    # our problem is a property + a model
    problem = ProblemContainer(
        tag_id=1,
        model=model,
        stability_property=property,
    )

    # evaluate the problem on local regions: a problem + bounds + methods (blocks)
    global_verif = full_verification_pipeline(
        problem=problem,
        input_points=cells,
        output_points=cells,
        blocks_verifier=blocks,
        verbose=verbose == 1,
        batch_split=batch_split
    )
    
    if verbose:
        print("Summary of global verification:")
        print(np.sum(global_verif.status == StatusVerif.VERIFIED), "Verified points")
        print(np.sum(global_verif.status == StatusVerif.VIOLATED), "Violated points")
        print(np.sum(global_verif.status == StatusVerif.TIMEOUT), "Timeout points")
        print(np.sum(global_verif.status == StatusVerif.UNKNOWN), "Unknown points")

    is_verified = np.where(global_verif.status == StatusVerif.VERIFIED)[0]
    is_not_verified = np.where((global_verif.status == StatusVerif.UNKNOWN) | (global_verif.status == StatusVerif.TIMEOUT))[0]
    is_unsafe = np.where(global_verif.status == StatusVerif.VIOLATED)[0]

    return is_verified, is_not_verified, is_unsafe



