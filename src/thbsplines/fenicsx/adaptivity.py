import numpy as np
from thbsplines.hierarchical_space import HierarchicalSpace
import dolfinx.fem as dolfinx_fem

def dorfler_marking(hierarchical_space: HierarchicalSpace, theta: float, 
                    local_error_form):

    local_error_vector = dolfinx_fem.assemble_vector(local_error_form)
    local_error_vector.scatter_forward()

    # cell_errors = np.sqrt(local_error_vector.array)
    squared_errors = np.abs(local_error_vector.array)# cell_errors**2
    total_squared_error = np.sum(squared_errors)
    descending_indices = np.flip(np.argsort(squared_errors))
    sorted_squared_errors = squared_errors[descending_indices]
    cumulative_errors = np.cumsum(sorted_squared_errors)
    threshold_value = theta*total_squared_error
    num_cells_to_mark = max(2, np.searchsorted(cumulative_errors, threshold_value)+1)
    top_error_indices = descending_indices[:num_cells_to_mark]
    print(f"Total cells marked via Dörfler (theta={theta}): {num_cells_to_mark} out of {len(squared_errors)}")
    print(f"Indices to refine: {top_error_indices[:10]}")
    
    my_arr = []
    hs = hierarchical_space
    for value in hs.hmesh.aelem_level.values():
        my_arr.extend(value)
    err_cells = {}
    sorted_top_error_indices = np.sort(top_error_indices)
    level=0
    current_length = len(hs.hmesh.aelem_level[0])
    for i in sorted_top_error_indices:
        while i > current_length-1:
            level+=1
            current_length+=len(hs.hmesh.aelem_level[level])
        
        if level not in err_cells:
            err_cells[level]=[]
        
        err_cells[level].append(my_arr[i])

    return err_cells
