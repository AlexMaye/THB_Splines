import numpy as np
from THBSplines.src.cartesian_mesh import CartesianMesh
from THBSplines.src.hierarchical_space import HierarchicalSpace

import dolfinx.mesh as dolfinx_mesh
import dolfinx.fem as dolfinx_fem
import dolfinx.cpp as dolfinx_cpp
from dolfinx.common import IndexMap as dolfinx_common_IndexMap
from basix.ufl import element as basix_ufl_element
from mpi4py import MPI
from dolfinx import default_scalar_type
dtype = default_scalar_type
from ufl import TestFunction, Measure, inner, CellDiameter

import numpy.typing as npt

def refine(knots: npt.ArrayLike, p: int, n_times: int=1)->npt.NDArray:
    """Given `knots`, returns its dyadic refinement with multiplicity `p+1`
    at the extremities."""
    knots: npt.NDArray[np.float_] = np.asarray(knots)
    mult_left: int = np.searchsorted(knots, knots[0], side='right')
    mult_right: int = len(knots) - np.searchsorted(knots, knots[-1], side='left')
    pad_left: int = max(0, p + 1 - mult_left)
    pad_right: int = max(0, p + 1 - mult_right)
    if pad_left > 0 or pad_right > 0:
        knots = np.concatenate((
            np.full(pad_left, knots[0], dtype=knots.dtype),
            knots,
            np.full(pad_right, knots[-1], dtype=knots.dtype)
        ))
    if n_times == 0:
        return knots
    
    # Find indices where the knot value changes
    jump_idx: npt.NDArray[np.int_] = np.where(knots[1:] > knots[:-1])[0]
    left_vals = knots[jump_idx]
    right_vals =knots[jump_idx + 1]
    num_new_points = (1<<n_times)-1
    fractions = np.linspace(0.,1.,num_new_points+2)[1:-1]
    new_points = left_vals[:, None] + (right_vals - left_vals)[:, None] * fractions[None, :]
    new_points = new_points.ravel()
    insert_positions = np.repeat(jump_idx + 1, num_new_points)
    
    return np.insert(knots, insert_positions, new_points)

def build_mesh_2d(hs, mapping= lambda x: x):
    total_active_cells = sum(len(hs.hmesh.aelem_level[l]) for l in range(hs.nlevels))

    all_cells = np.empty((4*total_active_cells, 2), dtype=np.float64) # will have coarser cells on top and finer on bottom
    thb_operators: dict[tuple[int, int], npt.NDArray[np.float64]] = {}
    N_max = 0 # maximum amount of dofs in a cell
    current_idx = 0
    for l in range(hs.nlevels):
        active_cells_l = hs.hmesh.aelem_level[l]
        if len(active_cells_l)==0:
            continue
    
        thb_operators_list = hs.local_multi_level_extraction_operator2(active_cells_l, l, l)
        thb_operators.update({(l, cell): op for cell, op in zip(active_cells_l, thb_operators_list)})
        
        if thb_operators_list:
            level_max = max(op.shape[0] for op in thb_operators_list)
            N_max = max(N_max, level_max)

        mesh = CartesianMesh(hs.hmesh.one_d_indices[l], len(hs.hmesh.one_d_indices[l]))
        my_cells_l = mesh.cells[active_cells_l]

        n_cells = len(my_cells_l)
        x_coords = my_cells_l[:, 0, :]
        y_coords = my_cells_l[:, 1, :] 

        start, end = current_idx, current_idx+(4*n_cells)

        view = all_cells[start:end]
        #points = np.zeros((4 * len(my_cells_l), 2), dtype=np.float64)
        view[::4] = np.column_stack((x_coords[:, 0], y_coords[:, 0]))  # Bottom-left
        view[1::4] = np.column_stack((x_coords[:, 1], y_coords[:, 0]))  # Bottom-right
        view[2::4] = np.column_stack((x_coords[:, 0], y_coords[:, 1]))  # Top-left
        view[3::4] = np.column_stack((x_coords[:, 1], y_coords[:, 1]))  # Top-right

        current_idx=end
    pass
    all_cells = np.array(all_cells).reshape(-1, 2)
    all_cells = mapping(all_cells)

    coordinates = np.arange(len(all_cells), dtype=np.int32).reshape(-1, 4)
    coordinate_element = basix_ufl_element("Q", "quadrilateral", 1, shape=(2,))
    disconnected_mesh = dolfinx_mesh.create_mesh(MPI.COMM_WORLD, cells=coordinates, e=coordinate_element, x=all_cells)
    midpoints = np.mean(all_cells.reshape(-1, 4, 2), axis=1)

    return disconnected_mesh, thb_operators, N_max, midpoints

def build_dofmap(hierarchical_space, mesh, N_max, morton=False):
    hs = hierarchical_space
    disconnected_mesh = mesh

    if morton:
        print("Building dof map with Morton code ordering")
        dofmap, dummy_dof_index = hs.build_morton_dof_map()
    else:
        print("Building dof map with hierarchical ordering.")
        dofmap, dummy_dof_index = hs.build_global_dof_map()
    dummy_dof_index += 1 

    # num_control_points_with_dummy = dummy_dof_index+1

    num_cells = disconnected_mesh.topology.index_map(disconnected_mesh.topology.dim).size_global

    # which BSpline functions are active on each cell, padded with the dummy index.
    padded_cells_to_dofs = np.full((num_cells, N_max), dummy_dof_index, dtype=np.int32)

    # Find the maximum local index for each level to size arrays
    max_local_indices = [0] * hs.nlevels
    for l, local_idx in dofmap.keys():
        if local_idx > max_local_indices[l]:
            max_local_indices[l] = local_idx
        pass
    pass

    # convert the dictionary into a list of numpy arrays for faster lookup
    dofmap_arrays = [np.full(size + 1, dummy_dof_index, dtype=np.int32) for size in max_local_indices]
    for (l, local_idx), global_idx in dofmap.items():
        dofmap_arrays[l][local_idx] = global_idx


    cell_index = 0
    for l in range(hs.nlevels):
        active_cells_l = hs.hmesh.aelem_level[l]
        for cell in active_cells_l:
            
            # Get the local functions on the cell
            active_funcs_dict: dict[int, npt.NDArray[np.int_]] = hs.get_all_active_functions_on_cell(l, cell)
            
                
            #  Vectorized conversion from local to global indices
            mapped_arrays = [
                dofmap_arrays[ll][local_funcs] for ll, local_funcs in active_funcs_dict.items() if len(local_funcs) > 0
            ]
            
            #
            if mapped_arrays: # Check if there are actually active functions
                global_dofs = np.concatenate(mapped_arrays)
                n_dofs = len(global_dofs)
                padded_cells_to_dofs[cell_index, :n_dofs] = global_dofs
                
            cell_index += 1

    return dofmap, padded_cells_to_dofs

def fill_function_space(hierachical_space, mesh, N_max, thb_operators: dict, mapping_function=None):
    disconnected_mesh = mesh
    hs = hierachical_space
    p0 = hs.degrees[0]

    M = hs._bezier_to_legendre(degree = p0)
    S_indices = np.arange(p0+1, dtype=np.float64)
    # scaling for unnormalised Legendre basis polynomials
    S_inv = (1./np.sqrt(2.*S_indices+1.))*np.identity(p0+1, dtype=np.float64) # Scaling factor, since fenicsx uses orthonormal legendre polynomials
    T = np.asfortranarray(np.kron(M, M).T @ np.kron(S_inv, S_inv), dtype=dtype)

    operator_shape = (N_max, T.shape[1])
    # Create a custom space that holds the content of each matrix for the relevant cell.
    # degree 0 because the value is constant over each cell
    C_space = dolfinx_fem.functionspace(disconnected_mesh, ("DG", 0, operator_shape))
    C_func = dolfinx_fem.Function(C_space, dtype=dtype)

    num_cells_local = disconnected_mesh.topology.index_map(disconnected_mesh.topology.dim).size_local
    indices = np.arange(num_cells_local, dtype=np.int32)
    # To make sure that each matrix is assigned to the correct cell
    midpoints: npt.NDArray[np.float_] = dolfinx_mesh.compute_midpoints(disconnected_mesh, disconnected_mesh.topology.dim, indices)

    c_values = C_func.x.array.reshape((-1, N_max, T.shape[1]))

    if mapping_function is None:
        mapping_function = hs.hmesh.find_active_cell

    for local_idx, midpoint in enumerate(midpoints):
        #print(f"midpoint = {midpoint}")
        level, idx = mapping_function(midpoint[:hs.dim])
        #print(f"midpoint = {midpoint}, level={level}, idx={idx}")
        mat: npt.NDArray[np.float_] = thb_operators[level, idx] @ hs.level_spaces[level].get_bezier_operator(idx)
        
        real_k, n_cols = mat.shape

        if real_k<N_max:
            padding_size = N_max - real_k
            
            #mat_padded = np.vstack((mat, np.zeros((padding_size, mat.shape[1])) ))
            Ci = mat#_padded
            to_stack = np.zeros((padding_size, n_cols), dtype=np.float64)
        else:
            Ci = mat
            to_stack = np.zeros((0, n_cols), dtype=np.float64)
        
        # is a view of C_func.x.array, therefore we modify the content of C_func.x.array
        # No new array is created, the matrix->cell mapping is done here.
        c_values[local_idx, :, :] = np.vstack((Ci@T, to_stack))
    C_func.x.scatter_forward()
    return C_func, C_space

def create_spline_space(cells_to_dofs: npt.NDArray[np.int_], mesh, N_max:int, mult_factor:int=1, cell_type="quadrilateral"):
    
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global

    num_control_points = np.max(cells_to_dofs)+1
    my_index_map = dolfinx_common_IndexMap(comm=mesh.comm, 
                                        local_size=mult_factor*num_control_points)

    dummy_element = basix_ufl_element(
        family="DG", 
        cell=cell_type, 
        degree=0, 
        shape=(mult_factor*N_max,)
    )
    dummy_space = dolfinx_fem.functionspace(mesh=mesh, element=dummy_element)
    element_layout = dummy_space.dofmap.dof_layout
    cpp_element = dummy_space.element._cpp_object

    #dof_indices = cells_to_dofs.ravel().astype(np.int32)
    vector_dofs = np.zeros((num_cells, mult_factor*N_max), dtype=np.int32)
    for i in range(mult_factor):
        vector_dofs[:, i::mult_factor]=mult_factor*cells_to_dofs+i
    dof_indices = vector_dofs.ravel().astype(np.int32)

    offsets = (np.arange(num_cells + 1, dtype=np.int32) * mult_factor*N_max).astype(np.int32)

    adj = dolfinx_cpp.graph.AdjacencyList_int32(data=dof_indices, offsets=offsets)

    dofmap = dolfinx_cpp.fem.DofMap(
        element_dof_layout=element_layout, 
        index_map=my_index_map,  
        index_map_bs=1, 
        dofmap=adj, 
        bs=1
    )

    V_spline_cpp = dolfinx_cpp.fem.FunctionSpace_float64(
        mesh=mesh._cpp_object, 
        element=cpp_element, 
        dofmap=dofmap
    )

    V_spline = dolfinx_fem.FunctionSpace(
        mesh=mesh, 
        element=dummy_element, 
        cppV=V_spline_cpp
    )

    return V_spline

def dorfler_marking(hierarchical_space: HierarchicalSpace, theta: float, 
                    local_error_form):

    local_error_vector = dolfinx_fem.assemble_vector(local_error_form)
    local_error_vector.scatter_forward()

    cell_errors = np.sqrt(local_error_vector.array)
    squared_errors = cell_errors**2
    total_squared_error = np.sum(squared_errors)
    descending_indices = np.flip(np.argsort(squared_errors))
    sorted_squared_errors = squared_errors[descending_indices]
    cumulative_errors = np.cumsum(sorted_squared_errors)
    threshold_value = theta*total_squared_error
    num_cells_to_mark = max(2, np.searchsorted(cumulative_errors, threshold_value)+1)
    top_error_indices = descending_indices[:num_cells_to_mark]
    print(f"Total cells marked via Dörfler (theta={theta}): {num_cells_to_mark} out of {len(cell_errors)}")
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