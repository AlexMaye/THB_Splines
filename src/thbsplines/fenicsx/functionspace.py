import numpy as np
import numpy.typing as npt

from thbsplines.hierarchical_space import HierarchicalSpace

import dolfinx.fem as dolfinx_fem
import dolfinx.mesh as dolfinx_mesh
import dolfinx.fem as dolfinx_fem
import dolfinx.cpp as dolfinx_cpp

from dolfinx.common import IndexMap as dolfinx_common_IndexMap
from basix.ufl import element as basix_ufl_element

from dolfinx import default_scalar_type
dtype = default_scalar_type

def build_dofmap(hierarchical_space, mesh, N_max, morton=False)->dict[tuple[int, int], int]:
    """Builds a degree of freedom map to indicate where each active function
    should be mapped to in the mass/stiffness matrix. 
    
    :param hierarchical_space: HierarchicalSpace which holds the B-Splines informations
    :param: mesh: dolfinx mesh
    :param N_max: maximum amount of active B-Splines on a cell in `mesh`.
    :param morton: whether to use a morton code for the dofmap or not. This usually
     has better locality than the default mapping.
      
    :returns dof_map: dictionary that indicates how each active function is mapped
    :returns padded_cells_to_dofs: numpy array that shows active functions on each cell.
     This is padded with a dummy index if there are more than one active level in the mesh
      so that it can be a viable numpy array of shape Ncells x `N_max`."""
    hs = hierarchical_space
    disconnected_mesh = mesh

    if morton:
        # print("Building dof map with Morton code ordering")
        dofmap, dummy_dof_index = hs.build_morton_dof_map()
    else:
        # print("Building dof map with hierarchical ordering.")
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
    """Defines a function space with constant values that holds the local multi-level extraction operator
    for each cell.
    
    :param hierarchical_space: HierarchicalSpace
    :param mesh: dolfinx mesh
    :param N_max: maximum amount of active B-Splines on a cell in `mesh`.
    :param thb_operators: local multi-level extraction operators. These will be padded with zeros 
    until they have an appropriate shape, depending on `N_max`, transformed and given to the custom space
    :param mapping_function: if the physical domain is not equal to the parametric domain, this function 
    should map physical cell midpoints to parametric cell midpoints

    :returns C_func: dolfinx.fem.Function, constant function with appropriate operators for each cell
    :returns C_space: dolfinx.fem.functionspace, space corresponding to `C_func`.
    """
    disconnected_mesh: dolfinx_fem.mesh = mesh
    hs: HierarchicalSpace = hierachical_space
    p0: int = hs.degrees[0]

    M: npt.NDArray[np.float64] = hs._bezier_to_legendre(degree = p0)
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
    c_values.fill(0.)

    if mapping_function is None:
        mapping_function = hs.hmesh.find_active_cell

    for local_idx, midpoint in enumerate(midpoints):
        #print(f"midpoint = {midpoint}")
        level, idx = mapping_function(midpoint[:hs.dim])
        # thb_operators: dict[tuple[int, int], npt.NDArray[np.float64]]
        mat: npt.NDArray[np.float_] = thb_operators[level, idx] @ hs.level_spaces[level].get_bezier_operator(idx)
        
        real_k, n_cols = mat.shape

        # if real_k<N_max:
        #     padding_size = N_max - real_k
            
        #     #mat_padded = np.vstack((mat, np.zeros((padding_size, mat.shape[1])) ))
        #     Ci = mat#_padded
        #     to_stack = np.zeros((padding_size, n_cols), dtype=np.float64)
        # else:
        #     Ci = mat
        #     to_stack = np.zeros((0, n_cols), dtype=np.float64)
        
        # is a view of C_func.x.array, therefore we modify the content of C_func.x.array
        # No new array is created, the matrix->cell mapping is done here.
        #c_values[local_idx, :, :] = np.vstack((Ci@T, to_stack))
        c_values[local_idx, :real_k, :] = mat@T
    C_func.x.scatter_forward()
    return C_func, C_space

def create_spline_space(cells_to_dofs: npt.NDArray[np.int_], mesh, N_max:int, mult_factor:int=1, cell_type="quadrilateral",
                        dtype=np.float64):
    
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_global

    num_control_points = np.max(cells_to_dofs)+1
    my_index_map = dolfinx_common_IndexMap(comm=mesh.comm, 
                                        local_size=mult_factor*num_control_points)

    dummy_element = basix_ufl_element(
        family="DG", 
        cell=cell_type, 
        degree=0, 
        shape=(mult_factor*N_max,),
        dtype=dtype
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