import numpy as np
from thbsplines.cartesian_mesh import CartesianMesh
import numpy.typing as npt
import dolfinx.mesh as dolfinx_mesh
from basix.ufl import element as basix_ufl_element
from mpi4py import MPI

from scipy.spatial import KDTree

class FastMidpointMapper:
    def __init__(self, hs, all_midpoints: npt.NDArray[np.float64]):
        self.hs = hs
        # Build a KD-Tree for O(N log N) nearest-neighbor lookups
        self.tree = KDTree(all_midpoints)
        
        # Precompute the prefix sums of elements per level to avoid the while-loop
        self.level_offsets = [0]
        for lvl in range(len(hs.hmesh.aelem_level)):
            self.level_offsets.append(self.level_offsets[-1] + len(hs.hmesh.aelem_level[lvl]))
        self.level_offsets = np.array(self.level_offsets)

    def __call__(self, midpoint):
        # Find the global index in O(log N)
        _, global_index = self.tree.query(midpoint)
        
        # Binary search to instantly find the correct level
        level = np.searchsorted(self.level_offsets, global_index, side='right') - 1
        
        # Calculate local index within that level
        local_index = global_index - self.level_offsets[level]
        
        return level, self.hs.hmesh.aelem_level[level][local_index]

def build_mesh_2d(hs, mapping= lambda x: x):
    """Builds a 2D space and deforms it with the provided mapping. 
    The local multi-level extraction operators are also computed and returned.
    
    :returns mesh: dolfinx mesh
    :returns thb_operators: dictionary of local multi-level extraction operator by level 
    and active element
    :returns N_max: integer that represents the maximum amount of active B-Splines on a cell.
    :returns midpoints: middle points of cells, which is handy to fill the function space when a deformation was passed."""
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