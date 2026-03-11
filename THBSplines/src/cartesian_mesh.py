import numpy as np

from THBSplines.src.abstract_mesh import Mesh


class CartesianMesh(Mesh):
    """
    Attributes
    ---------
    knots: np.ndarray
        knot vectors defining the mesh. They are not repeated.
    dim: int
        number of dimensions
    cells: np.ndarray
        represented as AABB, i.e each cell is identified by its bottom left and top right index [[min1, max1], [min2, max2], ..., ]. 
        Is a list of N cells of shape (N, dim, 2).
    nelems: int
        number N of cells
    cell_areas: np.ndarray
        area of each cell of shape (N), 

    Methods
    ----------
    refine(): 
        returns a `CartesianMesh` with midpoints inserted in the current mesh.
    get_sub_elements(box: np.ndarray):
        Returns the indices of the cells that are contained in the region delimited by `box`,
        where `box` contains the endpoints of a box.

    """

    def plot_cells(self) -> None:
        pass

    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, knots: list, parametric_dimension: int):
        """
        Represents a regular cartesian mesh in ``parametric_dimension`` dimensions.

        :param knots: knot vectors defining the mesh, a list of lists
        :param parametric_dimension: number of parametric directions
        """
        assert len(knots)==parametric_dimension, 'Dimension and knots do not match.'
        self.knots = [np.array(np.unique(knot_v)) for knot_v in knots]
        for knot_v in self.knots:
            assert len(knot_v)>1, 'Given mesh is degenerate.'
        self.dim = parametric_dimension
        self.cells = self._compute_cells()
        self.nelems = len(self.cells)
        self.cell_areas = np.prod(np.diff(self.cells).reshape(-1, self.dim), axis=1)

    def _compute_cells(self) -> np.ndarray:
        """
        Computes an array of cells, represented as AABBs with each cell as [[min1, max1], [min2, max2], ..., ]
        :return: a list of N cells of shape (N, dim, 2).
        """
        knots_left = [k[:-1] for k in self.knots]
        knots_right = [k[1:] for k in self.knots]
        cells_bottom_left = np.stack(np.meshgrid(*knots_left), -1).reshape(-1, self.dim)
        cells_top_right = np.stack(np.meshgrid(*knots_right), -1).reshape(-1, self.dim)
        cells = np.transpose(np.stack((cells_bottom_left, cells_top_right)), (1,2,0))

        return np.squeeze(cells)
    
    def get_neighbours(self, indices: np.ndarray):
        """
        Compute the neighbours of given cell indices, counting those along the 
        diagonals in dimension 2 or higher.

        :param indices: An integer, a list, or an array of cell indices.
        :return: If `indices` is a scalar, returns a 1D np.ndarray of neighbour indices.
                 If `indices` is an array, returns a list of 1D np.ndarrays of neighbour indices.
        """
        # Determine if a single cell or multiple were requested
        is_scalar = np.isscalar(indices) or np.asarray(indices).ndim == 0
        indices = np.atleast_1d(indices)
        
        # Determine the number of cells spanning along each dimension
        L = [len(k) - 1 for k in self.knots]
        
        # 1. Determine the grid's multi-index shape. 
        # _compute_cells uses np.meshgrid's default 'xy' indexing, mapping dimensions accordingly:
        if self.dim == 1:
            shape = (L[0],)
        else:
            shape = tuple([L[1], L[0]] + L[2:])
            
        # 2. Map linear cell indices to physical grid coordinate indices
        unraveled = np.unravel_index(indices, shape)
        coords = np.empty((len(indices), self.dim), dtype=int)
        
        if self.dim == 1:
            coords[:, 0] = unraveled[0]
        else:
            coords[:, 0] = unraveled[1]  # x
            coords[:, 1] = unraveled[0]  # y
            for d in range(2, self.dim):
                coords[:, d] = unraveled[d]
                
        # 3. Generate all possible neighbour coordinate offsets for Chebyshev distance <= 1 
        #    This covers up to 3^dim combinations (includes straight edges + diagonals)
        shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * self.dim)).T.reshape(-1, self.dim)
        
        # Broadcast offsets to get all potential neighbours forming an array of shape (N, 3^dim, dim)
        neigh_coords = coords[:, None, :] + shifts[None, :, :]
        
        # 4. Filter out coordinate instances mapped strictly outside the grid's external boundaries
        valid = np.all((neigh_coords >= 0) & (neigh_coords < L), axis=2)
        
        # Flattened validation masks strictly to optimize runtime processing overhead
        flat_valid = valid.ravel()
        flat_neigh_coords = neigh_coords.reshape(-1, self.dim)[flat_valid]
        
        # 5. Map validated neighbour grid coordinates reversely into original linear cell indices
        if self.dim == 1:
            ur_coords = (flat_neigh_coords[:, 0],)
        else:
            ur_coords = [None] * self.dim
            ur_coords[0] = flat_neigh_coords[:, 1]  # y (mapped to shape[0])
            ur_coords[1] = flat_neigh_coords[:, 0]  # x (mapped to shape[1])
            for d in range(2, self.dim):
                ur_coords[d] = flat_neigh_coords[:, d]
            ur_coords = tuple(ur_coords)
            
        flat_neigh_indices = np.ravel_multi_index(ur_coords, shape)
        
        # 6. Re-split flattened validated neighbour indices back into their independent starting groupings
        valid_counts = valid.sum(axis=1)
        split_indices = np.cumsum(valid_counts)[:-1]
        neighbors_list = np.split(flat_neigh_indices, split_indices)
        
        # 7. For each input cell, exclude itself from its neighbour array and securely return them safely
        final_list = [np.sort(arr[arr != indices[i]]) for i, arr in enumerate(neighbors_list)]
        
        # Deliver a single array output if the initial payload was just a single index
        if is_scalar:
            return final_list[0]
        
        return final_list

    def refine(self) -> 'CartesianMesh':
        """
        Dyadic refinement of the mesh, by inserting midpoints in each knot vector.
        :return: a refined CartesianMesh object.
        """
        refined_knots = [
            np.sort(np.concatenate((knot_v, (knot_v[1:] + knot_v[:-1]) / 2.))) for knot_v in self.knots
        ]
        return CartesianMesh(refined_knots, self.dim)

    def get_sub_elements(self, box):
        """
        Returns the indices of the cells that are contained in the region delimited by `box`.

        :param box: numpy array containing endpoints of the box / rectangle
        :return: indices of cells contained in box
        """

        indices = []
        for i, element in enumerate(self.cells):
            condition = (element[:, 0] >= box[:, 0]) & (element[:, 1] <= box[:, 1])
            if np.all(condition):
                indices.append(i)
        return indices

if __name__ == '__main__':
    knots = [
        [0, 1, 2],
        [0, 1, 2]
    ]
    C = CartesianMesh(knots, 2)
    C1 = C.refine().refine().refine()
