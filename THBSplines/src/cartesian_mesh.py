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
        self.knots = [np.array(np.unique(knot_v)) for knot_v in knots]
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
