from functools import lru_cache
from typing import Union, List, Tuple

import numpy as np
import scipy.sparse as sp
from THBSplines.src_c.BSpline import TensorProductBSpline
from THBSplines.src.abstract_space import Space
from THBSplines.src.b_spline import augment_knots, find_knot_index
from THBSplines.src.cartesian_mesh import CartesianMesh
from scipy.interpolate import NdBSpline


class TensorProductSpace(Space):
    """
    Attributes
    ---------
    knots: np.ndarray
        knots for each dimension
    degrees: np.ndarray
        spline degrees in each direction
    dim: int 
        the number of parametric dimensions
    mesh: CartesianMesh
        underlying mesh
    nfuncs: int 
        number of functions in `basis`.
    basis_supports: np.ndarray(nfuncs x dim x 2)
        the support of each function in AABB format 
    basis: np.ndarray(nfuncs x dim x p+2)
        the knots that define the support of each function. Knots 
        are repeated if passed so.
    basis_end_evals: np.ndarray(nfuncs x dim)
        Evaluation of basis functions at the end of their cells
    nfuncs_onedim: list
        number of functions per dimension
    cell_areas: np.ndarray
        area of each cell

    Methods
    ---------
    cell_to_basis(cell_indices):
        Returns the indices of basis-functions active on the corresponding cells.
    basis_to_cell(basis_indices):
        Returns the indices of cells in the support of the passed basis
    get_cells(basis_function_list):
        Given a list of basis function indices, returns the support for each basis
        function as well as a dictionary mapping the basis function indices
        to the corresponding cell indices.
    construct_basis():
        Initializes the tensor product space by creating all the BSplines and
        setting the corresponding cell/support pointers.
    refine():
        dyadic refinement of the current knots. Returns a new TensorProductSpace
        and the projection matrix to go from one space to the other.
    construct_function(coefficients):
        constructs the spline using the basis and the provided coefficients.
    get_function_on_rectangle(rectangle):
        Returns the indices of the functions whose support contains the
        rectangle.
    """

    def __init__(self, knots: np.ndarray, degrees: np.ndarray, dim: int):
        """
        Initialize a new TensorProductSpace.

        :param knots: a np.ndarray of knot vectors 
        :param degrees: a np.ndarray of degrees corresponding to each parametric dimension
        :param dim: the number of parametric directions
        """
        self.knots = [np.array(k, dtype=np.float64) for k in knots]
        self.degrees = np.array(degrees, dtype=np.int16)
        self.dim = dim
        self.mesh = CartesianMesh(knots, dim)
        self.basis_supports = None  # a list of supports
        self.basis = None

        self.construct_basis()
        self.nfuncs = len(self.basis)
        self.nfuncs_onedim = [len(k) - d - 1 for k, d in zip(self.knots, self.degrees)]
        self.cell_areas = self.mesh.cell_areas

    def basis_to_cell(self, basis_indices: np.ndarray, tol=1e-7) -> np.ndarray:
        """
        Returns the indices of cells in the support of the passed basis
        indices. The 'inverse' of cell_to_basis.

        :param basis_indices: a list/array of indices
        :return: a nested list of index-sets corresponding to cells in the support of the provided basis functions.
        """
        if len(basis_indices)==0:
            print('No basis list was provided in basis_to_cell.')
            return np.array([], dtype=np.int32)
        selected_basis_supports = self.basis_supports[basis_indices]
        cell_starts = self.mesh.cells[:, None, :, 0]
        cell_ends = self.mesh.cells[:, None, :, 1]

        basis_starts = selected_basis_supports[None, :, :, 0]
        basis_ends = selected_basis_supports[None, :, :, 1]

        condition_dims = (cell_starts>=basis_starts-tol) & (cell_ends<=basis_ends+tol)
        valid_per_support = np.all(condition_dims, axis=2)
        valid_cell_mask = np.any(valid_per_support, axis=1)
        return np.flatnonzero(valid_cell_mask)
    
    def cell_to_basis(self, cell_list: np.ndarray, tol=1e-7) -> np.ndarray:
        """
        Returns the indices of basis functions supported over the given list of cells.
        A basis function is returned if its support contains a cell in `cell_list`.

        :param cell_list: Numpy array containing the indices of cells.
        :return: numpy array containing the indices of basis functions.
        """
        if len(cell_list) == 0:
            print('No cell list was provided in get_basis_function')
            return np.array([], np.int32)
        
        selected_cells = self.mesh.cells[cell_list]

        basis_starts = self.basis_supports[:, None, :, 0] #shape: (N_basis, 1, dim)
        # [[[b_{1,1}, b_{1,2}]], [[b_{2,1}, b_{2,2}]], ...]
        basis_ends = self.basis_supports[:, None, :, 1]

        cell_starts = selected_cells[None, :, :, 0] #shape: (1, n_cells, dim)
        # [[[c_{1,1}, c_{1,2}], 
        #   [c_{2,1}, c_{2,2}],
        # ...]]
        cell_ends = selected_cells[None, :, :, 1] 

        # broadcasting magic
        condition_dims = (basis_starts <= cell_starts+tol) & (basis_ends >= cell_ends-tol)
        # [[[b_{1,1}<=c_{1,1}, b_{1,2}<=c_{1,2}],
        #    b_{1,1}<=c_{2,1}, b_{1,2}<=c_{2,2}], ...],

        # [[b_{2,1}<=c_{1,1}, b_{2,2}<=c_{1,2}], ...] ]

        valid_per_cell = np.all(condition_dims, axis=2)
        # [[are all b_{i,j}<=c_{i,j}?]]

        # We want indices of basis functions that support ANY of the provided cells.
        # Shape: (N_basis,)
        valid_basis_mask = np.any(valid_per_cell, axis=1)
        # [is there any basis contained in at least one cell?]

        # flatnonzero returns sorted unique indices automatically because the mask is ordered.
        return np.flatnonzero(valid_basis_mask)

    def construct_basis(self):
        """
        Initializes the tensor product space by creating all the BSplines and
        setting the corresponding cell/support pointers. Only works when degrees are equal across all dimensions

        Example:
        >>> knots = [0,0,0,1,2,3,3,3]
        >>> self.basis
        [[[0. 0. 0. 1.]]
        [[0. 0. 1. 2.]]
        [[0. 1. 2. 3.]]
        [[1. 2. 3. 3.]]
        [[2. 3. 3. 3.]]]
        >>> self.basis_supports
        [[[0. 1.]]
        [[0. 2.]]
        [[0. 3.]]
        [[1. 3.]]
        [[2. 3.]]]
        >>> self.basis_end_evals
        [array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([0], dtype=int32), array([1], dtype=int32)]
        """
        dim = self.dim

        # range of indices for each B-spline in each dimension (spline i goes from knots[idx_start[i]] to knots[idx_stop[i]])
        # >>> idx_start
        # [[0 1 2 3 4]]
        idx_start_per_dim = [
            np.arange(len(self.knots[j]) - self.degrees[j] - 1) 
            for j in range(self.dim)
        ]
        # All combinations of starting and stopping indices across all dimensions
        # >>> idx_start_perm
        #[[0]
        # [1]
        # [2]
        # [3]
        # [4]]
        idx_start_perm = np.stack(np.meshgrid(*idx_start_per_dim, indexing='ij'), -1).reshape(-1, self.dim)
        offsets = self.degrees + 2
        idx_stop_perm = idx_start_perm + offsets[None, :]

        n = len(idx_start_perm)

        supports_per_dim = []
        evals_per_dim = []
        basis_per_dim = []
        for j in range(dim):
            starts = self.knots[j][idx_start_perm[:, j]]
            ends = self.knots[j][idx_stop_perm[:, j]-1]
            supports_per_dim.append(np.stack((starts, ends), axis=1))

            is_at_end = idx_stop_perm[:, j] == len(self.knots[j])
            evals_per_dim.append(is_at_end)

            d = self.degrees[j]
            # Create the window offsets: [0, 1, ..., degree+1]
            offsets = np.arange(d + 2)
            # Create a grid of indices: Shape (N, degree+2)
            # We broadcast (N, 1) + (1, degree+2)
            grid_indices = idx_start_perm[:, j, None] + offsets[None, :]
            # Fancy index into the knot array for dimension j
            # This retrieves the slices for all N functions simultaneously
            basis_per_dim.append(self.knots[j][grid_indices])

        self.basis_supports = np.stack(supports_per_dim, axis = 1)
        self.basis_end_evals = np.stack(evals_per_dim, axis=1).astype(np.float32)
        if np.all(self.degrees==self.degrees[0]):
            self.basis = np.stack(basis_per_dim, axis=1)
        else: #if degrees vary
            flat_basis = np.empty((len(idx_start_perm), self.dim), dtype=object)
            for j in range(self.dim):
                flat_basis[:, j] = list(basis_per_dim[j])
            self.basis = flat_basis
        self.nfuncs = len(self.basis)
        # print(f'basis = \n {self.basis}, \n supports = \n{self.basis_supports}, \n basis end evals = \n {self.basis_end_evals}')

    def refine(self) -> Tuple["TensorProductSpace", np.ndarray, List]:
        """
        Refine the space by dyadically inserting midpoints in the knot
        vectors, and computing the knot-insertion matrix (the projection
        matrix form coarse to fine space).

        :return: the refined TensorProductSpace along with the projection matrix that connects this space with the refined space.
        """

        coarse_knots = self.knots
        fine_knots = [insert_midpoints(knot_vector) for knot_vector in self.knots]
        #knots = np.array(knots)
        #midpoints = (knots[..., 1:]+knots[..., :-1])/2.
        #fine_knots = np.sort(np.concatenate((knots, midpoints), axis=1), axis=1)

        projection_onedim = self.compute_projection_matrix(coarse_knots, fine_knots, self.degrees)
        fine_space = TensorProductSpace(fine_knots, self.degrees, self.dim)

        return fine_space, projection_onedim

    @staticmethod #does not need to have access to class data or methods
    def compute_projection_matrix(coarse_knots, fine_knots, degrees):
        """
        Computes the full 1D projection matrix of the space corresponding to the
        fine knots with respect to the coarse knots and the spline degree.

        :param coarse_knots: list/array of coarse knot vectors, one for each parametric direction
        :param fine_knots: list/array of fine knot vectors, one for each parametric direction
        :param degrees: list of spline degrees, one for each parametri direction.

        :return: a list of 1D projection matrices corresponding to each parametric dimension
        """
        matrices = []
        for fine, coarse, degree in zip(fine_knots, coarse_knots, degrees):
            coarse = augment_knots(coarse, degree)
            fine = augment_knots(fine, degree)
            m = len(fine) - (degree + 1)
            n = len(coarse) - (degree + 1)

            a = sp.lil_matrix((m, n), dtype=np.float64)
            fine = np.array(fine, dtype=np.float64)
            coarse = np.array(coarse, dtype=np.float64)
            for i in range(m):
                mu = find_knot_index(fine[i], coarse)
                b = 1
                for k in range(1, degree + 1):
                    tau1 = coarse[mu - k + 1:mu + 1]
                    tau2 = coarse[mu + 1:mu + k + 1]
                    omega = (fine[i + k] - tau1) / (tau2 - tau1)
                    b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)
                a[i, mu - degree:mu + 1] = b
            matrices.append(a[degree + 1:-degree - 1, degree + 1:-degree - 1])

        return matrices

    def get_cells(self, basis_function_list: np.ndarray, tol: float=1e-7) -> Tuple[np.ndarray, dict]:
        """
        Given a list of indices corresponding to basis functions, return the
        union of the support-cells, and a dictionary mapping basis_function
        to cell index.

        :param basis_function_list: A list/array of indices corresponding to basis functions
        :return: the set of cells in the support of at least one basis function, and a dictionary mapping basis functions to their cell indices.
        """

        # Ensure input is a numpy array
        basis_function_list = np.asarray(basis_function_list)
        
        if len(basis_function_list) == 0:
            return np.array([], dtype=int), {}

        # 1. Retrieve Geometries
        # subset_supports shape: (N_subset, dim, 2)
        subset_supports = self.basis_supports[basis_function_list]
        # all_cells shape: (N_cells, dim, 2)
        all_cells = self.mesh.cells

        basis_start = subset_supports[:, :, 0][:, None, :]
        basis_end   = subset_supports[:, :, 1][:, None, :]

        # Cells: (1, N_cells, dim)
        cells_start = all_cells[:, :, 0][None, :, :]
        cells_end   = all_cells[:, :, 1][None, :, :]

        condition_dims = (cells_start + tol >= basis_start) & (cells_end <= basis_end + tol)
        is_supported_mask = np.all(condition_dims, axis=2)

        valid_cells_mask = np.any(is_supported_mask, axis=0)
        unique_cells = np.flatnonzero(valid_cells_mask)

        # rows tells how many times a B-spline is supported on the cells, e.g rows=array([0, 1, 1, 1, 1, 2, 2, 2]) means that given B-spline 0's
        # support spans 1 cell, given B-spline 1 support spans 4 cells given spline 2 support spans 3 cells.
        # cols tells us which cell is in the support of the B-spline in `row`, e.g cols=array([6, 0, 1, 3, 4, 0, 1, 2]) means that
        # B-spline 0 is supported on cell 6, B-spline 1 is supported on cells 0,1,3,4 and B-spline 2 is supported on cells 0,1,2.
        rows, cols = np.nonzero(is_supported_mask)

        # unique rows stores which B-splines are active on the cells, and split_indices tells us where we change spline in `rows`, e.g
        # it will be split_indices=arrays([0,1,5]) in the example above.
        unique_rows, split_indices = np.unique(rows, return_index=True)
        # grouped cells are the indices of cells who are included in the support of B-spline j, e.g
        # grouped_cells = [array([6]), array([0, 1, 3, 4]), array([0, 1, 2])]
        grouped_cells = np.split(cols, split_indices[1:]) #

        # Map back to real Basis IDs
        cells_map = {}
        for i, row_idx in enumerate(unique_rows):
            real_basis_id = basis_function_list[row_idx]
            cells_map[real_basis_id] = grouped_cells[i]

        return unique_cells, cells_map

    def construct_function(self, coefficients):
        """
        Constructs a linear combination of THB-splines, given coefficients.
        
        :param coefficients: an array of spline coefficients
        :return: a callable spline-function
        """

        assert len(coefficients) == len(self.basis)

        # def f(x):
        #     return sum([c * self.construct_B_spline(i)(x) for i, c in enumerate(coefficients)])

        return NdBSpline(self.knots, coefficients, self.degrees, extrapolate=False)

    def get_functions_on_rectangle(self, rectangle):
        """
        Returns the indices of the functions whose support contains the
        rectangle.

        :param rectangle: np.array containing rectangle endpoints
        :return: function indices
        """
        condition = (self.basis_supports[:, :, 0] <= rectangle[:, 0]) & (
                self.basis_supports[:, :, 1] >= rectangle[:, 1])
        i = np.flatnonzero(np.all(condition, axis=1))
        return i

    @lru_cache()
    def construct_B_spline(self, i):
        """
        Return a Callable TensorProductBSpline. This is cached to avoid
        re-creating BSplines.

        :param i: index of basis function
        :return: a callable TensorProductBSpline
        """

        return TensorProductBSpline(self.degrees, self.basis[i], self.basis_end_evals[i])


class TensorProductSpace2D(TensorProductSpace):


    # def construct_basis(self):
    #     """

    #     """

    #     knots_u = self.knots[0]
    #     knots_v = self.knots[1]
    #     deg_u = self.degrees[0]
    #     deg_v = self.degrees[1]

    #     lenu = len(knots_u)
    #     n = lenu - deg_u - 1
    #     lenv = len(knots_v)
    #     m = lenv - deg_v - 1

    #     b_splines_end_evals = np.zeros((n * m, 2), dtype=np.intc)
    #     b_support = np.zeros((n * m, self.dim, 2))

    #     index = 0
    #     for j in range(m):
    #         for i in range(n):
    #             b_support[index] = [[knots_u[i], knots_u[i + deg_u + 1]], [knots_v[j], knots_v[j + deg_v + 1]]]
    #             offset_u = (i + deg_u + 2)
    #             offset_v = (j + deg_v + 2)
    #             b_splines_end_evals[index] = [lenu == offset_u, lenv == offset_v]

    #             index += 1
    #     self.basis_supports = b_support
    #     self.basis_end_evals = b_splines_end_evals
    #     self.nfuncs = n * m
    #     self.dim_u = n
    #     self.dim_v = m
    #     self.nfuncs_onedim = [n, m]
    #     self.basis = [0]*(n * m)

    def refine(self) -> Tuple["TensorProductSpace2D", np.ndarray, List]:
        """
        Refine the space by dyadically inserting midpoints in the knot
        vectors, and computing the knot-insertion
        matrix (the projection matrix form coarse to fine space).
        
        :return: refined TensorProductSpace and a projection matrix
        """

        coarse_knots = self.knots
        fine_knots = [insert_midpoints(knot_vector, degree) for knot_vector, degree in zip(self.knots, self.degrees)]

        projection_onedim = self.compute_projection_matrix(coarse_knots, fine_knots, self.degrees)
        fine_space = TensorProductSpace2D(fine_knots, self.degrees, self.dim)

        return fine_space, projection_onedim

    @lru_cache()
    def construct_B_spline(self, i):
        """
        Return a Callable TensorProductBSpline

        :param i: index of B-spline
        :return: callable B-spline function
        """

        ind_v = i // self.dim_u
        ind_u = i % self.dim_u

        knots = np.array([self.knots[0][ind_u : ind_u + self.degrees[0] + 2],
            self.knots[1][ind_v : ind_v + self.degrees[1] + 2]], dtype=np.float64)
        return TensorProductBSpline(self.degrees, knots, self.basis_end_evals[i])


def insert_midpoints(knots):
    """
    Inserts midpoints in all interior knot intervals of a p+1 regular knot
    vector.

    :param knots: p + 1 regular knot vector to be refined
    :param p: spline degree
    :return: refined_knots
    """

    knots = np.array(knots, dtype=np.float64)
    unique_knots = np.unique(knots)
    midpoints = (unique_knots[:-1] + unique_knots[1:]) / 2.

    return np.sort(np.concatenate((knots, midpoints)))



if __name__ == '__main__':
    knots = [
        [0,0,0,1.1,2.9,3,3,3],
        #[-3,-3,-3, -1, 0.2, 2,2,2],
        #[0,0,0,1,2,3,3,3]
    ]
    d = 2*np.ones(len(knots))
    dim = len(knots)

    T = TensorProductSpace(knots, d, dim)
    #T.construct_basis
    print(T.mesh.cells)
    print(T.cell_areas)

    
