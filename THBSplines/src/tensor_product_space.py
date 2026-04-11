from typing import Optional
import numpy.typing as npt

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import BSpline, NdBSpline
from THBSplines.src.cartesian_mesh import CartesianMesh
from numba import jit

#@jit(nopython=True)
def _bezier_extraction_impl(p: int, knots: npt.ArrayLike)->npt.NDArray[np.float64]:
    """ Based on the paper 'Isogeometric finite element data structures based on Bézier extraction of T-splines'
      ( https://doi.org/10.1002/nme.2968), algorithm 1."""
    
    #p = self.degree
    #knots = self.knots
    
    uniquekn = np.unique(knots)
    m=len(knots)
    a=p+1
    b=a+1
    nb = 1
    num_elements = len(uniquekn) - 1
    # Create 3d array with identity matrix for each entry
    C: npt.NDArray[np.float64] = np.zeros((num_elements, p + 1, p + 1))
    for i in range(num_elements): #for numba
        C[i] = np.eye(p + 1, dtype=np.float64)
    #C = np.repeat(np.identity(p+1, dtype=np.float64)[None, ...], len(uniquekn)-1, axis=0)
    #C = np.array([np.eye(p+1, dtype=np.float64) for _ in range(len(uniquekn)-1)])
    while b<m:
        i=b
        while b<m and knots[b]==knots[b-1]: #needed for break condition
            b+=1
        mult=b-i+1
        if mult<p:
            numer = knots[b-1]-knots[a-1]
            # alphas = np.zeros((p-mult, ))
            knots2 = knots[a-1]
            knots1 = knots[a+mult:a+p]
            denom = knots1-knots2
            alphas = numer/denom
            r=p-mult
            for j in range(1, r+1):
                save=r-j+1
                s=mult+j
                this_alphas = alphas[:p-s+1]
                #C[nb-1, :, s:p+1] = this_alphas*C[nb-1, :, s:p+1] + (1.-this_alphas)*C[nb-1, :, s-1:p]
                for row in range(p+1): #for numba
                    C[nb-1, row, s:p+1] = this_alphas*C[nb-1, row, s:p+1] + (1.-this_alphas)*C[nb-1, row, s-1:p]
                pass
                if b<m:
                    C[nb, save-1:j+save, save-1]=C[nb-1, p-j:p+1, p]
                pass
            pass
        pass
        nb+=1
        if b<m:
            a=b
            b+=1
        pass
    pass
    return C

#@jit(nopython=True)
def _oslo1(p: int, coarsekn: npt.ArrayLike, finekn: npt.ArrayLike, cf: int, rf: int)->npt.NDArray[np.float64]:
    """ 
    From [Multi-level Bézier extraction for hierarchical local refinement of Isogeometric Analysis](https://doi.org/10.1016/j.cma.2017.08.017),
    algorithm 1.
    """
    p1: int = p+1
    if cf >= len(coarsekn) - 1:
        return np.zeros(p1)

    if not (coarsekn[cf] <= finekn[rf] < coarsekn[cf+1]):
        return np.zeros(p1)
    b: npt.NDArray[np.float64] = np.zeros(p1)
    b_temp: npt.NDArray[np.float64] = np.zeros_like(b)
    b[0]=1.
    for k in range(p):
        t1: npt.ArrayLike = coarsekn[cf-k:cf+1]
        t2: npt.ArrayLike = coarsekn[cf+1:cf+k+2]
        denom: npt.ArrayLike = t2-t1
        x: npt.ArrayLike = finekn[rf+k+1]
        w: npt.NDArray[np.float64] = np.zeros(k+1) #k+1=len(t1)
        nnz: npt.NDArray[np.bool_] = np.abs(denom)>1e-10
        w[nnz] = (x-t1)[nnz]/denom[nnz]
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     w = (x - t1) / denom
        #     w = np.nan_to_num(w)
        b_temp[:k+1] = (1.-w)*b[:k+1]
        b_temp[1:k+2] = b_temp[1:k+2] + w*b[:k+1]
        b[:k+2] = b_temp[:k+2]
        b_temp[:k+2] = 0.

    return b 

#@jit(nopython=True)
def _knot_insertion_impl(p: int, coarsekn: np.ndarray, finekn: np.ndarray)->np.ndarray:
    """
    From [Multi-level Bézier extraction for hierarchical local refinement of Isogeometric Analysis](https://doi.org/10.1016/j.cma.2017.08.017),
    algorithm 3.
    """
    uniquekn = np.unique(finekn)
    rf = 0
    e = 0
    p1 = p+1
    m = len(finekn)
    R = np.zeros((len(uniquekn)-1, p1, p1))
    all_cfs = np.searchsorted(coarsekn, finekn, side='right') - 1
    
    max_cf = len(coarsekn) - 2 
    # all_cfs = np.clip(all_cfs, p, max_cf)[:m-p-1]
    all_cfs = np.minimum(max_cf, np.maximum(all_cfs, p))[:m-p-1]
    cf = all_cfs[p] 
    
    while rf<m-p-1:
        mult = 1
        while ((rf+mult<m)and(finekn[rf+mult] == finekn[rf])):
            mult+=1
        pass
        
        lastcf = cf
        cf = all_cfs[rf] 
        
        if e>0:
            offs = cf-lastcf
        
            R[e, 0:p1-offs, 0:p1-mult] = R[e-1, offs:p1, mult:p1]
        pass

        for col in range(p1-mult, p1):
            R[e, :, col] = _oslo1(p, coarsekn=coarsekn, finekn=finekn, cf=cf, rf=rf)
            rf+=1
        pass

        e+=1
    pass

    return R, e

class UnivariateSplineSpace():
    """

    This class represents a space with univariate B-Splines based on a knot vector. 
    It also implements many utility functions that serve truncated hierarchical B-Splines.

    Attributes
    --------------
    - degree: int
        degree of the B-Splines.

    - knots: np.ndarray
        knots associated to the B-Splines, with the provided multiplicities.
    
    - nfuncs: int
        number of functions associated to these knots and degree.

    - unique_knots: np.ndarray
        knots without the multiplicity.

    - knot_to_unique: np.ndarray
        array to find back `self.knots` for `self.unique_knots(return_inverse=True)`. See `np.unique` for more detail.
    
    - n_cells: int
        number of cells in this space

    - cell_to_last_knot: np.ndarray

    - grid_indices: np.ndarray(n_funcsx(degree+1), dtype=np.int32)
        knots indices that hold each basis function

    - local_knots: np.ndarray(nfuncsx(degree+1), dtype=np.float64)
        knots values that hold each basis function

    - supports: np.ndarray(nfuncsx2, dtype=np.float64)
        each basis function support interval's boundaries

    - end_evals: np.ndarray(nfuncs)
        value of each basis function at the end of its support

    - Rs: np.ndarray(n_cellsx(degree+1)x(degree+1), dtype=np.float64)
        local multi-level extraction operators to go from `self.knots` to their dyadic refinement.
    
    - bezier: np.ndarray(n_cellsx(degree+1)x(degree+1), dtype=np.float64)
        local bezier extraction matrices



    Methods
    ------------
    - cell_to_basis_indices(cell_indices)->np.ndarray:
        returns the indices of basis functions supported over the given cell indices
    
    - basis_to_cell_indices(basis_indices)->list[np.array(dtype=np.int32)]:
        returns the indices of the cells who make the support of the given basis.
        This function returns a list of arrays since all basis do not have the same amount of 
        cells in their support

    - refine(knots, p, n_times)->np.ndarray:
        given `knots` and a degree `p`, computes the dyadic refinement `n_times` of the knots with 
        respect to `p`. Boundary multiplicities are handled automatically.

    - element_knot_insertion_operator():
        Computes local refinement matrices to go from current level to next one.

    - evaluate_BSpline(point, coeffs):
        performs the evaluation of the BSplines at the given point.
    
    - get_element_extraction_matrix(element_idx):
        Constructs the (p+1)x(p+1) matrix M for a specific fine element 'e' such that:
        N_coarse_local(u) = M @ N_fine_local(u).

    - get_children_functions(indices):
        Given the indices of BSplines in the current space, returns a list of indices of children BSplines
        in the dyadic refinement of the current space.


    """

    def __init__(self, degree: int, knots: npt.ArrayLike):
        self.degree = degree
        self.knots: npt.NDArray = np.sort(knots)
        uniquekn, mults = np.unique(self.knots, return_counts=True)
        p1 = self.degree+1
        assert np.all(np.diff(uniquekn)==np.diff(uniquekn)[0]), "Only equally spaced knots are supported."
        if mults[0]<p1:
            self.knots = np.append(self.knots[0]*np.ones(p1-mults[0], dtype=self.knots.dtype), self.knots)
        elif mults[0]>p1:
            self.knots = np.delete(self.knots, np.arange(mults[0]-p1))
        if mults[-1]<p1:
            self.knots = np.append(self.knots, self.knots[-1]*np.ones(p1-mults[-1], dtype=self.knots.dtype))
        elif mults[-1]>self.degree+1:
            self.knots = np.delete(self.knots, np.arange(len(self.knots)-1, len(self.knots)-mults[-1]+p1-1, -1))

        self.nfuncs: int = len(self.knots)-p1
        self.unique_knots, self.knot_to_unique = np.unique(self.knots, return_inverse=True)
        self.n_cells: int = len(self.unique_knots)-1
        left_boundaries = self.unique_knots[:-1]
        # 
        self.cell_to_last_knot: npt.NDArray = np.searchsorted(self.knots, left_boundaries, side='right')-1

         # --- 1D Basis Construction ---
        
        # Starting indices for each basis function: [0, 1, ..., nfuncs-1]
        starts: npt.NDArray[np.int_] = np.arange(self.nfuncs, dtype=np.int32)
        
        # Local knots for each basis function: shape (nfuncs, p+2)
        offsets: npt.NDArray[np.int_] = np.arange(p1+1, dtype=np.int32)
        self.grid_indices = starts[:, None] + offsets[None, :]
        # coordinates of supported knots (e.g [[-0.5, -0.5, -0.5, -0.25], ...])
        self.local_knots: npt.NDArray = self.knots[self.grid_indices]
        
        # Parametric support bounds [start_knot, end_knot]: shape (nfuncs, 2)
        self.supports: npt.NDArray = np.column_stack((self.local_knots[:, 0], self.local_knots[:, -1]))
        
        # End evaluations (touches the very last knot): shape (nfuncs,)
        self.end_evals: npt.NDArray[np.bool_] = (starts + p1+1) == len(self.knots)

        self.Rs, _ = self.element_knot_insertion_operator()
        self.bezier = self.bezier_extraction_operator()

    def cell_to_basis_indices(self, cell_indices: int | npt.NDArray[np.int_] | list[int]) -> npt.NDArray[np.int_]:
        """
        Takes a single cell index or an array of N cell indices.
        Returns an array of shape (N, p+1) containing the basis indices.
        """
        cell_indices = np.atleast_1d(cell_indices)
        assert np.max(cell_indices)<self.n_cells, 'The space does not have as many cells.'
        # j has shape (N,)
        j = self.cell_to_last_knot[cell_indices]
        
        # offsets has shape (p+1,) -> [-p, -p+1, ..., 0]
        offsets = np.arange(-self.degree, 1, dtype=j.dtype)
        
        # Broadcasting: (N, 1) + (1, p+1) -> shape (N, p+1)
        basis_idx = j[:, None] + offsets[None, :]
        
        return basis_idx
    
    def basis_to_cell_indices(self, basis_indices: int|npt.NDArray[np.int_]|list[int])->list[npt.NDArray[np.int32]]:
        """
        Returns the indices of the physical cells supported by these basis functions.
        """
        # Basis 'i' is defined by the knot span [u_i, u_{i+p+1}]
        basis_indices = np.atleast_1d(basis_indices)
        assert np.max(basis_indices)<self.nfuncs, "The space does not have as many functions."
        assert np.min(basis_indices)>=0, "Indices must be non-negative."
        start_knot_idxs = basis_indices
        end_knot_idxs = basis_indices + self.degree + 1
        
        # Map those knots to physical cell boundaries
        start_cells = self.knot_to_unique[start_knot_idxs]
        end_cells = self.knot_to_unique[end_knot_idxs]

        if len(basis_indices)==1:
            return [np.arange(start_cells, end_cells, dtype=np.int32)]
        
        # Return the range of cells
        # A list comprehension is necessary since BSplines do not span over the same amount of cells
        return [np.arange(start_cells[i], end_cells[i], dtype=np.int32) for i in range(len(basis_indices))]
    
    def bezier_extraction_operator(self) -> npt.NDArray[np.float64]:
        # Pass only the raw data (int and numpy array) to the JIT function
        return _bezier_extraction_impl(self.degree, self.knots)
    
    @staticmethod
    def refine(knots: npt.ArrayLike, p: int, n_times: int=1)->npt.NDArray:
        """Given `knots`, returns its dyadic refinements done `n_times` with multiplicity `p+1`
        at the extremities."""
        knots = np.asarray(knots, dtype=np.float64)
        mult_left = np.searchsorted(knots, knots[0], side='right')
        mult_right = len(knots) - np.searchsorted(knots, knots[-1], side='left')
        pad_left = max(0, p + 1 - mult_left)
        pad_right = max(0, p + 1 - mult_right)
        if pad_left > 0 or pad_right > 0:
            knots = np.concatenate((
                np.full(pad_left, knots[0], dtype=knots.dtype),
                knots,
                np.full(pad_right, knots[-1], dtype=knots.dtype)
            ))
        if n_times == 0:
            return knots
        
        # Find indices where the knot value changes
        jump_idx = np.where(knots[1:] > knots[:-1])[0]
        left_vals = knots[jump_idx]
        right_vals =knots[jump_idx + 1]
        num_new_points = (1<<n_times)-1
        fractions = np.linspace(0.,1.,num_new_points+2)[1:-1]
        new_points = left_vals[:, None] + (right_vals - left_vals)[:, None] * fractions[None, :]
        new_points = new_points.ravel()
        insert_positions = np.repeat(jump_idx + 1, num_new_points)
        
        return np.insert(knots, insert_positions, new_points)

    def element_knot_insertion_operator(self, coarsekn: np.ndarray=None)->np.ndarray:
        """
        Computes local refinement matrices to go from current level to next one. 

        :return: np.ndarray(#cells_level_l, p+1, p+1) of insertion operators 
        """
        p = self.degree
        coarsekn = self.knots if coarsekn is None else coarsekn
        finekn = self.refine(coarsekn, p, n_times=1)
        # uniquekn = np.unique(finekn)
        
        m = len(finekn)
        assert m>len(coarsekn), "finekn is not finer than coarsekn"
        R, e = _knot_insertion_impl(p=p, coarsekn=coarsekn, finekn=finekn)
        return R,e
    
    def evaluate_BSpline(self, point, coeffs=None):
        assert np.min(self.knots)<=point<np.max(self.knots), "Point is outside the domain."

        design_matrix = BSpline.design_matrix(point, self.knots, self.degree, extrapolate=False)      
        rows, cols = design_matrix.nonzero()
        nnz_design_matrix = design_matrix[rows, cols]

        if coeffs is None:
            coeffs = np.ones(design_matrix.shape[1])
        else:
            assert len(coeffs)==self.degree+1, "Different amount of BSplines and coefficients."
        return nnz_design_matrix@coeffs
    
    
    def get_element_extraction_matrix(self, element_idx: int, coarsekn: np.ndarray=None):
        """
        Constructs the (p+1)x(p+1) matrix M for a specific fine element 'e' such that:
        N_coarse_local(u) = M @ N_fine_local(u).

        Given an element_idx = [e_n, e_{n+1}), the function finds the p associated basis spline functions {b_{n_1}, ..., b_{n_p}} whose support
        contains element_idx. Each time we go forward a knot in finekn, we discard a basis function and pick the next one.
        Therefore, to find b_{n_1}, we need to know how many functions previous functions are not active on element_idx. Since
        each basis function is associated to a knot, counting multiplicity, we need to count how many knots were already discarded.
        Using that the empty sum is equal to 0, and that the first basis function b_0 only has support on element 0, we must 
        skip the first p+1 terms in finekn when counting the discarded functions.

        Functions {b_{n_2}, ... b_{n_p}} follow b_{n_1} in order.

        Since knots and basis functions are linked, we can also get the starting point of each b_{n_i}, that we use to compute in which
        element of the coarse knots each basis function begins. We know have the relevant cfs and rfs to give to the Oslo function.

        To populate the matrix, one can observe that if element_idx belongs to coarse element l-1, then only its last p entry should
        be put into the first p entries of the relevant column of the matrix.
        
        Rows of M: Active Coarse Basis Functions
        Cols of M: Active Fine Basis Functions
        """
        coarsekn = self.knots if coarsekn is None else coarsekn
        p=self.degree
        finekn = self.refine(knots=coarsekn, p=p)
        _, fine_multiplicities = np.unique(finekn[p+1:], return_counts=True) # Assuming knot 0 has multiplicity p+1
        rf1 = fine_multiplicities[:element_idx].sum() #Find first fine function that has support on element. The sum is to take knot mults into account
        rfs = range(rf1, rf1+p+1) # Each interval supports p+1 basis functions
        cfs = np.searchsorted(coarsekn, finekn[rfs], 'right') -1 # Find starting point of coarse interval corresponding to starting point of each fine function.
        a = p+1 - (np.max(cfs) - cfs) # Determine how many elements from oslo to take and how to put them into the matrix
        M = np.zeros((p+1, p+1), dtype=np.float64) #Prepare output matrix
        for i in range(p+1):
            weights = self._oslo1(p, coarsekn=coarsekn, finekn=finekn, cf=cfs[i], rf=rfs[i])
            M[:a[i], i] = weights[p+1-a[i]:]

        return M
    
    @staticmethod
    def _check_subspace_property(coarse_knots: np.ndarray, fine_knots: np.ndarray, tol=1e-10) -> bool:
        """
        Verifies if V_coarse is a subspace of V_fine by checking knot multiplicities.
        Returns True if valid, False otherwise.
        """
        # 1. Get unique knots and their counts (multiplicities)
        # Using np.unique with return_counts is efficient and handles repeats
        # Note: rounding is crucial for float comparison
        c_unique, c_counts = np.unique(np.round(coarse_knots, decimals=10), return_counts=True)
        f_unique, f_counts = np.unique(np.round(fine_knots, decimals=10), return_counts=True)
        
        # 2. Iterate through each unique coarse knot
        for i, knot_val in enumerate(c_unique):
            # A. Check existence in fine vector
            # searchsorted finds the insertion index; we verify value match
            idx = np.searchsorted(f_unique, knot_val)
            
            if idx >= len(f_unique) or abs(f_unique[idx] - knot_val) > tol:
                print(f"Violation: Knot {knot_val} exists in Coarse but NOT in Fine.")
                return False
                
            # B. Check multiplicity
            # Coarse multiplicity must be <= Fine multiplicity
            if c_counts[i] > f_counts[idx]:
                print(f"Violation at {knot_val}: Coarse Multiplicity ({c_counts[i]}) > Fine Multiplicity ({f_counts[idx]})")
                return False
                
        return True
    

    def _get_fine_cells_in_coarse_cell(self, coarse_cell: int, cell_size: float):
        "Given the indices in the current space, returns the indices of the children cells in the dyadic refinement."
        finekn = self.knots
        factor = int(cell_size/np.max(np.diff(finekn)))
        if factor==1:
            print('Finer knots are as fine as coarse knots')
            return np.atleast_1d(coarse_cell)
        
        return np.arange(factor)+coarse_cell*factor

    def get_children_functions(self, indices: list[int]|npt.NDArray[np.int_]|int)->list[npt.NDArray[np.int32]]:
        """Given the indices of BSplines in the current space, returns a list of indices of children BSplines
        in the dyadic refinement of the current space."""
        
        p1 = self.degree+1
        indices = np.atleast_1d(indices)
        knots = self.knots
        unique_knots = self.unique_knots
        left_knots = knots[indices]
        right_knots = knots[indices+p1]

        insert_left = np.searchsorted(unique_knots, left_knots, side='left')
        insert_right = np.searchsorted(unique_knots, right_knots, side='left')

        j_min = indices + insert_left
        j_max = indices + insert_right

        return [np.arange(start, end + 1, dtype=np.int32) 
            for start, end in zip(j_min, j_max)]


class TensorProductSpace():
    """
    Aggregates d Univariate spaces. It represents one global grid resolution.
    This class does not currently support spaces of dimension 4 or more.

    Attributes
    ---------
    dim: int 
        the number of parametric dimensions

    mesh: CartesianMesh
        underlying mesh

    spaces: list[UnivariateSplineSpace]
        holds the knots of each dimension and 1D information about the corresponding BSplines.

    nfuncs_onedim: list[int]
        number of functions in each space

    nfuncs_total: int
        number of function in the current TensorProductSpace
    
    nfuncs_nextdim: list[int]
        number of functions per dimension after a dyadic refinement of the current mesh    

    cell_supports: np.ndarray(np.ndarray, dtype=np.int32), dtype=object)
        For each basis function, holds its support cells. Since each basis function 
        does not have the same amount of support cells, this array cannot have a fixed
        width.

    basis_indices_supports = np.ndarray((#cells)x(#supported_functions), dtype=np.int32)
        row j contains the indices of basis functions whose support includes cell j

    refinement_operators: list[scipy.sparse.csc_array]
        holds the kronecker products of local multi-level refinement operators of the 1D spaces.

    bezier_operators: list[scipy.sparse.bsr_array]
        holds the kronecker products of local bezier extraction operators.

        
    Methods
    ---------
    cell_to_basis(cell_indices):
        Returns the indices of basis-functions active on the corresponding cells.
    
    basis_to_cell(basis_indices):
        Returns the indices of cells in the support of the passed basis

    get_children_functions(indices):
        Given indices of functions in the current space, returns the indices
        of the corresponding children functions if a dyadic refinement of the 
        current space was made.
    
    construct_basis():
        Initializes the tensor product space by (implicitly) creating all the BSplines and
        setting their support as intervals [[[a_1^1, a_1^2], ..., [a_1^d, a_2^d]], ..., [[a_n^1, a_n^2], ..., [a_n^d, a_n^d]]]
        and with respect to the knots.
    
    refine():
        dyadic refinement of the current knots. Returns a new TensorProductSpace.
        
    evaluate_BSpline(point, coeffs):
        evaluates B-Splines a the current point with the given coefficients for each
        B-Spline.
    
    get_element_extraction_matrix(element_idx, p, coarsekn, finekn):
        Constructs the (p+1)x(p+1) matrix M for a specific fine element 'e' such that:
        N_coarse_local(u) = M @ N_fine_local(u).

    get_cells(basis_function_list):
        Given a list of basis function indices, returns the support for each basis
        function as well as a dictionary mapping the basis function indices
        to the corresponding cell indices.
    
    
    
    """

    def __init__(self, dim: int, univariate_spaces: list[UnivariateSplineSpace]):
        """
        Represents ONE flat level in the THB hierarchy.

        :param knots: a np.ndarray of knot vectors 
        :param degrees: a np.ndarray of degrees corresponding to each parametric dimension
        :param dim: the number of parametric directions
        """
        assert len(univariate_spaces) == dim
        assert dim<=3, "Dimensions higher than 3 are not supported."
        mode = "sparse" #"dense"
        self.dim: int = dim
        self.spaces: list[UnivariateSplineSpace] = univariate_spaces
        self.degrees: npt.NDArray[np.int_] = np.array([space.degree for space in self.spaces])
        # The total number of basis functions in this flat space
        self.nfuncs_onedim: npt.NDArray[np.int32] = np.array([space.nfuncs for space in self.spaces], dtype=np.int32)
        self.nfuncs_nextdim = self.nfuncs_onedim + np.array([len(np.where(self.spaces[d].knots[1:]>self.spaces[d].knots[:-1])[0]) for d in range(self.dim)])
        self.nfuncs_total: int = np.prod(self.nfuncs_onedim)
        self.mesh = CartesianMesh([space.knots for space in self.spaces], self.dim)
        self.cell_supports = np.array(self._basis_to_cell(np.arange(self.nfuncs_total, dtype=np.int32)), dtype=object)
        self.basis_indices_supports = self._cell_to_basis(np.arange(self.mesh.nelems, dtype=np.int32))
        
        self._set_bezier_and_refinements(mode=mode)

    def _set_bezier_and_refinements(self, mode: str="sparse"):
        """Initialises the attributes `self.refinement_operators` and `self.bezier_operators`, either with dense
        numpy arrays or sparse scipy arrays depending on `mode`. 
        """

        # In dense mode, these are stored in Fortran order, since they are only on the right-hand side during matrix-matrix multiplication
        # and need a lot of slicing during the truncation phase.
        refinement_operators: list[sp.csc_array[np.float64]] = []
        # In dense mode, these are in C-order because we mainly multiply from the left
        bezier_operators: list[sp.bsr_array[np.float64]] = []

        if self.dim==1:
            # store everything in dense format since the arrays are not sparse.
            self.bezier_operators = self.spaces[0].bezier                          
            self.refinement_operators = np.asfortranarray(self.spaces[0].Rs)

        else: #dim = 2 or 3
            if mode == "sparse":
                sp0 = self.spaces[0]
                sp1 = self.spaces[1]
            
                bezier_operators = [sp.bsr_array(sp.kron(bz0, bz1, format='bsr'), blocksize=(self.degrees[1]+1, self.degrees[1]+1)) 
                                    for bz0 in sp0.bezier for bz1 in sp1.bezier]
                
                refinement_operators = [sp.csc_array(sp.kron(rs0, rs1, format='csc')) 
                                        for rs0 in sp0.Rs for rs1 in sp1.Rs]

            else: # dense mode
                refinement_operators = np.kron(self.spaces[0].Rs, self.spaces[1].Rs)
                bezier_operators = np.kron(self.spaces[0].bezier, self.spaces[1].bezier)
            
            if self.dim==3:
                sp2 = self.spaces[2]
                if mode=="sparse":
                    self.refinement_operators = [sp.csc_array(sp.kron(rsi, rs2, format='csc')) 
                                                 for rsi in refinement_operators for rs2 in sp2.Rs]
                    
                    self.bezier_operators = [sp.bsr_array(sp.kron(bzi, bz2, format='bsr'), blocksize=(self.degrees[2]+1, self.degrees[2]+1)) 
                                             for bzi in bezier_operators for bz2 in sp2.bezier]
                    
                
                else: # dense mode
                    refinement_operators3d = np.kron(refinement_operators, self.spaces[2].Rs)
                    self.refinement_operators = np.asfortranarray(refinement_operators3d)
                    bezier_operators3d = np.kron(bezier_operators, self.spaces[2].bezier)
                    self.bezier_operators = bezier_operators3d
                    
            else: # if self.dim==2
                self.bezier_operators = bezier_operators

                if mode=="sparse":
                    self.refinement_operators = refinement_operators
                else:
                    self.refinement_operators = np.asfortranarray(refinement_operators)

                


    def basis_to_cell(self, basis_indices: int|list[int]|npt.NDArray[np.int_])->npt.NDArray:
        """
        Returns the indices of cells in the support of the passed basis
        indices. The 'inverse' of cell_to_basis.

        :param basis_indices: a list/array of indices
        :return: a nested list of index-sets corresponding to cells in the support of the provided basis functions.
        """
        return self.cell_supports[basis_indices]

    def _basis_to_cell(self, basis_indices: int|list[int]|npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """
        Returns the indices of cells in the support of the passed basis
        indices. The 'inverse' of cell_to_basis.

        :param basis_indices: a list/array of indices
        :return: a nested list of index-sets corresponding to cells in the support of the provided basis functions.
        """
        if len(basis_indices)==0:
            print('No basis list was provided in basis_to_cell.')
            return np.array([], dtype=np.int32)
        
        #grid_shape = self.mesh.shape
        tensor_idx = np.unravel_index(basis_indices, self.nfuncs_onedim)
        cells_1d = []
        for d in range(self.dim):
            cells_1d.append(self.spaces[d].basis_to_cell_indices(tensor_idx[d]))
        
        nd_cells_list = []

        for i in range(len(basis_indices)):
            # Extract the 1D cell arrays for the i-th basis function across all dimensions
            grids_1d = [cells_1d[d][i] for d in range(self.dim)]
            
            # Create the Cartesian product of these 1D cell arrays
            mesh_grids = np.meshgrid(*grids_1d, indexing='ij')
            
            flat_cells = np.ravel_multi_index(tuple(mesh_grids), self.mesh.shape)
            
            nd_cells_list.append(flat_cells.ravel())
            
        return nd_cells_list
    
    def cell_to_basis(self, cell_indices: int|list[int]|npt.NDArray[np.int_])->npt.NDArray[np.int_]:
        """Returns the indices of the function supported in the given list of cells.
        A basis function is returned if its support contains a cell in `cell_indices`.
        The "inverse" of basis_to_cell.
        
        :param cell_indices: a int/list/array of indices
        """
        return self.basis_indices_supports[cell_indices]

    def _cell_to_basis(self, cell_list: np.ndarray) -> np.ndarray:
        """
        Returns the indices of basis functions supported over the given list of cells.
        A basis function is returned if its support contains a cell in `cell_list`.

        :param cell_list: Numpy array containing the indices of cells.
        :return: numpy array containing the indices of basis functions.
        """
        cell_list = np.atleast_1d(cell_list)
        if len(cell_list) == 0:
            print('No cell list was provided in cell_to_basis().')
            return np.array([], dtype=np.int32)
        
        grid_shape: tuple[int]=self.mesh.shape

        # Convert flat cell index to (ix, iy) tuple
        tensor_idx = np.unravel_index(cell_list, grid_shape)
        
        # Get 1D basis functions for each dimension
        expanded_basis_1d = []
        for d in range(self.dim):
            b_1d = self.spaces[d].cell_to_basis_indices(tensor_idx[d])
            target_shape = [len(cell_list)] + [1] * self.dim
            target_shape[d + 1] = self.spaces[d].degree + 1
            
            expanded_basis_1d.append(b_1d.reshape(target_shape))
            
        # Convert nD tensor indices back to flat basis indices
        # The result has shape (N, p0+1, p1+1, ...)
        flat_basis_nd = np.ravel_multi_index(
            tuple(expanded_basis_1d), 
            dims=self.nfuncs_onedim,
            mode='raise'
        )
        return flat_basis_nd.reshape(len(cell_list), -1).astype(np.int32)
        # Flatten the array and return unique basis indices
        #return np.unique(flat_basis_nd)

    def get_children_functions(self, indices: int|list[int]|npt.NDArray[np.int_])->list[npt.NDArray[np.int32]]:
        """
        Returns functions that are children of the given functions, given a dyadic refinement of the mesh.
        Using the inclusion property V^l ⊂ V^{l+1}, a BSpline b^l ∈ V^l can be expressed as a linear combinations
        of BSplines {b_i^{l+1} ∈ V^{l+1}} with some real coefficients {c_i^{l+1}}: 

        b^l(x) = Σ_i c_i^{l+1} b_i^{l+1}(x).

        This functions returns the indices of the basis functions b_i^{l+1} that have non-zero coefficients in the sum above.
        
        """
        
        indices = np.atleast_1d(indices)
        multi_index = np.unravel_index(indices, self.nfuncs_onedim) # does not exist in numba
        children_funcs = []
        children_1d = []
        for d in range(self.dim):
            # spaces[d].get_children_functions(multi_index[d]) returns a 
            # list of 1D arrays containing the 1D fine indices.
            children_1d.append(self.spaces[d].get_children_functions(multi_index[d]))

        children_funcs = []
        for k in range(len(indices)):
            # Extract the 1D children for the k-th B-spline across all dimensions
            # For a 2D spline, this gets c_x and c_y.
            c_k = [children_1d[d][k] for d in range(self.dim)]
            
            # Create a D-dimensional grid of fine indices (the Tensor Product)
            # indexing='ij' ensures the matrix layout matches numpy's standard unravel/ravel rules.
            grids = np.meshgrid(*c_k, indexing='ij')
            
            # Ravel the D-dimensional fine indices back into 1D flat fine indices
            flat_children_k = np.ravel_multi_index(tuple(grids), dims=self.nfuncs_nextdim) # does not exist in numba
            
            # Flatten the grid into a single 1D array of integers
            children_funcs.append(flat_children_k.flatten().astype(np.int32))
            
        return children_funcs


    def construct_basis(self):
        """
        Initializes the tensor product space by creating all the BSplines and
        setting the corresponding cell/support pointers.

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
        basis_function_list = np.arange(self.nfuncs_total, dtype=np.int32)
        tensor_idx = np.unravel_index(basis_function_list, self.nfuncs_onedim)
        basis_supports=[]
        basis=[]
        for d in range(self.dim-1, -1, -1): #reverse order for lexicographic ordering
            current_space = self.spaces[d]
            supports_d = current_space.supports[tensor_idx[d]]
            basis_supports.append(supports_d)

            basis_d = current_space.knots[np.arange(current_space.nfuncs, dtype=np.int32)[:, None] + np.arange(current_space.degree+2, dtype=np.int32)]
            basis.append(basis_d[tensor_idx[d]])
        pass
        self.basis_supports = np.stack(basis_supports, axis=1)
        if np.all(self.degrees==self.degrees[0]):
            self.basis = np.stack(basis, axis=1)
        else:
            flat_basis = np.empty((self.nfuncs_total, self.dim), dtype=object)
            for d in range(self.dim):
                flat_basis[:, d] = list(basis[d])
            self.basis = flat_basis
        return

    def refine(self, dims: list[int])-> "TensorProductSpace":
        """
        Refine the space by dyadically inserting midpoints in the knot
        vectors, and computing the knot-insertion matrix (the projection
        matrix form coarse to fine space).

        :return: the refined TensorProductSpace along with the projection matrix that connects this space with the refined space.
        """
        dims = np.atleast_1d(dims)
        assert len(dims)<=self.dim
        uni_spline_spaces = self.spaces
        fine_uss = []
        for i, spline_space in enumerate(uni_spline_spaces):
            new_spline_space: np.ndarray = spline_space.refine(knots=spline_space.knots, p=spline_space.degree) if i in dims else spline_space
            fine_uss.append(UnivariateSplineSpace(degree=spline_space.degree, knots=new_spline_space))

        #projection_onedim = self.compute_projection_matrix(coarse_knots, fine_knots, self.degrees)
        fine_space = TensorProductSpace(dim=self.dim, univariate_spaces=fine_uss)

        return fine_space#, projection_onedim
    
    def evaluate_BSpline(self, point: float| npt.ArrayLike, coeffs: Optional[npt.ArrayLike]=None)->npt.NDArray[np.float_]:
        point = np.atleast_1d(point)
        assert len(point) == self.dim, "Provided point does not have the appropriate amount of dimensions."
        
        if self.dim==1:
            return self.spaces[0].evaluate_BSpline(point, coeffs)
        else:
            design_matrix = NdBSpline.design_matrix(point, tuple([self.spaces[i] for i in range(self.dim)]), self.degrees[0], extrapolate=False)
            nnz1, nnz2 = design_matrix.nonzero()
            if coeffs is not None:
                return design_matrix[nnz1, nnz2] @ coeffs
            else:
                return design_matrix[nnz1, nnz2]
        
    
    
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
        pass
        # matrices = []
        # for fine, coarse, degree in zip(fine_knots, coarse_knots, degrees):
        #     coarse = augment_knots(coarse, degree)
        #     fine = augment_knots(fine, degree)
        #     m = len(fine) - (degree + 1)
        #     n = len(coarse) - (degree + 1)

        #     a = sp.lil_matrix((m, n), dtype=np.float64)
        #     fine = np.array(fine, dtype=np.float64)
        #     coarse = np.array(coarse, dtype=np.float64)
        #     for i in range(m):
        #         mu = find_knot_index(fine[i], coarse)
        #         b = 1.
        #         for k in range(1, degree + 1):
        #             tau1 = coarse[mu - k + 1:mu + 1]
        #             tau2 = coarse[mu + 1:mu + k + 1]
        #             omega = (fine[i + k] - tau1) / (tau2 - tau1)
        #             b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)
        #         a[i, mu - degree:mu + 1] = b
        #     matrices.append(a[degree + 1:-degree - 1, degree + 1:-degree - 1])

        # return matrices

    def get_cells(self, basis_functions_list: np.ndarray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Given a list of indices corresponding to basis functions, return the
        union of the support-cells.

        :param basis_function_list: A list/array of indices corresponding to basis functions
        :return: the set of cells in the support of at least one basis function.
        """

        # Ensure input is a numpy array
        basis_function_list = np.atleast_1d(basis_functions_list)
        tensor_idx = np.unravel_index(basis_function_list, self.nfuncs_onedim)

        supports = []
        for d in range(self.dim-1, -1, -1):
            supports_d = self.spaces[d].supports[tensor_idx[d]]
            supports.append(supports_d)
        cell_supports = np.stack(supports, axis=1)


        return cell_supports, self.basis_to_cell(basis_function_list)



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

    
