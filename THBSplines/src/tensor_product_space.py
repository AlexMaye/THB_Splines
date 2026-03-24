from functools import lru_cache
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from THBSplines.src_c.BSpline import TensorProductBSpline
from THBSplines.src.abstract_space import Space
from THBSplines.src.b_spline import augment_knots, find_knot_index
from THBSplines.src.cartesian_mesh import CartesianMesh
from numba import jit

#@jit(nopython=True)
def _bezier_extraction_impl(p: int, knots: np.ndarray)->np.ndarray:

    #p = self.degree
    #knots = self.knots
    
    uniquekn = np.unique(knots)
    m=len(knots)
    a=p+1
    b=a+1
    nb = 1
    num_elements = len(uniquekn) - 1
    # Create 3d array with identity matrix for each entry
    C = np.zeros((num_elements, p + 1, p + 1))
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

class UnivariateSplineSpace(Space):
    """
    Attributes
    --------------
    - degree: int
        degree of the BSplines 

    - knots: np.ndarray
        knots associated to the BSplines, with the provided multiplicities.
    
    - nfuncs: int
        number of functions associated to these knots and degree

    - unique_knots: np.ndarray
        knots without the multiplicity

    - knot_to_unique: np.ndarray
        array to find back `self.knots` for `self.unique_knots(return_inverse=True)`. See `np.unique` for more detail
    
    - n_cells: int
        number of cells in this space

    - cell_to_last_knot: np.ndarray

    - grid_indices: np.ndarray(n_funcsx(degree+1), dtype=np.uintp)
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
    
    - basis_to_cell_indices(basis_indices)->list[np.array(dtype=np.uintp)]:
        returns the indices of the cells who make the support of the given basis.
        This function returns a list of arrays since all basis do not have the same amount of 
        cells in their support

    - refine(knots, p)->np.ndarray:
        given `knots` and a degree `p`, computes the dyadic refinement of the knots with 
        respect to `p`. Boundary multiplicities are handled automatically.

    """

    def __init__(self, degree: int, knots: np.ndarray):
        self.degree = degree
        self.knots = np.sort(knots)
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

        self.nfuncs = len(self.knots)-p1
        self.unique_knots, self.knot_to_unique = np.unique(self.knots, return_inverse=True)
        self.n_cells = len(self.unique_knots)-1
        left_boundaries = self.unique_knots[:-1]
        # 
        self.cell_to_last_knot = np.searchsorted(self.knots, left_boundaries, side='right')-1

         # --- 1D Basis Construction ---
        
        # 1. Starting indices for each basis function: [0, 1, ..., nfuncs-1]
        starts = np.arange(self.nfuncs, dtype=np.uintp)
        
        # 2. Local knots for each basis function: shape (nfuncs, p+2)
        offsets = np.arange(p1+1, dtype=np.uintp)
        self.grid_indices = starts[:, None] + offsets[None, :]
        # coordinates of supported knots (e.g [[-0.5, -0.5, -0.5, -0.25], ...])
        self.local_knots = self.knots[self.grid_indices]
        
        # 3. Parametric support bounds [start_knot, end_knot]: shape (nfuncs, 2)
        self.supports = np.column_stack((self.local_knots[:, 0], self.local_knots[:, -1]))
        
        # 4. End evaluations (touches the very last knot): shape (nfuncs,)
        # Equivalent to your: is_at_end = idx_stop_perm[:, j] == len(self.knots[j])
        self.end_evals = (starts + p1+1) == len(self.knots)

        self.Rs, _ = self.element_knot_insertion_operator()
        self.bezier = self.bezier_extraction_operator()

    def cell_to_basis(self, cell_indices):
        pass

    def basis_to_cell(self, basis_indices):
        pass

    def cell_to_basis_indices(self, cell_indices: int | np.ndarray) -> np.ndarray:
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
    
    def basis_to_cell_indices(self, basis_indices: np.ndarray)->list:
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
            return [np.arange(start_cells, end_cells, dtype=np.uintp)]
        
        # Return the range of cells
        # A list comprehension is necessary since BSplines do not span over the same amount of cells
        return [np.arange(start_cells[i], end_cells[i], dtype=np.uintp) for i in range(len(basis_indices))]
    
    def bezier_extraction_operator(self) -> np.ndarray:
        # Pass only the raw data (int and numpy array) to the JIT function
        return _bezier_extraction_impl(self.degree, self.knots)
    
    @staticmethod
    # @jit(nopython=True)
    def _oslo1(p: int, coarsekn: np.ndarray, finekn: np.ndarray, cf: int, rf: int)->np.ndarray:
        if cf >= len(coarsekn) - 1:
            return np.zeros(p + 1)

        if not (coarsekn[cf] <= finekn[rf] < coarsekn[cf+1]):
            return np.zeros(p + 1)
        b = np.zeros(p+1)
        b_temp = np.zeros_like(b)
        b[0]=1.
        for k in range(p):
            t1 = coarsekn[cf-k:cf+1]
            t2 = coarsekn[cf+1:cf+k+2]
            denom = t2-t1
            x = finekn[rf+k+1]
            w = np.zeros(len(t1))
            mask = np.abs(denom)>1e-10
            w[mask] = (x-t1)[mask]/denom[mask]
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     w = (x - t1) / denom
            #     w = np.nan_to_num(w)
            b_temp[:k+1] = (1.-w)*b[:k+1]
            b_temp[1:k+2] = b_temp[1:k+2] + w*b[:k+1]
            b[:k+2] = b_temp[:k+2]
            b_temp[:k+2] = 0.

        return b 
    
    @staticmethod
    def refine(knots: np.ndarray, p: int)->np.ndarray:
        """Given `knots`, returns its dyadic refinement with multiplicity `p+1`
        at the extremities."""
        unique_knots, mults = np.unique(knots, return_counts=True)
        q1,q2=0,0
        if mults[0]!=p+1:
            q1 = p+1-mults[0]
        if mults[-1]!=p+1:
            q2=p+1-mults[-1]
        refined_knots = (unique_knots[1:]+unique_knots[:-1])/2.
        return np.hstack((unique_knots[0]*np.ones(q1, dtype=unique_knots.dtype), np.hstack((np.sort(np.hstack((knots,refined_knots))), unique_knots[-1]*np.ones(q2, dtype=unique_knots.dtype)))))
        
    def element_knot_insertion_operator(self, coarsekn: np.ndarray=None)->np.ndarray:
        """
        Computes local refinement matrices to go from current level to next one. 

        :return: np.ndarray(#cells_level_l, p+1, p+1) of insertion operators 
        """
        p = self.degree
        coarsekn = self.knots if coarsekn is None else coarsekn
        finekn = self.refine(coarsekn, p)
        uniquekn = np.unique(finekn)
        
        m = len(finekn)
        assert m>len(coarsekn), "finekn is not finer than coarsekn"
        rf = 0
        e = 0
        #_, mults = np.unique(finekn, return_counts=True)
        R = np.zeros((len(uniquekn)-1, p+1, p+1))
        all_cfs = np.searchsorted(coarsekn, finekn, side='right') - 1
        
        max_cf = len(coarsekn) - 2 
        all_cfs = np.clip(all_cfs, p, max_cf)[:m-p-1]
        cf = all_cfs[p] 
        while rf<m-p-1:
            mult = 1
            while ((rf+mult<m)and(finekn[rf+mult] == finekn[rf])):
                mult+=1
            #mult = fine_mults[e].astype(int)
            lastcf = cf
            # while cf+2<len(coarsekn) and coarsekn[cf+1]<=finekn[rf]:
            #     cf+=1
            cf = all_cfs[rf] 
            
            if e>0:
                offs = cf-lastcf
                rows_prev = slice(offs, p + 1)
                cols_prev = slice(mult, p + 1)
                
                rows_curr = slice(0, p + 1 - offs)
                cols_curr = slice(0, p + 1 - mult)
            
                R[e,rows_curr, cols_curr] = R[e-1,rows_prev, cols_prev]
            start_col = p+1-mult
            end_col = p+1
            for col in range(start_col, end_col):
                R[e, :, col] = self._oslo1(p, coarsekn=coarsekn, finekn=finekn, cf=cf, rf=rf)
                rf+=1
            e+=1
        return R, e
    
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
        "Get the indices "
        finekn = self.knots
        factor = int(cell_size/np.max(np.diff(finekn)))
        if factor==1:
            print('Finer knots are as fine as coarse knots')
            return np.atleast_1d(coarse_cell)
        
        return np.arange(factor)+coarse_cell*factor

    def get_children_functions(self, coarse_func_idx: int, fine_space: "UnivariateSplineSpace")->np.ndarray:
        "Does not work"
        return
        coarse_cells = self.basis_to_cell_indices(basis_indices=coarse_func_idx)[0]

        children_1d=set()

        for c_cell in coarse_cells:
            R_local = self.Rs[c_cell]
            active_coarse_funcs = self.cell_to_basis_indices(cell_indices=c_cell)[0]
            local_row = np.where(active_coarse_funcs == coarse_func_idx)[0][0]
            non_zero_local_cols = np.where(R_local[local_row, :] > 1e-12)[0]
            fine_cells_in_c = fine_space._get_fine_cells_in_coarse_cell(coarse_cell=c_cell, cell_size=np.max(np.diff(self.unique_knots)))
            active_fine_funcs = fine_space.cell_to_basis_indices(cell_indices=fine_cells_in_c)
        
            # Map the non-zero local columns to global fine indices
            global_fine_children = active_fine_funcs[:, non_zero_local_cols]
            children_1d.update(global_fine_children.ravel())
            
        return np.array(list(children_1d), dtype=int)


class TensorProductSpace(Space):
    """
    Aggregates d Univariate spaces. It represents one global grid resolution

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

    cell_supports: np.ndarray(np.ndarray, dtype=np.uintp), dtype=object)
        For each basis function, holds its support cells. Since each basis function 
        does not have the same amount of support cells, this array cannot have a fixed
        width.

    basis_indices_supports = np.ndarray((#cells)x(#supported_functions), dtype=np.uintp)
        row j contains the indices of basis functions whose support includes cell j

    refinement_operators: list[scipy.sparse.bsr_array]
        holds the kronecker products of local multi-level refinement operators of the 1D spaces.

        
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
        Initializes the tensor product space by (implicitly) creating all the BSplines and
        setting their support as intervals [[[a_1^1, a_1^2], ..., [a_1^d, a_2^d]], ..., [[a_n^1, a_n^2], ..., [a_n^d, a_n^d]]]
        and with respect to the knots.
    
    refine():
        dyadic refinement of the current knots. Returns a new TensorProductSpace
        and the projection matrix to go from one space to the other.
    
    bezier_extraction_operator():
        For each dimension, computes the local Bézier extraction operators.
        The result is saved in a new attribute `self.bezier_extraction_operators`.
    
    element_knot_insertion_operators():
        For each dimension, computes local extraction operators for current knots to the next refinement level.
        The result is saved in a new attribute `self.insertion_operators`.
    
    get_element_extraction_matrix(element_idx, p, coarsekn, finekn):
        Constructs the (p+1)x(p+1) matrix M for a specific fine element 'e' such that:
        N_coarse_local(u) = M @ N_fine_local(u).
    
    
    
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
        self.dim: int = dim
        self.spaces: list[UnivariateSplineSpace] = univariate_spaces
        self.degrees: np.ndarray = np.array([space.degree for space in self.spaces])
        # The total number of basis functions in this flat space
        self.nfuncs_onedim = [space.nfuncs for space in self.spaces]
        self.nfuncs_total = np.prod(self.nfuncs_onedim)
        self.mesh = CartesianMesh([space.knots for space in self.spaces], self.dim)
        self.cell_supports = np.array(self._basis_to_cell(np.arange(self.nfuncs_total, dtype=np.uintp)), dtype=object)
        self.basis_indices_supports = self._cell_to_basis(np.arange(self.mesh.nelems, dtype=np.uintp))
        refinement_operators = []
        if self.dim==1:
            self.refinement_operators = [sp.bsr_array(self.spaces[0].Rs[i], 
                                                      blocksize=(self.degrees[0]+1, self.degrees[0]+1),
                                                      ) for i in range(len(self.spaces[0].Rs))]
        else:
            for rs0 in self.spaces[0].Rs:
                for rs1 in self.spaces[1].Rs:
                    refinement_operators.append(sp.bsr_array(sp.kron(rs0, rs1, format='bsr'), 
                                                             blocksize=(self.degrees[1]+1, self.degrees[1]+1)))
            if self.dim==3:
                refinement_operators3d = []
                for i in range(len(refinement_operators)):
                    for rs2 in self.spaces[2].Rs:
                        refinement_operators3d.append(sp.bsr_array(sp.kron(refinement_operators[i], rs2, format='bsr'),
                                                                   blocksize=(self.degrees[2]+1, self.degrees[2]+1)))
                    pass
                pass
                self.refinement_operators = refinement_operators3d
            else:
                self.refinement_operators = refinement_operators


    def basis_to_cell(self, basis_indices: np.ndarray)->np.ndarray:
        """
        Returns the indices of cells in the support of the passed basis
        indices. The 'inverse' of cell_to_basis.

        :param basis_indices: a list/array of indices
        :return: a nested list of index-sets corresponding to cells in the support of the provided basis functions.
        """
        return self.cell_supports[basis_indices]

    def _basis_to_cell(self, basis_indices: np.ndarray) -> np.ndarray:
        """
        Returns the indices of cells in the support of the passed basis
        indices. The 'inverse' of cell_to_basis.

        :param basis_indices: a list/array of indices
        :return: a nested list of index-sets corresponding to cells in the support of the provided basis functions.
        """
        if len(basis_indices)==0:
            print('No basis list was provided in basis_to_cell.')
            return np.array([], dtype=np.uintp)
        
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
    
    def cell_to_basis(self, cell_indices)->np.ndarray:
        """Returns the indices of the function supported in the given list of cells.
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
            return np.array([], dtype=np.uintp)
        
        grid_shape: tuple[int]=self.mesh.shape

        # 1. Convert flat cell index to (ix, iy) tuple
        tensor_idx = np.unravel_index(cell_list, grid_shape)
        
        # 2. Get 1D basis functions for each dimension
        expanded_basis_1d = []
        for d in range(self.dim):
            b_1d = self.spaces[d].cell_to_basis_indices(tensor_idx[d])
            target_shape = [len(cell_list)] + [1] * self.dim
            target_shape[d + 1] = self.spaces[d].degree + 1
            
            expanded_basis_1d.append(b_1d.reshape(target_shape))
            
        # 3. Convert nD tensor indices back to flat basis indices
        # The result has shape (N, p0+1, p1+1, ...)
        flat_basis_nd = np.ravel_multi_index(
            tuple(expanded_basis_1d), 
            dims=self.nfuncs_onedim,
            mode='raise'
        )
        return flat_basis_nd.reshape(len(cell_list), -1).astype(np.uintp)
        # 4. Flatten the array and return unique basis indices
        #return np.unique(flat_basis_nd)

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
        basis_function_list = np.arange(self.nfuncs_total, dtype=np.uintp)
        tensor_idx = np.unravel_index(basis_function_list, self.nfuncs_onedim)
        basis_supports=[]
        basis=[]
        for d in range(self.dim-1, -1, -1): #reverse order for lexicographic ordering
            current_space = self.spaces[d]
            supports_d = current_space.supports[tensor_idx[d]]
            basis_supports.append(supports_d)

            basis_d = current_space.knots[np.arange(current_space.nfuncs, dtype=np.uintp)[:, None] + np.arange(current_space.degree+2, dtype=np.uintp)]
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

        # 1. Generate all combinations of 1D function indices
        # idx_1d = [ [0,1,2,3], [0,1,2,3,4], ... ]
        idx_1d = [np.arange(s.nfuncs) for s in self.spaces]
        
        # Meshgrid creates the tensor topology. Shape becomes (nfuncs_total, dim)
        grid = np.meshgrid(*idx_1d, indexing='ij')
        idx_nd = np.stack(grid, axis=-1).reshape(-1, self.dim)
        
        # 2. Aggregate the nD properties
        supports_per_dim = []
        evals_per_dim = []
        basis_per_dim = []
        
        for d in range(self.dim):
            space = self.spaces[d]
            # Get the 1D indices corresponding to this specific dimension 'd'
            indices_for_dim = idx_nd[:, d]
            
            # Fetch and append the 1D properties using fancy indexing
            supports_per_dim.append(space.supports[indices_for_dim])
            evals_per_dim.append(space.end_evals[indices_for_dim])
            basis_per_dim.append(space.local_knots[indices_for_dim])
            
        # 3. Stack into the final nD arrays
        self.basis_supports = self.get_cells(range(self.nfuncs_total))
        self.basis_end_evals = np.stack(evals_per_dim, axis=1).astype(np.float32)
        
        # 4. Handle identical vs mixed degrees for self.basis
        if np.all(self.degrees == self.degrees[0]):
            self.basis = np.stack(basis_per_dim, axis=1)
        else:
            flat_basis = np.empty((self.nfuncs, self.dim), dtype=object)
            for d in range(self.dim):
                flat_basis[:, d] = list(basis_per_dim[d])
            self.basis = flat_basis

    def refine(self, dims: list)-> "TensorProductSpace":
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
        
    
    def bezier_extraction_operator(self):
        """
        For each dimension, computes the local Bézier extraction operators.
        The result is saved in a new attribute `self.bezier_extraction_operators`.
        """
        self.bezier_extraction_operators = []
        for space in range(self.spaces):
            self.bezier_extraction_operator.append(space.bezier_extraction_operator())
            
        #return bezier_extraction_operators
    
    def element_knot_insertion_operators(self):
        """
        For each dimension, computes local extraction operators for current knots to the next refinement level.
        The result is saved in a new attribute `self.insertion_operators`.
        """
        self.insertion_operators = []
        for space in self.spaces:
            self.insertion_operators.append(space.element_knot_insertion_operator())
        #return insertion_operators

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
                b = 1.
                for k in range(1, degree + 1):
                    tau1 = coarse[mu - k + 1:mu + 1]
                    tau2 = coarse[mu + 1:mu + k + 1]
                    omega = (fine[i + k] - tau1) / (tau2 - tau1)
                    b = np.append((1 - omega) * b, 0) + np.insert((omega * b), 0, 0)
                a[i, mu - degree:mu + 1] = b
            matrices.append(a[degree + 1:-degree - 1, degree + 1:-degree - 1])

        return matrices

    def get_cells(self, basis_function_list: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Given a list of indices corresponding to basis functions, return the
        union of the support-cells.

        :param basis_function_list: A list/array of indices corresponding to basis functions
        :return: the set of cells in the support of at least one basis function.
        """

        # Ensure input is a numpy array
        basis_function_list = np.atleast_1d(basis_function_list)
        tensor_idx = np.unravel_index(basis_function_list, self.nfuncs_onedim)

        supports = []
        for d in range(self.dim-1, -1, -1):
            supports_d = self.spaces[d].supports[tensor_idx[d]]
            supports.append(supports_d)
        cell_supports = np.stack(supports, axis=1)


        return cell_supports, self.basis_to_cell(basis_function_list)
        
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

    
