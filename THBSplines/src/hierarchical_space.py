import numpy.typing as npt

import numpy as np
import scipy.sparse as sp
from THBSplines.src.hierarchical_mesh import HierarchicalMesh, sorted_isin
from THBSplines.src.tensor_product_space import TensorProductSpace, UnivariateSplineSpace
from copy import deepcopy
import itertools
from math import prod

from numba import njit

@njit 
def sorted_isin_2d(ar1_2d, ar2_1d):
    """Vectorized isin for 2D ar1 and sorted 1D ar2."""
    rows, cols = ar1_2d.shape
    total_size = rows*cols
    flat_ar1 = ar1_2d.reshape(-1)
    idx = np.searchsorted(ar2_1d, flat_ar1)

    #valid_mask = idx<len(ar2_1d)
    my_len = len(ar2_1d)

    result_flat = np.zeros(flat_ar1.shape, dtype=np.bool_)
    for i in range(total_size):
        if idx[i]<my_len:
            if ar2_1d[idx[i]] == flat_ar1[i]:
                result_flat[i] = True
                
    return result_flat.reshape((rows, cols))


class HierarchicalSpace():
    """
    This class holds everything related to a B-Spline hierarchical space, with the meshes of all different levels,
    active and inactive cells/functions, and change of basis matrices to go from univariate Legendre basis polynomials to Bernstein
    basis polynomials and vice-versa, from multivariate Bernstein basis polynomials to multivariate B-Splines, and the local 
    truncation refinement operators to go from coarse B-Splines to their finer truncated counterparts.

    When considering a sequence of nested B-Spline spaces V^0 ⊂ ... ⊂ V^N defined on a domain Ω, each space V^l spanned by the normalised 
    B-Spline basis B^l attached to a knots vector k^l. One can consider a subset of elements e^l that partitions the underlying domain, 
    which will be referred to as active cells. The union of these active cells is denoted as Ω^l.

    Attributes
    ---------------------
    - degree: list[int]
        list of degrees of the univariate B-Splines in [x,y,z] order.

    - dim: int
        number of dimensions in the space. This is currently capped to 3.

    - nlevels: int
        number of refinement levels. Starts at 1.

    - hmesh: HierarchicalMesh
        custom class which keeps track of active/inactive cells in the tensor product mesh.

    - level_spaces: dict[int, TensorProductSpace]
        holds full tensor product spaces with associated mesh, B-Splines and operators for a given level.

    - active_functions: dict[int, np.ndarray(np.int32)]
        keeps track of active functions at each level. A function of level l is said to be _active_ if 
        it has support on Ω^l. 

    - deactivated_functions: dict[int, np.ndarray(np.int32)]
        keeps track of inactive functions at each level. A function of level l is said to be _inactive_
        if it does not have support on Ω^l.

    - Bl_minus: dict[int, np.ndarray(np.int32)]
        keeps track of functions of level l whose support overlaps the coarser domain Ω^{l-1}.

    - truly_active: dict[int, np.ndarray(np.int32)]
        keeps track of active functions of level l whose support does not overlap the coarser domain Ω^{l-1}.

    - active_cell_counts: dict[int, np.ndarray(np.int32)]
        keeps track of the number of active cell at each level.

    - refinement_operators: dict[int, list[scipy.sparse.csc_array]]
        Stores the operators that express a B-Spline of level l on cell i_{l+1} as a linear
        combinations of B-Splines of level l+1 on cell i_{l+1}.

    - bezier_operators: dict[int, list[scipy.sparse.bsr_array]]
        Stores the operators that express a Bernstein basis polynomial of level l on cell i
        as a linear combination of B-Splines of level l on cell i.
    """

    def __init__(self, knots: list, degrees: list[int]):
        """Initialise one level """
        self.degrees = np.atleast_1d(degrees)
        assert np.min(degrees)>=0, "Negative degrees are not allowed."
        if len(self.degrees)==1:
            self.degrees = np.full(len(knots), self.degrees, dtype=np.int32) #self.degrees*np.ones(len(knots), dtype=np.intp)
        else:
            assert len(knots)==len(self.degrees), "There are not enough degrees for the given knots."
        self.dim: int = len(knots)
        self.nlevels = 1

        self.hmesh: HierarchicalMesh = HierarchicalMesh(knots=knots, degrees=degrees)

        univariate_spline_spaces = [UnivariateSplineSpace(degree=self.degrees[d], knots=knots[d]) for d in range(self.dim)]
        self.level_spaces: dict[int, TensorProductSpace] = {0: TensorProductSpace(dim=self.dim, univariate_spaces=univariate_spline_spaces)}

        self.active_functions: dict[int, npt.NDArray[np.int32]] = {0: np.arange(self.level_spaces[0].nfuncs_total, dtype=np.int32)}
        self.deactivated_functions: dict[int, npt.NDArray[np.int32]] = {0: np.array([], dtype=np.int32)}

        # Functions of level l that are supported on Ω^l_{-}
        self.Bl_minus: dict[int, npt.NDArray[np.int32]] = {0: np.array([], dtype=np.int32)}
        # Active functions that are NOT supported on Ω^l_{-}
        self.truly_active: dict[int, npt.NDArray[np.int32]] = {0: np.arange(self.level_spaces[0].nfuncs_total, dtype=np.int32)}

        # 0: inactive, 1: truly_active, 2: Bl_minus
        self.function_status: dict[int, np.NDArray[np.uint8]] = {0: np.ones_like(self.active_functions[0], dtype=np.int8)}

        l0_space = self.level_spaces[0]
        n_funcs0 = l0_space.nfuncs_total
        # self.active_cell_counts: dict[int, npt.NDArray[np.int_]] = {0: np.array([len(l0_space.basis_to_cell(i)) for i in range(n_funcs0)], dtype=np.uint16)}

        # self.refinement_operators: dict[int, list[sp.csc_array]] = {0: self.level_spaces[0].refinement_operators}
        # self.bezier_operators: dict[int, list[sp.bsr_array]] = {0: self.level_spaces[0].bezier_operators}

        # self.get_all_active_functions_on_cell
        

    def refine(self, marked_cells: list[int], level: int, axes=None, refine_neighbours=False, buffer_zone_size=None,
               refine_T_neighbours=False, m=2):
        """
        Refines the specified cells at a given level.

       
        
        :param marked_cells: List of flat cell indices at 'level' to refine.
        :param level: The level at which the marked_cells currently exist.
        :param axes: Optional list of axes to refine.
        :param refine_neighbours: Optional bool to refine around marked cells to be sure to get new degrees of freedom
        :param refine_T_neigbours: Optional bool to refine neighbours such that at most splines of `m` different levels
        act on the same cell.
        :param m: Optional int to specify that splines of at most `m` different levels can act on each cell.
        """
        # if m:
        #     assert m>=2
        assert level>=0, "Provided level must be non-negative."
        marked_cells = np.atleast_1d(marked_cells)
        if len(marked_cells) == 0:
            return
        assert np.min(marked_cells)>=0, "Cell indices start at 0."
        
        
        marked_cells = np.unique(marked_cells)
        if refine_T_neighbours:
            T_marked_cells = {level: marked_cells}
            T_marked_cells = self.mark_recursive(T_marked_cells, l=level, m=m)
            # Sort by level
            T_marked_cells = {k: v for k, v in sorted(T_marked_cells.items(), key=lambda item: item[0])}
        pass
        current_level = self.nlevels-1
        
        # If we are refining cells on the currently finest level, 
        # generate the next level.
        if level >= current_level:
            while(self.nlevels<=level+1):
                self._add_level(axes=axes)
            pass
        pass

        # Update the physical mesh topology
        # This includes refining the current finest mesh, creating new CellNodes if necessary, 
        # and updating active/deactivated cells
        if refine_T_neighbours:
            for l in T_marked_cells.keys():
                self.hmesh.refine(T_marked_cells[l], at_level=l, refine_neighbours=refine_neighbours, admissible_m=buffer_zone_size)
            pass
        else:
            self.hmesh.refine(marked_cells=marked_cells, at_level=level, refine_neighbours=refine_neighbours, admissible_m=buffer_zone_size)
        
        self._update_active_functions()
        
        return T_marked_cells
        

    def _add_level(self, axes=None):
        """
        Adds a level l of refinement to the hierarchical space.
        """
        if axes is None:
            axes = range(self.dim)
        pass
        l = self.nlevels-1
        self.nlevels+=1
        
        new_space: TensorProductSpace = self.level_spaces[l].refine(dims=axes)
        self.level_spaces[l+1] = new_space

        self.active_functions[l + 1] = np.array([], dtype=self.active_functions[0].dtype)
        self.deactivated_functions[l + 1] = np.array([], dtype=self.deactivated_functions[0].dtype)
        self.Bl_minus[l+1] = np.array([], dtype=self.Bl_minus[0].dtype)
        self.truly_active[l+1] = np.array([], dtype=self.truly_active[0].dtype)
        # self.bezier_operators[l+1] = self.level_spaces[l+1].bezier_operators
        # self.refinement_operators[l+1] = self.level_spaces[l+1].refinement_operators

    def _update_active_functions(self):
        """
        Updates the set of active and deactivated functions.
        A function of level l is active if its supports intersects at least one active cell of level l. 
        It is inactive otherwise.
        A cell is said to be active if it was not refined.
        """
        # Dictionaries for each refinement level
        self.active_functions: dict[int, npt.NDArray[np.int32]] = {}
        self.deactivated_functions: dict[int, npt.NDArray[np.int32]] = {}
        self.Bl_minus: dict[int, npt.NDArray[np.int32]] = {}
        self.truly_active: dict[int, npt.NDArray[np.int32]] = {}
        
        for l in range(self.nlevels):
            space_l: TensorProductSpace = self.level_spaces[l]
            nfuncs = space_l.nfuncs_total            
            active_cells_arr: npt.NDArray[np.int32] = self.hmesh.aelem_level[l]
            
            if active_cells_arr.size>0:
                
                # Get all functions that could be truly active
                candidate_funcs: npt.NDArray[np.int32] = np.unique(space_l.cell_to_basis(active_cells_arr))
                # And compute their support cells
                extended_support: list[npt.NDArray[np.int32]] = space_l.basis_to_cell(candidate_funcs)
                has_active_parent = np.zeros_like(candidate_funcs, dtype=bool)
                
                if isinstance(extended_support, list):
                    lens = np.fromiter((len(s) for s in extended_support), count=len(extended_support), dtype=np.int32)
                    flat_support = np.concatenate(extended_support) if len(extended_support) > 0 else np.array([], dtype=np.int32)
                else:
                    extended_support = np.asarray(extended_support)
                    lens = np.full(extended_support.shape[0], extended_support.shape[1], dtype=np.int32)
                    flat_support = extended_support.ravel()

                # rep_idx maps each cell in flat_support back to its parent candidate_func index
                rep_idx = np.repeat(np.arange(len(candidate_funcs), dtype=np.int32), lens)
                # rep_idx = [0,1,1, 2,2,2, ...]
                # fine_shape = tuple(self.meshes_shape[l])

                for ll in range(l): # Check on each level if a candidate function is supported on a coarser cell
                    active_cells_up: npt.NDArray[np.int32] = self.hmesh.aelem_level[ll]
                    if active_cells_up.size==0:
                        continue # skip the level if there are no active cells at this level
                    
                    # Don't check functions for which an active support cell of a coarser level has already been found
                    unresolved_mask = ~has_active_parent 
                    if not np.any(unresolved_mask):
                        break # Early stopping: all candidates are already flagged

                    # flat_support contains all individual cells that have at least one function supported on them.
                    # Therefore, valid_flat_mask filters out the cells that correspond to functions which were already
                    # resolved, e.g. we already know that the functions corresponding to those cells have an active parent
                    valid_flat_mask = unresolved_mask[rep_idx]
                    check_support = flat_support[valid_flat_mask]
                    # keeps the arrays synchronised by removing function indices that were already resolved
                    check_rep_idx = rep_idx[valid_flat_mask]

                    parents = self.hmesh.get_parent_at_level(start_level=l, stop_level=ll, marked_cells_at_start_level=check_support)
                    is_active = sorted_isin(parents, active_cells_up)

                    if np.any(is_active):
                        has_active_parent[check_rep_idx[is_active]] = True


                    # for candidate_idx, support in enumerate(extended_support):
                    #     if has_active_parent[candidate_idx]:
                    #         continue # don't modify the entry if an active parent has already been found
                    #     # Get the parents at a coarser level
                    #     # If two cells share the same parent, the index of the parent is returned twice
                    #     parents: npt.NDArray[np.int32] = self.hmesh.get_parent_at_level(start_level=l, stop_level=ll, marked_cells_at_start_level=support)
                    #     # Verify which are active
                    #     active_parents = sorted_isin(parents, active_cells_up)
                    #     # If at least one parent is active, the function is not `truly active`
                    #     has_active_parent[candidate_idx] = np.any(active_parents)
                    # pass
                pass
            else:
                candidate_funcs = np.array([], dtype=np.int32)
                has_active_parent = np.array([], dtype=bool)
            pass
            
            
            is_truly_active = np.zeros(nfuncs, dtype=bool)
            self.truly_active[l] = candidate_funcs[np.nonzero(~has_active_parent)[0]]
            
            is_truly_active[self.truly_active[l]] = True
            self.Bl_minus[l] = np.nonzero(~is_truly_active)[0].astype(np.int32)
            self.active_functions[l] = candidate_funcs
            self.deactivated_functions[l] = np.array([], dtype=np.int32)
           
        pass #for loop on number of levels

    def mark_recursive(self, marked: dict, l: int, m: int)->npt.ArrayLike:
        """
        Helper function to mark cells belong to the T-neighbourhood of cells in `marked` for 
        `m` admissible meshes.

        Algorithm implemented from Carraturo et. al. from Suitably graded THB-spline refinement and coarsening: Towards an adaptive 
        isogeometric analysis of additive manufacturing processes

        :param marked: cells whose T-neighbourhood is to be computed
        :param l: initial level of cells in `marked` that can influence the extenstion to an `m`-admissible mesh.
        :param m: parameter to form the `m`-admissible mesh. 

        :return marked: Extended dictionary
        """
        assert m>=2
        neighbours = self.get_T_neighbourhood(marked[l], l, m)
        if neighbours.size>0:
            k = l-m+1
            if k not in marked:
                marked[k] = neighbours
            else:
                marked[k] = np.union1d(marked[k], neighbours)
            marked = self.mark_recursive(marked=marked, l=k, m=m)
        return marked

    def get_multilevel_support_extension(self, cell_indices: int|list[int]|npt.NDArray[np.int_], cell_level: int, extension_level: int)->npt.NDArray[np.int_]:
        """Given a cell `Q` of level `l`, there are basis functions {`b_k`} of level 0<=`k`<=`l` that have support on `Q`. 
        This function returns all cells `Q'` of level `k`that are included in the support of at least one basis function
        in {`b_k`}.
        This is denoted by S(`Q`, k).

        Algorithm implemented from Carraturo et. al. from Suitably graded THB-spline refinement and coarsening: Towards an adaptive 
        isogeometric analysis of additive manufacturing processes

        :parameter cell_indices: All cells `Q`
        :parameter cell_level: Level of cells in `cell_indices`
        :parameter extension_level: k

        :return S(`Q`, k): all cells `Q'` of level `k` that are included in the support of at least one basis function
        in {`b_k`}
        """
        l=cell_level
        k=extension_level
        assert l>=k, "Extension level must be finer than cell level."
        if k==l:
            extension: npt.NDArray[np.int_] = self.level_spaces[l].get_support_extension(cell_indices)
        else:
            ancestors: npt.NDArray[np.int_] = self.hmesh.get_parent_at_level(start_level=l, stop_level=k, marked_cells_at_start_level=cell_indices)
            extension = self.level_spaces[k].get_support_extension(ancestors)
        return extension
    
    def get_T_neighbourhood(self, cell_indices: int|list[int]|npt.NDArray[np.int_], cell_level: int, m: int):
        """
        For an element `Q` of level `l`, this function returns all active elements `Q'` of level `l-m+1` such that 
        there are elements `Q''` in the multi-level support extension S(`Q`, `l-m+2`) included in `Q'`. 

        Algorithm implemented from Carraturo et. al. from Suitably graded THB-spline refinement and coarsening: Towards an adaptive 
        isogeometric analysis of additive manufacturing processes

        """
        assert cell_level>=0
        assert cell_level<self.nlevels
        l=cell_level
        k = l-m+2
        if k-1<0:
            return np.array([], dtype=np.int32)
        extension = self.get_multilevel_support_extension(cell_indices=cell_indices, cell_level=cell_level, extension_level=k)
        parents: npt.NDArray[np.int_] = self.hmesh.get_parent(level=k, marked_cells_at_level=extension)
        active_elts: npt.NDArray[np.int_] = self.hmesh.aelem_level[k-1]
        # neighbourhood = np.intersect1d(parents, active_elts)
        unique_parents = np.unique(parents)
        if len(unique_parents)<len(active_elts):
            mask = sorted_isin(unique_parents, active_elts)
            neighbourhood = unique_parents[mask]
        else:
            mask = sorted_isin(active_elts, unique_parents)
            neighbourhood = active_elts[mask]
            
        return neighbourhood


    def _truncation_operator(self, element_idx: int, element_level: int, l: int)->sp.csc_array:
        """
        it represents with functions of level l
        that have support on `element` and B^{l}_{-} functions of level l-1 that have support on `element`.
        The columns that are kept are those who correspond to basis functions whose support is included in 
        B^{l}_{-} and `element`.

        Algorithm implemented from D'angella et. al. Multi-level Bézier extraction for hierarchical local 
        refinement of Isogeometric Analysis

        Parameters
        -------------
        - element_idx: int
            index of the element on which the truncation
            operator is to be computed, with respect to level l.
            This is the global element_idx, i.e its position in the fully refined mesh
            of level l.
        - l: int
            The level of truncation operator, i.e it goes from level l-1 to l
        - element_level: int
            The level of `element_idx`.

        Returns
        ---------------
        - relevant columns of the refinement operator
        - indices of the relevant columns
        - number of columns of the original refinement operator
        """
        assert l>=0, "l has to be positive, as the returned truncation matrix goes from level `l-1` to `l`."
        #space_coarse = self.level_spaces[l-1]
        space_fine: TensorProductSpace = self.level_spaces[l+1]
        assert element_idx<prod(self.level_spaces[element_level].mesh_shape), "There aren't as many nodes at that level."
        #assert element_level>=l, "Not sure if l can be smaller than element_level"

        coarse_element_idx = self.hmesh.get_parent_at_level(start_level=element_level, stop_level=l+1, marked_cells_at_start_level=element_idx)
        # R_local = self.refinement_operators[l-1][coarse_element_idx] #go from level l-1 to l on coarse_element_idx
        try:
            R_local = self.level_spaces[l].get_refinement_operator(index = coarse_element_idx)
        except TypeError:
            self.level_spaces[l].get_refinement_operator(index = coarse_element_idx)
            pass

        fine_funcs_on_elem: np.ndarray = space_fine.cell_to_basis(coarse_element_idx)
        Bl_minus_arr = self.Bl_minus[l+1]
        # nodes_fine: list[CellNode] = self.hmesh.nodes[l]

        # keep functions that overlap coarser cells, i.e discard columns corresponding to truly active functions.
        keep_mask = (sorted_isin(fine_funcs_on_elem, Bl_minus_arr)).astype(R_local.dtype)
        D = sp.diags_array(keep_mask, format='csc', dtype=R_local.dtype)
        R_sliced = R_local@D
        return R_sliced
        # return R_local[:, cols_to_truncate], cols_to_truncate, R_local.shape[1]
        
    
    def _compute_J(self, element_idx: int, element_level: int, l: int) -> sp.csr_array:
        """
        Keeps functions in Bl_minus.

        For `element` at level `element_idx` and a level `l`, the matrix J^l selects the element active functions of level l 
        that do NOT have support on Ω^l_{-} and whose support is not entirely contained in Ω^l_{+}.

        For example, if we look at element 3, level l=2 and degree p=2, suppose that functions 4,5,6 of level 2 are active on element 3.
        However, function 4 has partial support on Ω^2_{-} and function 6's support is contained in Ω^2_{+}. The function will return 
        [0 1 0].

        If we look at element 2, level l=1 and degree p=2, suppose that functions 1,2,3 of level 1 are active on element 2.
        However, function 3's support is contained in Ω^1_{+}. The function will return 

        [[1 0 0]]<br>
        [0 1 0]].

        Algorithm implemented from D'angella et. al. Multi-level Bézier extraction for hierarchical local 
        refinement of Isogeometric Analysis
        """
        assert (l>=0) and (element_level>=0), "Requested level does not exist."
        assert (element_level<=self.nlevels-1) and (l<=self.nlevels-1), "Requested level does not exist."
        assert element_idx< prod(self.level_spaces[element_level].mesh_shape), "Requested element does not exist." #len(self.hmesh.nodes[element_level])
        # element_space = self.level_spaces[element_level]
        space_l: TensorProductSpace = self.level_spaces[l]

        # Get index of element at level of considered functions, i.e the parent('s parent) of the given element.
        coarse_ancestor_idx: int=self.hmesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)
        funcs_on_elem: np.ndarray = space_l.cell_to_basis(coarse_ancestor_idx)
        my_size=len(funcs_on_elem)
        
        # The globally active functions in the THB basis at level l
        active_thb_funcs_l: np.ndarray = self.active_functions[l] #discards functions only active on Ω^l_{+} and Ω^l_{-}

        # Get active functions on relevant element
        active_funcs_on_elem = sorted_isin(ar1=funcs_on_elem, ar2=active_thb_funcs_l)

        if not np.any(active_funcs_on_elem):
            return sp.csr_array((0, my_size), dtype=np.float64)
            # return np.eye(0, my_size, k=0)
        
        # keep functions that are active and are not supported on the coarser mesh
        # for l=0, Bl_minus[0] is empty, therefore the second argument evaluates to all True
        rows_to_keep = active_funcs_on_elem & (~sorted_isin(funcs_on_elem, self.Bl_minus[l]))# (~np.isin(funcs_on_elem, self.Bl_minus[l], assume_unique=True))

        indices = np.flatnonzero(rows_to_keep)
        num_rows = len(indices)
        num_cols = len(rows_to_keep)

        data = np.ones(num_rows)
        row_indices = np.arange(num_rows)
        col_indices = indices
        
        J = sp.csr_array((data, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.float64)
        return J
        

    
    def local_multi_level_extraction_operator(self, element_idx: int, element_level: int, l: int):
        """
        Builds the local multi-level extraction operator M_{l, epsilon}^{loc}

        Algorithm implemented from D'angella et. al. Multi-level Bézier extraction for hierarchical local 
        refinement of Isogeometric Analysis
        
        """
        # Base case: Level 0
        M = self._compute_J(element_idx=element_idx, element_level=element_level, l=0)
        
        # Iteratively apply the algorithm: M_{l} = [ M_{l-1} * trunc(R) ]
        #                                          [        J^l         ]
        for ll in range(0, l):
            
            # R_sliced, mask, num_cols = self._truncation_operator(element_idx=element_idx, element_level=element_level, l=ll)
            R_sliced = self._truncation_operator(element_idx=element_idx, element_level=element_level, l=ll)
            # M_top = np.zeros((M.shape[0], num_cols), dtype=M.dtype)
            # M_top[:, mask] = M@R_sliced
            M_top = M@R_sliced
            # Get J for current level
            J_l = self._compute_J(element_idx=element_idx, element_level=element_level, l=ll+1)
            M = sp.vstack((M_top, J_l), format='csr')
            
        return M
    
    def local_multi_level_extraction_operator2(self, element_indices: list[int], element_level: int, l: int)->list[npt.NDArray[np.float64]]:
        """Returns the local multi-level extraction operators for all elements listed in `element_indices` of the same level 
        `element_level`.
        
        This method is more efficient than calling `local_multi_level_extraction_operator()` several times, as many computations are vectorised 
        in the current method..
        
        :param element_indices: indices of all levels for which the local multi-level extration operators are to be computed
        :param element_level: level to which `element_indices` belong to
        :param l: level of the extraction matrices. This is usually set to the finest level of the mesh.

        :returns Ms: list of numpy arrays, with all the local multi-level extraction operators.

        Algorithm implemented from D'angella et. al. Multi-level Bézier extraction for hierarchical local 
        refinement of Isogeometric Analysis
        """
        element_indices = np.atleast_1d(element_indices)
        if element_indices.size==0:
            return
        assert (element_level>=0)
        assert (element_level<=self.nlevels-1)
        assert np.max(element_indices)<prod(self.level_spaces[element_level].mesh_shape)

        def compute_J( active_funcs_level: npt.NDArray[np.int_], l: int):
            
            active_thb_funcs_l: npt.NDArray[np.int32] = self.truly_active[l]
            active_funcs_level=np.atleast_2d(active_funcs_level)
            active_funcs_on_elem: npt.NDArray[np.bool_] = sorted_isin_2d(ar1_2d=active_funcs_level, ar2_1d=active_thb_funcs_l)
            _, num_cols = np.atleast_2d(active_funcs_on_elem).shape
            base_eye = np.eye(num_cols, dtype=np.float64)
            Js = [base_eye[mask] for mask in active_funcs_on_elem]
            return Js

        def truncation_operator(cells_at_level, active_funcs_at_level, l):
            
            coarse_element_indices = cells_at_level
            if coarse_element_indices.size<2:
                Rs_local = self.level_spaces[l].get_refinement_operator(index=coarse_element_indices)
            else:
                Rs_local: list = self.level_spaces[l].get_refinement_operators(indices=coarse_element_indices)
            
            fine_funcs_on_elems = np.atleast_2d(active_funcs_at_level)# space_fine.cell_to_basis(coarse_element_indices)
            Bl_minus_arr: npt.NDArray[np.int_] = self.Bl_minus[l+1]
            keep_masks:npt.NDArray = sorted_isin_2d(ar1_2d=fine_funcs_on_elems, ar2_1d=Bl_minus_arr).astype(np.bool_)
            #Rs_sliced = [R_local@sp.diags_array(keep_mask, format='csc', dtype=dtype) 
            #             for R_local, keep_mask in zip(Rs_local, keep_masks)]
            Rs_sliced = []
            if coarse_element_indices.size>1:
                for R_local, keep_mask in zip(Rs_local, keep_masks):
                    R_local[:, ~keep_mask] = 0.
                    Rs_sliced.append(R_local.astype(np.float64))
                #Rs_sliced = [R_local[:, keep_mask] for R_local, keep_mask in zip(Rs_local, keep_masks)]
            else:
                R_local = Rs_local
                R_local[:, ~keep_masks[0]]=0.
                Rs_sliced.append(R_local.astype(np.float64))
            
            return Rs_sliced
        
        # get parents of cells at coarsest level
        # If n cells have the same parent, the index of the parent is returned n times.
        cells_level: npt.NDArray[np.int_] = self.hmesh.get_parent_at_level(start_level=element_level, 
                                                                           stop_level=0, 
                                                                           marked_cells_at_start_level=element_indices)
        
        # and the functions of level 0 that have support on these cells
        active_funcs_level: npt.NDArray[np.int_] = self.level_spaces[0].cell_to_basis(cells_level)

        # Discard rows corresponding to functions that have support on a coarser level (passive functions)
        # or are fully contained in a finer level (inactive functions)
        Ms: list = compute_J(active_funcs_level=active_funcs_level, l=0)

        for ll in range(0,l):
            cells_level = self.hmesh.get_parent_at_level(start_level=element_level, 
                                                         stop_level=ll+1, 
                                                         marked_cells_at_start_level=element_indices)
            
            active_funcs_level = self.level_spaces[ll+1].cell_to_basis(cells_level)
            
            Rs_sliced: list = truncation_operator(cells_at_level=cells_level, active_funcs_at_level=active_funcs_level, l=ll)
            M_tops:list = [M@R_sliced for M, R_sliced in zip(Ms, Rs_sliced)]
            
            Jls: list = compute_J(active_funcs_level=active_funcs_level, l=ll+1)
            #Ms: list = [sp.vstack((M_top, Jl), format='csr') for M_top, Jl in zip(M_tops, Jls)]
            Ms: list = [np.vstack((M_top, Jl), dtype=np.float32) for M_top, Jl in zip(M_tops, Jls)]

        return Ms

    def local_multi_level_extraction_operator3(self, element_indices:npt.NDArray[np.int_], element_level:int , l: int,
                                               vectorised=False):
        """
        This method should be more efficient than local_multi_level_extraction_operator2 and achieves the same result.
        
        Algorithm implemented from D'angella et. al. Multi-level Bézier extraction for hierarchical local 
        refinement of Isogeometric Analysis
        """
        element_indices = np.atleast_1d(element_indices)
        N = element_indices.size
        if N==0:
            return []
        assert 0 <=element_level<= self.nlevels-1

        masks = []
        Rs_3d_list = []
        
        cells_0 = self.hmesh.get_parent_at_level(start_level=element_level, stop_level=0, marked_cells_at_start_level=element_indices)
        funcs_0 = np.atleast_2d(self.level_spaces[0].cell_to_basis(cells_0))
        
        C = funcs_0.shape[1]  # Constant number of local basis functions per element: (p+1)^d
        
        # Store boolean mask of active functions for level 0
        masks.append(sorted_isin_2d(funcs_0, self.truly_active[0]))
        any_mask = [np.any(masks[0], axis=1)]
        
        for ll in range(l):
            cells_next = self.hmesh.get_parent_at_level(start_level=element_level, stop_level=ll+1, marked_cells_at_start_level=element_indices)
            funcs_next = np.atleast_2d(self.level_spaces[ll+1].cell_to_basis(cells_next))
            
            # Fetch Local Refinement Matrices
            if cells_next.size == 1:
                Rs_local = [self.level_spaces[ll].get_refinement_operator(index=cells_next)]
            else:
                Rs_local = self.level_spaces[ll].get_refinement_operators(indices=cells_next)
                
            # Convert to a contiguous 3D array for Batched Tensor math: Shape (N, C, C)
            Rs_3d = np.array(Rs_local, dtype=np.float64)
            
            # Vectorized Truncation: Zero out columns across all N matrices simultaneously
            # keep_masks = sorted_isin_2d(funcs_next, self.Bl_minus[ll+1])
            keep_masks = ~sorted_isin_2d(funcs_next, self.truly_active[ll+1])
            Rs_3d *= keep_masks[:, np.newaxis, :]
            
            Rs_3d_list.append(Rs_3d)
            
            # Store boolean mask of active functions for level ll+1
            masks.append(sorted_isin_2d(funcs_next, self.truly_active[ll+1]))
            any_mask.append(np.any(masks[-1], axis=1))

        # R_cums[ll] will store the batched cumulative refinement matrix from level ll -> l, i.e
        # R_cums[0] holds trunc(R^{0,1}) @ trunc(R^{1,2}) @ ... @ trunc(R^{l-1, l})
        # R_cums[1] holds trunc(R^{1,2}) @ trunc(R^{2,3}) @ ... @ trunc(R^{l-1, l})
        def backwards_mult(Rs_3d_list):
            R_cums = [None] * l
            if l > 0:
                R_cums[l-1] = Rs_3d_list[l-1]
                for ll in range(l-2, -1, -1):
                    R_cums[ll] = np.matmul(Rs_3d_list[ll], R_cums[ll+1])
                pass
            pass
            return R_cums
        R_cums = backwards_mult(Rs_3d_list=Rs_3d_list)
        
        def final_loop_vectorised(N, masks, R_cums, l, C):
            all_rows = []
            all_i = []
            
            I_C = np.eye(C, dtype=np.float64)
            
            for ll in range(l + 1):
                mask = masks[ll]
                
                # Find the elements (i_idx) and basis functions (c_idx) where mask is True
                i_idx, c_idx = np.nonzero(mask)
                
                if len(i_idx) == 0:
                    continue
                    
                if ll == l:
                    # The finest level extracts directly from the Identity matrix
                    rows = I_C[c_idx]
                else:
                    # Coarser levels extract directly from their pre-calculated cumulative tensor
                    rows = R_cums[ll][i_idx, c_idx, :]
                    
                all_rows.append(rows)
                all_i.append(i_idx)
                
            if len(all_rows) > 0:
                # A single vertical stack globally replaces N rounds of cascading stacks
                all_rows = np.vstack(all_rows)
                all_i = np.concatenate(all_i)
                
                # Stable sort ensures levels (0, 1, .. l) remain stacked sequentially per element
                sort_idx = np.argsort(all_i, kind='stable')
                all_rows = all_rows[sort_idx]
                all_i = all_i[sort_idx]
                
                # Group element boundaries
                counts = np.bincount(all_i, minlength=N)
                offsets = np.zeros(N + 1, dtype=np.intp)
                np.cumsum(counts, out=offsets[1:])
                
                # Slice large matrix into list of matrices (returns fast memory views)
                Ms = [all_rows[offsets[i]:offsets[i+1]] for i in range(N)]
            else:
                Ms = [np.empty((0, C), dtype=np.float64) for _ in range(N)]
                
            return Ms
                
                
        def final_loop(N, masks, any_mask, R_cums, l, C):
            Ms = []
            I_C = np.eye(C, dtype=np.float64)
            
            for i in range(N):
                blocks = []
                
                for ll in range(l + 1):
                    mask = masks[ll][i]
                    
                    # if not mask.any():
                    #     continue
                    if not any_mask[ll][i]:
                        continue
                        
                    if ll == l:
                        # The finest level extracts directly from the Identity matrix
                        blocks.append(I_C[mask, :])
                    else:
                        # Coarser levels extract directly from their pre-calculated cumulative tensor
                        blocks.append(R_cums[ll][i, mask, :])
                        # This corresponds to J^{ll} @ (trunc(R^{ll, ll+1}) @ ... @ trunc(R^{l-1, l}))
                        
                # A single vertical stack per element replaces L rounds of cascading stacks
                if len(blocks) == 1:
                    Ms.append(blocks[0])
                elif len(blocks) > 1:
                    Ms.append(np.vstack(blocks))
                    # blocks[0] corresponds to J^0 @ trunc(R^{0,1}) @ trunc(R^{1,2}) @ ... @ trunc(R^{l-1, l})
                    # blocks[1] corresponds to J^1 @ trunc(R^{1,2}) @ trunc(R^{2,3}) @ ... @ trunc(R^{l-1, l})
                    # blocks[-1] corresponds to J^{l+1}
                    # Stacking everything corresponds to the recursion formula in D'Angella et al. 2017, section 3.6.1
                else:
                    Ms.append(np.empty((0, C), dtype=np.float64))
                    
            return Ms
        if not vectorised:
            return final_loop(N, masks=masks, any_mask=any_mask, R_cums=R_cums, l=l, C=C)
        else:
            return final_loop_vectorised(N=N, masks=masks, R_cums=R_cums, l=l, C=C)

                
    def build_global_dof_map(self)->tuple[dict[tuple[int, int], int], int]:
        """Numbers active functions that are not active on a coarser level.

        Returns
        --------------
        - dof_map: dict[(level, local_func_idx), global_func_idx]
            global numbering of active functions (those who have support on Ω^l_{-} are not in `dof_map`).
        - int:
            Greatest element in `dof_map`

        """
                
        dof_map = {}
        next_id = 0
        for l in range(self.nlevels):
            considered_functions = self.truly_active[l]
            for func_idx in considered_functions:
                key = (l, func_idx)
                dof_map[key] = next_id
                next_id += 1
            pass
        pass
        return dof_map, next_id-1

    def build_morton_dof_map(self)->tuple[dict[int, int], int]:
        from THBSplines.src.tensor_product_space import quantised_to_morton
        
        total_active = sum(len(self.truly_active.get(l, [])) for l in range(self.nlevels))
        quantised = np.empty((total_active, self.dim), dtype=np.int64)
        a = 0
        truly_active_list = []
        for l in range(self.nlevels):
            active_l = self.truly_active.get(l, [])
            n_l = len(active_l)
            if n_l==0:
                continue
            b = a+n_l
            midpoints_l: npt.NDArray[np.float64] = self.level_spaces[l].middle_points(active_l)
            quantised_l: npt.NDArray[np.float64] = self.level_spaces[l].quantisation(midpoints_l, k=31)
            quantised[a:b] = quantised_l.astype(np.int64)
            truly_active_list.extend((l, idx) for idx in active_l)
            a = b
        pass

        morton: npt.NDArray[np.int64] = quantised_to_morton(quantised_midpoints=quantised)
        dof_order = np.argsort(morton)

        dof_map = {truly_active_list[idx]: i for i, idx in enumerate(dof_order)}
    
        return dof_map, len(dof_map) - 1
        
    
    def element_global_indices(self, element_idx: int, element_level: int, dof_map: dict)->list[int]:
        """
        produces list of active functions that are supported on a specific level but not on coarser levels.
        """
        global_indices: list = []
        for l in range(element_level+1):

            coarse_ancestor_idx: int=self.hmesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)

            funcs_on_elem: np.ndarray = self.level_spaces[l].cell_to_basis(coarse_ancestor_idx)
            active_thb_funcs_l: np.ndarray = self.active_functions[l]
            active_funcs_on_elem: np.ndarray = np.intersect1d(funcs_on_elem, active_thb_funcs_l, assume_unique=True)

            considered_functions: np.ndarray = np.setdiff1d(active_funcs_on_elem, self.Bl_minus[l], assume_unique=True)

            for func_idx in considered_functions:
                key = (l, func_idx)
                global_indices.append(dof_map[key])
            pass
        pass
        return global_indices
    
    def get_all_active_functions_on_cell(self, level: int, cell_idx: int)->dict[int, npt.NDArray[np.int32]]:
        """Given a cell at `level`, returns functions from all levels that are active on it.
        
        Returns
        ------------
        - dict[int, np.ndarray]: all active functions with their respective levels as the keys.
        
        """
        # - dict[int, np.ndarray]: all functions that have support on this cell, whether they are active or not.

        assert self.hmesh._get_node(level, cell_idx).is_active, "Provided cell is inactive."
        # functions = self.level_spaces[level].cell_to_basis(cell_idx)
        # functions_on_cell: np.ndarray = np.intersect1d(functions, self.active_functions[level], assume_unique=True)
        active_functions_l = {}
        #all_functions = {}
        current_cell = cell_idx

        for l in range(level, -1, -1): 
            # All functions with support on the relevant cell
            functions_on_cell: npt.NDArray[np.int_] = self.level_spaces[l].cell_to_basis(current_cell)
            #all_functions[l] = functions

            # active functions with no support on Ω^l_{-}
            truly_active: npt.NDArray[np.int32] = self.truly_active[l]

            if len(functions_on_cell)>0 and len(truly_active)>0:
                # Given the functions that have support on this cell, which ones are active?
                active_mask: npt.NDArray[np.bool_] = sorted_isin(functions_on_cell, truly_active)
                active_functions_l[l] = functions_on_cell[active_mask]

            else:
                active_functions_l[l] = np.array([], dtype=functions_on_cell.dtype)

            if l>0:
                current_cell = self.hmesh.get_parent(level=l, marked_cells_at_level=current_cell)
            pass
        pass

        return dict(reversed(active_functions_l.items()))#, all_functions
                
    
    def evaluate_thb_spline(self, x_eval, coefficients: np.ndarray=None):
        """evaluate the THB splines on the current mesh at point `x_eval` with coefficients `coefficients`. """
        level, cell_idx = self.hmesh.find_active_cell(x_eval)
        # active_functions, all_functions = self.get_all_active_functions_on_cell(level, cell_idx=cell_idx)
        refinement_operator = self.local_multi_level_extraction_operator(element_idx=cell_idx, element_level=level, l=level)
        if coefficients is None:
            scaled_operator = refinement_operator
        else:
            scaled_operator = coefficients * refinement_operator

        if self.dim==1:
            from scipy.interpolate import BSpline
            design_matrix = BSpline.design_matrix(x_eval, self.level_spaces[level].spaces[0].knots, self.degrees[0], extrapolate=False)
            nnz1, nnz2 = design_matrix.nonzero()
            return scaled_operator@design_matrix[nnz1, nnz2]
        else:
            from scipy.interpolate import NdBSpline
            design_matrix = NdBSpline.design_matrix(x_eval, tuple([self.level_spaces[level].spaces[i] for i in range(self.dim)]), self.degrees, extrapolate=False)
            nnz1, nnz2 = design_matrix.nonzero()
            return scaled_operator@design_matrix[nnz1, nnz2]

        
    def _legendre_to_bezier(self, degree: int)->npt.NDArray[np.float_]:
        """Consider a degree n, unnormalised shifted Legendre basis functions L_k on [0,1] with coefficients l_k and Bernstein basis polynomials b_{k, n}
        with coefficients c_k such that, for a polynomial t_n we have

        t_n(x) = Σ_k l_k L_k(x) = Σ_k c_k b_{k,n}(x),

        or in matrix form

        t_n(x) = L(x).l = B_n(x).b .

        This function returns a matrix B_{L->B} such that 

        B_{L->B}l = b.


        The formula is a vectorised version of the one presented in "Legendre-Bernstein basis transformations" by Rida T. Farouki (eq. 20).
        """
        from scipy.special import comb
        n = degree
        indices = np.arange(n + 1)
        
        # G[j, i] = comb(j, i)
        G = comb(indices[:, None], indices[None, :])
        
        # F[i, k] = (-1)**(k+i) * comb(k, i) * comb(k+i, i) / comb(n, i)
        I = indices[:, None]
        K = indices[None, :]
        F = ((-1)**(K + I)) * comb(K, I) * comb(K + I, I) / comb(n, I)
        
        return G @ F
    
    def _bezier_to_legendre(self, degree: int) -> npt.NDArray[np.float_]:
        """Consider a degree n, unnormalised shifted Legendre basis functions L_k on [0,1] with coefficients l_k and Bernstein basis polynomials b_{k, n}
        with coefficients c_k such that, for a polynomial t_n we have

        t_n(x) = Σ_k l_k L_k(x) = Σ_k c_k b_{k,n}(x),

        or in matrix form

        t_n(x) = L(x).l = B_n(x).b .

        This function returns a matrix B_{B->L} such that 

        B_{B->L}b = l.

        The formula is a vectorised version of the one presented in "Legendre-Bernstein basis transformations" by Rida T. Farouki (eq. 21).
        """
        from scipy.special import comb
        n = degree+1
        js = np.arange(n)
        ks = np.arange(n)
        i_vals = np.arange(n)

        # lefts[j] = (2j+1) / ((n+j) * comb(n-1+j, n-1))
        lefts = (2 * js + 1) / ((n + js) * comb(degree + js, degree))

        J = js[:, np.newaxis, np.newaxis]
        K = ks[np.newaxis, :, np.newaxis]
        I = i_vals[np.newaxis, np.newaxis, :]

        # Mask to handle the i <= j constraint
        mask = (I <= J)
        sign = (-1.0)**(J - I)
      
        c1 = comb(J, I)
        
        c2 = comb(K + I, K)
        
        c3 = comb(degree - K + J - I, degree - K)

        # Multiply and sum over the I axis 
        # The mask ensures terms where i > j are zeroed out
        inner_sum = np.sum(sign * c1 * c2 * c3 * mask, axis=2)

        M = lefts[:, np.newaxis] * inner_sum

        return M

    def refine_in_rectangle(self, rectangle: npt.NDArray, level: int, refine_neighbours=False, refine_T_neighbours=False, m=2):
        """
        Refines the mesh from `level` to `level`+1. All cells that intersect with the rectangle are refined up to a suitable level.
        This is to avoid L-shaped domains.

        :param rectangle: array containing endpoints of rectangle as [[a,b], [cd]] = [a,b]x[c,d]
        :param level: refinement level
        :return: indices of active cells marked for refinement.
        """
        
        rectangle=np.atleast_2d(rectangle).astype(np.float64)
        assert rectangle.shape == (2,2), "Provided rectangle does not have an appropriate shape."
        eps = 1e-13

        def find_intersecting_geometrically(level: int)->npt.NDArray[np.int_]:
                # multi_index = np.unravel_index(cell_indices, tuple(self.hmesh.meshes_shape[level]))
                intersecting_1d_indices = []
                for d in range(self.dim):

                    knots_l_d = self.hmesh.one_d_indices[level][d]
                    rect_min = rectangle[d, 0]  
                    rect_max = rectangle[d, 1]
                    c_mins, c_maxs = knots_l_d[:-1], knots_l_d[1:]
                    mask =(c_maxs>=rect_min+eps)& (c_mins<=rect_max-eps)
                    intersecting_1d_indices.append(np.flatnonzero(mask))
                
                if any(len(idx)==0 for idx in intersecting_1d_indices):
                    print("No cells intersect the provided rectangle.")
                    return
                if self.dim==1:
                    marked_cells = intersecting_1d_indices[0]
                else:
                    mesh_indices = np.meshgrid(*intersecting_1d_indices, indexing='ij')
                    multi_indices = tuple(m.flatten() for m in mesh_indices)
                    marked_cells = np.ravel_multi_index(multi_indices, self.hmesh.meshes_shape[level])
                    return marked_cells
            
        
        active_indices = find_intersecting_geometrically(0)
        if active_indices is None:
            return
        self.refine(marked_cells=active_indices, level=0, refine_neighbours=refine_neighbours,
                    refine_T_neighbours=refine_T_neighbours, m=m)

        for l in range(1, level+1):
                # _, children_cells = self.hmesh.get_children(level=l-1, marked_cells_at_level=active_indices)
                #active_indices = np.intersect1d(children_cells, find_intersecting_geometrically(l))
                active_indices = find_intersecting_geometrically(l)
                if active_indices is None:
                    return
                self.refine(marked_cells=active_indices, level=l, refine_neighbours=refine_neighbours,
                            refine_T_neighbours=refine_T_neighbours, m=m)

    def refine_in_rectangles(self, rectangles, level, refine_neighbours=False, refine_T_neighbours=False, m=2):
        """
        Refines the mesh from `level` to `level`+1. All cells that intersect with 
        ANY of the provided rectangles are refined up to a suitable level.

        :param rectangles: array of shape (N, dim, 2) containing endpoints of N rectangles.
                           Also accepts a single rectangle of shape (dim, 2).
        :param level: refinement level
        :return: indices of active cells marked for refinement.
        """
        rectangles = np.asarray(rectangles, dtype=np.float64)
        if rectangles.ndim==2:
            rectangles = rectangles[None, :, :]
        pass

        assert rectangles.ndim==3 and rectangles.shape[1]==self.dim and rectangles.shape[2] == 2, \
            f"Provided rectangles must have shape (N, {self.dim}, 2)."
        
        eps = 1e-13
        N = rectangles.shape[0]

        def find_intersecting_geometrically(level: int) -> npt.NDArray[np.int_]:
            # Initialize an (N, 1, 1, ...) boolean mask that we will expand via broadcasting
            combined_mask = np.ones([N] + [1] * self.dim, dtype=bool)
            
            for d in range(self.dim):
                knots_l_d = self.hmesh.one_d_indices[level][d]
                c_mins, c_maxs = knots_l_d[:-1], knots_l_d[1:]
                
                rect_mins = rectangles[:, d, 0]  # Shape: (N,)
                rect_maxs = rectangles[:, d, 1]  # Shape: (N,)
                
                # 1D Intersection mask. Shape: (N, num_cells_in_d)
                mask_1d = (c_maxs[None, :]>=rect_mins[:, None]+eps) & (c_mins[None, :]<=rect_maxs[:, None]-eps)
                
                # Dynamically set the reshape tuple to broadcast the current dimension
                # e.g., for dim=2 and d=0 -> (N, num_cells_x, 1)
                # e.g., for dim=2 and d=1 -> (N, 1, num_cells_y)
                broadcast_shape = [N] + [1] * self.dim
                broadcast_shape[d + 1] = len(c_mins)
                
                # Accumulate the ND mask via bitwise AND
                combined_mask = combined_mask & mask_1d.reshape(broadcast_shape)
            
            # Collapse the N axis (True if a cell intersects ANY of the rectangles)
            any_intersect = combined_mask.any(axis=0)
            
            if not any_intersect.any():
                print(f"No cells intersect the provided rectangles at level {level}.")
                return None
                
            return np.flatnonzero(any_intersect)
            
        # Refinement Execution
        active_indices = find_intersecting_geometrically(0)
        if active_indices is None:
            return
            
        self.refine(marked_cells=active_indices, level=0, refine_neighbours=refine_neighbours,
                    refine_T_neighbours=refine_T_neighbours, m=m)

        for l in range(1, level + 1):
            active_indices = find_intersecting_geometrically(l)
            if active_indices is None:
                return
            self.refine(marked_cells=active_indices, level=l, refine_neighbours=refine_neighbours,
                        refine_T_neighbours=refine_T_neighbours, m=m)
        


if __name__ == '__main__':
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3, 4]}
    T.hmesh.plot_cells()
