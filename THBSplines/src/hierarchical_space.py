from functools import reduce
from typing import Union, List

import numpy as np
import scipy.sparse as sp
from THBSplines.src.abstract_space import Space
from THBSplines.src.hierarchical_mesh import HierarchicalMesh, CellNode
from THBSplines.src.tensor_product_space import TensorProductSpace, UnivariateSplineSpace

import warnings
warnings.filterwarnings('error')

class HierarchicalSpace(Space):

    def cell_to_basis(self, cell_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        pass

    def basis_to_cell(self, basis_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        pass

    def __init__(self, knots: list, degrees: list):
        """Initialise one level """
        self.degrees = np.atleast_1d(degrees)
        if len(self.degrees)==1:
            self.degrees = self.degrees*np.ones(len(knots), dtype=np.intp)
        else:
            assert len(knots)==len(self.degrees), "There are not enough degrees for the given knots."
        self.dim: int = len(knots)
        self.mesh: HierarchicalMesh = HierarchicalMesh(knots=knots, dim=self.dim)
        univariate_spline_spaces = [UnivariateSplineSpace(degree=self.degrees[d], knots=knots[d]) for d in range(self.dim)]
        self.level_spaces: dict[int, TensorProductSpace] = {0: TensorProductSpace(dim=self.dim, univariate_spaces=univariate_spline_spaces)}
        self.nlevels = 1
        self.active_functions: dict[int, np.ndarray] = {0: np.arange(self.level_spaces[0].nfuncs_total, dtype=np.int32)}
        self.deactivated_functions: dict[int, np.ndarray] = {0: np.array([], dtype=np.int32)}
        # Functions of level l that are supported on \Omega^l_{-}
        self.Bl_minus: dict[int, np.ndarray] = {0: np.array([], dtype=np.int32)}
        self.truly_active: dict[int, np.ndarray] = self.active_functions

        l0_space = self.level_spaces[0]
        n_funcs0 = l0_space.nfuncs_total
        self.active_cell_counts: dict[int, np.ndarray] = {0: np.array([len(l0_space.basis_to_cell(i)) for i in range(n_funcs0)], dtype=np.int32)}

        self.refinement_operators: dict[int, np.ndarray] = {0: self.level_spaces[0].refinement_operators}
        self.bezier_operators: dict[int, np.ndarray] = {0: self.level_spaces[0].bezier_operators}

        self.get_all_active_functions_on_cell
        

    def refine(self, marked_cells: list, level: int, axes=None):
        """
        Refines the specified cells at a given level.
        
        :param marked_cells: List of flat cell indices at 'level' to refine.
        :param level: The level at which the marked_cells currently exist.
        :param axes: Optional list of axes to refine.
        """

        marked_cells = np.atleast_1d(marked_cells)
        current_level = self.nlevels-1
        if len(marked_cells) == 0:
            return

        # 1. If we are refining cells on the currently finest level, 
        # we MUST generate the next level mathematically and physically.
        if level >= current_level:
            while(self.nlevels<=level+1):
                self.add_level(axes=axes)
            pass
        pass

        # 2. Update the physical mesh topology
        # This includes refining the current finest mesh and creating new CellNodes if necessary, 
        # and updating active/deactivated cells
        self.mesh.refine(marked_cells=marked_cells, at_level=level)
        
        # 3. Re-evaluate basis function statuses (Active vs Deactivated)
        # if level<=current_level:
        #     # Faster function if we are not skipping levels
        #     self._update_active_functions_incremental(marked_cells, level)
        # else:
        #     pass
        self._update_active_functions()
        
        

    def add_level(self, axes=None):
        """
        Adds a level l of refinement to the hierarchical space.
        """
        if axes is None:
            axes = range(self.dim)
        pass
        l = self.nlevels-1
        self.nlevels+=1
        #univariate_spline_spaces = [self.level_spaces[l].spaces[d].refine() for d in range(self.dim)]
        new_space: TensorProductSpace = self.level_spaces[l].refine(dims=axes)
        self.level_spaces[l+1] = new_space

        # new_knots = [uni.knots for uni in new_space.spaces]

        # self.mesh.add_level(new_knots)

        # self.refinement_operators[l+1] = new_space.compute_refinement_matrix(self.level_spaces[l_fine])
        self.active_functions[l + 1] = np.array([], dtype=self.active_functions[l].dtype)
        self.deactivated_functions[l + 1] = np.array([], dtype=self.active_functions[l].dtype)
        self.bezier_operators[l+1] = self.level_spaces[l+1].bezier_operators
        self.refinement_operators[l+1] = self.level_spaces[l+1].refinement_operators

    def _update_active_functions_incremental(self, marked_cells: np.ndarray, level: int):
        """Updates active and deactivated functions when `marked_cells` at `level` have been refined to `level+1`. """

        space_l = self.level_spaces[level]
        # all affected functions
        impacted_basis_l = np.unique(np.concatenate(space_l.cell_to_basis(marked_cells)))
        # supports = space_l.basis_to_cell(impacted_basis_l)
        # not sure if this can be vectorized since basis_to_cell returns a list of numpy arrays
        for b_idx in impacted_basis_l:
            support = space_l.basis_to_cell(b_idx)[0]
            num_removed = np.isin(support, marked_cells).sum()
            self.active_cell_counts[level][b_idx]-=num_removed
        pass
        self.active_functions[level] = np.where(self.active_cell_counts[level]>0)[0]
        self.deactivated_functions[level] = np.where(self.active_cell_counts[level] == 0)[0]

        next_level = level+1
        if next_level in self.level_spaces:
            space_lp1 = self.level_spaces[next_level]
            
            # New children cells are created in the mesh during refine()
            # We find which functions at L+1 cover the children of marked_cells
            # In B-Splines, refinement is predictable: 1 cell splits into 2^dim children
            # We ask the mesh for the children indices
            children_cells = []
            for c in marked_cells:
                children_cells.extend([child.index for child in self.mesh.nodes[level][c].children])
            pass
            
            children_cells = np.array(children_cells)
            # all functions that are supported on the newly refined cells
            impacted_basis_lp1 = np.unique(np.concatenate(space_lp1.cell_to_basis(children_cells)))
            
            # If this is a newly created level, initialize counts to 0
            if next_level not in self.active_cell_counts:
                self.active_cell_counts[next_level] = np.zeros(space_lp1.nfuncs_total, dtype=self.active_cell_counts[0].dtype)
            pass
            Bl_minus = []
            mesh_nodes_l1 = self.mesh.nodes[next_level]
            for b_idx in impacted_basis_lp1:
                support = space_lp1.basis_to_cell(b_idx)
                # How many of the new children are in this basis support?
                num_added = np.isin(support, children_cells).sum()
                self.active_cell_counts[next_level][b_idx] += num_added
                for cell in support:
                    if mesh_nodes_l1[cell].parent.is_active:
                        Bl_minus.append(b_idx)
                        break #break if at least one support cell has an active parent.
                    pass
                pass
            pass

            self.active_functions[next_level] = np.where(self.active_cell_counts[next_level] > 0)[0]
            self.deactivated_functions[next_level] = np.where(self.active_cell_counts[next_level] == 0)[0]
            self.Bl_minus[next_level] = Bl_minus
        pass
    pass

    def _update_active_functions(self):
        """
        Updates the set of active and deactivated functions.
        A function of level l is active if its supports intersects at least one active cell of level l. 
        It is inactive otherwise.
        A cell is said to be active if it was not refined.
        """
        # Dictionaries for each refinement level
        self.active_functions = {}
        self.deactivated_functions = {}
        self.Bl_minus = {}
        
        for l in range(self.nlevels):
            space_l: TensorProductSpace = self.level_spaces[l]
            nfuncs = space_l.nfuncs_total
            # Quick lookup dictionary for level l cells
            nodes_l: list[CellNode] = self.mesh.nodes[l]
            
            is_active_mask = np.zeros(nfuncs, dtype=bool)
            is_minus_mask = np.zeros_like(is_active_mask)

            for cell_idx, node in enumerate(nodes_l):
                cell_active = node.is_active
                parent_active = node.parent.is_active if node.parent else False

                if cell_active or parent_active:
                    affected_funcs: np.ndarray = space_l.cell_to_basis(cell_idx)
                    if cell_active:
                        is_active_mask[affected_funcs]=True
                    if parent_active:
                        is_minus_mask[affected_funcs]=True
                        
                    pass
                pass
            pass
            
            all_indices = np.arange(nfuncs, dtype=np.int32)
            self.active_functions[l] = all_indices[is_active_mask]
            self.deactivated_functions[l]=all_indices[~is_active_mask]
            self.Bl_minus[l]=all_indices[is_minus_mask&is_active_mask]
            self.truly_active[l] = all_indices[is_active_mask&(~is_minus_mask)] #np.setdiff1d(self.active_functions[l], self.Bl_minus[l], assume_unique=True)
            
        pass

    def get_children(self, level, marked_functions_at_level, tol: float=1e-7):
        """
        Return the indices of the children function of the marked functions
        at the given level.

        :param level: refinement level
        :param marked_functions_at_level: list of function indices marked at given refinement level.
        :return: np.array of indices
        """

        supports_of_marked_functions: np.ndarray = self.spaces[level].basis_supports[marked_functions_at_level]
        supports_of_finer_functions: np.ndarray = self.spaces[level+1].basis_supports
        min_finer = supports_of_finer_functions[:, :, 0]
        max_finer = supports_of_finer_functions[:, :, 1]
        
        min_marked = supports_of_marked_functions[:, :, 0]
        max_marked = supports_of_marked_functions[:, :, 1]
        is_min_inside = min_finer >= (min_marked[:, None, :] - tol)
        is_max_inside = max_finer <= (max_marked[:, None, :] + tol)
        is_child = np.all(is_min_inside & is_max_inside, axis=2)
        # A finer function is a child if it is contained in AT LEAST ONE marked function.
        is_child_of_any = np.any(is_child, axis=0)

        # Get the 1D indices of the finer functions that returned True
        indices = np.where(is_child_of_any)[0]
        
        return indices

    def _truncation_operator(self, element_idx: int, element_level: int, l: int)->sp.csc_array:
        """
        it represents with functions of level l
        that have support on `element` and B^{l}_{-} functions of level l-1 that have support on `element`.
        The columns that are kept are those who correspond to basis functions whose support is included in 
        B^{l}_{-} and `element`.

        Parameters
        -------------
        - element_idx: int
            index of the element on which the truncation
            operator is to be computed, with respect to level l.
            This is the global element_idx, i.e its position in the fully refined mesh
            of level l.
        - l: int
            The level of truncation operator, i.e it goes from level l-1 to l

        Returns
        ---------------
        - relevant columns of the refinement operator
        - indices of the relevant columns
        - number of columns of the original refinement operator
        """
        assert l>0, "l has to be positive, as the returned truncation matrix goes from level `l-1` to `l`."
        #space_coarse = self.level_spaces[l-1]
        space_fine: TensorProductSpace = self.level_spaces[l]
        assert element_idx<len(self.mesh.nodes[element_level]), "There aren't as many nodes at that level."
        #assert element_level>=l, "Not sure if l can be smaller than element_level"

        coarse_element_idx = self.mesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)
        R_local = self.refinement_operators[l-1][coarse_element_idx] #go from level l-1 to l on coarse_element_idx

        fine_funcs_on_elem: np.ndarray = space_fine.cell_to_basis(coarse_element_idx)
        Bl_minus_arr = self.Bl_minus[l]
        # nodes_fine: list[CellNode] = self.mesh.nodes[l]

        # cols_to_truncate = np.isin(fine_funcs_on_elem, Bl_minus_arr, assume_unique=True)
        keep_mask = (self._my_isin(fine_funcs_on_elem, Bl_minus_arr)).astype(np.float64)
        D = sp.diags_array(keep_mask, format='csc', dtype=np.float64)
        R_sliced = R_local.dot(D)
        return R_sliced
        # return R_local[:, cols_to_truncate], cols_to_truncate, R_local.shape[1]
        
    
    def _compute_J(self, element_idx: int, element_level: int, l: int) -> sp.csr_array:
        """
        For `element` at level `element_idx` and a level `l`, the matrix J^l selects the element active functions of level l 
        that do NOT have support on Ω^l_{-} and whose support is not entirely contained in Ω^l_{+}.

        For example, if we look at element 3, level l=2 and degree p=2, suppose that functions 4,5,6 of level 2 are active on element 3.
        However, function 4 has partial support on Ω^2_{-} and function 6's support is contained in Ω^2_{+}. The function will return 
        [0 1 0].

        If we look at element 2, level l=1 and degree p=2, suppose that functions 1,2,3 of level 1 are active on element 2.
        However, function 3's support is contained in Ω^1_{+}. The function will return 

        [[1 0 0]]<br>
        [0 1 0]].
        """
        assert (l>=0) and (element_level>=0), "Requested level does not exist."
        assert (element_level<=self.nlevels-1) and (l<=self.nlevels-1), "Requested level does not exist."
        assert element_idx<len(self.mesh.nodes[element_level]), "Requested element does not exist."
        # element_space = self.level_spaces[element_level]
        space_l: TensorProductSpace = self.level_spaces[l]

        # Get index of element at level of considered functions, i.e the parent('s parent) of the given element.
        coarse_ancestor_idx: int=self.mesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)
        funcs_on_elem: np.ndarray = space_l.cell_to_basis(coarse_ancestor_idx)
        my_size=len(funcs_on_elem)
        
        # The globally active functions in the THB basis at level l
        active_thb_funcs_l: np.ndarray = self.active_functions[l] #discards functions only active on Ω^l_{+} and Ω^l_{-}

        # Get active functions on relevant element
        active_funcs_on_elem = self._my_isin(ar1=funcs_on_elem, ar2=active_thb_funcs_l)

        if not np.any(active_funcs_on_elem):
            return sp.csr_array((0, my_size), dtype=np.float64)
            # return np.eye(0, my_size, k=0)
        
        # keep functions that are active and are not supported on the coarser mesh
        # for l=0, Bl_minus[0] is empty, therefore the second argument evaluates to all True
        rows_to_keep = active_funcs_on_elem & (~self._my_isin(funcs_on_elem, self.Bl_minus[l]))# (~np.isin(funcs_on_elem, self.Bl_minus[l], assume_unique=True))

        indices = np.flatnonzero(rows_to_keep)
        num_rows = len(indices)
        num_cols = len(rows_to_keep)
        
        #J = np.zeros((num_rows, num_cols))

        data = np.ones(num_rows)
        row_indices = np.arange(num_rows)
        col_indices = indices
        
        J = sp.csr_array((data, (row_indices, col_indices)), shape=(num_rows, num_cols), dtype=np.float64)
        return J
        
        #J[np.arange(num_rows), indices] = 1.0
        
        #return J
    
    def _my_isin(self, ar1: np.ndarray, ar2: np.ndarray)->np.ndarray[bool]:
        """`np.isin` when both arrays are sorted and have unique values.
        No checks are performed to ensure this.
        """
        idx = np.searchsorted(ar2, ar1)
        valid_mask = idx < len(ar2)
        cols_to_keep = np.zeros(len(ar1), dtype=bool)
        cols_to_keep[valid_mask] = ar2[idx[valid_mask]]==ar1[valid_mask]
        return cols_to_keep

    
    def local_multi_level_extraction_operator(self, element_idx: int, element_level: int, l: int):
        """
        Builds the local multi-level extraction operator M_{L, epsilon}^{loc}
        
        """
        # Base case: Level 0
        M = self._compute_J(element_idx=element_idx, element_level=element_level, l=0)
        
        # Iteratively apply the algorithm: M_{l} = [ M_{l-1} * trunc(R) ]
        #                                          [        J^l         ]
        for ll in range(1, l+1):
            
            # R_sliced, mask, num_cols = self._truncation_operator(element_idx=element_idx, element_level=element_level, l=ll)
            R_sliced = self._truncation_operator(element_idx=element_idx, element_level=element_level, l=ll)
            # M_top = np.zeros((M.shape[0], num_cols), dtype=M.dtype)
            # M_top[:, mask] = M@R_sliced
            M_top = M.dot(R_sliced)
            # Get J for current level
            J_l = self._compute_J(element_idx=element_idx, element_level=element_level, l=ll)
            M = sp.vstack((M_top, J_l), format='csr')
            
        return M
    
    def build_global_dof_map(self):
        """Numbers active functions that are not active on a coarser level.

        Returns
        --------------
        - dof_map: dict[level, func_idx]
            global numbering of active functions (those who have support on \Omega^l_{-} are not in `dof_map`).
        - int:
            Greatest element in `dof_map`

        Example
        -----------------
        Consider the space defined in D'Angella's paper, then the dofmap is: <br>
        {(0, np.int32(0)): 0, <br>
        (0, np.int32(1)): 1, <br>
        (0, np.int32(2)): 2, <br>
        (0, np.int32(3)): 3, <br>
        (1, np.int32(6)): 4, <br>
        (2, np.int32(12)): 5, <br>
        (2, np.int32(13)): 6, <br>
        (2, np.int32(14)): 7, <br>
        (2, np.int32(15)): 8, <br>
        (2, np.int32(16)): 9, <br>
        (2, np.int32(17)): 10}
        """
        dof_map = {}
        next_id = 0
        for l in range(self.nlevels):
            considered_functions = self.truly_active[l] #np.setdiff1d(self.active_functions[l], self.Bl_minus[l], assume_unique=True)
            for func_idx in considered_functions:
                key = (l, func_idx)
                dof_map[key] = next_id
                next_id += 1
            pass
        pass
        return dof_map, next_id-1
    
    def element_global_indices(self, element_idx: int, element_level: int, dof_map: dict)->list[int]:
        """
        produces list of active functions that are supported on a specific level but not on coarser levels.
        """
        global_indices: list = []
        for l in range(element_level+1):

            coarse_ancestor_idx: int=self.mesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)

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
    
    def get_all_active_functions_on_cell(self, level: int, cell_idx: int)->dict:
        """Given a cell at `level`, returns functions from all levels that are active on it.
        
        Returns
        ------------
        - dict[int, np.ndarray]: all active functions with their respective levels as the keys.
        
        """
        # - dict[int, np.ndarray]: all functions that have support on this cell, whether they are active or not.

        assert self.mesh.nodes[level][cell_idx].is_active, "Provided cell is inactive."
        # functions = self.level_spaces[level].cell_to_basis(cell_idx)
        # functions_on_cell: np.ndarray = np.intersect1d(functions, self.active_functions[level], assume_unique=True)
        active_functions_l = {}
        #all_functions = {}
        current_cell = cell_idx

        for l in range(level, -1, -1): 
            # All functions with support on the relevant cell
            functions_on_cell = self.level_spaces[l].cell_to_basis(current_cell)
            #all_functions[l] = functions

            # active functions with no support on \Omega^l_{-}
            truly_active = self.truly_active[l]

            if len(functions_on_cell)>0 and len(truly_active)>0:
                # Given the functions that have support on this cell, which ones are active?
                active_mask: np.ndarray[bool] = self._my_isin(functions_on_cell, truly_active)
                active_functions_l[l] = functions_on_cell[active_mask]

                # idx = np.searchsorted(truly_active, functions)

                # idx[idx == len(truly_active)] = len(truly_active) - 1
                # # Keep only the functions that actually match the value at the found index
                # mask = truly_active[idx] == functions
                # active_functions_l[l] = functions[mask]
            else:
                active_functions_l[l] = np.array([], dtype=functions_on_cell.dtype)

            if l>0:
                current_cell = self.mesh.get_parent(level=l, marked_cells_at_level=current_cell)
            pass
        pass

        return dict(reversed(active_functions_l.items()))#, all_functions
                
    
    def evaluate_thb_spline(self, x_eval, coefficients: np.ndarray=None):
        """evaluate the THB splines on the current mesh at point `x_eval` with coefficients `coefficients`. """
        level, cell_idx = self.mesh.find_active_cell(x_eval)
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

        
    def _legendre_to_bezier(self, degree: int):
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
    
    def _bezier_to_legendre(self, degree: int):
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
        

    def get_basis_conversion_matrix(self, level, coarse_indices=None):
        """
        Returns the basis conversion matrix taking you from level to the
        next.

        :param level: refinement level of the underlying tensor product space
        :param coarse_indices: indices of the coarse active functions.
        :return: a basis conversion matrix
        """
        print('Space dimensions', self.nfuncs_level[level], self.spaces[level].nfuncs,
              self.spaces[level].nfuncs_onedim)
        if coarse_indices is None:
            c = self.compute_full_projection_matrix(level)
        else:

            prod = np.prod(self.spaces[level + 1].degrees + 2) * len(coarse_indices)  # allocate space for data
            rows = np.zeros(prod)

            columns = np.zeros_like(rows)
            values = np.zeros_like(rows)

            ncounter = 0
            sub_coarse = np.unravel_index(coarse_indices, self.spaces[level].nfuncs_onedim)
            # TODO: Ugly hack!!!
            # unravel_index gives basis functions in the opposite order from my implementation.
            if self.dim == 2:
                sub_coarse = [sub_coarse[1], sub_coarse[0]]
            for i in range(len(coarse_indices)):
                C = 1
                for dim in range(self.dim):
                    C = sp.kron(self.projections_onedim[level][dim][:, sub_coarse[dim][i]], C)
                ir, ic, iv = sp.find(C)
                rows[ncounter: len(ir) + ncounter] = ir
                columns[ncounter: len(ir) + ncounter] = i
                values[ncounter: len(ir) + ncounter] = iv

                ncounter += len(ir)

            rows = rows[:ncounter]
            columns = columns[:ncounter]
            values = values[:ncounter]
            c = sp.coo_matrix((values, (rows, columns)),
                              shape=(self.spaces[level + 1].nfuncs, self.spaces[level].nfuncs)).tolil()

        if self.truncated:
            i = np.union1d(self.afunc_level[level + 1], self.dfunc_level[level + 1])
            c[i, :] = 0
        return c

    def compute_full_projection_matrix(self, level):
        """
        Computes the kroneker product of the 1D basis-projections at a given
        level and stores it as a sparse matrix.

        :param level: refinement level in question
        :return: sparse projection matrix
        """
        c = 1
        for dim in range(self.dim):
            c = sp.kron(self.projections_onedim[level][dim], c)
        c = c.tolil()
        return c

    def functions_to_deactivate_from_cells(self, marked_cells: dict):
        """
        Returns the indices of functions that have no active cells within
        their support.

        :param marked_cells: cell indices to check against
        :return: function indices
        """
        marked_functions = {}
        for level in range(self.nlevels):
            func_to_deact = self.spaces[level].get_basis_functions(marked_cells[level])
            func_to_deact = np.intersect1d(func_to_deact, self.afunc_level[level])

            func_to_keep = np.array([], dtype=np.int)
            _, func_cells_map = self.spaces[level].get_cells(func_to_deact)
            for f in func_to_deact:
                func_cells = func_cells_map[f]
                common_elements = np.intersect1d(func_cells, self.mesh.aelem_level[level])
                if common_elements.size != 0:
                    func_to_keep = np.append(func_to_keep, f)
            func_to_deact = np.setdiff1d(func_to_deact, func_to_keep)
            marked_functions[level] = func_to_deact
        return marked_functions

    def _get_all_cells(self):
        all_cells = []
        for i, mesh in enumerate(self.mesh.meshes):
            for cell in mesh.cells[self.mesh.aelem_level[i]]:
                all_cells.append(cell)
        return np.array(all_cells)

    def create_subdivision_matrix(self, mode='reduced') -> dict:
        """
        Returns hspace.nlevels-1 matrices used for representing coarse
        B-splines in terms of the finer B-splines.
        
        :param mode: whether to create a full or a reduced subdivision matrix. Only reduced supported at the moment.
        :return: a dictionary mapping
        """

        mesh = self.mesh

        C = {}
        C[0] = sp.identity(self.spaces[0].nfuncs, format='lil')
        C[0] = C[0][:, self.afunc_level[0]]

        if mode == 'reduced':
            func_on_active_elements = self.spaces[0].get_basis_functions(mesh.aelem_level[0])
            func_on_deact_elements = self.spaces[0].get_basis_functions(mesh.delem_level[0])
            func_on_deact_elements = np.union1d(func_on_deact_elements, func_on_active_elements)
            for level in range(1, self.nlevels):
                I_row_idx = self.afunc_level[level]
                I_col_idx = list(range(self.nfuncs_level[level]))

                data = np.ones(len(I_col_idx))
                I = sp.coo_matrix((data, (I_row_idx, I_col_idx)),
                                  shape=(self.spaces[level].nfuncs, self.nfuncs_level[level]))
                aux = self.get_basis_conversion_matrix(level - 1, coarse_indices=func_on_deact_elements)
                func_on_active_elements = self.spaces[level].get_basis_functions(mesh.aelem_level[level])
                func_on_deact_elements = self.spaces[level].get_basis_functions(mesh.delem_level[level])
                func_on_deact_elements = np.union1d(func_on_deact_elements, func_on_active_elements)
                C[level] = sp.hstack([aux @ C[level - 1], I])
            return C
        else:
            for level in range(1, self.nlevels):
                I = sp.identity(self.spaces[level].nfuncs, format='lil')
                aux = self.get_basis_conversion_matrix(level - 1)
                C[level] = sp.hstack([aux @ C[level - 1], I[:, self.afunc_level[level]]])
            return C

    def _get_truncated_supports(self):
        """
        Returns the indices at the fine level that constitutes the supports of
        each truncated basis function.

        :return: indices of cells that support each truncated basis function.
        """
        C = self.create_subdivision_matrix(mode='full')[self.nlevels - 1].toarray()
        trunc_basis_to_fine_idx = {}
        for i, trunc_coeff in enumerate(C.T):
            fine_active_basis = np.flatnonzero(trunc_coeff)
            basis_support_cells = self._get_fine_basis_support_cells(fine_active_basis)
            trunc_basis_to_fine_idx[i] = self.mesh.meshes[-1].cells[basis_support_cells]
        return trunc_basis_to_fine_idx

    def _get_fine_basis_support_cells(self, fine_active_basis):
        """
        Given a list of functions, returns the unique cells at the finest level
        in their common support.

        :param fine_active_basis: list of function indices
        :return: a set of cells at the finest refinement level in their common support
        """

        cells = np.array([], dtype=np.int)
        fine_cells = self.mesh.meshes[-1].cells
        for i in fine_active_basis:
            supp = self.spaces[-1].basis_supports[i]
            condition = np.all((supp[:, 0] <= fine_cells[:, :, 0]) & (supp[:, 1] >= fine_cells[:, :, 1]), axis=1)
            cells = np.union1d(cells, np.flatnonzero(condition))
        return cells

    def refine_in_rectangle(self, rectangle, level):
        """
        Returns the set active of indices marked for refinement contained in
        the given rectangle

        :param rectangle: array containing endpoints of rectangle
        :param level: refinement level
        :return: indices of active cells markde for refinement.
        """
        eps = np.spacing(1)
        cells = self.mesh.meshes[level].cells
        cells_to_mark = np.all((rectangle[:, 0] <= cells[:, :, 0] + eps) & (eps + rectangle[:, 1] >= cells[:, :, 1]),
                               axis=1)
        return np.intersect1d(np.flatnonzero(cells_to_mark), self.mesh.aelem_level[level])

    def plot_overloading(self, filename=None, text=False, fontsize=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as plp
        fig = plt.figure()
        axs = fig.add_subplot(1, 1, 1)
        u_min = 0
        u_max = 1
        v_min = 0
        v_max = 1
        C = self.create_subdivision_matrix('full')
        max_degree = np.prod([d + 1 for d in self.degrees])
        for level in range(self.nlevels):
            indices = self.spaces[level].cell_to_basis(self.mesh.aelem_level[level])
            Csub = sp.lil_matrix(C[level])
            mesh = self.mesh.meshes[level]
            for cell, i_elem in zip(self.mesh.aelem_level[level], indices):
                _, col, _ = sp.find(Csub[i_elem, :])
                n = len(np.unique(col))
                if n > max_degree:
                    color = 'black'
                    fill = True
                    hatch = '//'
                    dotext = True
                else:
                    color = 'black'
                    hatch = None
                    fill = False
                    dotext = False

                e = mesh.cells[cell]
                w = e[0, 1] - e[0, 0]
                h = e[1, 1] - e[1, 0]
                mp1 = e[0, 0] + w / 2
                mp2 = e[1, 0] + h / 2

                axs.add_patch(plp.Rectangle((e[0, 0], e[1, 0]), w, h, fill=fill, color=color, alpha=0.2, hatch=None,
                                            linewidth=0.2))
                if text:
                    if not dotext:
                        continue
                    if fontsize is None:
                        fontsize = 50

                    axs.text(mp1, mp2, '{}'.format(n - max_degree), ha='center',
                             va='center', fontsize=fontsize * w)
                v_min = min(v_min, e[1, 0])
                v_max = max(v_max, e[1, 1])
                u_min = min(u_min, e[0, 0])
                u_max = max(u_max, e[0, 1])

        plt.xlim(u_min, u_max)
        plt.ylim(v_min, v_max)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_basis_weights(self, weights, filename=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as plp
        fig = plt.figure()
        axs = fig.add_subplot(1, 1, 1)
        u_min = 0
        u_max = 1
        v_min = 0
        v_max = 1
        C = self.create_subdivision_matrix('full')
        Csub_last = C[self.nlevels - 1].toarray()
        max_degree = np.prod([d + 1 for d in self.degrees])
        for level in range(self.nlevels):
            indices = self.spaces[level].cell_to_basis(self.mesh.aelem_level[level])
            print(indices)
            Csub = sp.lil_matrix(C[level])
            mesh = self.mesh.meshes[level]
            for cell, i_elem in zip(self.mesh.aelem_level[level], indices):
                _, col, _ = sp.find(Csub[i_elem, :])
                print(col)
                n = len(np.unique(col))
                print(Csub_last)
                if n > max_degree:
                    color = 'black'
                    fill = True
                    hatch = '//'
                    dotext = True
                else:
                    color = 'black'
                    hatch = None
                    fill = False
                    dotext = False

                e = mesh.cells[cell]
                w = e[0, 1] - e[0, 0]
                h = e[1, 1] - e[1, 0]

                axs.add_patch(plp.Rectangle((e[0, 0], e[1, 0]), w, h, fill=fill, color=color, alpha=0.2, hatch=None,
                                            linewidth=0.2))

                v_min = min(v_min, e[1, 0])
                v_max = max(v_max, e[1, 1])
                u_min = min(u_min, e[0, 0])
                u_max = max(u_max, e[0, 1])

        plt.xlim(u_min, u_max)
        plt.ylim(v_min, v_max)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


if __name__ == '__main__':
    knots = [
        [0, 0, 1, 2, 3, 3],
        [0, 0, 1, 2, 3, 3]
    ]
    d = 2
    degrees = [1, 1]
    T = HierarchicalSpace(knots, degrees, d)
    marked_cells = {0: [0, 1, 2, 3, 4]}
    T.mesh.plot_cells()
    T.mesh.plot_cells()
    weights = np.random.random(T.nfuncs)
    T.plot_basis_weights(weights)
