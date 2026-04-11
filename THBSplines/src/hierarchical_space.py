import numpy.typing as npt

import numpy as np
import scipy.sparse as sp
from THBSplines.src.hierarchical_mesh import HierarchicalMesh, CellNode
from THBSplines.src.tensor_product_space import TensorProductSpace, UnivariateSplineSpace
from copy import deepcopy

import warnings
warnings.filterwarnings('error')

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

    - mesh: HierarchicalMesh
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
        self.active_functions: dict[int, npt.NDArray[np.int32]] = {0: np.arange(self.level_spaces[0].nfuncs_total, dtype=np.int32)}
        self.deactivated_functions: dict[int, npt.NDArray[np.int32]] = {0: np.array([], dtype=np.int32)}
        # Functions of level l that are supported on Ω^l_{-}
        self.Bl_minus: dict[int, npt.NDArray[np.int32]] = {0: np.array([], dtype=np.int32)}
        self.truly_active: dict[int, npt.NDArray[np.int32]] = {0: np.arange(self.level_spaces[0].nfuncs_total, dtype=np.int32)}

        l0_space = self.level_spaces[0]
        n_funcs0 = l0_space.nfuncs_total
        self.active_cell_counts: dict[int, npt.NDArray[np.int32]] = {0: np.array([len(l0_space.basis_to_cell(i)) for i in range(n_funcs0)], dtype=np.int16)}

        self.refinement_operators: dict[int, list[sp.csc_array]] = {0: self.level_spaces[0].refinement_operators}
        self.bezier_operators: dict[int, list[sp.bsr_array]] = {0: self.level_spaces[0].bezier_operators}

        # self.get_all_active_functions_on_cell
        

    def refine(self, marked_cells: list[int], level: int, axes=None):
        """
        Refines the specified cells at a given level.
        
        :param marked_cells: List of flat cell indices at 'level' to refine.
        :param level: The level at which the marked_cells currently exist.
        :param axes: Optional list of axes to refine.
        """
        assert level>=0, "Provided level must be non.negative."
        assert np.min(marked_cells)>=0, "Cell indices start at 0."
        
        marked_cells = np.atleast_1d(marked_cells)
        if len(marked_cells) == 0:
            return
        marked_cells = np.unique(marked_cells)
        current_level = self.nlevels-1
        
        # If we are refining cells on the currently finest level, 
        # generate the next level.
        if level >= current_level:
            while(self.nlevels<=level+1):
                self._add_level(axes=axes)
            pass
        pass
         

        # Keep a copy of cells that were already refined, so that they don't get
        # analysed an additional time when active functions are updated.
        # The deepcopy is needed because the content must not be overwritten 
        # by the call to self.mesh.refine() later on.
        previously_inactive_cells = deepcopy(self.mesh.delem_level)

        # Update the physical mesh topology
        # This includes refining the current finest mesh, creating new CellNodes if necessary, 
        # and updating active/deactivated cells
        self.mesh.refine(marked_cells=marked_cells, at_level=level)
        
    
        # Make sure that we are not refining an already refined cell.
        to_refine_mask = np.ones_like(marked_cells, dtype=bool)
        nodes_l = self.mesh.nodes[level]
        for idx, cell in enumerate(marked_cells):
                if nodes_l[cell].is_active:
                    to_refine_mask[idx] = False
                pass
        pass
        if not np.any(to_refine_mask):
            return
        marked_cells = marked_cells[to_refine_mask]

        for l in range(level+1):
            cells = self.mesh.get_parent_at_level(start_level=level, stop_level=l, marked_cells_at_start_level=marked_cells)
            # Don't work with cells that were already refined, i.e the cells that were
            # inactive during the previous call of refine
            cells = np.setdiff1d(cells, previously_inactive_cells[l], assume_unique=False)
            if cells.size!=0:
                self._update_active_functions_incremental(marked_cells=cells, level=l)
        
       
        # self._update_active_functions()
        

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

        self.active_functions[l + 1] = np.array([], dtype=self.active_functions[l].dtype)
        self.deactivated_functions[l + 1] = np.array([], dtype=self.active_functions[l].dtype)
        self.bezier_operators[l+1] = self.level_spaces[l+1].bezier_operators
        self.refinement_operators[l+1] = self.level_spaces[l+1].refinement_operators

    def _update_active_functions_incremental(self, marked_cells: npt.NDArray[np.int_], level: int):
        """Updates active and deactivated functions when `marked_cells` at `level` have been refined to `level`+1. """

        space_l = self.level_spaces[level]
        # all affected functions

        # Find the functions that have some support on the refined cells, 
        # then find all the support cells of these functions.
        # If a function does not have any active cells left in its support, 
        # remove it from the list of active cells at that level.
        impacted_basis_l = np.unique(space_l.cell_to_basis(marked_cells))
        marked_cells = np.atleast_1d(marked_cells)
        for b_idx in impacted_basis_l:
            support = space_l.basis_to_cell(b_idx)
            
            num_removed = self._my_isin(support, marked_cells).sum()
            self.active_cell_counts[level][b_idx] -= num_removed
        pass
        function_stays = self.active_cell_counts[level]>0
        self.active_functions[level] = np.flatnonzero(function_stays).astype(np.int32)
        self.deactivated_functions[level] = np.flatnonzero(~function_stays).astype(np.int32)
        self.Bl_minus[level] = np.intersect1d(self.Bl_minus[level], self.active_functions[level], assume_unique=True)
        self.truly_active[level] = np.setdiff1d(self.active_functions[level], self.Bl_minus[level], assume_unique=True)

        # Then move on to the next level, and update active functions by
        # 1) computing which cells are now active, 
        # 2) computing which functions have support on these cells,
        # 3) computing how many active cells a function has in its support,
        # 4) updating active functions based on this new information.
        
        next_level = level+1
        if next_level in self.level_spaces:
            space_lp1 = self.level_spaces[next_level]
            
            # New children cells are created in the mesh during refine()
            # We find which functions at L+1 cover the children of marked_cells
            # Here, refinement is predictable: 1 cell splits into 2^dim children
            # We ask the mesh for the children indices
            children_cells = np.empty(shape=(len(marked_cells), 2**self.dim), dtype=np.int32)
            nodes_l = self.mesh.nodes[level]
            for idx, c in enumerate(marked_cells):
                children_cells[idx, :] = np.array([child.index for child in nodes_l[c].children], dtype=np.int32)
                #children_cells.extend([child.index for child in nodes_l[c].children])
            pass
            
            # children_cells = np.array(children_cells)
            # all functions that are supported on the newly refined cells
            impacted_basis_lp1 = np.unique(np.concatenate(space_lp1.cell_to_basis(children_cells.ravel())))
            
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

            function_stays = self.active_cell_counts[next_level]>0
            self.active_functions[next_level] = np.flatnonzero(function_stays).astype(np.int32)
            #print(f"active functions = {self.active_functions}")
            self.deactivated_functions[next_level] = np.flatnonzero(~function_stays).astype(np.int32)
            self.Bl_minus[next_level] = Bl_minus
            self.truly_active[next_level] = np.setdiff1d(self.active_functions[next_level], self.Bl_minus[next_level], assume_unique=True)
            #print(f"active functions = {self.active_functions}")
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
        self.active_functions: dict[int, npt.NDArray[np.int32]] = {}
        self.deactivated_functions: dict[int, npt.NDArray[np.int32]] = {}
        self.Bl_minus: dict[int, npt.NDArray[np.int32]] = {}
        self.truly_active: dict[int, npt.NDArray[np.int32]] = {}
        
        for l in range(self.nlevels):
            space_l: TensorProductSpace = self.level_spaces[l]
            nfuncs = space_l.nfuncs_total
            # Quick lookup dictionary for level l cells
            nodes_l: list[CellNode] = self.mesh.nodes[l]
            
            is_active_mask: npt.NDArray[np.bool_] = np.zeros(nfuncs, dtype=bool)
            is_minus_mask: npt.NDArray[np.bool_] = np.zeros_like(is_active_mask)

            for cell_idx, node in enumerate(nodes_l):
                cell_active: bool = node.is_active
                parent_active: bool = node.parent.is_active if node.parent else False

                if cell_active or parent_active:
                    affected_funcs: npt.NDArray[np.int_] = space_l.cell_to_basis(cell_idx)
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
        R_sliced = R_local@D
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
    
    def _my_isin(self, ar1: npt.NDArray, ar2: npt.NDArray)->npt.NDArray[np.bool_]:
        """`np.isin` when `ar2` is sorted.
        No checks are performed to ensure this.
        """
        ar1 = np.array(ar1)
        ar2 = np.atleast_1d(ar2)
        if len(ar2)<2:
            # searchsorted fails for short arrays
            return np.isin(ar1, ar2)
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
            M_top = M@R_sliced
            # Get J for current level
            J_l = self._compute_J(element_idx=element_idx, element_level=element_level, l=ll)
            M = sp.vstack((M_top, J_l), format='csr')
            
        return M
    
    def _deduplicate_dofs(self, dofs:list[tuple])->list[tuple]:
        """Helper method to remove duplicates while strictly preserving insertion order."""
        seen = set()
        dedup = []
        for item in dofs:
            if item not in seen:
                seen.add(item)
                dedup.append(item)
        return dedup

    def build_better_dof_map(self)->tuple[dict[tuple[int, int], int], int]:
        max_level = len(self.level_spaces) - 1
        
        current_dofs: list[tuple[int, int]] = [(0, f) for f in np.arange(self.level_spaces[0].nfuncs_total, dtype=self.truly_active[0].dtype)]
        for l in range(max_level):
            next_dofs = []
            # Extract all level 'l' functions currently in the list to check their activity
            funcs_l: list[int] = [f for (lvl, f) in current_dofs if lvl == l]
                
            # If there are no active functions of level l, make sure to eliminate possible duplicates.
            #Then, move on to the lext level.
            if not funcs_l:
                current_dofs: list[tuple[int, int]] = self._deduplicate_dofs(current_dofs)
                continue

            # Find which level 'l' functions are inactive.
            # This is not the same as deactivated functions.
            active_mask: npt.NDArray[np.bool_] = self._my_isin(funcs_l, self.truly_active[l])
            inactive_funcs: npt.NDArray[np.int32] = np.array(funcs_l)[~active_mask]

            # Batch fetch children for the inactive functions
            children_map: dict[int, npt.NDArray[np.int32]] = {}
            if len(inactive_funcs) > 0:
                sort_idx = np.argsort(inactive_funcs)
                sorted_inactive = inactive_funcs[sort_idx]
                
                children_list_sorted: list[npt.NDArray[np.int32]] = self.level_spaces[l].get_children_functions(sorted_inactive)
                
                for i, s_idx in enumerate(sort_idx):
                    # Map the parent function ID directly to its list of children
                    children_map[inactive_funcs[s_idx]] = children_list_sorted[i]
                pass
            pass

            # Expand the list: replace inactive functions with their children in-place
            for lvl, f in current_dofs:
                # Checking the level makes sure that we replace the inactive functions of the previous iteration
                # by the newly found children function.
                # The previous iteration, we replaced the inactive functions with each of their children in 
                # current_dofs. 
                # Now, we look at these children (filtered by lvl==l), 
                # and see which of them have to be replaced (f in children_map).
                # When such a function is found, we skip this function in next_dofs, and 
                # append the children instead.
                if lvl == l and f in children_map:
                    for child in children_map[f]:
                        # next dofs is initialised to [] at the beginning of each l iteration
                        next_dofs.append((l + 1, child))
                else:
                    next_dofs.append((lvl, f))
            pass
                    
            # Update and deduplicate, strictly preserving the order of first appearance
            # if a function is the child of several functions, we keep the first occurence.
            current_dofs = self._deduplicate_dofs(next_dofs)

        pass

        # Final filtering to remove any max_level functions that are not truly active
        max_level_funcs: list[np.int32] = [f for (lvl, f) in current_dofs if lvl == max_level]
        if max_level_funcs:
            active_max_mask: npt.NDArray[np.bool_] = self._my_isin(max_level_funcs, self.truly_active[max_level])
            active_max_set = set(np.array(max_level_funcs)[active_max_mask])
        else:
            active_max_set = set()

        # Build the final dictionary mapping (lvl, f) -> global ID
        dof_map = {}
        next_id = 0
        for lvl, f in current_dofs:
            if lvl == max_level and f not in active_max_set:
                continue
            
            k = (lvl, f)
            if k not in dof_map:
                dof_map[k] = next_id
                next_id += 1
                
        greatest_id = next_id - 1 if next_id > 0 else 0
        return dof_map, greatest_id
                
    def build_global_dof_map(self)->tuple[dict[tuple[int, int], int], int]:
        """Numbers active functions that are not active on a coarser level.

        Returns
        --------------
        - dof_map: dict[(level, local_func_idx), global_func_idx]
            global numbering of active functions (those who have support on Ω^l_{-} are not in `dof_map`).
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
    
    def get_all_active_functions_on_cell(self, level: int, cell_idx: int)->dict[int, npt.NDArray[np.int32]]:
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
            functions_on_cell: npt.NDArray[np.int_] = self.level_spaces[l].cell_to_basis(current_cell)
            #all_functions[l] = functions

            # active functions with no support on Ω^l_{-}
            truly_active: npt.NDArray[np.int32] = self.truly_active[l]

            if len(functions_on_cell)>0 and len(truly_active)>0:
                # Given the functions that have support on this cell, which ones are active?
                active_mask: npt.NDArray[np.bool_] = self._my_isin(functions_on_cell, truly_active)
                active_functions_l[l] = functions_on_cell[active_mask]

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

    def refine_in_rectangle(self, rectangle: npt.NDArray, level: int):
        """
        Refines the mesh from `level` to `level`+1. All cells that intersect with the rectangle are refined up to a suitable level.
        This is to avoid L-shaped domains.

        :param rectangle: array containing endpoints of rectangle as [[a,b], [cd]] = [a,b]x[c,d]
        :param level: refinement level
        :return: indices of active cells marked for refinement.
        """
        
        rectangle=np.atleast_2d(rectangle)
        assert rectangle.shape == (2,2), "Provided rectangle does not have an appropriate shape."
        eps = 1e-10

        def find_intersecting_geometrically(level: int, cell_indices: npt.NDArray[np.int_]):
                #start with coarse mesh to eliminate many cells
                cells = self.mesh.meshes[level].cells[cell_indices]
                rect_min = rectangle[:, 0]  
                rect_max = rectangle[:, 1]
                cell_min = cells[..., 0]    
                cell_max = cells[..., 1]
                # overlap condition: (min1 <= max2) AND (max1 >= min2)
                overlap_dims = (rect_min <= cell_max - eps) & (rect_max >= cell_min + eps)
                intersecting_cells = np.all(overlap_dims, axis=-1)

                #  Get the active indices of those cells
                active_indices = np.where(intersecting_cells)[0]
                return active_indices
        
        active_indices = find_intersecting_geometrically(0, np.arange(self.mesh.nel_per_level[0], dtype=np.int32))
        self.refine(marked_cells=active_indices, level=0)

        for l in range(1, level+1):
                _, children_cells = self.mesh.get_children(level=l-1, marked_cells_at_level=active_indices)
                active_indices = children_cells[find_intersecting_geometrically(l, children_cells)]
                self.refine(marked_cells=active_indices, level=l)
        

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
