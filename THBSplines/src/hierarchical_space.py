from functools import reduce
from typing import Union, List

import numpy as np
import scipy.sparse as sp
from THBSplines.src.abstract_space import Space
from THBSplines.src.hierarchical_mesh import HierarchicalMesh, CellNode
from THBSplines.src.tensor_product_space import TensorProductSpace, UnivariateSplineSpace


class HierarchicalSpace(Space):

    def cell_to_basis(self, cell_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        pass

    def basis_to_cell(self, basis_indices: Union[np.ndarray, List[int]]) -> np.ndarray:
        pass

    def __init__(self, knots: list, degrees: list):
        """Initialise one level """
        self.degrees = np.atleast_1d(degrees)
        if len(self.degrees)==1:
            self.degrees = self.degrees*np.ones(len(knots), dtype=int)
        else:
            assert len(knots)==len(self.degrees), "There are not enough degrees for the given knots."
        self.dim: int = len(knots)
        self.mesh: HierarchicalMesh = HierarchicalMesh(knots=knots, dim=self.dim)
        univariate_spline_spaces = [UnivariateSplineSpace(degree=self.degrees[d], knots=knots[d]) for d in range(self.dim)]
        self.level_spaces: dict[int, TensorProductSpace] = {0: TensorProductSpace(dim=self.dim, univariate_spaces=univariate_spline_spaces)}
        self.nlevels = 1
        self.active_functions: dict[int, np.ndarray] = {0: np.arange(self.level_spaces[0].nfuncs_total)}
        self.deactivated_functions: dict[int, np.ndarray] = {0: np.array([], dtype=int)}
        self.refinement_operators: dict[int, np.ndarray] = {0: self.level_spaces[0].refinement_operators}
        self.bezier_operators: dict[int, np.ndarray] = {0: [uni_spline_space.bezier for uni_spline_space in univariate_spline_spaces]}
        

    def refine(self, marked_cells: list, level: int, axes=None):
        """
        Refines the specified cells at a given level.
        
        :param marked_cells: List of flat cell indices at 'level' to refine.
        :param level: The level at which the marked_cells currently exist.
        :param axes: Optional list of axes to refine.
        """

        marked_cells = np.atleast_1d(marked_cells)
        if len(marked_cells) == 0:
            return

        # 1. If we are refining cells on the currently finest level, 
        # we MUST generate the next level mathematically and physically.
        if level >= self.nlevels-1:
            while(self.nlevels<=level+1):
                self.add_level(axes=axes)
            pass
        pass

        # 2. Update the physical mesh topology
        # This includes refining the current finest mesh and creating new CellNodes if necessary, 
        # and updating active/deactivated cells
        self.mesh.refine(marked_cells=marked_cells, at_level=level)
        
        # 3. Re-evaluate basis function statuses (Active vs Deactivated)
        self._update_active_functions()
        
        # 4. Re-evaluate Truncation Operators for THB-splines
        # self._update_truncation_operators()

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
        self.active_functions[l + 1] = np.array([], dtype=int)
        self.deactivated_functions[l + 1] = np.array([], dtype=int)
        self.bezier_operators[l+1] = [self.level_spaces[l+1].spaces[d].bezier for d in range(self.dim)]
        self.refinement_operators[l+1] = self.level_spaces[l+1].refinement_operators

    def _update_active_functions(self):
        """
        Updates the set of active and deactivated functions.
        A function of level l is active if its supports intersects at least one active cell of level l. 
        It is inactive otherwise.
        An cell is said to be active if it was not refined.
        """
        # Dictionaries for each refinement level
        self.active_functions = {}
        self.deactivated_functions = {}
        
        for l in range(self.nlevels):
            space_l = self.level_spaces[l] #TensorProductSpace
            active_l = []
            deactivated_l = []
            
            # Quick lookup dictionary for level l cells
            nodes_l: list[CellNode] = self.mesh.nodes[l]
            
            for i in range(space_l.nfuncs_total):
                # Get flat cell indices at level l spanning the support of function i
                support_cells = space_l.basis_to_cell(np.array([i]))[0] 
                
                overlaps_Omega_l = False
                #overlaps_Omega_minus = False
                
                for cell_idx in support_cells:
                    if nodes_l[cell_idx].is_active:
                        overlaps_Omega_l = True
                        # Stop as soon as an active cell is found in the function's support
                        break
                
                # The mathematical condition for inclusion in HB^l
                if overlaps_Omega_l:
                    active_l.append(i)
                else:
                    deactivated_l.append(i)
                    
            self.active_functions[l] = np.array(active_l, dtype=int)
            self.deactivated_functions[l] = np.array(deactivated_l, dtype=int)

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

    def _truncation_operator(self, element_idx: int, element_level: int, l: int)->sp.bsr_array:
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
        """
        assert l>0, "l has to be positive, as the returned truncation matrix goes from level `l-1` to `l`."
        #space_coarse = self.level_spaces[l-1]
        space_fine: TensorProductSpace = self.level_spaces[l]
        assert element_idx<len(self.mesh.nodes[element_level]), "There aren't as many nodes at that level."
        assert element_level>=l, "Not sure if l can be smaller than element_level"

        coarse_element_idx = self.mesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)
        R_local: sp.bsr_array = self.refinement_operators[l-1][coarse_element_idx] #go from level l-1 to l on coarse_element_idx
        R_trunc = R_local.copy()
        my_blocksize = R_local.blocksize
        R_shape = R_trunc.shape
        fine_funcs_on_elem: np.ndarray = space_fine.cell_to_basis(np.array([coarse_element_idx]))
        nodes_fine: list[CellNode] = self.mesh.nodes[l]

        cols_to_truncate = np.ones(shape=(R_shape[0]), dtype=bool)
        for col_idx, fine_func in enumerate(fine_funcs_on_elem):
            # Get the full support of this fine function
            support_cells: np.ndarray = space_fine.basis_to_cell(np.array([fine_func]))[0]
            
            overlaps_coarse_domain = False
            for cell in support_cells:
                if not nodes_fine[cell].parent.is_refined:
                    # If the support cell doesn't exist in the level l mesh, 
                    # it physically resides in the unrefined coarse domain (\Omega^l_-).
                    overlaps_coarse_domain = True
                    break
            
            # If it does not overlap the coarse domain, truncate it
            if not overlaps_coarse_domain:
                cols_to_truncate[col_idx]=0
                #R_trunc[:, col_idx] = 0.
        if np.all(cols_to_truncate):
            return sp.bsr_array(R_trunc, blocksize=my_blocksize)
        if not np.any(cols_to_truncate):
            return sp.bsr_array(R_shape, dtype=R_trunc.dtype, blocksize=my_blocksize)
        
        # bsr_arrays cannot be sliced, so replace with this matrix-matrix multiplication instead which sould be fast. 
        mask = sp.diags_array(cols_to_truncate, format='bsr', dtype=bool)
        return sp.bsr_array(R_trunc.dot(mask), blocksize=my_blocksize)
    
    def _compute_J(self, element_idx: int, element_level: int, l: int) -> sp.dia_array:
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

        coarse_ancestor_idx: int=self.mesh.get_parent_at_level(start_level=element_level, stop_level=l, marked_cells_at_start_level=element_idx)
        # Get ancestor of level `l` of the considered element
        # for ll in range(element_level, l, -1):
        #     coarse_ancestor_idx = self.mesh.get_parent(level=ll, marked_cells_at_level=coarse_ancestor_idx) # Parent cell of level l-1
        funcs_on_elem: np.ndarray = space_l.cell_to_basis(np.array([coarse_ancestor_idx]))
        supports_of_funcs_on_elem: np.ndarray = space_l.basis_to_cell(funcs_on_elem)
        
        # The globally active functions in the THB basis at level l
        active_thb_funcs_l = set(self.active_functions[l]) #discards functions only active on Ω^l_{+} and Ω^l_{-}
        
        # Keep only the rows corresponding to functions that are actually in the THB basis
        rows_to_keep = np.ones(len(funcs_on_elem), dtype=bool)
        # Of all the functions whose support includes the considered element
        for i, func in enumerate(funcs_on_elem):
            #func_is_okay = []
            # Only keep those whose support overlaps their level space
            if func in active_thb_funcs_l:
                # Look at all cells in the support of these functions
                for cell in supports_of_funcs_on_elem[i]:
                    # Finally, only keep those whose parent cell is not active (i.e their support does not intersect Ω^l_{-})
                    node_cell = self.mesh.nodes[l][cell]
                    if node_cell.parent is not None and node_cell.parent.is_active:
                        rows_to_keep[i]=False
                        break
            else:
                rows_to_keep[i]=False
                
        # J is initially the identity matrix for all functions on the element
        # We slice it to keep only the active THB functions
        #J = sp.eye(len(funcs_on_elem), dtype=float, format='csr')
        #return sp.bsr_array(J[rows_to_keep, :])
        my_size=len(funcs_on_elem)
        rows_to_keep_sparse = np.arange(my_size)[rows_to_keep]
        return sp.diags_array(np.ones(my_size, dtype=bool), offsets=np.min(rows_to_keep_sparse), shape=(len(rows_to_keep_sparse), my_size), dtype=bool, format='csr')
    
    def local_multi_level_extraction_operator(self, element_idx: int, element_level: int, l: int):
        """
        Builds the local multi-level extraction operator M_{L, epsilon}^{loc}
        
        """
        # Base case: Level 0
        M = self._compute_J(element_idx=element_idx, element_level=element_level, l=0)
        
        # Iteratively apply the algorithm: M_{l} = [ M_{l-1} * trunc(R) ]
        #                                          [        J^l         ]
        for ll in range(1, l+1):
            
            trunc_R = self._truncation_operator(element_idx=element_idx, element_level=element_level, l=ll)
            M_top = M.dot(trunc_R)
            # Get J for current level
            J_l = self._compute_J(element_idx=element_idx, element_level=element_level, l=ll)
            M = sp.vstack((M_top, J_l), format='csr')
            
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
