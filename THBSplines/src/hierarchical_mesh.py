import numpy as np
from THBSplines.src.cartesian_mesh import CartesianMesh
from scipy.spatial import KDTree
from collections import deque
import numpy.typing as npt
from numba import jit

def sorted_isin(ar1: npt.NDArray, ar2: npt.NDArray)->npt.NDArray[np.bool_]:
        """`np.isin` when `ar2` is sorted.
        No checks are performed to ensure this.
        """
        ar1 = np.array(ar1)
        ar2 = np.array(ar2)
        idx = np.searchsorted(ar2, ar1)
        valid_mask = idx < len(ar2)
        vals_to_keep = np.zeros(len(ar1), dtype=bool)
        vals_to_keep[valid_mask] = ar2[idx[valid_mask]]==ar1[valid_mask]
        return vals_to_keep

def refine_1d(knots: npt.ArrayLike, n_times: int=1)->npt.NDArray:
        """Given `knots`, returns its dyadic refinements done `n_times`."""
        knots = np.asarray(knots, dtype=np.float64)
        
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

class CellNode:
    "Tree node representing a cell in a hierarchical mesh."

    __slots__ = ['level', 'index', 'parent', 'children', 'is_active',
                 'is_refined']
    
    def __init__(self, level: int, index: int, parent: "CellNode"):
        self.level: int=level
        self.index: int=index
        self.parent: CellNode=parent
        self.children: list[CellNode]=[]
        self.is_active: bool = False
        self.is_refined: bool=False

    def add_child(self, child_node: "CellNode"):
        self.children.append(child_node)


class HierarchicalMesh():
    """
    Attributes
    -------------
    - nlevels: int
        number of levels in the hierarchy
    - one_d_indices: dict[int, list[np.ndarray]]
        coordinates vector for each level and each dimension
    - meshes_shape:
        number of cells per level and per dimension
    - nodes: dict{int, CellNode}
        lists of cells for each level
    - aelem_level: dict{int, np.ndarray}
        list of active cells for each level l
    - delem_level: dict{int, np.ndarray}
        list of deactivated cells for each level l
    - nel_per_level: dict
        number of active cells per level
    - cell_area_per_level: dict
        area of each cell for each level
    - nel: int
        number of cells
    
    Methods
    -------------
    - plot_cells():
        plots the hierarchical mesh
    - add_level():
        adds a new level to the hierarchy by refining the current finest mesh
    - refine(marked_cells):
        refines the hierarchical mesh at the provided cells.
    - refine_in_rectangle(rect, level):
        Refines to level+1 all active cells at the specified level that are 
        contained in or intersect the given rectangle.
    -  get_children(level, marked_cells):
        get children of marked cells at given level, i.e for cells Q_i^l, returns fine cells 
        {Q_k^{l+1}|there exists i such that Q_k^{l+1}⊆Q_i^l}
    - get_parent(level, marked_cells):
        For given level l and marked cells {Q_i^l}, returns coarse cells {Q_k^{l-1}| there exists k such that Q_i^l⊆Q_k^{l-1}}

    """

    def __init__(self, knots: list[npt.NDArray]):
        dim = len(knots)
        assert dim>0

        self.one_d_indices: dict[int, list[npt.NDArray]] = {0: [np.unique(knot) for knot in knots]}
        #self.meshes: list[CartesianMesh] = [CartesianMesh(knots, dim)]
        self.meshes_shape: dict[int, list[npt.NDArray[np.int32]]] = {0: np.array([len(knot)-1 for knot in self.one_d_indices[0]], dtype=np.int32)}
        self.nlevels: int = 1
        self.dim: int=dim
        self.nel = np.prod(self.meshes_shape[0])
        

        self.nodes: dict[int, list[CellNode]] = {0: [CellNode(level=0,index=i, parent=None) for i in range(self.nel)]}
        for node in self.nodes[0]:
            node.is_active=True

        self._aelem_level_set: dict[int, set[int]] = {0: set(range(self.nel))}  # active elements on level
        self._delem_level_set:dict[int, set[int]] = {0: set()}  # deactivated elements on level

        self.aelem_level:dict[int, npt.NDArray[np.int_]] = {0: np.array(list(self._aelem_level_set[0]), dtype=np.int32)}
        self.delem_level:dict[int, npt.NDArray[np.int_]] = {0: np.array(list(self._delem_level_set[0]), dtype=np.int32)}
        self.nel_per_level:dict[int, int] = {0: self.nel}

        # self.cell_area_per_level = {0: self.meshes[0].cell_areas}
        
    pass

    def is_active(self, level: int, indices: list[int])->npt.NDArray[np.bool_]:
        return sorted_isin(indices, self.aelem_level[level])
    
    def is_refined(self, level: int, indices: list[int])->npt.NDArray[np.bool_]:
        """Checks if given indices of level `level` are refined.
        
        :return refined: boolean array of same size as `indices` with `True` if 
        corresponding node was refined, and `False` otherwise.
        """
        indices = np.atleast_1d(indices)
        refined = np.zeros_like(indices, dtype=bool)
        #considered_nodes = self.nodes[level][indices]
        nodes_l = self.nodes[level]
        #refined = [node.is_refined for node in considered_nodes]
        for i, index in enumerate(indices):
            if nodes_l[index].is_refined:
                refined[i] = True
        
        return refined


    def refine(self, marked_cells: npt.NDArray[np.int_]|list[int], at_level: int):
        """
        Refines the hierarchical mesh, and updates the global element indices
        of active elements for each level.

        :param marked_cells: indices of cells marked for refinement
        :param at_level: level at which the refinement should take place
        :return: updated elements
        """
        assert at_level>=0, "Cells of negative level do not exist."
        marked_cells = np.atleast_1d(marked_cells)
        if marked_cells.size==0:
            return
        
        if at_level>=self.nlevels-1:
            while(self.nlevels<=at_level+1):
                self.add_level()
            pass
        pass
        
        marked_cells = np.unique(marked_cells)
        already_refined = self.is_refined(level=at_level, indices=marked_cells)
        if np.all(already_refined):
            return 
        marked_cells = marked_cells[~already_refined]
        

        # old_active_cells = self.aelem_level
        self._update_active_cells(marked_cells, at_level=at_level)
        self.nel=0
        for l in range(self.nlevels):
            self.aelem_level[l] = np.array(sorted(list(self._aelem_level_set[l])), dtype=self.aelem_level[0].dtype)
            self.delem_level[l] = np.array(sorted(list(self._delem_level_set[l])), dtype=self.aelem_level[0].dtype)
            self.nel_per_level[l] = len(self.aelem_level[l])
            self.nel += self.nel_per_level[l]
        pass
    pass

    def add_level(self):
        """
        Adds a new level and constructs tree edges (parent-child relationships) .
        """
        coarse_level=self.nlevels-1
        fine_level=self.nlevels
        #coarse_mesh: CartesianMesh = self.meshes[-1]
        #fine_mesh: CartesianMesh = coarse_mesh.refine() # refine method of CartesianMesh
        #self.meshes.append(fine_mesh)
        coarse_shape: npt.NDArray[np.int32] = self.meshes_shape[coarse_level]
        fine_shape = 2*coarse_shape
        self.meshes_shape[fine_level] = fine_shape
        num_fine_cells = np.prod(fine_shape)
        self.nlevels += 1
        self.one_d_indices[fine_level] = [refine_1d(knot, n_times=1) for knot in self.one_d_indices[coarse_level]]
        self.nodes[fine_level] = []

        # Initialise new attributes
        self._aelem_level_set[fine_level] = set()
        self._delem_level_set[fine_level] = set()
        self.aelem_level[fine_level] = np.array([], dtype=self.aelem_level[coarse_level].dtype)
        self.delem_level[fine_level] = np.array([], dtype=self.delem_level[coarse_level].dtype)
        self.nel_per_level[fine_level]=0
        # self.cell_area_per_level[fine_level] = fine_mesh.cell_areas

        f_flat_indices = np.arange(num_fine_cells, dtype=np.uint32)
        f_multi_indices: tuple[npt.NDArray[np.int_]] = np.unravel_index(f_flat_indices, tuple(fine_shape))

        c_multi_indices = tuple(idx//2 for idx in f_multi_indices)

        parent_flat_indices = np.ravel_multi_index(c_multi_indices, tuple(coarse_shape))
        coarse_nodes = self.nodes[coarse_level]
        for f_idx, p_idx in zip(f_flat_indices, parent_flat_indices):
            parent_node = coarse_nodes[p_idx]

            child_node = CellNode(level=fine_level, index=f_idx, parent=parent_node)
            parent_node.add_child(child_node)
            self.nodes[fine_level].append(child_node)



    def _update_active_cells(self, marked_cells: list[int]|npt.NDArray[np.int_], at_level: int):
        """
        Updates the set of active cells and deactivated cells.

        :param marked_cells: indices of cells marked for refinement
        :return: returns the newly added cells
        """
        # Uniquely identify all cells to be refined at a certain level
        nodes_to_refine: set[CellNode] = set()
        for idx in np.atleast_1d(marked_cells):
            nodes_to_refine.add(self.nodes[at_level][idx])
        pass
            

        # 1. BOTTOM-UP: Identify all ancestors that need to be refined.
        queue: deque[CellNode] = deque(nodes_to_refine)
        while queue:
            # Identify a cell that needs to be refined
            node: CellNode = queue.popleft()
            # If the cell's parent is not refined
            if node.parent and not node.parent.is_refined:
                # Add the parent node to the refinement list
                nodes_to_refine.add(node.parent)
                # Check if its parents need to be refined as well
                queue.append(node.parent)
            pass
        pass


        # 2. TOP-DOWN: Apply refinements iteratively
        sorted_nodes = sorted(list(nodes_to_refine), key=lambda x: x.level)
        for node in sorted_nodes:
            # If cell is not already refined 
            if not node.is_refined:
                # Deactivate it and mark it as refined
                node.is_active = False
                node.is_refined = True
                
                # Add the parent to the set of deactivated cells
                self._delem_level_set[node.level].add(node.index)
                # Remove it from the set of active cells if necessary
                if node.index in self._aelem_level_set[node.level]:
                    self._aelem_level_set[node.level].remove(node.index)
                pass
                
                # Activate children and add indices to the set of active cells.
                for child in node.children:
                    child.is_active = True
                    self._aelem_level_set[child.level].add(child.index)
                pass
            pass
        pass
    pass

    def get_children(self, level: int, marked_cells_at_level: list[int]|npt.NDArray[np.int_]|int) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """
        For given level l and marked cells {Q_i^l}, returns fine cells {Q_k^{l+1}| there exists i such that Q_k^{l+1}⊆Q_i^l}

        Parameters
        -------------
        - level: level of parent cells. The returned cells will be those corresponding to `level+1`
        - marked_cells_at_level: indices of cells whose children are sought

        Returns
        --------------------
        - [coarse_cells, indices]: tuple of numpy array with local indices of coarse cells and their corresponding children global indices.
        If only the children of the given cells are wanted, one should only use the second return argument and 
        get the children cells with fine_mesh.cells[indices].
        """
        coarse_cells, indices = [], []
        for idx in np.atleast_1d(marked_cells_at_level):
            node = self.nodes[level][idx]
            for child in node.children:
                coarse_cells.append(idx)
                indices.append(child.index)
            pass
        pass

        return np.array(coarse_cells, dtype=np.int32), np.array(indices, dtype=np.int32)
    
    def get_parent(self, level:int, marked_cells_at_level: list[int]|npt.NDArray[np.int_]|int, skip_assert=False) -> npt.NDArray[np.int_]:
        """
        For given level l and marked cells {Q_i^l}, returns coarse cells {Q_k^{l-1}| there exists k such that Q_i^l⊆Q_k^{l-1}}

        Parameters
        -------------
        - level: level of children cells. The returned cells will be those corresponding to `level-1`
        - marked_cells_at_level: indices of cells whose parents are sought

        Returns
        --------------------
        - indices: np.ndarray(int) parent global at level `level-1`.
        """
        if not skip_assert:
            assert level>0, "Parents of cells at level 0 do not exist."
            assert np.max(marked_cells_at_level)<len(self.nodes[level]), "There aren't as many cells at that level."
            assert np.min(marked_cells_at_level)>=0, "Cells indices cannot be negative."

        marked_cells_at_level = np.atleast_1d(marked_cells_at_level)
        if len(marked_cells_at_level)==1:
            return self.nodes[level][marked_cells_at_level[0]].parent.index

        level_nodes = self.nodes[level]
        indices = np.empty(len(marked_cells_at_level), dtype=int)
        for i, idx in enumerate(marked_cells_at_level):
            node: CellNode = level_nodes[idx]
            parent_node: CellNode = node.parent
            indices[i] = parent_node.index
        pass
        return indices
    
    def get_parent_at_level(self, start_level: int, stop_level: int, marked_cells_at_start_level: npt.NDArray[np.int_])->npt.NDArray[np.int_]:
        """Returs parents of `marked_cells_at_start_level` at level `stop_level`. 
        
        :param start_level: level of provided cells
        :param stop_level: level of sought parents
        :param marked_cells_at_start_level: indices of cells whose parents are to be found

        :return parent_indices: indices of parents at level `stop_level`. This method returns an int if `marked_cells_at_start_level` is
        an int or a list/array with only one index.  
        """
        assert start_level>=0 and stop_level>=0 and stop_level<=start_level and start_level<=self.nlevels
        assert np.max(marked_cells_at_start_level)<len(self.nodes[start_level]), "There aren't as many cells at that level."
        assert np.min(marked_cells_at_start_level)>=0, "Cells indices cannot be negative."

        if stop_level==start_level:
            return np.squeeze(marked_cells_at_start_level)
        if stop_level==start_level-1:
            return self.get_parent(level=start_level, marked_cells_at_level=marked_cells_at_start_level, skip_assert=True)
        
        marked_cells_at_start_level = np.atleast_1d(marked_cells_at_start_level)

        if len(marked_cells_at_start_level)==1:
            parent_node = self.nodes[start_level][marked_cells_at_start_level[0]].parent
            for _ in range(1, start_level-stop_level):
                parent_node = parent_node.parent
            pass
            return parent_node.index
        pass
        
        for level in range(start_level, stop_level, -1):
            marked_cells_at_start_level = self.get_parent(level=level, marked_cells_at_level=marked_cells_at_start_level, skip_assert=True)
        return marked_cells_at_start_level
    
    def _is_point_in_cell_geometry(self, node: CellNode, point: npt.NDArray)->bool:
        """Checks if a point is inside the bounding box of a specific node."""
        shape = self.meshes_shape[node.level]
        multi_indices = np.unravel_index(node.index, shape)

        eps = np.spacing(1.) 
        point = np.atleast_1d(point)
        
        for d, idx in enumerate(multi_indices):
            # Retrieve the 1D grid points for this dimension at this level
            nodes_1d = self.one_d_indices[node.level][d]
            
            # The 1D bounds of this specific cell along axis d
            c_min = nodes_1d[idx]
            c_max = nodes_1d[idx + 1]
            
            # If it falls outside any 1D bound, it's not in the cell
            if point[d] < c_min - eps or point[d] > c_max + eps:
                return False
                
        return True

    def find_active_cell(self, point: npt.NDArray)->tuple[int, int]:
        """Given `point`, returns the level and the finest cell that contains it.
        
        Returns
        ---------------
        - (level, cell_index), the level of the finest active cell containing `point` and its index 
        in that level.
        - None, if the point is outside the mesh
        """
        point = np.atleast_1d(point)
        assert len(point) == self.dim, "Point does not have the correct amount of dimensions."

        # Geometric checks for coarsest mesh.
        indices = np.empty(self.dim, dtype=np.int32)
        for d in range(self.dim):
            knots_d = self.one_d_indices[0][d]
            point_d = point[d]
            if point_d>knots_d[-1] or point_d<knots_d[0]:
                return None
            idx = np.searchsorted(knots_d, point_d, side='right')-1
            indices[d] = idx
        
        l0_index = np.ravel_multi_index(indices, tuple(self.meshes_shape[0]))

        current_node = self.nodes[0][l0_index]
        while not current_node.is_active:
            found_in_child = False
            for child in current_node.children:
                if self._is_point_in_cell_geometry(child, point):
                    current_node = child
                    found_in_child = True
                    break
                pass
            pass
            if not found_in_child:
                return None
            pass
        pass
        return (current_node.level, current_node.index)
    
    def refine_in_rectangle(self, rect, level: int):
        """
        Refines to level+1 all active cells at the specified level that are 
        contained in or intersect the given rectangle. The region is naturally 
        expanded to the cell boundaries to avoid L-shaped refinement patches.
        
        :param rect: list or np.ndarray of shape (dim, 2) defining the rectangle.
                     Format: [[min1, max1], [min2, max2], ..., [mind, maxd]]
        :param level: the level of cells to be evaluated and refined
        """
        if level >= self.nlevels-1:
            while(self.nlevels<=level+1):
                self.add_level()
            pass
        pass
        
        rect = np.atleast_2d(rect)
        if rect.shape != (self.dim, 2):
            raise ValueError(f"Rectangle must have shape ({self.dim}, 2). Got {rect.shape}")
            
        shape = self.meshes_shape[level]
        eps = np.spacing(1.)
        intersecting_1d_indices = []

        for d in range(self.dim):
            nodes_1d = self.one_d_indices[level][d]
            r_min, r_max = rect[d, 0], rect[d,1]

            c_mins = nodes_1d[:-1]
            c_maxs = nodes_1d[1:]

            mask = (c_maxs>r_min+eps) & (c_mins<r_max-eps)
            intersecting_1d_indices.append(np.flatnonzero(mask))

        if any(len(idx)==0 for idx in intersecting_1d_indices):
            print("No cells intersect the provided rectangle.")
            return
        
        if self.dim == 1:
            marked_cells = intersecting_1d_indices[0]
        else:
            # meshgrid generates all combinations of the intersecting 1D indices
            mesh_indices = np.meshgrid(*intersecting_1d_indices, indexing='ij')
            
            # Flatten each dimension's grid and convert the multi-indices back to flat indices
            multi_indices = tuple(m.flatten() for m in mesh_indices)
            marked_cells = np.ravel_multi_index(multi_indices, shape)
            
        # Refine
        self.refine(marked_cells, at_level=level)

    def plot_cells(self, figsize:tuple=(10,5), return_fig = False) -> None:
        """
        Plots the hierarchical mesh, and optionally returns the figure handle.
        Otherwise, the figure is displayed.

        :param return_fig: if true, return the figure handle
        :return: plt.fig handle if return_fig is true, None otherwise.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.get_cmap('tab10', max(10, self.nlevels))

        if getattr(self, 'dim', 1) not in [1,2]:
            print('Dimensions 3 and higher are not displayed.')
            plt.close(fig)
            return 
        
        for level in range(self.nlevels):
            active_indices = self.aelem_level[level]
            if len(active_indices)==0:
                continue

            shape = tuple(len(nodes)-1 for nodes in self.one_d_indices[level])
            multi_indices = np.unravel_index(active_indices, shape)

            if self.dim == 1:
                idx_x = multi_indices[0]
                x_nodes = self.one_d_indices[level][0]
                
                for ix in idx_x:
                    
                    xmin, xmax = x_nodes[ix], x_nodes[ix + 1]
                    width = xmax - xmin
                    height = 1.0 # arbitrary
                    
                    rect = plt.Rectangle((xmin, -height/2), width, height,
                                         edgecolor='black',
                                         facecolor=colors(level),
                                         alpha=0.6,
                                         linewidth=1.5)
                    ax.add_patch(rect)

            elif self.dim == 2:
                idx_x, idx_y = multi_indices
                x_nodes = self.one_d_indices[level][0]
                y_nodes = self.one_d_indices[level][1]
                
                for ix, iy in zip(idx_x, idx_y):
                    # Look up the 2D bounds from the 1D arrays
                    xmin, xmax = x_nodes[ix], x_nodes[ix + 1]
                    ymin, ymax = y_nodes[iy], y_nodes[iy + 1]
                    
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    # Draw a colored rectangle for the 2D element
                    rect = plt.Rectangle((xmin, ymin), width, height,
                                         edgecolor='black',
                                         facecolor=colors(level),
                                         alpha=0.6,
                                         linewidth=1.5)
                    ax.add_patch(rect)
                pass
            pass
        pass # end of for loop

        if self.dim == 1:
            ax.set_ylim(-2, 2)
            ax.set_yticks([]) # Hide the arbitrary y-axis values
            ax.set_xlabel('Parametric Domain')
        elif self.dim == 2:
            ax.set_aspect('equal', 'box') # Ensures squares look like squares
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        if self.nlevels > 1:
            handles = [mpatches.Patch(color=colors(l), alpha=0.6, label=f'Level {l}') 
                       for l in range(self.nlevels)]
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.autoscale_view()
        plt.tight_layout()
        
        if return_fig:
            return fig
        else:
            plt.show()
            plt.close(fig)



if __name__ == '__main__':
    knots = [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ]
    dim = 2
    M = HierarchicalMesh(knots, dim)
    marked_cells = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3, 4, 5, 6],
                    2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
    M.refine(marked_cells)
    M.plot_cells()
