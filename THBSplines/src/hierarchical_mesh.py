import numpy as np
from THBSplines.src.abstract_mesh import Mesh
from THBSplines.src.cartesian_mesh import CartesianMesh
from scipy.spatial import KDTree
from collections import deque
import numpy.typing as npt

class CellNode:
    "Tree node representing a cell in a hierarchical mesh."

    __slots__ = ['level', 'index', 'parent', 'children', 'is_active',
                 'is_refined']
    
    def __init__(self, level: int, index: int, parent=None):
        self.level: int=level
        self.index: int=index
        self.parent: CellNode=parent
        self.children: list[CellNode]=[]
        self.is_active: bool = False
        self.is_refined: bool=False

    def add_child(self, child_node: "CellNode"):
        self.children.append(child_node)


class HierarchicalMesh(Mesh):
    """
    Attributes
    -------------
    - nlevels: int
        number of levels in the hierarhy
    - meshes: CartesianMesh
        Cartesian mesh structure for each level l and information about cells
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

    def __init__(self, knots, dim):
        self.meshes: list[CartesianMesh] = [CartesianMesh(knots, dim)]
        self.nlevels: int = 1
        self.dim: int=dim

        self.nodes: dict[int, list[CellNode]] = {0: [CellNode(level=0,index=i) for i in range(self.meshes[0].nelems)]}
        for node in self.nodes[0]:
            node.is_active=True

        self._aelem_level_set: dict[int, set[int]] = {0: set(range(self.meshes[0].nelems))}  # active elements on level
        self._delem_level_set:dict[int, set[int]] = {0: set()}  # deactivated elements on level

        self.aelem_level:dict[int, npt.NDArray[np.int_]] = {0: np.array(list(self._aelem_level_set[0]), dtype=np.int32)}
        self.delem_level:dict[int, npt.NDArray[np.int_]] = {0: np.array(list(self._delem_level_set[0]), dtype=np.int32)}
        self.nel_per_level:dict[int, int] = {0: self.meshes[0].nelems}

        self.cell_area_per_level = {0: self.meshes[0].cell_areas}
        self.nel = self.meshes[0].nelems
    pass


    def add_level(self):
        """
        Adds a new level and constructs tree edges (parent-child relationships) .
        """
        coarse_level=self.nlevels-1
        fine_level=self.nlevels
        coarse_mesh: CartesianMesh = self.meshes[-1]
        fine_mesh: CartesianMesh = coarse_mesh.refine() # refine method of CartesianMesh
        self.meshes.append(fine_mesh)
        self.nlevels += 1

        # Initialise new attributes
        self._aelem_level_set[fine_level] = set()
        self._delem_level_set[fine_level] = set()
        self.aelem_level[fine_level] = np.array([], dtype=np.int32)
        self.delem_level[fine_level] = np.array([], dtype=np.int32)
        self.nel_per_level[fine_level]=0
        self.cell_area_per_level[fine_level] = fine_mesh.cell_areas

        coarse_centers = np.mean(coarse_mesh.cells, axis=-1)
        fine_centers = np.mean(fine_mesh.cells, axis=-1)
        if self.dim == 1:
            coarse_centers = coarse_centers[:, None]
            fine_centers = fine_centers[:, None]
            
        tree = KDTree(coarse_centers)

        k_neighbours = min(3**self.dim, len(coarse_centers))
        #a 1D mesh has two neighbours, which makes three elts including itself
        # a 2D mesh has 8 neighbours (including corners), which makes nine elts including itself
        # a 3D mesh has 26 neighbouts, etc

        # Returns index of each neighbour in tree.data
        # The parent cell is guaranteed to be in the closest k_neighbours coarse centroids
        _, indices = tree.query(fine_centers, k=k_neighbours)
        if k_neighbours == 1:
            indices = indices[:, None]

        c_min = coarse_mesh.cells[..., 0]
        c_max = coarse_mesh.cells[..., 1]
        eps = np.spacing(1)
        self.nodes[fine_level] = []
        coarse_nodes: list[CellNode] = self.nodes[coarse_level]

        # Find parent cell of each fine cell
        for i, fine_center in enumerate(fine_centers):
            parent_idx = -1
            # Verify exact geometrical inclusion to discard unwanted neighbours
            for candidate in indices[i]:
                if np.all(fine_center >= c_min[candidate] - eps) and np.all(fine_center <= c_max[candidate] + eps):
                    parent_idx = candidate
                    break #only breaks the inner loop
                pass
            pass
            # Geometric checking if tree based approach failed
            if parent_idx == -1:
                fine_cell_min = fine_mesh.cells[i, :, 0] + eps
                fine_cell_max = fine_mesh.cells[i, :, 1] - eps
                valid = np.all((fine_cell_min >= c_min) & (fine_cell_max <= c_max), axis=-1)
                parent_idx = np.argmax(valid)
            pass
            # Get identified parent node
            parent_node: CellNode = coarse_nodes[parent_idx] # Create corresponding child_node
            child_node = CellNode(level=fine_level, index=i, parent=parent_node)
            # Make parent keep record of created child
            parent_node.add_child(child_node)
            self.nodes[fine_level].append(child_node) #Dict of nodes
        pass
    pass

    def refine(self, marked_cells: npt.NDArray[np.int_]|list[int], at_level: int):
        """
        Refines the hierarchical mesh, and updates the global element indices
        of active elements for each level.

        :param marked_cells: indices of cells marked for refinement
        :param at_level: level at which the refinement should take place
        :return: updated elements
        """

        if at_level>=self.nlevels-1:
            while(self.nlevels<=at_level+1):
                self.add_level()
            pass
        pass

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
            assert level>0, "Parents of cells at level 0 do not exist"
            assert np.max(marked_cells_at_level)<len(self.nodes[level]), "There aren't as many cells at that level."
            assert np.min(marked_cells_at_level)>=0, "Cells indices cannot be negative."

        marked_cells_at_level = np.atleast_1d(marked_cells_at_level)
        if len(marked_cells_at_level)==1:
            return self.nodes[level][marked_cells_at_level[0]].parent.index

        level_nodes = self.nodes[level]
        indices = np.empty(len(marked_cells_at_level), dtype=level_nodes[0].index.dtype)
        for i, idx in enumerate(marked_cells_at_level):
            node: CellNode = level_nodes[idx]
            parent_node: CellNode = node.parent
            indices[i] = parent_node.index
        pass
        return indices
    
    def get_parent_at_level(self, start_level: int, stop_level: int, marked_cells_at_start_level: npt.NDArray[np.int_])->np.ndarray:
        """Returs parents of `marked_cells_at_start_level` at level `stop_level`. """
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
        # Retrieve the cell's AABB: shape (dim, 2) -> [[min_x, max_x], [min_y, max_y]...]
        cell_bounds = np.atleast_2d(self.meshes[node.level].cells[node.index])
        
        eps = np.spacing(1.) # Tolerance for floating point boundaries
        return np.all( (cell_bounds[:, 0]-eps<point) & (cell_bounds[:, 1]+eps>point) )
        

    def find_active_cell(self, point: npt.NDArray)->tuple[int, int]:
        """Given `point`, returns the level and the finest cell that contains it.
        
        Returns
        ---------------
        - (level, cell_index), the level of the finest active cell containing `point` and its index 
        in that level.
        """
        point = np.atleast_1d(point)
        l0_index = self.meshes[0].find_index(point)
    
        if l0_index is None:
            return None #point is outside the mesh domain
        pass

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
            
        cells: np.ndarray = self.meshes[level].cells
        cell_mins = cells[..., 0]
        cell_maxs = cells[..., 1]
        rect_mins = rect[:, 0]
        rect_maxs = rect[:, 1]

        eps = np.spacing(1) * 10
        
        if self.dim==1:
            intersect_mask = (cell_maxs>rect_mins+eps)&(cell_mins<rect_maxs-eps) 
        else:
            intersect_mask = np.all(
                (cell_maxs > rect_mins + eps) & (cell_mins < rect_maxs - eps), 
                axis=1
            )
        pass
        marked_cells = np.where(intersect_mask)[0]
        
        if len(marked_cells) > 0:
            # Our existing `refine` method gracefully handles overlapping 
            # or previously refined cells thanks to the graph-based top-down logic.
            self.refine(marked_cells, at_level=level)
        else:
            print("No cells intersected the provided rectangle.")
        pass
        

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
        for level in range(self.nlevels):
            active_cells = self.meshes[level].cells[self.aelem_level[level]]
            if self.meshes[0].dim==2:
                for cell in active_cells:
                    x = cell[0, [0, 1, 1, 0, 0]]
                    y = cell[1, [0, 0, 1, 1, 0]]
                    plt.plot(x, y, color='black')
            elif self.meshes[0].dim==1:
                for cell in active_cells:
                    xmin, xmax = cell[0], cell[1]
                    width = xmax - xmin
                    height = 1.0 # arbitrary
                    
                    # Draw a colored rectangle for each 1D element
                    rect = plt.Rectangle((xmin, -height/2), width, height,
                                         edgecolor='black',
                                         facecolor=colors(level),
                                         alpha=0.6,
                                         linewidth=1.5)
                    ax.add_patch(rect)
            else:
                print('Dimensions 3 and higher are not displayed')
                plt.close(fig)
                return
            pass
        pass
        if self.meshes[0].dim == 1:
            ax.set_ylim(-2, 2)
            ax.set_yticks([]) # Hide the arbitrary y-axis values
            ax.set_xlabel('Parametric Domain')
            
            # Create a custom legend to know which color is which level
            handles = [mpatches.Patch(color=colors(l), alpha=0.6, label=f'Level {l}') 
                       for l in range(self.nlevels)]
            # Only show legend if we have more than 1 level
            if self.nlevels > 1:
                ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
                
            ax.autoscale_view()

        plt.tight_layout()
        if return_fig:
            return fig
        else:
            plt.show()

    def get_gauss_points(self, cell_indices: np.ndarray) -> np.ndarray:
        pass


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
