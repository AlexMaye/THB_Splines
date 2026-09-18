from thbsplines.hierarchical_space import HierarchicalSpace
import numpy as np
import scipy.sparse as sp
import dolfinx
from mpi4py import MPI
import basix.ufl
import pyvista

from dolfinx import default_real_type, default_scalar_type
rtype = default_real_type
dtype = default_scalar_type
import ufl

from thbsplines.refinement import refine
from thbsplines.fenicsx.mesh import build_mesh, FastMidpointMapper
from thbsplines.fenicsx.functionspace import build_dofmap, fill_function_space, create_spline_space
from thbsplines.fenicsx.solvers import solve_problem, enforce_dirichlet_boundary
from thbsplines.fenicsx.adaptivity import dorfler_marking
from thbsplines.fenicsx.kernels import make_linear_kernel, make_bilinear_kernel
from thbsplines.fenicsx.postprocessing import map_spline_to_legendre, convergence_plot
from thbsplines.fenicsx.forms import mark_cells, make_bilinear_form, make_linear_form, mark_exterior_boundary_entities, exterior_facets

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# This example reproduces the result of the following paper
# Multi-level Bézier extraction for hierarchical local refinement of Isogeometric Analysis
# (https://doi.org/10.1016/j.cma.2017.08.017).
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

def map_uv_to_xy_turn(uv_points, nodes_per_cell=4):
    """Maps parametric [0,1]^2 to a 4-cell L-shape without polar singularities."""
    # Ensure points are structured as complete cells
    assert len(uv_points) % nodes_per_cell == 0, "uv_points must be grouped by cells."
    
    # Reshape points to (N_cells, 4, 2)
    uv_cells = uv_points.reshape(-1, nodes_per_cell, 2)
    xy_cells = np.zeros_like(uv_cells)
    
    def bilinear(u_loc, v_loc, C00, C10, C01, C11):
        """Standard isoparametric Q1 interpolation"""
        return ((1-u_loc)*(1-v_loc)*C00 + 
                u_loc*(1-v_loc)*C10 + 
                (1-u_loc)*v_loc*C01 + 
                u_loc*v_loc*C11)
    
    # Map cell by cell using the cell center to uniquely identify the block
    for i in range(len(uv_cells)):
        u = uv_cells[i, :, 0]
        v = uv_cells[i, :, 1]
        
        # Cell centers bypass boundary ambiguities
        u_c = np.mean(u)
        v_c = np.mean(v)
        
        if u_c < 0.25:
            if v_c >= 0.5:
                # Cell 0 -> P0 (Bottom-Left)
                C00, C10, C01, C11 = [-0.5, -1], [-0.5, -0.25], [-1, -1], [-1, 0]
                u_loc, v_loc = u * 4.0, (v-0.5) * 2.0
            else:
                # Cell 2 -> P1 
                C00, C10, C01, C11 = [0, -1], [0, -0.5], [-0.5, -1], [-0.5, -0.25]
                u_loc, v_loc = u * 4.0, v * 2.0
                
                
        elif u_c < 0.5:
            if v_c >= 0.5:
                # Cell 1 -> P2
                C00, C10, C01, C11 = [-0.5, -0.25], [-0.5, 0.5], [-1, 0], [-1, 1]
                u_loc, v_loc = (u-0.25) * 4.0, (v - 0.5) * 2.0
            else:
                # Cell 3 -> P3 
                C00, C10, C01, C11 = [0, -0.5], [0, 0], [-0.5, -0.25], [-0.5, 0.5]
                u_loc, v_loc = (u - 0.25) * 4.0, v * 2.0
                
        elif u_c < 0.75:
            if v_c < 0.5:
                # Cell 4 -> P4 (Bottom-Mid-Right) - Connects to P3's top edge!
                C00, C10, C01, C11 = [0, 0], [0.5, 0], [-0.5, 0.5], [0.25, 0.5]
                u_loc, v_loc = (u - 0.5) * 4.0, v * 2.0
            else:
                # Cell 5 -> P5 (Top-Mid-Right)
                C00, C10, C01, C11 = [-0.5, 0.5], [0.25, 0.5], [-1, 1], [0, 1]
                u_loc, v_loc = (u - 0.5) * 4.0, (v - 0.5) * 2.0
                
        else:
            if v_c < 0.5:
                # Cell 6 -> P6 (Bottom-Right)
                C00, C10, C01, C11 = [0.5, 0], [1, 0], [0.25, 0.5], [1, 0.5]
                u_loc, v_loc = (u - 0.75) * 4.0, v * 2.0
            else:
                # Cell 7 -> P7 (Top-Right)
                C00, C10, C01, C11 = [0.25, 0.5], [1, 0.5], [0, 1], [1, 1]
                u_loc, v_loc = (u - 0.75) * 4.0, (v - 0.5) * 2.0
                
        C00, C10 = np.array(C00), np.array(C10)
        C01, C11 = np.array(C01), np.array(C11)
        
        xy = bilinear(u_loc[:, None], v_loc[:, None], C00, C10, C01, C11)
        xy_cells[i] = xy
        
    return xy_cells.reshape(-1, 2)

if __name__=="__main__":
    print("The first iteration is a bit slow, because the numba code needs to be compiled...")
    n_refinements = 1
    p0 = 3
    # ==================================================
    # Make sure that the middle knot has a multiplicity of p0
    knotsx = np.array([0,0,0, 0.5,0.5,0.5, 1,1,1], dtype=np.float64)
    #===================================================
    knotsx = refine(knotsx, p=p0, n_times=n_refinements)
    knotsy = np.array([0,0,0, 0.5,1,1,1], dtype=np.float64)
    knotsy = refine(knotsy, p0, n_times=n_refinements-1)
    err_cells = {}
    hierarchical_space = HierarchicalSpace(knots=[knotsx, knotsy], degrees=[p0])

    err_cells = {}
    
    # Hyperparameters for the iterative refinement loop
    n_iterations = 0
    n_max_iterations = 11
    TOL=1e-7
    current_error=1000.
    errors_array = np.zeros((n_max_iterations, 2), dtype=float)

    while ((n_iterations<n_max_iterations) and (current_error>TOL)):
        print(f"Iteration {n_iterations}")
        for level, cells in err_cells.items():
            hierarchical_space.refine(cells, level, refine_neighbours=False, refine_T_neighbours=True, m=2)
        pass

        disconnected_mesh, thb_operators, N_max, physical_cells_midpoints = build_mesh(hs=hierarchical_space, mapping=map_uv_to_xy_turn)

        def outer_boundary(x):
            on_left = np.isclose(x[0], -1.)
            on_bottom = np.isclose(x[1], -1.)
            on_right = np.isclose(x[0], 1.) & (x[1]>=-1e-10)
            on_top = np.isclose(x[1], 1.) & (x[0]<=1.)
            return on_left|on_bottom|on_right|on_top

        legendre_elt = basix.ufl.element(
            "DG",
            "quadrilateral",
            degree=p0,
            lagrange_variant=basix.LagrangeVariant.legendre
        )
        V = dolfinx.fem.functionspace(disconnected_mesh, legendre_elt)
        
        facet_dim = disconnected_mesh.topology.dim-1
        boundary_facets = dolfinx.mesh.locate_entities_boundary(disconnected_mesh, facet_dim, outer_boundary)
        facet_values = np.full_like(boundary_facets, 1, dtype=np.int32)
        boundary_tags = dolfinx.mesh.meshtags(
            disconnected_mesh, 
            facet_dim,
            boundary_facets,
            facet_values
        )
        custom_metadata = {"quadrature_degree": 12}
        ds = ufl.Measure("ds", domain=disconnected_mesh, subdomain_data=boundary_tags, metadata=custom_metadata)
        dx_custom = ufl.Measure("dx", domain=disconnected_mesh, metadata=custom_metadata)


        u,v = ufl.TrialFunction(V), ufl.TestFunction(V) 
        a0 = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_custom

        my_x = ufl.SpatialCoordinate(disconnected_mesh)
        n_normal = ufl.FacetNormal(disconnected_mesh)
        r_radius = ufl.sqrt(my_x[0]**2+my_x[1]**2+1e-15)
        theta_angle = ufl.atan2(my_x[1], my_x[0])
        # restrict theta to [0, 2\pi)
        theta_angle = ufl.conditional(condition=ufl.lt(left= theta_angle,right= 0.), true_value= theta_angle+ 2*np.pi, false_value= theta_angle)
    
        u_bar = r_radius**(2./3.)*ufl.sin(2./3. * theta_angle)
        g = ufl.dot(ufl.grad(u_bar), n_normal)
        # ds(1) because we defined our subdomain_data to have label "1" at the boundaries of interest.
        L_neumann = ufl.inner(g, v)*ds(1)

        dofmap, padded_cells_to_dofs = build_dofmap(hierarchical_space=hierarchical_space, mesh=disconnected_mesh, N_max=N_max, morton=True)
        custom_mapping = FastMidpointMapper(hierarchical_space, physical_cells_midpoints)
        C_func, C_space = fill_function_space(hierachical_space=hierarchical_space, mesh=disconnected_mesh, N_max = N_max, 
                                            thb_operators=thb_operators, mapping_function=custom_mapping)
        V_spline = create_spline_space(cells_to_dofs=padded_cells_to_dofs, mesh=disconnected_mesh,
                                    N_max=N_max, mult_factor=1)
        local_dofs = (hierarchical_space.degrees[0]+1)**2
        tabulate_A = make_bilinear_kernel(disconnected_mesh, a0, padded_dofs=N_max, local_dofs=local_dofs)
        tabulate_L = make_linear_kernel(disconnected_mesh, L_neumann, padded_dofs=N_max, local_dofs=local_dofs)

        cell_domain = mark_cells(mesh=disconnected_mesh)
        boundary_entities = mark_exterior_boundary_entities(mesh=disconnected_mesh, 
                                                            boundary_function=outer_boundary)
        exterior_domain = exterior_facets(entities=boundary_entities)
        a_cond = make_bilinear_form(mesh=disconnected_mesh,
                                    ufl_form=a0,
                                    trial_space=V_spline, test_space=V_spline,
                                    coefficients=C_func, 
                                    integrals=[(cell_domain, tabulate_A)])
        l_cond = make_linear_form(
            mesh=disconnected_mesh,
            ufl_form=L_neumann,
            test_space=V_spline,
            coefficients=C_func,
            integrals=[(exterior_domain, tabulate_L)])

        forbidden_indices = enforce_dirichlet_boundary(hierarchical_space, dofmap, bottom=True)
        x_vec, A = solve_problem(hs=hierarchical_space, a=a_cond, rhs=l_cond, dirichlet_indices=forbidden_indices,
                                dummy_index=np.max(padded_cells_to_dofs), V_spline=V_spline,
                                iterative=False, return_A=True)

        u_dg = map_spline_to_legendre(hierarchical_space, V, C_func, N_max, disconnected_mesh, padded_cells_to_dofs, x_vec)
        u_dg.x.scatter_forward()

        # Define the exact analytic solution using UFL
        my_x = ufl.SpatialCoordinate(disconnected_mesh)
        r_radius = ufl.sqrt(my_x[0]**2 + my_x[1]**2 + 1e-15) # 1e-15 prevents singularity at origin
        theta_angle = ufl.atan2(my_x[1], my_x[0])

        # Restrict theta to [0, 2\pi)
        theta_angle = ufl.conditional(
            ufl.lt(theta_angle, 0.0), 
            theta_angle + 2*np.pi, 
            theta_angle
        )

        # Exact solution u_bar
        u_bar = (r_radius**(2.0/3.0)) * ufl.sin((2.0/3.0) * theta_angle)

        # Compute H1 Semi-norm (Gradient) Error: sqrt( \int |grad(u_bar) - grad(u_dg)|^2 dx )
        # This is the "energy" error and is crucial for elliptic PDEs!
        error_H1_form = dolfinx.fem.form(ufl.inner(ufl.grad(u_bar) - ufl.grad(u_dg), 
                                                ufl.grad(u_bar) - ufl.grad(u_dg)) * dx_custom)
        error_H1_sq = dolfinx.fem.assemble_scalar(error_H1_form)
        h1_error = np.sqrt(disconnected_mesh.comm.allreduce(error_H1_sq, op=MPI.SUM))

        norm_H1_form = dolfinx.fem.form(ufl.inner(ufl.grad(u_bar), ufl.grad(u_bar)) * ufl.dx)
        exact_H1_norm = np.sqrt(disconnected_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(norm_H1_form), op=MPI.SUM))
        
        print(f"Absolute H1 Error: {h1_error:.4e}")
        errors_array[n_iterations, 0] = x_vec.shape[0]
        errors_array[n_iterations, 1] = h1_error
        current_error=h1_error

        # Create a DG0 space (one value per cell)
        V_error = dolfinx.fem.functionspace(disconnected_mesh, ("DG", 0))
        v = ufl.TestFunction(V_error)

        hQ = ufl.CellDiameter(disconnected_mesh)
        volume_form = dolfinx.fem.form(1.0*v*dx_custom)
        cell_volumes = dolfinx.fem.assemble_vector(volume_form).array

        # Define the local L2 error form: integral of (f - u_dg)^2 per cell
        # Note: We multiply by the test function 'v' to pick out each cell's contribution
        local_error_form = dolfinx.fem.form(hQ**2*ufl.inner(ufl.grad(u_bar - u_dg), ufl.grad(u_bar - u_dg)) * v * dx_custom)
        err_cells = dorfler_marking(hierarchical_space=hierarchical_space, theta=0.5, 
                                    local_error_form=local_error_form)
        print("\n")
        n_iterations +=1
        pass
    convergence_plot(errors_array=errors_array, n_iterations=n_iterations, degree=p0)