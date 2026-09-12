from thbsplines.hierarchical_space import HierarchicalSpace
import numpy as np
import dolfinx
import basix.ufl
#import pyvista
from mpi4py import MPI

from dolfinx import default_real_type, default_scalar_type
rtype = default_real_type
dtype = default_scalar_type
import ufl


from thbsplines.refinement import refine
from thbsplines.fenicsx.mesh import build_mesh
from thbsplines.fenicsx.functionspace import build_dofmap, fill_function_space, create_spline_space
from thbsplines.fenicsx.solvers import solve_problem
from thbsplines.fenicsx.adaptivity import dorfler_marking
from thbsplines.fenicsx.kernels import make_linear_kernel, make_bilinear_kernel
from thbsplines.fenicsx.postprocessing import map_spline_to_legendre, convergence_plot
from thbsplines.fenicsx.forms import mark_cells, make_bilinear_form, make_linear_form

if __name__=="__main__":

    degree = 2 #polynomial degree
    n_admissible_mesh = 3 # Admissibility of the refined mesh
    n_initial_refinement = 0
    knots_x = refine(np.array([-1, 0, 1], dtype=float), p=degree, n_times=n_initial_refinement)
    knots_y = refine(np.array([-1, 0, 1], dtype=float), p=degree, n_times=n_initial_refinement)

    # Define the space that holds all information related to THB-Splines
    print("The first iteration is a bit slow, because the numba code needs to be compiled...\n")
    hierarchical_space = HierarchicalSpace(knots=[knots_x, knots_y], degrees=[degree])
    err_cells = {}

    # Hyperparameters for the iterative refinement loop
    n_iterations = 0
    n_max_iterations = 4
    TOL=5e-5
    current_error=1000.
    errors_array = np.zeros((n_max_iterations, 2), dtype=float)

    while ((n_iterations<n_max_iterations) and (current_error>TOL)):
        print(f"Iteration {n_iterations}")
        for level, cells in err_cells.items():
            hierarchical_space.refine(cells, level, refine_neighbours=False, refine_T_neighbours=True, m=n_admissible_mesh)
        pass

        # Define the mesh used by the THB-Splines, the operators used to transition from Legendre finite elements to THB-Splines and 
        # the maximum amount of dofs on a single cell.
        disconnected_mesh, thb_operators, N_max, _ = build_mesh(hs=hierarchical_space)

        # Define the L2 approximation problem
        legendre_elt = basix.ufl.element(
            "DG",
            "quadrilateral",
            degree=degree,
            lagrange_variant=basix.LagrangeVariant.legendre
        )
        V = dolfinx.fem.functionspace(disconnected_mesh, legendre_elt)
        dx_custom = ufl.Measure("dx", domain=disconnected_mesh, metadata={"quadrature_degree": 10})
        u,v = ufl.TrialFunction(V), ufl.TestFunction(V) 
        my_x = ufl.SpatialCoordinate(disconnected_mesh)

        # Custom function to approximate
        f = (ufl.tanh(9.*(my_x[1]-my_x[0]))+1)/9. + (2./3.)*ufl.exp(-(10.*my_x[0]-6.)**2-(10.*my_x[1]+7.)**2)

        # Define the linear and bilinear forms
        a0 = ufl.inner(u, v) * dx_custom
        f0 = ufl.inner(f, v)*dx_custom

        # Create all appropriate custom spaces that allow the usage of THB-Splines
        dofmap, padded_cells_to_dofs = build_dofmap(hierarchical_space=hierarchical_space, mesh=disconnected_mesh, 
                                                    N_max=N_max, morton=True)
        C_func, C_space = fill_function_space(hierachical_space=hierarchical_space, mesh=disconnected_mesh,
                                              N_max=N_max, thb_operators=thb_operators)
        V_spline = create_spline_space(cells_to_dofs=padded_cells_to_dofs, mesh=disconnected_mesh,
                                       N_max=N_max, mult_factor=1)

        local_dofs = (hierarchical_space.degrees[0]+1)**2
        tabulate_A = make_bilinear_kernel(disconnected_mesh, a0, padded_dofs=N_max, local_dofs=local_dofs)
        tabulate_b = make_linear_kernel(disconnected_mesh, f0, padded_dofs=N_max, local_dofs=local_dofs)

        cell_domain = mark_cells(mesh=disconnected_mesh)
        a_cond = make_bilinear_form(mesh=disconnected_mesh,
                                    ufl_form=a0,
                                    trial_space=V_spline, test_space=V_spline,
                                    coefficients=C_func, 
                                    integrals=[(cell_domain, tabulate_A)])
        
        l_cond = make_linear_form(mesh=disconnected_mesh, 
                                ufl_form=f0,
                                test_space=V_spline,
                                coefficients=C_func,
                                integrals=[(cell_domain, tabulate_b)])
        
        x_vec, A = solve_problem(hs=hierarchical_space, a=a_cond, rhs=l_cond, dirichlet_indices=None, 
                            dummy_index=np.max(padded_cells_to_dofs), V_spline=V_spline,
                            iterative=False, return_A=True)

        u_dg = map_spline_to_legendre(hierarchical_space, V, C_func, N_max, disconnected_mesh, padded_cells_to_dofs, x_vec)
        u_dg.x.scatter_forward()


        # Compute exact L2 error using FEniCSx standard UFL
        error_form = dolfinx.fem.form(ufl.inner(f - u_dg, f - u_dg) * dx_custom)
        error_sq = dolfinx.fem.assemble_scalar(error_form)
        exact_l2_error = np.sqrt(disconnected_mesh.comm.allreduce(error_sq, op=MPI.SUM))

        print(f"Exact L2 Error (via DG projection): {exact_l2_error:.4e}")
        current_error=exact_l2_error
        errors_array[n_iterations, 0] = x_vec.shape[0]
        errors_array[n_iterations, 1] = exact_l2_error
        #rel_err = exact_l2_error/f_sq_integral
        #print(f"Relative error = {rel_err:.2e}")

        V_error = dolfinx.fem.functionspace(disconnected_mesh, ("DG", 0))
        v = ufl.TestFunction(V_error)
        local_error_form = dolfinx.fem.form(ufl.inner(f - u_dg, f - u_dg) * v * dx_custom)


        err_cells = dorfler_marking(hierarchical_space=hierarchical_space, theta=0.7, local_error_form=local_error_form)
        print("\n")
        n_iterations +=1
        pass
    convergence_plot(errors_array=errors_array, n_iterations=n_iterations, degree=degree)