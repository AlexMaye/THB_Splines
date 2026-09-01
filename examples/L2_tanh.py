from thbsplines.hierarchical_space import HierarchicalSpace
import numpy as np
import dolfinx
from mpi4py import MPI
import basix.ufl

import numba.core.typing.cffi_utils as cffi_support
from dolfinx.jit import ffcx_jit
from dolfinx import default_real_type, default_scalar_type, geometry
rtype = default_real_type
dtype = default_scalar_type
import ufl

from THBSplines.src.fenicsx.kernels import make_assembly_kernels

from solve_utils import (refine, build_mesh_2d, build_dofmap, fill_function_space, create_spline_space,\
                          solve_problem, dorfler_marking)

if __name__=="__main__":
    degree = 2
    n_admissible_mesh = 3
    n_initial_refinement = 0
    knots_x = refine(np.array([-1, 0, 1], dtype=float), p=degree, n_times=n_initial_refinement)
    knots_y = refine(np.array([-1, 0, 1], dtype=float), p=degree, n_times=n_initial_refinement)
    hierarchical_space = HierarchicalSpace(knots=[knots_x, knots_y], degrees=[degree])
    err_cells = {}

    n_iterations = 0
    n_max_iterations = 5
    TOL=1e-5
    current_error=1000.

    while ((n_iterations<n_max_iterations) or (current_error>TOL)):
        for level, cells in err_cells.items():
            hierarchical_space.refine(cells, level, refine_neighbours=False, refine_T_neighbours=True, m=n_admissible_mesh)
        pass

        disconnected_mesh, thb_operators, N_max, _ = build_mesh_2d(hs=hierarchical_space)

        # Define the L2 approximation problem
        legendre_elt = basix.ufl.element(
            "DG",
            "quadrilateral",
            degree=degree,
            lagrange_variant=basix.LagrangeVariant.legendre
        )
        V = dolfinx.fem.functionspace(disconnected_mesh, legendre_elt)
        print(f"Number of degrees of freedom: {V.dofmap.index_map.size_global}")
        dx_custom = ufl.Measure("dx", domain=disconnected_mesh, metadata={"quadrature_degree": 12})
        u,v = ufl.TrialFunction(V), ufl.TestFunction(V) 
        my_x = ufl.SpatialCoordinate(disconnected_mesh)
        # Custom function to approximate
        f = (ufl.tanh(9.*(my_x[1]-my_x[0]))+1)/9. + (2./3.)*ufl.exp(-(10.*my_x[0]-6.)**2-(10.*my_x[1]+7.)**2)
        a0 = ufl.inner(u, v) * dx_custom
        f0 = ufl.inner(f, v)*dx_custom

        f_square_integral = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(f,f)*dx_custom))
        f_sq_integral = np.sqrt(disconnected_mesh.comm.allreduce(f_square_integral, op=MPI.SUM))

        msh = disconnected_mesh
        # Intercept the kernel formation for a custom definition later on
        ufcxa0, _, _ = ffcx_jit(msh.comm, a0, form_compiler_options={"scalar_type": dtype})  # type: ignore
        kernela0 = getattr(ufcxa0.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")  # type: ignore

        ufcxf0, _, _ = ffcx_jit(msh.comm, f0, form_compiler_options={"scalar_type": dtype})  # type: ignore
        kernelf0 = getattr(ufcxf0.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")  # type: ignore

        # Create all appropriate custom spaces that allow the usage of THB-Splines
        dofmap, padded_cells_to_dofs = build_dofmap(hierarchical_space=hs, mesh=disconnected_mesh, 
                                            N_max=N_max, morton=True)
        C_func, C_space = fill_function_space(hierachical_space=hierarchical_space, mesh=disconnected_mesh,
                                            N_max=N_max, thb_operators=thb_operators)
        V_spline = create_spline_space(cells_to_dofs=padded_cells_to_dofs, mesh=disconnected_mesh,
                                    N_max=N_max, mult_factor=1) #mult factor is dimensionality of problem (scalar valued, vector valued, etc.)


        local_dofs = (hierarchical_space.degrees[0]+1)**2
        tabulate_A, tabulate_b = make_assembly_kernels(
        dtype=dtype,
        rtype=rtype,
        kernela0=kernela0,
        kernelf0=kernelf0,
        padded_dofs=N_max,
        local_dofs=local_dofs
        )

        formtype = dolfinx.fem.form_cpp_class(dtype)  # type: ignore
        # Gets the number of cells for which each individual core is responsible for.
        cells = np.arange(msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32)

        # The 4th argument np.array([...], dtype=np.int8) is the 
        # active coefficients array. It lists which indices from the 
        # coefficients list should be packed into the w_ pointer that the kernel receives.
        integrals = {dolfinx.fem.IntegralType.cell: [
            (0, tabulate_A.address, cells, np.array([0], dtype=np.int8))]}

        a_cond = dolfinx.fem.Form( # We are not forming anything yet, this is a recipe
            formtype( # selectes the correct floating-point precision
                spaces=[V_spline._cpp_object, 
                        V_spline._cpp_object]
                    , # trial and test spaces, determines the size of A_
                integrals=integrals, #this is a dictionary, and we are passing the adress of tabulate_A() here
                coefficients=[C_func._cpp_object
                            ], # weights w_, holds C@T
                    constants=[],
                    need_permutation_data=False,
                    entity_maps=[], 
                    mesh=msh._cpp_object)
        )

        integrals_rhs = {dolfinx.fem.IntegralType.cell: [(0, tabulate_b.address, cells, np.array([0], dtype=np.int8))]}
        l_cond = dolfinx.fem.Form(
            formtype(
                spaces=[V_spline._cpp_object], # test space, determines the size of b_
                integrals=integrals_rhs, #give the adress of tabulate_b
                coefficients=[C_func._cpp_object], # holds the evaluations of f at the correct points, as well as C@T
                constants=[], need_permutation_data=False, entity_maps=[], mesh=msh._cpp_object
            )
        )
        x_vec, A = solve_problem(hs=hierarchical_space, a=a_cond, rhs=l_cond, dirichlet_indices=None, 
                      dummy_index=np.max(padded_cells_to_dofs), V_spline=V_spline,
                      iterative=False, return_A=True)


        u_dg = dolfinx.fem.Function(V)
        c_values = C_func.x.array.reshape((-1, N_max, local_dofs))
        num_cells_local = disconnected_mesh.topology.index_map(disconnected_mesh.topology.dim).size_local

        # Map the global B-spline coefficients back to local Legendre coefficients
        for local_idx in range(num_cells_local):
            # Get global B-spline dof indices for this cell
            spline_dofs = padded_cells_to_dofs[local_idx]
            
            # Extract the B-spline coefficients for this cell
            u_spline_local = x_vec[spline_dofs]
            
            # Get the local transformation matrix G for this cell
            G = c_values[local_idx, :, :]
            
            # Transform B-spline to DG: mathematically, the kernel does A = G @ A0 @ G.T
            # This implies the coefficient mapping is u_dg = G.T @ u_spline
            u_dg_local = G.T @ u_spline_local
            
            # Assign to the standard DG function
            dg_dofs = V.dofmap.cell_dofs(local_idx)
            u_dg.x.array[dg_dofs] = u_dg_local

        u_dg.x.scatter_forward()


        # Compute exact L2 error using FEniCSx standard UFL
        error_form = dolfinx.fem.form(ufl.inner(f - u_dg, f - u_dg) * dx_custom)
        error_sq = dolfinx.fem.assemble_scalar(error_form)
        exact_l2_error = np.sqrt(disconnected_mesh.comm.allreduce(error_sq, op=MPI.SUM))

        print(f"Exact L2 Error (via DG projection): {exact_l2_error:.2e}")
        rel_err = exact_l2_error/f_sq_integral
        print(f"Relative error = {rel_err:.2e}")