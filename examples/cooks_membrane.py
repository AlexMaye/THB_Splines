from thbsplines.hierarchical_space import HierarchicalSpace
import numpy as np
import dolfinx
import basix.ufl

from dolfinx import default_real_type, default_scalar_type
rtype = default_real_type
dtype = default_scalar_type
import ufl
from copy import deepcopy
from scipy.integrate import quad

import matplotlib.pyplot as plt

from thbsplines.refinement import refine
from thbsplines.fenicsx.mesh import build_mesh, FastMidpointMapper
from thbsplines.fenicsx.functionspace import build_dofmap, fill_function_space, create_spline_space
from thbsplines.fenicsx.solvers import solve_problem_vector_field, enforce_dirichlet_boundary
from thbsplines.fenicsx.postprocessing import map_spline_to_legendre
from thbsplines.fenicsx.kernels import make_vector_bilinear_kernel, make_vector_linear_kernel
from thbsplines.fenicsx.adaptivity import dorfler_marking
from thbsplines.fenicsx.forms import mark_exterior_boundary_entities, mark_cells, exterior_facets, make_bilinear_form, make_linear_form

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# This example was inspired by the results on the following webpages
# https://www.solids4foam.com/tutorials/more-tutorials/solid-mechanics/linearElasticity/cooksMembrane.html
# https://cofea.readthedocs.io/en/latest/benchmarks/002-cook-membrane/results.html
# They were last consulted on September 18th, 2026
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

def mollifier(rho):
            return np.exp(-1./(1.-rho**2))*rho

def my_dict_update(a,b):
    for level in a.keys():
        if level in b:
            b[level] = np.concatenate((a[level], b[level]))
        else:
            b[level] = a[level]
    return b

def mapping_to_trapezoid(uv_points):
    original_shape = uv_points.shape
    uv_flat = uv_points.reshape(-1, 2)
    xy_flat = np.zeros_like(uv_flat)

    def bilinear(u_loc, v_loc, C00, C10, C01, C11):
        """Standard isoparametric Q1 interpolation"""
        return ((1-u_loc)*(1-v_loc)*C00 + 
                u_loc*(1-v_loc)*C10 + 
                (1-u_loc)*v_loc*C01 + 
                u_loc*v_loc*C11)
    
    P00 = np.array([0.0, 0.0])
    P10 = np.array([4.8, 4.4])
    P01 = np.array([0.0, 4.4])
    P11 = np.array([4.8, 6.0])

    xy_flat[:, 0] = bilinear(u_loc=uv_flat[:, 0], v_loc=uv_flat[:, 1],
                             C00=P00[0], C10=P10[0], C01=P01[0], C11=P11[0])
    xy_flat[:, 1] = bilinear(u_loc=uv_flat[:, 0], v_loc=uv_flat[:, 1],
                             C00=P00[1], C10=P10[1], C01=P01[1], C11=P11[1])
    
    
    return xy_flat.reshape(original_shape)

if __name__=="__main__":

    n_refinements = 2
    p0 = 2
    knotsx = np.array([0,0,0.5,1,1], dtype=np.float64)
    knotsx = refine(knotsx, p=p0, n_times=n_refinements)
    knotsy = np.array([0,0,0.5,1,1], dtype=np.float64)
    knotsy = refine(knotsy, p0, n_times=n_refinements)
    err_cells = {}
    T_err_cells = {}
    print("The first iteration is a bit slow, because the numba code needs to be compiled...\n")

    hs = HierarchicalSpace(knots=[knotsx, knotsy], degrees=[p0])
    p0_dual = p0+1
    m=3
    knotsx_dual = refine(knotsx, p0_dual, 0)
    knotsy_dual = refine(knotsy, p0_dual, 0)
    hs_dual = HierarchicalSpace([knotsx_dual, knotsy_dual], [p0_dual])

    # Hyperparameters for the iterative refinement loop
    n_iterations = 0
    n_max_iterations = 12
    conv_array = np.zeros((n_max_iterations, 2), dtype=float)

    while (n_iterations<n_max_iterations):
        print(f"Iteration {n_iterations}")
        T_err_cells = deepcopy(err_cells)
        for level, cells in err_cells.items():
            Temp_cells = hs_dual.refine(cells, level, refine_T_neighbours=True, m=m)
            T_err_cells.update(my_dict_update(Temp_cells, T_err_cells))
        for level, cells in T_err_cells.items():
            _ = hs.refine(cells, level, refine_T_neighbours=True, m=m)

        disconnected_mesh, thb_operators, N_max, trapez_midpoints = build_mesh(hs=hs, mapping=mapping_to_trapezoid)

        tdim = disconnected_mesh.topology.dim
        fdim = tdim - 1
        disconnected_mesh.topology.create_connectivity(fdim, tdim)

        coords = disconnected_mesh.geometry.x
        # midpoints = np.mean(coords.reshape(-1, 4, 2), axis=1)
        x_min = np.min(coords[:, 0])  # Find the leftmost x-value dynamically
        x_max = np.max(coords[:, 0])
        y_min = np.min(coords[:, 1])
        y_max = np.max(coords[:, 1])
        y_min2 = np.min(coords[:, 1][np.isclose(coords[:, 0], x_max)])
        y_max2 = np.max(coords[:, 1][np.isclose(coords[:, 0], x_min)])

        T00 = np.array([x_min, y_min], dtype=float)
        T01 = np.array([x_max, y_min2], dtype=float)
        T10 = np.array([x_min, y_max2], dtype=float)
        T11 = np.array([x_max, y_max], dtype=float)

        def left_boundary(x):
            return np.isclose(x[0], x_min)
        def right_boundary(x):
            return np.isclose(x[0], x_max)
        def bottom_boundary(x):
            return np.isclose(x[1], x[0]*T01[1]/T01[0])
        def top_boundary(x):
            return np.isclose(x[1], T10[1]+x[0]*(T11[1]-T10[1])/T11[0])
        def bottom_right_top_boundary(x):
            return right_boundary(x)|bottom_boundary(x)|top_boundary(x)
        def not_right(x):
            return ~right_boundary(x)

        left_facets = dolfinx.mesh.locate_entities_boundary(
            disconnected_mesh, 
            fdim, 
            left_boundary
        )

        right_facets = dolfinx.mesh.locate_entities_boundary(
            disconnected_mesh, 
            fdim, 
            right_boundary
        )

        #sorted_facets = np.sort(right_facets)

        # Assign a marker ID (e.g., 1) to these facets
        facet_values = np.full_like(right_facets, 1, dtype=np.int32)

        # Create the MeshTags object
        facet_tags = dolfinx.mesh.meshtags(
            disconnected_mesh, fdim, right_facets, facet_values
        )

        dim = disconnected_mesh.topology.dim
        legendre_elt = basix.ufl.element(
            "DG",
            "quadrilateral",
            degree=p0,
            shape=(dim,),
            lagrange_variant=basix.LagrangeVariant.legendre
        )
        V = dolfinx.fem.functionspace(disconnected_mesh, legendre_elt)
        # print(f"Number of degrees of freedom: {V.dofmap.index_map.size_global}")

        #boundary_facets = dolfinx.mesh.locate_entities_boundary(disconnected_mesh, facet_dim, Gamma_N)
        custom_metadata = {"quadrature_degree": 6}
        ds_custom = ufl.Measure("ds", domain=disconnected_mesh, subdomain_data=facet_tags, metadata=custom_metadata)
        dx_custom = ufl.Measure("dx", domain=disconnected_mesh, metadata=custom_metadata)

        E=70.0
        nu = 1./3.
        lmbda = E*nu/(1.+nu)/(1.-2.*nu)
        #mu = E / 2. / (1. + nu)
        mu = E/2./(1.+nu)
        lmbda_c = dolfinx.fem.Constant(disconnected_mesh, lmbda)
        mu_c = dolfinx.fem.Constant(disconnected_mesh, mu)

        def epsilon(v):
            return ufl.sym(ufl.grad(v))

        def sigma(v):
            return lmbda_c * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2. * mu_c * epsilon(v)


        u,v = ufl.TrialFunction(V), ufl.TestFunction(V) 
        T = dolfinx.fem.Constant(
            disconnected_mesh, 
            dolfinx.default_scalar_type((0., 6.25))
        )
        a = ufl.inner(sigma(u),epsilon(v))*dx_custom
        L_facet = ufl.inner(T, v)*ds_custom(1)

        dofmap, padded_cells_to_dofs = build_dofmap(hierarchical_space=hs, mesh=disconnected_mesh, N_max=N_max, morton=False)
        custom_mapping = FastMidpointMapper(hs, trapez_midpoints)
        C_func, C_space = fill_function_space(hierachical_space=hs, mesh=disconnected_mesh, N_max = N_max, 
                                            thb_operators=thb_operators, mapping_function=custom_mapping)
        V_spline = create_spline_space(cells_to_dofs=padded_cells_to_dofs, mesh=disconnected_mesh,
                                    N_max=N_max, mult_factor=2)

        local_dofs = (hs.degrees[0]+1)**2
        tabulate_A = make_vector_bilinear_kernel(disconnected_mesh, a, padded_dofs=N_max, local_dofs=local_dofs)
        tabulate_L_facet = make_vector_linear_kernel(disconnected_mesh, L_facet, padded_dofs=N_max, local_dofs=local_dofs)
        boundary_entities = mark_exterior_boundary_entities(disconnected_mesh, right_boundary)
        cell_domain = mark_cells(disconnected_mesh)
        traction_domain = exterior_facets(boundary_entities)
        a_cond = make_bilinear_form(mesh=disconnected_mesh, ufl_form=a,
                                    trial_space=V_spline, test_space=V_spline,
                                    coefficients=C_func, integrals=[(cell_domain, tabulate_A)])
        l_cond = make_linear_form(
            mesh=disconnected_mesh,
            ufl_form=L_facet,
            test_space=V_spline,
            coefficients=C_func,
            integrals=[(traction_domain, tabulate_L_facet)]
        )

        forbidden_indices = enforce_dirichlet_boundary(hs, dofmap, left=True)
        if forbidden_indices is not None and len(forbidden_indices) > 0:
            forbidden_indices_vec = np.empty(2 * len(forbidden_indices), dtype=np.int32)
            forbidden_indices_vec[0::2] = 2 * forbidden_indices      # X DOFs
            forbidden_indices_vec[1::2] = 2 * forbidden_indices + 1  # Y DOFs
            forbidden_indices = forbidden_indices_vec

        x_vec = solve_problem_vector_field(hs=hs, a = a_cond, lhs=l_cond, dirichlet_indices=forbidden_indices, 
                                        dummy_index=np.max(padded_cells_to_dofs), V_spline=V_spline)
        u_dg = map_spline_to_legendre(hs=hs, V=V, C_func=C_func, N_max=N_max, mesh=disconnected_mesh, 
                                                cells_to_dofs=padded_cells_to_dofs, u_sol=x_vec, vector_field=True)


        vertical_displacement = np.max(u_dg.x.array.reshape(-1, 2)[:,1])*10.
        print(f"Top right corner vertical displacement: {vertical_displacement:.5f} mm")
        conv_array[n_iterations, 0] = int(x_vec.shape[0]/2)
        conv_array[n_iterations, 1] = vertical_displacement

        legendre_elt_dual = basix.ufl.element(
            "DG",
            "quadrilateral",
            degree=p0_dual,
            shape=(dim,),
            lagrange_variant=basix.LagrangeVariant.legendre
        )
        V_dual = dolfinx.fem.functionspace(disconnected_mesh, legendre_elt_dual)

        z,v_dual = ufl.TrialFunction(V_dual), ufl.TestFunction(V_dual)

        x = ufl.SpatialCoordinate(disconnected_mesh)
        # constant_zero = dolfinx.fem.Constant(disconnected_mesh, 0.)
        s = 0.075
        
        integral, _ = quad(mollifier, 0., 1.)
        integral_c = dolfinx.fem.Constant(disconnected_mesh, 2.*np.pi*s**2*integral)
        r2 = (x[0]-x_max)**2+(x[1]-y_max)**2
        mollifier_inside = ufl.exp(-1./(1.-(r2/(s**2) ) ))
        rhs_dual = ufl.as_vector((0., ufl.conditional(
                                condition=ufl.lt(left=r2, right=s**2), 
                                true_value=mollifier_inside / integral_c, 
                                false_value=0.)
        ))
        a_dual = ufl.inner(sigma(v_dual), epsilon(z)) * dx_custom
        L_cell_dual = ufl.inner(rhs_dual, v_dual) * dx_custom 

        _, thb_operators_dual, N_max_dual, trapez_midpoints_dual = build_mesh(hs=hs_dual, mapping=mapping_to_trapezoid)

        custom_mapping_dual = FastMidpointMapper(hs_dual, trapez_midpoints_dual)

        dof_map_dual, padded_cells_to_dofs_dual = build_dofmap(hierarchical_space=hs_dual, mesh=disconnected_mesh, N_max=N_max_dual, morton=False)

        C_func_dual, C_space_dual = fill_function_space(hierachical_space=hs_dual, mesh=disconnected_mesh, N_max=N_max_dual, 
                                                        thb_operators=thb_operators_dual, mapping_function=custom_mapping_dual)

        V_spline_dual = create_spline_space(cells_to_dofs=padded_cells_to_dofs_dual, mesh=disconnected_mesh,
                                    N_max=N_max_dual, mult_factor=2)

        local_dofs_dual = (p0_dual+1)**2
        tabulate_A_dual = make_vector_bilinear_kernel(disconnected_mesh, a_dual, padded_dofs=N_max_dual, local_dofs=local_dofs_dual)
        tabulate_L_cell_dual = make_vector_linear_kernel(disconnected_mesh, L_cell_dual, padded_dofs=N_max_dual, local_dofs=local_dofs_dual)

        a_cond_dual = make_bilinear_form(mesh=disconnected_mesh, ufl_form=a_dual,
                                        trial_space=V_spline_dual,
                                        test_space=V_spline_dual, coefficients=(C_func_dual,),
                                        integrals=[(cell_domain, tabulate_A_dual)])
        l_cond_dual = make_linear_form(mesh=disconnected_mesh, ufl_form=L_cell_dual,
                                        test_space=V_spline_dual,
                                    coefficients=C_func_dual, 
                                    integrals=[(cell_domain, tabulate_L_cell_dual)])

        forbidden_indices_dual = enforce_dirichlet_boundary(hs_dual, dof_map_dual, left=True)
        if forbidden_indices_dual is not None and len(forbidden_indices_dual) > 0:
            forbidden_indices_vec_dual = np.empty(2 * len(forbidden_indices_dual), dtype=np.int32)
            forbidden_indices_vec_dual[0::2] = 2 * forbidden_indices_dual      # X DOFs
            forbidden_indices_vec_dual[1::2] = 2 * forbidden_indices_dual + 1  # Y DOFs
            forbidden_indices_dual = forbidden_indices_vec_dual

        z_vec = solve_problem_vector_field(hs=hs_dual, a = a_cond_dual, lhs=l_cond_dual, dirichlet_indices=forbidden_indices_dual, 
                                   dummy_index=np.max(padded_cells_to_dofs_dual), V_spline=V_spline_dual)
        z_dg = map_spline_to_legendre(hs=hs_dual, V=V_dual, C_func=C_func_dual,
                                                N_max=N_max_dual, mesh=disconnected_mesh, 
                                                cells_to_dofs=padded_cells_to_dofs_dual, u_sol=z_vec, vector_field=True)

        DG0 = dolfinx.fem.functionspace(disconnected_mesh, ("DG", 0))
        v_dg = ufl.TestFunction(DG0)

        i_h_z_h = dolfinx.fem.Function(V)
        i_h_z_h.interpolate(z_dg)
        dual_weight = z_dg - i_h_z_h

        R_expr = ufl.div(sigma(u_dg))
        cell_form = ufl.inner(R_expr, dual_weight)*v_dg*dx_custom

        n_normal = ufl.FacetNormal(disconnected_mesh)
        flux_jump = ufl.jump(sigma(u_dg), n_normal)
        r_val = 0.5 * flux_jump
        custom_dS = ufl.Measure("dS", domain=disconnected_mesh, metadata=custom_metadata)



        facet_form_dS = (
            ufl.inner(r_val, dual_weight('+')) * v_dg('+')
            + ufl.inner(r_val, dual_weight('-')) * v_dg('-')
        ) * custom_dS 

        traction_computed = ufl.dot(sigma(u_dg), n_normal)
        facet_form_ds = ufl.inner(dual_weight,T-traction_computed)*v_dg*ds_custom(1)

        eta_form = cell_form+ facet_form_dS + facet_form_ds
        eta_vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(eta_form))
        err_cells = dorfler_marking(hs, 0.6, dolfinx.fem.form(eta_form))
        print("\n")
        n_iterations +=1

        pass

    fig, ax = plt.subplots()
    plot_dofs = conv_array[:, 0]
    plot_displacement = conv_array[:, 1]
    ax.plot(plot_dofs, plot_displacement, linewidth=3, marker='x', markersize=13, mew=3)
    ax.hlines(32., xmin=np.min(plot_dofs), xmax=np.max(plot_dofs), linestyles="dashed", label="target displacement",
              color="black", lw=2.5)
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Top right corner vertical displacement (millimetres)")
    ax.set_title(f"Cook's membrane displacement plot for degree {p0} THB-Splines")
    ax.grid(True, which="both", axis="both", lw=2, alpha=0.4, color="gray", ls="--")
    ax.legend()
    plt.show()