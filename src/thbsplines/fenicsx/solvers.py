from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
import dolfinx.fem as dolfinx_fem
import numpy as np

def solve_problem(hs, a, rhs, dirichlet_indices, dummy_index, V_spline, iterative=False, return_A=False):
    A = assemble_matrix(a, bcs=[])
    A.assemble()
    one_active=False
    two_active= False
    for level in range(hs.nlevels):
        if level in hs.truly_active and hs.truly_active[level].size>0:
            if one_active:
                two_active=True
            one_active=True
    A_mat = A
    if two_active:
        dummy_dof_index = np.max(dummy_index)
        A_mat.setValue(dummy_dof_index, dummy_dof_index, 1., addv=PETSc.InsertMode.INSERT_VALUES)
        A_mat.assemble()
        A_mat.assemblyBegin()
        A_mat.assemblyEnd()

    b = assemble_vector(rhs)
    if two_active:
        b[dummy_dof_index] = 0.0
        b.assemblyBegin()
        b.assemblyEnd()

    if dirichlet_indices is not None:
        A_mat.zeroRowsColumns(dirichlet_indices, diag=1.0, x=None, b=b)
        b.array_w[dirichlet_indices]=0.
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    ksp = PETSc.KSP().create(A_mat.comm)
    ksp.setOperators(A_mat)
    if iterative:
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        ksp.setTolerances(rtol=1e-12, atol=1e-12, max_it=2000)
    else:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.LU)
        ksp.getPC().setFactorSolverType("mumps")
    u_sol = dolfinx_fem.Function(V_spline)
    ksp.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    u_vec=u_sol.x.array
    print(f"Solve complete. Reason: {ksp.getConvergedReason()}, Iterations: {ksp.getIterationNumber()}")
    
    ksp.destroy()
    b.destroy()

    if not return_A:
        A.destroy()
        return u_vec
    else:
        return u_vec, A

def solve_problem_vector_field(hs, a, lhs, dirichlet_indices, dummy_index, V_spline, iterative=False):
    A = assemble_matrix(a, bcs=[])
    A.assemble()
    one_active=False
    two_active= False
    for level in range(hs.nlevels):
        if level in hs.truly_active and hs.truly_active[level].size>0:
            if one_active:
                two_active=True
            one_active=True

    A_mat = A
    if two_active:
        dummy_dof_index = np.max(dummy_index)
        dummy_x = 2 * dummy_dof_index
        dummy_y = 2 * dummy_dof_index + 1
        A_mat.setValue(dummy_x, dummy_x, 1., addv=PETSc.InsertMode.INSERT_VALUES)
        A_mat.setValue(dummy_y, dummy_y, 1., addv=PETSc.InsertMode.INSERT_VALUES)
        A_mat.assemble()
        A_mat.assemblyBegin()
        A_mat.assemblyEnd()

    b = assemble_vector(lhs)
    if two_active:
        b[dummy_x] = 0.0
        b[dummy_y] = 0.0
        b.assemblyBegin()
        b.assemblyEnd()

    if dirichlet_indices is not None:
        A_mat.zeroRowsColumns(dirichlet_indices, diag=1.0, x=None, b=b)
        b.array_w[dirichlet_indices]=0.
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    A_mat.setOption(PETSc.Mat.Option.SPD, True)
    ksp = PETSc.KSP().create(A_mat.comm)
    ksp.setOperators(A_mat)
    if iterative:
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.JACOBI)
    else:
        ksp.setType(PETSc.KSP.Type.PREONLY)
        ksp.getPC().setType(PETSc.PC.Type.CHOLESKY)
        ksp.getPC().setFactorSolverType("mumps")
    u_sol = dolfinx_fem.Function(V_spline)
    ksp.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    u_vec=u_sol.x.array

    ksp.destroy()
    A.destroy()
    b.destroy()

    return u_vec