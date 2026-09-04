import numpy as np
import numba
from dolfinx.jit import ffcx_jit
import cffi
from dolfinx import default_real_type, default_scalar_type
rtype = default_real_type
dtype = default_scalar_type
from ffcx.codegeneration.utils import (
    empty_void_pointer,
    numba_ufcx_kernel_signature as ufcx_signature,
)

def _make_kernel(mesh, ufl_form):
    ufcx, _, _ = ffcx_jit(mesh.comm, ufl_form, form_compiler_options={"scalar_type": dtype})  # type: ignore
    return getattr(ufcx.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}")  # type: ignore

def make_bilinear_kernel(mesh, ufl_form, padded_dofs, local_dofs):
    ffi = cffi.FFI()
    ufcx_kernel = _make_kernel(mesh, ufl_form)
    
    @numba.cfunc(ufcx_signature(dtype, rtype), nopython=True)
    def tabulate_A(
        A_, w_, c_, coords_, entity_local_index,
        permutation=ffi.NULL,
        custom_data=None,
    ):
        A = numba.carray(A_, (padded_dofs, padded_dofs), dtype=dtype)

        G = numba.carray(w_, (padded_dofs, local_dofs), dtype=dtype)

        A0 = np.zeros((local_dofs, local_dofs), dtype=dtype)

        ufcx_kernel(
            ffi.from_buffer(A0),
            w_,
            c_,
            coords_,
            entity_local_index,
            permutation,
            empty_void_pointer(),
        )

        A[:, :] = G @ A0 @ G.T
    return tabulate_A

def make_linear_kernel(mesh, ufl_form, padded_dofs, local_dofs):
    ufcx_kernel = _make_kernel(mesh, ufl_form)
    ffi = cffi.FFI()

    @numba.cfunc(ufcx_signature(dtype, rtype), nopython=True)
    def tabulate_b(
        b_, w_, c_, coords_, entity_local_index,
        permutation=ffi.NULL,
        custom_data=None,
    ):
        b = numba.carray(
            b_, (padded_dofs,),
            dtype=dtype
        )

        G = numba.carray(
            w_, (padded_dofs, local_dofs),
            dtype=dtype
        )

        b0 = np.zeros(
            (local_dofs,),
            dtype=dtype
        )

        ufcx_kernel(
            ffi.from_buffer(b0),
            w_,
            c_,
            coords_,
            entity_local_index,
            permutation,
            empty_void_pointer(),
        )

        b[:] = G @ b0
    return tabulate_b


@numba.njit(inline="always")
def _lift_vector_operator(G):
    padded_dofs, local_dofs = G.shape
    G_vec = np.zeros((2*padded_dofs, 2*local_dofs), dtype=G.dtype)
    for i in range(padded_dofs):
        for j in range(local_dofs):
            G_vec[2*i, 2*j] = G[i,j]
            G_vec[i+i+1, j+j+1] = G[i,j]
        pass
    pass

    return G_vec

def make_vector_bilinear_kernel(mesh, ufl_form, padded_dofs, local_dofs):
    ufcx_kernel = _make_kernel(mesh, ufl_form)
    ffi = cffi.FFI()
    local_dofs_vec = 2*local_dofs
    padded_dofs_vec = 2*padded_dofs

    @numba.cfunc(ufcx_signature(dtype, rtype), nopython=True)
    def tabulate_A(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL, custom_data=None):
        A = numba.carray(A_, (padded_dofs_vec, padded_dofs_vec), dtype=dtype)
        G = numba.carray(w_, (padded_dofs, local_dofs), dtype=dtype)
        A0 = np.zeros((local_dofs_vec, local_dofs_vec), dtype=dtype)
        ufcx_kernel(
            ffi.from_buffer(A0),
            w_,
            c_,
            coords_,
            entity_local_index,
            permutation,
            empty_void_pointer(),
        )

        G_vec = _lift_vector_operator(G)
        A[:, :] = G_vec@A0@(G_vec.T)

    return tabulate_A

def make_vector_linear_kernel(mesh, ufl_form, padded_dofs, local_dofs):
    ufcx_kernel = _make_kernel(mesh, ufl_form)
    ffi = cffi.FFI()
    local_dofs_vec = 2*local_dofs
    padded_dofs_vec = 2*padded_dofs

    @numba.cfunc(ufcx_signature(dtype, rtype), nopython=True)
    def tabulate_b(b_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL, custom_data=None):
        b = numba.carray(b_, (padded_dofs_vec,), dtype=dtype)
        G = numba.carray(w_, (padded_dofs, local_dofs), dtype=dtype)
    
        b0 = np.zeros((local_dofs_vec,), dtype=dtype)
        ufcx_kernel(ffi.from_buffer(b0), 
                        w_, 
                        c_, 
                        coords_, 
                        entity_local_index, 
                        permutation, 
                        empty_void_pointer())

        G_vec = _lift_vector_operator(G)
                
        b[:] = G_vec @ b0

    return tabulate_b