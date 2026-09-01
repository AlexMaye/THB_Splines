import numpy as np
import numba

import cffi

from ffcx.codegeneration.utils import (
    empty_void_pointer,
    numba_ufcx_kernel_signature as ufcx_signature,
)


def make_assembly_kernels(dtype, rtype, kernela0, kernelf0,
                          padded_dofs, local_dofs):

    ffi = cffi.FFI()

    @numba.cfunc(ufcx_signature(dtype, rtype), nopython=True)
    def tabulate_A(
        A_, w_, c_, coords_, entity_local_index,
        permutation=ffi.NULL,
        custom_data=None,
    ):
        A = numba.carray(
            A_, (padded_dofs, padded_dofs), dtype=dtype
        )

        G = numba.carray(
            w_, (padded_dofs, local_dofs), dtype=dtype
        )

        A0 = np.zeros(
            (local_dofs, local_dofs), dtype=dtype
        )

        kernela0(
            ffi.from_buffer(A0),
            w_,
            c_,
            coords_,
            entity_local_index,
            permutation,
            empty_void_pointer(),
        )

        A[:, :] = G @ A0 @ G.T

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

        kernelf0(
            ffi.from_buffer(b0),
            w_,
            c_,
            coords_,
            entity_local_index,
            permutation,
            empty_void_pointer(),
        )

        b[:] = G @ b0

    return tabulate_A, tabulate_b