import numpy as np
import dolfinx.fem as dolfinx_fem
from dolfinx import default_scalar_type
dtype = default_scalar_type

def map_spline_to_legendre_vector_field(hs, V, C_func, N_max, mesh, cells_to_dofs, u_sol):
    u_dg = dolfinx_fem.Function(V)
    block_size = V.dofmap.bs
    c_values = C_func.x.array.reshape((-1, N_max, (hs.degrees[0]+1)**2))
    vector_dofs = np.zeros((cells_to_dofs.shape[0], 2*N_max), dtype=np.int32)
    for i in range(2):
        vector_dofs[:, i::2]=2*cells_to_dofs+i

    for local_idx in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        spline_dofs = vector_dofs[local_idx]
        u_spline_local = u_sol[spline_dofs]
        
        G = c_values[local_idx, :, :]
        G_vec = np.zeros((2*G.shape[0], 2*G.shape[1]), dtype=dtype)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                G_vec[2*i, 2*j]     = G[i, j] # X-component mapping
                G_vec[2*i+1, 2*j+1] = G[i, j] # Y-component mapping
            pass
        pass
        u_dg_local = G_vec.T @ u_spline_local
        
        dg_dofs = V.dofmap.cell_dofs(local_idx)
        unrolled_dg_dofs = np.empty(len(dg_dofs) * block_size, dtype=np.int32)
        for i in range(block_size):
            unrolled_dg_dofs[i::block_size] = dg_dofs * block_size + i
            
        u_dg.x.array[unrolled_dg_dofs] = u_dg_local

    return u_dg