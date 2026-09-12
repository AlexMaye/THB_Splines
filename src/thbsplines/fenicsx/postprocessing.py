import numpy as np
import dolfinx.fem as dolfinx_fem
from dolfinx import default_scalar_type
dtype = default_scalar_type

def map_spline_to_legendre(hs, V, C_func, N_max, mesh, cells_to_dofs, u_sol, vector_field=False):
    u_dg = dolfinx_fem.Function(V)
    block_size = V.dofmap.bs
    c_values = C_func.x.array.reshape((-1, N_max, (hs.degrees[0]+1)**hs.dim))
    if vector_field:
        vector_dofs = np.zeros((cells_to_dofs.shape[0], 2*N_max), dtype=np.int32)
        for i in range(2):
            vector_dofs[:, i::2]=2*cells_to_dofs+i
    else:
        vector_dofs = cells_to_dofs

    for local_idx in range(mesh.topology.index_map(mesh.topology.dim).size_local):
        spline_dofs = vector_dofs[local_idx]
        u_spline_local = u_sol[spline_dofs]
        
        G = c_values[local_idx, :, :]
        if vector_field:
            G_vec = np.zeros((2*G.shape[0], 2*G.shape[1]), dtype=dtype)
            for i in range(G.shape[0]):
                for j in range(G.shape[1]):
                    G_vec[2*i, 2*j]     = G[i, j] # X-component mapping
                    G_vec[2*i+1, 2*j+1] = G[i, j] # Y-component mapping
                pass
            pass
        else:
            G_vec = G
        u_dg_local = G_vec.T @ u_spline_local
        
        dg_dofs = V.dofmap.cell_dofs(local_idx)
        if vector_field:
            unrolled_dg_dofs = np.empty(len(dg_dofs) * block_size, dtype=np.int32)
            for i in range(block_size):
                unrolled_dg_dofs[i::block_size] = dg_dofs * block_size + i
        else:
            unrolled_dg_dofs = dg_dofs
        u_dg.x.array[unrolled_dg_dofs] = u_dg_local

    return u_dg

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
def convergence_plot(errors_array, n_iterations, degree):
    if n_iterations<3:
        print(f"Cannot provide convergence plot for less than 3 iterations ({n_iterations} done).")
        return
    
    errors_array = errors_array[:n_iterations, :]
    x = errors_array[:, 0]   # degrees of freedom
    y = errors_array[:, 1]   # error

    fig, ax = plt.subplots()
    ax.loglog(x, y, linewidth=3, marker='x', markersize=13, mew=3, label=f"Degree {degree}")
    ax.grid(True, which="both", axis="both", lw=2, alpha=0.6, color="gray", ls="--")
    ax.set_xlabel("Degrees of freedom")
    ax.set_ylabel("Approximation error")
    ax.set_title(f"Error of L2 approximation problem with degree {degree} THB-Splines")

    x_last = x[-min(5, n_iterations):]
    y_last = y[-min(5, n_iterations):]

    log_x = np.log(x_last)
    log_y = np.log(y_last)

    slope, intercept = np.polyfit(log_x, log_y, 1)

    # convergence rate
    rate = -slope
    x_fit = np.linspace(x_last[0], x_last[-1], 100)
    y_fit = np.exp(intercept) * x_fit**slope

    ax.loglog(
        x_fit,
        y_fit,
        "--",
        linewidth=2.5,
        alpha=0.7,
        label=f"Fit, rate = {rate:.2f}"
    )

    # triangle_ax = inset_axes(
    # ax,
    # width="25%",
    # height="25%",
    # loc="lower left",
    # bbox_to_anchor=(0.67, 0.05, 1, 1),
    # bbox_transform=ax.transAxes,
    # borderpad=0
    # )

    # # Triangle vertices
    # x0, y0 = 0, 0
    # x1, y1 = 1, 0
    # x2, y2 = 0, rate

    # # Draw the three edges explicitly
    # triangle_ax.plot(
    #     [x0, x1], [y0, y1],
    #     color="purple", linewidth=3
    # )

    # triangle_ax.plot(
    #     [x0, x2], [y0, y2],
    #     color="purple", linewidth=3
    # )

    # triangle_ax.plot(
    #     [x1, x2], [y1, y2],
    #     color="purple", linewidth=3
    # )

    # # Labels
    # triangle_ax.text(
    #     0.5, -0.12,
    #     "1",
    #     ha="center",
    #     va="top",
    #     fontsize=16,
    #     fontweight="bold",
    #     transform=triangle_ax.transAxes
    # )

    # triangle_ax.text(
    #     -0.12, rate/2,
    #     f"{rate:.2f}",
    #     ha="right",
    #     va="center",
    #     fontsize=16,
    #     fontweight="bold",
    #     #transform=triangle_ax.transAxes
    # )

    # # Make it look like a clean annotation
    # triangle_ax.set_xlim(-0.25, 1.25)
    # triangle_ax.set_ylim(-0.1, max(rate*1.15, 1))
    # triangle_ax.set_aspect("equal", adjustable="box")
    # triangle_ax.axis("off")

    ax.legend()
    plt.show()
