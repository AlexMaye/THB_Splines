import numpy as np
import numpy.typing as npt

def refine(knots: npt.ArrayLike, p: int, n_times: int=1)->npt.NDArray:
    """Given `knots`, returns its dyadic refinement with multiplicity `p+1`
    at the extremities."""
    knots: npt.NDArray[np.float_] = np.asarray(knots)
    mult_left: int = np.searchsorted(knots, knots[0], side='right')
    mult_right: int = len(knots) - np.searchsorted(knots, knots[-1], side='left')
    pad_left: int = max(0, p + 1 - mult_left)
    pad_right: int = max(0, p + 1 - mult_right)
    if pad_left > 0 or pad_right > 0:
        knots = np.concatenate((
            np.full(pad_left, knots[0], dtype=knots.dtype),
            knots,
            np.full(pad_right, knots[-1], dtype=knots.dtype)
        ))
    if n_times == 0:
        return knots
    
    # Find indices where the knot value changes
    jump_idx: npt.NDArray[np.int_] = np.where(knots[1:] > knots[:-1])[0]
    left_vals = knots[jump_idx]
    right_vals =knots[jump_idx + 1]
    num_new_points = (1<<n_times)-1
    fractions = np.linspace(0.,1.,num_new_points+2)[1:-1]
    new_points = left_vals[:, None] + (right_vals - left_vals)[:, None] * fractions[None, :]
    new_points = new_points.ravel()
    insert_positions = np.repeat(jump_idx + 1, num_new_points)
    
    return np.insert(knots, insert_positions, new_points)
