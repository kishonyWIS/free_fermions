import numpy as np


def essential_singularity_at_zero(t, tail_length=1):
    if isinstance(t, np.ndarray):
        t[t==0] = -1e-10
        return np.exp(-tail_length/t) * (t>0)
    else:
        if t <= 0:
            return 0.
        else:
            return np.exp(-tail_length/t)


def smooth_transition_from_0_0_to_1_1(t, tail_length_start=1, tail_length_end=1):
    return essential_singularity_at_zero(t, tail_length_start) / (essential_singularity_at_zero(t, tail_length_start) +
                                                                  essential_singularity_at_zero(1-t, tail_length_end))


def smooth_transition(t, t0, t1, y0, y1, tail_length_ratio_start=1, tail_length_ratio_end=1):
    return y0 + (y1-y0)*smooth_transition_from_0_0_to_1_1((t-t0)/(t1-t0),
                                                          tail_length_start=tail_length_ratio_start,
                                                          tail_length_end=tail_length_ratio_end)


def smooth_bump(t, t0, t1, t2, t3, y1, tail_ratio_0=1, tail_ratio_1=1, tail_ratio_2=1, tail_ratio_3=1):
    return smooth_transition(t, t0, t1, 0, y1, tail_length_ratio_start=tail_ratio_0, tail_length_ratio_end=tail_ratio_1) * \
           smooth_transition(t, t2, t3, 1, 0, tail_length_ratio_start=tail_ratio_2, tail_length_ratio_end=tail_ratio_3)


def get_g(t, g0, T, t1):
    return smooth_bump(t, t0=0, t1=t1, t2=T-t1, t3=T, y1=g0, tail_ratio_0=1, tail_ratio_1=1, tail_ratio_2=1, tail_ratio_3=1)


def get_B(t, B0, B1, T):
    return smooth_transition(t, t0=0, t1=T, y0=B0, y1=B1, tail_length_ratio_start=1, tail_length_ratio_end=1)