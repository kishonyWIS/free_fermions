import numpy as np

# interpolate a hyperplane between points in nD space
def interpolate_hyperplane(anchors: list[np.ndarray], point:np.ndarray):
    # anchors: list of points in nD space
    # point: point in n-1 D space (without the last coordinate)
    # returns: point in nD space
    # the hyperplane is defined by the points in anchors

    # calculate the distance in n-1 D space between the point and the anchors
    distances = np.zeros(len(anchors))
    for i_anchor, anchor in enumerate(anchors):
        distances[i_anchor] = np.linalg.norm(anchor[:-1] - point)
    if np.any(distances == 0):
        # if the point is one of the anchors, return the anchor
        return anchors[np.argmin(distances)]
    # return the weighted average of the anchors with weights given by the inverse of the distances
    return np.average(anchors, axis=0, weights=1/distances)


def find_nearest_index(array, value):
    return (np.abs(array - value)).argmin()