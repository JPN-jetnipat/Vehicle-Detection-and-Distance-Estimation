REAL_WIDTHS = {
    "Car": 1.8,
    "Van": 2.0,
    "Truck": 2.5
}


def estimate_distance(
    focal_length_px,
    real_width_m,
    bbox_width_px
):
    """
    Distance estimation using pinhole camera model.

    D = fW / w

    Parameters
    ----------
    focal_length_px : float
    real_width_m : float
    bbox_width_px : float

    Returns
    -------
    distance_m : float
    """

    if bbox_width_px <= 0:
        return None

    return (
        focal_length_px * real_width_m
    ) / bbox_width_px


def estimate_distance_by_class(
    class_name,
    focal_length_px,
    bbox_width_px
):

    if class_name not in REAL_WIDTHS:
        return None

    real_width = REAL_WIDTHS[class_name]

    return estimate_distance(
        focal_length_px,
        real_width,
        bbox_width_px
    )