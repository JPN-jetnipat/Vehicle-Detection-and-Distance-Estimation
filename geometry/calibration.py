# TODO: Parse KITTI calibration files

import numpy as np


def read_kitti_calibration(calib_file):
    """
    Read a KITTI calibration file.

    Returns:
        dict containing:
        - P0, P1, P2, P3
        - R0_rect
        - Tr_velo_to_cam
        - Tr_imu_to_velo
    """

    data = {}

    with open(calib_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            key, value = line.split(":", 1)

            values = np.array(
                [float(x) for x in value.strip().split()],
                dtype=np.float64
            )

            if key.startswith("P"):
                data[key] = values.reshape(3, 4)

            elif key == "R0_rect":
                data[key] = values.reshape(3, 3)

            elif key in ["Tr_velo_to_cam", "Tr_imu_to_velo"]:
                data[key] = values.reshape(3, 4)

    return data


def get_focal_length_px(calib_file):
    """
    Extract focal length fx from KITTI P2 matrix.
    """

    calib = read_kitti_calibration(calib_file)

    p2 = calib["P2"]

    fx = float(p2[0, 0])

    return fx


if __name__ == "__main__":

    calib_path = (
        "../dataset/raw/KITTI/"
        "data_object_calib/testing/calib/000002.txt"
    )

    fx = get_focal_length_px(calib_path)

    print("Focal Length (px):", fx)