import numpy as np
from lib import utils, calib

def determine_link_distance(scene_file: str, cam_a: int, cam_b: int, cam_a_points: np.ndarray, cam_b_points: np.ndarray):
    # Triangulate points between two cameras to obtain the 3D point location.
    k_arr, d_arr, r_arr, t_arr, _, _, _ = utils.find_scene_file(scene_file, verbose=False)
    points_3d = calib.triangulate_points_fisheye(cam_a_points, cam_b_points,
                                                k_arr[cam_a - 1], d_arr[cam_a - 1], r_arr[cam_a - 1], t_arr[cam_a - 1],
                                                k_arr[cam_b - 1], d_arr[cam_b - 1], r_arr[cam_b - 1], t_arr[cam_b - 1])
    # Calcuate the distance between the 3D points.
    assert len(points_3d) == 2
    distance = np.linalg.norm(points_3d[0,:] - points_3d[1,:])

    return distance

if __name__ == "__main__":
    """
    Simple function that calculates the distance between two locations (joints in this case) from two different views.
    In particular this calculates the distance between the ankle and paw on the cheetah. The following frames (from the labelled dataset) were used:
        - 2019_03_03/Menya/Run1: Frame 068 from CAM3 and CAM2. Measured the r_back and r_front.
        - 2019_03_03/Phantom/Run1: Frame 104 from CAM1 and CAM2. Measured the r_back and r_front.
        - 2019_03_05/Jules/Run1: Frame 091 from CAM1 and CAM2. Measured the r_back and r_front.

    The pixel locations for each of the points used are shown below.
    """
    scene_file_1 = "/Users/zico/msc/dev/AcinoSet/data/2019_03_03/extrinsic_calib/4_cam_scene_sba.json"
    scene_file_2 = "/Users/zico/msc/dev/AcinoSet/data/2019_03_05/extrinsic_calib/6_cam_scene_sba.json"

    dist_front_1 = determine_link_distance(scene_file_1, cam_a=3, cam_b=2, cam_a_points=np.array([[711.7309733799430, 833.8034071456910], [729.8153409020460, 859.7080417043780]]), cam_b_points=np.array([[1781.3063123955600, 886.5134433880010], [1757.1799109718900, 924.7135789754660]]))
    dist_back_1 = determine_link_distance(scene_file_1, cam_a=3, cam_b=2, cam_a_points=np.array([[502.5388301512950, 862.6406418430980], [489.3421295270580, 894.8992433690110]]), cam_b_points=np.array([[1319.8888851680100, 926.7241124274380], [1253.5412812529400, 986.034849260609]]))
    print(f"The distance between front joint A and joint B is {dist_front_1:.2f}[m]")
    print(f"The distance between back joint A and joint B is {dist_back_1:.2f}[m]")

    dist_front_2 = determine_link_distance(scene_file_1, cam_a=1, cam_b=2, cam_a_points=np.array([[1031.1723772841100, 903.5127706817280], [1036.149441202250, 924.0431593440270]]), cam_b_points=np.array([[595.0751316740850, 901.0736990526530], [595.0751316740850, 926.3447100434370]]))
    dist_back_2 = determine_link_distance(scene_file_1, cam_a=1, cam_b=2, cam_a_points=np.array([[702.1992732585720, 898.8727398008090], [726.7799952181200, 934.4597551750800]]), cam_b_points=np.array([[363.8113874085810, 882.8915913427540], [368.6036762452500, 918.2347215131930]]))
    print(f"The distance between front joint A and joint B is {dist_front_2:.2f}[m]")
    print(f"The distance between back joint A and joint B is {dist_back_2:.2f}[m]")

    dist_back_3 = determine_link_distance(scene_file_2, cam_a=1, cam_b=2, cam_a_points=np.array([[1221.373501132450, 1050.9641659920300], [1260.2099439361400, 1061.9853186795600]]), cam_b_points=np.array([[627.414865480014, 762.1076194084940], [669.5376674725290, 775.1456295490350]]))
    dist_front_3 = determine_link_distance(scene_file_2, cam_a=1, cam_b=2, cam_a_points=np.array([[1307.96827224878, 1031.5459445901800], [1292.7485852040900, 1052.538616375960]]), cam_b_points=np.array([[716.6750887498660, 745.559375768578], [700.1268451099500, 771.1339341211760]]))
    print(f"The distance between front joint A and joint B is {dist_front_3:.2f}[m]")
    print(f"The distance between back joint A and joint B is {dist_back_3:.2f}[m]")

    print(f"The average distance between front joint A and joint B is {np.mean([dist_front_1, dist_front_2, dist_front_3]):.2f}[m]")
    print(f"The average distance between back joint A and joint B is {np.mean([dist_back_1, dist_back_2, dist_back_3]):.2f}[m]")
