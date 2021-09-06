from file_utils import *
from image_utils import *


###################################################################################################################
######################Functions using LIDAR data to build and visualize pointclouds and filter out the relevant####
######################points reflected by the road#################################################################
###################################################################################################################


def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)


def load_velo_scan(velo_filename):
    """
    Loads LIDAR files (.bin) of KITTI dataset
    :param velo_filename: Lidar filename
    :return: Pointcloud with x, y, z coordinates and reflectance
    """
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def project_velo_to_cam2(calib):
    """
    KITTI
    :param calib: Calibration file provided in the KITTI software development kit. (Depends on the date of recording!)
    :return: The projection matrix for projecting 3D LIDAR points into 2D camera (pixel) cooridnates.
    """
    P_velo2cam_ref = calib['R_T'].reshape(4, 4)  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R_rect_00'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P_rect_02'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref

    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection for KITTI.
    :param points: 3D points in camera coordinate [3, number of points]
    :param proj_mat: Projection matrix [3, 4] returned by project_velo_to_cam2
    :return: LIDAR points in image coordinates.
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def prepare_data(depth_filename, calib, dataset, img_nbr=None):
    """
    Prepare the LIDAR data from the different datasets to streamline the analysis and reuse the same functions. For
    KITTI and Oxford the LIDAR points that are high above ground are already filtered out, since we are only interested
    in the points reflected by the free road.
    :param depth_filename:
    :param calib: Calibration file for the KITTI dataset (see read_calib_file())
    :return: Dictionary containing the different LIDAR datapoints
    """
    #lidar_points = nx4 (x, y, z, reflectance)

    if dataset == "KITTI":
        lidar = load_velo_scan(depth_filename)
        img_filename = extract_image_file_name_from_depth_file_name(depth_filename, dataset="KITTI")
        img = load_image(img_filename, dataset)

        img_height = img.shape[0]
        img_width = img.shape[1]
        points = lidar[:, 0:3]
        reflectance = lidar[:, 3]

        road_points = points[points[:, 2] < -1] #since the LIDAR scanner sits at an elevation of 1.73m above ground
        road_points = road_points[np.abs(road_points[:, 1]) < np.percentile(np.abs(road_points[:, 1]), 80)]
        # we are interested in the corridor in front of the car, not the region to the sides
        depth = points[:, 0]
        road_depth = road_points[:, 0]
        distance = np.linalg.norm(points, axis=1)
        road_distance = np.linalg.norm(road_points, axis=1)

        # projection matrix (project from velo2cam2)
        proj_velo2cam2 = project_velo_to_cam2(calib)
        pixel_coord = project_to_image(points.transpose(), proj_velo2cam2)
        road_pixel_coord = project_to_image(road_points.transpose(), proj_velo2cam2)

        # Filter out the Lidar Points on the image
        inds = np.where((pixel_coord[0, :] < img_width) & (pixel_coord[0, :] >= 0) &
                        (pixel_coord[1, :] < img_height) & (pixel_coord[1, :] >= 0) &
                        (points[:, 0] > 0))[0]
        road_inds = np.where((road_pixel_coord[0, :] < img_width) & (road_pixel_coord[0, :] >= 0) &
                             (road_pixel_coord[1, :] < img_height) & (road_pixel_coord[1, :] >= 0) &
                             (road_points[:, 0] > 0))[0]

        pixel_coord = pixel_coord[:, inds]
        pixel_coord = np.transpose(pixel_coord)
        road_pixel_coord = road_pixel_coord[:, road_inds]
        road_pixel_coord = np.transpose(road_pixel_coord)
        points = points[inds, :]
        road_points = road_points[road_inds, :]
        depth = depth[inds]
        road_depth = road_depth[road_inds]
        distance = distance[inds]
        road_distance = road_distance[road_inds]
        reflectance = reflectance[inds]

        row = pixel_coord[:, 1]
        road_row = road_pixel_coord[:, 1]
        col = pixel_coord[:, 0]
        road_col = road_pixel_coord[:, 0]
        lidar_dict = {'points': road_points, 'reflectance': reflectance, 'row': road_row, 'col': road_col, 'depth':
            road_depth, 'distance': road_distance, 'all_points': points, 'all_row': row, 'all_col': col, 'all_depth':
            depth, 'all_distance': distance}

        return lidar_dict

    elif dataset=="Oxford":
        # lidar: XYZI pointcloud from the binary Velodyne data Nx4
        date = depth_filename.split('/')[6]
        lidar_dir = depth_filename.split('/')
        lidar_dir = lidar_dir[:len(lidar_dir)-1]
        lidar_dir = '/'.join(lidar_dir)
        pose_file = extract_ins_folder_name_from_depth_file_name(depth_filename)
        print(pose_file)
        extrinsic = depth_filename.split('/')
        extrinsic = extrinsic[:len(extrinsic)-4]
        extrinsic = '/'.join(extrinsic)
        extrinsic = extrinsic + '/extrinsics'
        time = depth_filename.split('.')[0]
        time = time.split('/')[-1]
        time = np.int(time)
        img_dir = depth_filename.split('/')
        img_dir = img_dir[:len(img_dir)-4]
        img_dir = '/'.join(img_dir)
        img_dir = img_dir + '/Camera/' + date + '/stereo/centre'
        models = "/Volumes/Extreme SSD/Bachelorarbeit/Oxford/camera-models"
        points, reflectance = build_pointcloud(lidar_dir, pose_file, extrinsic, start_time=time - 1e7, \
                                               end_time=time + 1e7, origin_time=time)

        points = np.array(points)
        points = points[:3, :]
        points = np.transpose(points)
        points = points[points[:, 2] > -2]
        points = points[np.abs(points[:, 1]) < np.percentile(np.abs(points[:, 1]), 5)]

        #get pixels from matrix in project laser into camera
        pixels, depth = project_laser_into_camera(img_dir, lidar_dir, pose_file, models, extrinsic, img_nbr, show_ptcld=
                                                  False, road=True)
        pixels = np.array(pixels)
        row = pixels[1, :]
        col = pixels[0, :]
        distance = np.linalg.norm(points, axis=1)

        lidar_dict = {'points': points, 'reflectance': reflectance, 'row': row, 'col': col, 'depth': depth,
                  'distance': distance}

        return lidar_dict


def create_open3d_pc(lidar, cam_image = None, dataset = "A2D2"):
    """
    Create a 3D pointcloud and obtain the lidar points (their coordinates in 3D) which are on the free road.
    Visualize the point cloud using o3.visualization.draw_geometries([pcd]).
    :param lidar: lidar dictionary, as provided by prepare_data()
    :param cam_image: either rgb image or semantic image with detected road
    :param dataset:
    :return: point cloud and road indices based on semantic image
    """
    #create opend3d point cloud pcd
    pcd = o3.geometry.PointCloud()

    #assign points to the cloud
    pcd.points = o3.utility.Vector3dVector(lidar['points'])

    #Color of the road in semantic images
    road_color = np.array([1, 0, 1])

    #assign colors
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance'])/(median_reflectance*5)

        #clip colors for visualisation on white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        if dataset == "A2D2":
            colours = cam_image[rows, cols, :]/255.0
        elif dataset == "KITTI":
            cam_image = cv2.resize(cam_image, dsize=(1242, 188))
            img = np.zeros((375,1242,3), np.uint8)
            yborder = int(img.shape[0] * 0.45)
            xborder = int(img.shape[1] * 0.3)
            img[yborder:, xborder:img.shape[1] - xborder, :] = cam_image[:, :]
            colours = img[rows-2, cols-2, :]/255.0
        elif dataset == "Oxford":
            img = np.zeros((960, 1275, 3), np.uint8)
            yborder = int(img.shape[0] * 0.45)
            xborder = int(img.shape[1] * 0.3)
            img[yborder:, xborder:img.shape[1] - xborder, :] = cam_image[:, :cam_image.shape[1]-1]
            colours = img[rows-6, cols-6, :]/255.0

    indices = []
    road_pts = np.zeros((100000, 3))
    if dataset == "A2D2":
        #Read in the semantic image
        pcd.colors = o3.utility.Vector3dVector(colours)
        for i in range(len(pcd.colors)):
            if (np.asarray(pcd.colors[i]) == road_color).all():
                #access pcd points in lidar coord
                road_pts[i, ] = pcd.points[i]
                indices.append(i)

    elif dataset == "KITTI":
        #Read in the binary image where the road is already filtered
        pcd.colors = o3.utility.Vector3dVector(colours)
        for i in range(len(pcd.colors)):
            if (np.asarray(pcd.colors[i]) == road_color).all():
                # access pcd points in lidar coord
                indices.append(i)

    elif dataset == "Oxford":
        # Read in the binary image where the road is already filtered
        pcd.colors = o3.utility.Vector3dVector(colours)
        for i in range(len(pcd.colors)):
            if (np.asarray(pcd.colors[i]) == road_color).all():
                # access pcd points in lidar coord
                indices.append(i)
    indices = np.array(indices)
    #road_pts = road_pts[np.all(road_pts != 0, axis=1)]

    return pcd, indices


#Visualize the mapping of point cloud
def map_lidar_points_onto_image(image_orig, lidar, dataset, pixel_size=3, pixel_opacity=1, road=None):
    """
    Maps LIDAR points on to image and colors them according to their distance.
    :param image_orig: RGB image
    :param lidar: LIDAR dictionary as provided by prepare_data()
    :param dataset: "A2D2", "KITTI", "Oxford"
    :param pixel_size: pixel size of LIDAR points mapped on to the image
    :param pixel_opacity: alpha value for LIDAR points mapped on to the image
    :param road: Boolean, indicates if only the LIDAR points reflected from the road, or the whole pointcloud should be
    displayed
    :return: image with LIDAR pointcloud
    """
    image = np.copy(image_orig)

    #get rows and cols of lidar points in 2d image coordinates
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)

    #lowest distance values to be accounted for in colour code
    MIN_DIST = np.min(lidar['distance'])
    #largest distance values to be accounted for in colour code
    MAX_DIST = np.max(lidar['distance'])

    distances = lidar['distance']
    colours = (distances - MIN_DIST)/(MAX_DIST - MIN_DIST)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75*c, np.sqrt(pixel_opacity), 1.0)) for c in colours])

    if dataset == "A2D2":
        pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
        pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
        canvas_rows = image.shape[0]
        canvas_cols = image.shape[1]
        for i in range (len(rows)):
            pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
            pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
            image[pixel_rows, pixel_cols, :] = (1. - pixel_opacity) * np.multiply(image[pixel_rows, pixel_cols, :],\
                                                colours[i]) + pixel_opacity * 255 * colours[i]
    elif dataset == "KITTI":
        if road==True:
            # get rows and cols of lidar points in 2d image coordinates
            rows = (lidar['road_row'] + 0.5).astype(np.int)
            cols = (lidar['road_col'] + 0.5).astype(np.int)

            # lowest distance values to be accounted for in colour code
            MIN_DIST = np.min(lidar['road_distance'])
            # largest distance values to be accounted for in colour code
            MAX_DIST = np.max(lidar['road_distance'])

            distances = lidar['road_distance']
            colours = (distances - MIN_DIST) / (MAX_DIST - MIN_DIST)
            colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, np.sqrt(pixel_opacity), 1.0)) for c in colours])

        for i in range (len(rows)):
            cv2.circle(image, (cols[i], rows[i]), 2, color=(255*colours[i, 0], 255*colours[i, 1], 255*colours[i, 2]),\
                       thickness=-1)
    return image.astype(np.uint8)


def get_lidar_data(depth_filename, dataset, calib, img_number):
    """

    :param img_number:
    :param depth_filename:
    :param calib:
    :param dataset:
    :return: Returns the LIDAR dictionary for every dataset and corresponding image.
    """
    if dataset=="A2D2":
        lidar = np.load(depth_filename)
    elif dataset=="KITTI":
        lidar = prepare_data(depth_filename, dataset=dataset, calib=calib)
    elif dataset=="Oxford":
        lidar = prepare_data(depth_filename, dataset=dataset, calib=calib, img_nbr=img_number)

    return lidar


################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy


def interpolate_vo_poses(vo_path, pose_timestamps, origin_timestamp):
    """Interpolate poses from visual odometry.

    Args:
        vo_path (str): path to file containing relative poses from visual odometry.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(vo_path) as vo_file:
        vo_reader = csv.reader(vo_file)
        headers = next(vo_file)

        vo_timestamps = [0]
        abs_poses = [matlib.identity(4)]

        lower_timestamp = min(min(pose_timestamps), origin_timestamp)
        upper_timestamp = max(max(pose_timestamps), origin_timestamp)

        for row in vo_reader:
            timestamp = int(row[0])
            if timestamp < lower_timestamp:
                vo_timestamps[0] = timestamp
                continue

            vo_timestamps.append(timestamp)

            xyzrpy = [float(v) for v in row[2:8]]
            rel_pose = build_se3_transform(xyzrpy)
            abs_pose = abs_poses[-1] * rel_pose
            abs_poses.append(abs_pose)

            if timestamp >= upper_timestamp:
                break

    return interpolate_poses(vo_timestamps, abs_poses, pose_timestamps, origin_timestamp)


def interpolate_ins_poses(ins_path, pose_timestamps, origin_timestamp, use_rtk=False):
    """Interpolate poses from INS.

    Args:
        ins_path (str): path to file containing poses from INS.
        pose_timestamps (list[int]): UNIX timestamps at which interpolated poses are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    """
    with open(ins_path) as ins_file:
        ins_reader = csv.reader(ins_file)
        headers = next(ins_file)

        ins_timestamps = [0]
        abs_poses = [matlib.identity(4)]

        upper_timestamp = max(max(pose_timestamps), origin_timestamp)

        for row in ins_reader:
            timestamp = int(row[0])
            ins_timestamps.append(timestamp)

            utm = row[5:8] if not use_rtk else row[4:7]
            rpy = row[-3:] if not use_rtk else row[11:14]
            xyzrpy = [float(v) for v in utm] + [float(v) for v in rpy]
            abs_pose = build_se3_transform(xyzrpy)
            abs_poses.append(abs_pose)

            if timestamp >= upper_timestamp:
                break

    ins_timestamps = ins_timestamps[1:]
    abs_poses = abs_poses[1:]

    return interpolate_poses(ins_timestamps, abs_poses, pose_timestamps, origin_timestamp)


def interpolate_poses(pose_timestamps, abs_poses, requested_timestamps, origin_timestamp):
    """Interpolate between absolute poses.

    Args:
        pose_timestamps (list[int]): Timestamps of supplied poses. Must be in ascending order.
        abs_poses (list[numpy.matrixlib.defmatrix.matrix]): SE3 matrices representing poses at the timestamps specified.
        requested_timestamps (list[int]): Timestamps for which interpolated timestamps are required.
        origin_timestamp (int): UNIX timestamp of origin frame. Poses will be reported relative to this frame.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    Raises:
        ValueError: if pose_timestamps and abs_poses are not the same length
        ValueError: if pose_timestamps is not in ascending order

    """
    requested_timestamps.insert(0, origin_timestamp)
    requested_timestamps = np.array(requested_timestamps)
    pose_timestamps = np.array(pose_timestamps)

    if len(pose_timestamps) != len(abs_poses):
        raise ValueError('Must supply same number of timestamps as poses')

    abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))
    for i, pose in enumerate(abs_poses):
        if i > 0 and pose_timestamps[i-1] >= pose_timestamps[i]:
            raise ValueError('Pose timestamps must be in ascending order')

        abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3])
        abs_positions[:, i] = np.ravel(pose[0:3, 3])

    upper_indices = [bisect.bisect(pose_timestamps, pt) for pt in requested_timestamps]
    lower_indices = [u - 1 for u in upper_indices]

    if max(upper_indices) >= len(pose_timestamps):
        upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices]

    fractions = (requested_timestamps - pose_timestamps[lower_indices]) // \
                (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]

    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)
    sin_interp_indices = np.nonzero(d_array < 1)

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))

    scale0_array[sin_interp_indices] = \
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = \
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0)
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower \
                         + np.tile(scale1_array, (4, 1)) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]

    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    poses_mat = matlib.zeros((4, 4 * len(requested_timestamps)))

    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])

    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])

    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    poses_mat = np.linalg.solve(poses_mat[0:4, 0:4], poses_mat)

    poses_out = [0] * (len(requested_timestamps) - 1)
    for i in range(1, len(requested_timestamps)):
        poses_out[i - 1] = poses_mat[0:4, i * 4:(i + 1) * 4]

    return poses_out


hdl32e_range_resolution = 0.002  # m / pixel
hdl32e_minimum_range = 1.0
hdl32e_elevations = np.array([-0.1862, -0.1628, -0.1396, -0.1164, -0.0930,
                              -0.0698, -0.0466, -0.0232, 0., 0.0232, 0.0466, 0.0698,
                              0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327,
                              0.2560, 0.2793, 0.3025, 0.3259, 0.3491, 0.3723, 0.3957,
                              0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353])[:, np.newaxis]
hdl32e_base_to_fire_height = 0.090805
hdl32e_cos_elevations = np.cos(hdl32e_elevations)
hdl32e_sin_elevations = np.sin(hdl32e_elevations)


def load_velodyne_binary(velodyne_bin_path: AnyStr):
    """Decode a binary Velodyne example (of the form '<timestamp>.bin')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset binary Velodyne pointcloud example path
    Returns:
        ptcld (np.ndarray): XYZI pointcloud from the binary Velodyne data Nx4
    Notes:
        - The pre computed points are *NOT* motion compensated.
        - Converting a raw velodyne scan to pointcloud can be done using the
            `velodyne_ranges_intensities_angles_to_pointcloud` function.
    """
    ext = os.path.splitext(velodyne_bin_path)[1]
    if ext != ".bin":
        raise RuntimeError("Velodyne binary pointcloud file should have `.bin` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_bin_path):
        raise FileNotFoundError("Could not find velodyne bin example: {}".format(velodyne_bin_path))
    data = np.fromfile(velodyne_bin_path, dtype=np.float32)
    ptcld = data.reshape((4, -1))
    return ptcld


def load_velodyne_raw(velodyne_raw_path: AnyStr):
    """Decode a raw Velodyne example. (of the form '<timestamp>.png')
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset raw Velodyne example path
    Returns:
        ranges (np.ndarray): Range of each measurement in meters where 0 == invalid, (32 x N)
        intensities (np.ndarray): Intensity of each measurement where 0 == invalid, (32 x N)
        angles (np.ndarray): Angle of each measurement in radians (1 x N)
        approximate_timestamps (np.ndarray): Approximate linearly interpolated timestamps of each mesaurement (1 x N).
            Approximate as we only receive timestamps for each packet. The timestamp of the next frame will was used to
            interpolate the last packet timestamps. If there was no next frame, the last packet timestamps was
            extrapolated. The original packet timestamps can be recovered with:
                approximate_timestamps(:, 1:12:end) (12 is the number of azimuth returns in each packet)
     Notes:
       Reference: https://velodynelidar.com/lidar/products/manual/63-9113%20HDL-32E%20manual_Rev%20E_NOV2012.pdf
    """
    ext = os.path.splitext(velodyne_raw_path)[1]
    if ext != ".png":
        raise RuntimeError("Velodyne raw file should have `.png` extension but had: {}".format(ext))
    if not os.path.isfile(velodyne_raw_path):
        raise FileNotFoundError("Could not find velodyne raw example: {}".format(velodyne_raw_path))
    example = cv2.imread(velodyne_raw_path, cv2.IMREAD_GRAYSCALE)
    intensities, ranges_raw, angles_raw, timestamps_raw = np.array_split(example, [32, 96, 98], 0)
    ranges = np.ascontiguousarray(ranges_raw.transpose()).view(np.uint16).transpose()
    ranges = ranges * hdl32e_range_resolution
    angles = np.ascontiguousarray(angles_raw.transpose()).view(np.uint16).transpose()
    angles = angles * (2. * np.pi) / 36000
    approximate_timestamps = np.ascontiguousarray(timestamps_raw.transpose()).view(np.int64).transpose()
    return ranges, intensities, angles, approximate_timestamps


def velodyne_raw_to_pointcloud(ranges: np.ndarray, intensities: np.ndarray, angles: np.ndarray):
    """ Convert raw Velodyne data (from load_velodyne_raw) into a pointcloud
    Args:
        ranges (np.ndarray): Raw Velodyne range readings
        intensities (np.ndarray): Raw Velodyne intensity readings
        angles (np.ndarray): Raw Velodyne angles
    Returns:
        pointcloud (np.ndarray): XYZI pointcloud generated from the raw Velodyne data Nx4

    Notes:
        - This implementation does *NOT* perform motion compensation on the generated pointcloud.
        - Accessing the pointclouds in binary form via `load_velodyne_pointcloud` is approximately 2x faster at the cost
            of 8x the storage space
    """
    valid = ranges > hdl32e_minimum_range
    z = hdl32e_sin_elevations * ranges - hdl32e_base_to_fire_height
    xy = hdl32e_cos_elevations * ranges
    x = np.sin(angles) * xy
    y = -np.cos(angles) * xy

    xf = x[valid].reshape(-1)
    yf = y[valid].reshape(-1)
    zf = z[valid].reshape(-1)
    intensityf = intensities[valid].reshape(-1).astype(np.float32)
    ptcld = np.stack((xf, yf, zf, intensityf), 0)
    return ptcld


def build_pointcloud(lidar_dir, poses_file, extrinsics_dir, start_time, end_time, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))

    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')
        if "velodyne" not in lidar:
            if not os.path.isfile(scan_path):
                continue

            scan_file = open(scan_path)
            scan = np.fromfile(scan_file, np.double)
            scan_file.close()

            scan = scan.reshape((len(scan) // 3, 3)).transpose()

            if lidar != 'ldmrs':
                # LMS scans are tuples of (x, y, reflectance)
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
                scan[2, :] = np.zeros((1, scan.shape[1]))
        else:
            if os.path.isfile(scan_path):
                ptcld = load_velodyne_binary(scan_path)
            else:
                scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
                if not os.path.isfile(scan_path):
                    continue
                ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

            reflectance = np.concatenate((reflectance, ptcld[3]))
            scan = ptcld[:3]

        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
        pointcloud = np.hstack([pointcloud, scan])

    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance


class CameraModel:
    """Provides intrinsic parameters and undistortion LUT for a camera.

    Attributes:
        camera (str): Name of the camera.
        camera sensor (str): Name of the sensor on the camera for multi-sensor cameras.
        focal_length (tuple[float]): Focal length of the camera in horizontal and vertical axis, in pixels.
        principal_point (tuple[float]): Principal point of camera for pinhole projection model, in pixels.
        G_camera_image (:obj: `numpy.matrixlib.defmatrix.matrix`): Transform from image frame to camera frame.
        bilinear_lut (:obj: `numpy.ndarray`): Look-up table for undistortion of images, mapping pixels in an undistorted
            image to pixels in the distorted image

    """

    def __init__(self, models_dir, images_dir):
        """Loads a camera model from disk.

        Args:
            models_dir (str): directory containing camera model files.
            images_dir (str): directory containing images for which to read camera model.

        """
        self.camera = None
        self.camera_sensor = None
        self.focal_length = None
        self.principal_point = None
        self.G_camera_image = None
        self.bilinear_lut = None

        self.__load_intrinsics(models_dir, images_dir)
        self.__load_lut(models_dir, images_dir)

    def project(self, xyz, image_size):
        """Projects a pointcloud into the camera using a pinhole camera model.

        Args:
            xyz (:obj: `numpy.ndarray`): 3xn array, where each column is (x, y, z) point relative to camera frame.
            image_size (tuple[int]): dimensions of image in pixels

        Returns:
            numpy.ndarray: 2xm array of points, where each column is the (u, v) pixel coordinates of a point in pixels.
            numpy.array: array of depth values for points in image.

        Note:
            Number of output points m will be less than or equal to number of input points n, as points that do not
            project into the image are discarded.

        """
        if xyz.shape[0] == 3:
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
        xyzw = np.linalg.solve(self.G_camera_image, xyz)

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0]
        xyzw = xyzw[:, in_front]

        uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0],
                        self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]))

        in_img = [i for i in range(0, uv.shape[1])
                  if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]]

        return uv[:, in_img], np.ravel(xyzw[2, in_img])

    def undistort(self, image):
        """Undistorts an image.

        Args:
            image (:obj: `numpy.ndarray`): A distorted image. Must be demosaiced - ie. must be a 3-channel RGB image.

        Returns:
            numpy.ndarray: Undistorted version of image.

        Raises:
            ValueError: if image size does not match camera model.
            ValueError: if image only has a single channel.

        """
        if image.shape[0] * image.shape[1] != self.bilinear_lut.shape[0]:
            raise ValueError('Incorrect image size for camera model')

        lut = self.bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))

        if len(image.shape) == 1:
            raise ValueError('Undistortion function only works with multi-channel images')

        undistorted = np.rollaxis(np.array([map_coordinates(image[:, :, channel], lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

        return undistorted.astype(image.dtype)

    def __get_model_name(self, images_dir):
        self.camera = re.search('(stereo|mono_(left|right|rear))', images_dir).group(0)
        if self.camera == 'stereo':
            self.camera_sensor = re.search('(left|centre|right)', images_dir).group(0)
            if self.camera_sensor == 'left':
                return 'stereo_wide_left'
            elif self.camera_sensor == 'right':
                return 'stereo_wide_right'
            elif self.camera_sensor == 'centre':
                return 'stereo_narrow_left'
            else:
                raise RuntimeError('Unknown camera model for given directory: ' + images_dir)
        else:
            return self.camera

    def __load_intrinsics(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)
        intrinsics_path = os.path.join(models_dir, model_name + '.txt')

        with open(intrinsics_path) as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            self.focal_length = (vals[0], vals[1])
            self.principal_point = (vals[2], vals[3])

            G_camera_image = []
            for line in intrinsics_file:
                G_camera_image.append([float(x) for x in line.split()])
            self.G_camera_image = np.array(G_camera_image)

    def __load_lut(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)
        lut_path = os.path.join(models_dir, model_name + '_distortion_lut.bin')

        lut = np.fromfile(lut_path, np.double)
        lut = lut.reshape([2, lut.size // 2])
        self.bilinear_lut = lut.transpose()


def project_laser_into_camera(image_directory, laser_directory, ins_or_vo_file, camera_models_directory,
                              extrinsics_directory, image_number, show_ptcld=False, road=False):
    model = CameraModel(models_dir=camera_models_directory, images_dir=image_directory)
    extrinsics_path = extrinsics_directory + "/" + model.camera + ".txt"
    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None

    poses_type = re.search('(vo|ins|rtk)\.csv', ins_or_vo_file).group(1)
    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_directory, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
    else:
        # VO frame and vehicle frame are the same
        G_camera_posesource = G_camera_vehicle

    timestamps_path = os.path.join(image_directory, os.pardir, os.pardir, model.camera + '.csv')
    if not os.path.isfile(timestamps_path):
        timestamps_path = os.path.join(image_directory, os.pardir, os.pardir, model.camera + '.timestamps')

    timestamp = 0
    with open(timestamps_path) as timestamps_file:
        for i, line in enumerate(timestamps_file):
            if i == image_number:
                timestamp = int(line.split(' ')[0])

    pointcloud, reflectance = build_pointcloud(laser_directory, ins_or_vo_file, extrinsics_directory,
                                               timestamp - 1e7, timestamp + 1e7, timestamp)
    pointcloud_road = np.array(pointcloud)
    pointcloud_road = np.transpose(pointcloud_road)
    pointcloud_road = pointcloud_road[pointcloud_road[:, 2] > -2]
    pointcloud_road = pointcloud_road[np.abs(pointcloud_road[:, 1]) < np.percentile(np.abs(pointcloud_road[:, 1]), 5)]
    pointcloud_road = np.transpose(pointcloud_road)
    pointcloud_road = np.dot(G_camera_posesource, pointcloud_road)
    pointcloud = np.dot(G_camera_posesource, pointcloud)

    image_path = os.path.join(image_directory, str(timestamp) + '.png')
    image = load_image(image_path, dataset="Oxford")

    uv, depth = model.project(pointcloud, image.shape)
    if road:
        uv, depth = model.project(pointcloud_road, image.shape)

    if show_ptcld:
        plt.imshow(image)
        plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=3, c=1 / depth, edgecolors='none', cmap='jet')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return [np.ravel(uv[0, :]), np.ravel(uv[1, :])], depth