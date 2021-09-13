from file_utils import *
#from lidar_utils import *


###################################################################################################################
######################Functions using image data for road detection and pointcloud visualization###################
###################################################################################################################


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


def load_image(img_filename, dataset):
    """

    :param img_filename: filename of the image to load.
    :param dataset: "A2D2", "Apolloscape_stereo", "Apolloscape_semantic" "KITTI", or "Oxford
    :return: undistorted RGB images of every type: RGB, semantic, grayscale, depth, disparity
    """
    if dataset == "A2D2":
        cam_name = 'front_center'
        image = cv2.imread(img_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    elif dataset in ["Apolloscape_stereo", "Apolloscape_semantic"]:
        image = cv2.imread(img_filename)
        type = re.search('(camera_5|fg_mask|ColorImage|Label|depth)', img_filename).group(0)
        if type in ["camera_5", "ColorImage"]:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # undistorted image is provided
        elif type == "fg_mask":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif type == "Label":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif type == "depth":
            image = cv2.imread(img_filename)
        return image
    elif dataset == "KITTI":
        image = cv2.imread(img_filename)  # undistorted image is provided
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    elif dataset == "Oxford":
        img_directory = get_img_directory(img_filename, dataset)
        model = CameraModel(models_dir=camera_models_directory, images_dir=img_directory)
        BAYER_STEREO = 'gbrg'
        BAYER_MONO = 'rggb'

        if model:
            camera = model.camera
        else:
            camera = re.search('(stereo|mono_(left|right|rear))', img_filename).group(0)
        if camera == 'stereo':
            pattern = BAYER_STEREO
        else:
            pattern = BAYER_MONO

        img = Image.open(img_filename)
        img = demosaic(img, pattern)
        if model:
            img = model.undistort(img)

        return np.array(img).astype(np.uint8)


def get_disparity_map(depth_filename):
    disparity = cv2.imread(depth_filename)
    disparity = cv2.cvtColor(disparity, cv2.COLOR_RGB2GRAY)

    return disparity


def undistort_image(image, cam_name):
    """
    Function is provided in the A2D2 tutorial. Undistorts images based on the camera they were recorded with.
    :param image:
    :param cam_name:
    :return:
    """
    if cam_name in ['front_left', 'front_right', 'front_center', 'side_right', 'side_left', 'rear_center']:
        intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if lens == 'Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens == 'Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


# Convert HSV to RGB values
def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def filter_road(semantic_image, dataset):
    """
    Takes in the semantic image in rgb format, converts it to hsv or grasycale and filters out the road pixels
    :param semantic_image:
    :return: Binary image with road pixel value = 255, rest = 0
    """
    if dataset in ["A2D2", "Apolloscape_stereo", "KITTI", "Oxford"]:
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 150) | (s != 255) & (s != 184) | (v != 255) & (v != 180)] = 0

    elif dataset == "Apolloscape_semantic":
        img = semantic_image

        binary = np.zeros_like(img)
        binary[:, :] = 0
        binary[img == 49] = 255

    return binary


def filter_cars(semantic_image, dataset):
    """

    :param semantic_image:
    :param dataset:
    :return: Binary image with car pixel value = 255, rest = 0
    """
    if dataset == "A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 0) & (h != 60)] = 0  # Cars & Small Vehicles

    elif dataset == "Apolloscape_semantic":
        img = semantic_image

        binary = np.zeros_like(img)
        binary[:, :] = 0
        binary[img == 33] = 255
        binary[img == 161] = 255

    return binary


def filter_trucks(semantic_image, dataset):
    """

    :param semantic_image:
    :param dataset:
    :return: Binary image with truck and bus pixel value = 255, rest = 0
    """
    if dataset == "A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 15) & (h != 19) & (h != 26) & (h != 120) & (h != 30) & (h != 15)] = 0

    elif dataset == "Apolloscape_semantic":
        img = semantic_image

        binary = np.zeros_like(img)
        binary[:, :] = 0
        binary[img == 38] = 255
        binary[img == 166] = 255
        binary[img == 39] = 255
        binary[img == 167] = 255

    return binary


def filter_ped_bc(semantic_image, dataset):
    """

    :param semantic_image:
    :param dataset:
    :return: Binary image with pedestrian and bicycle pixel value = 255, rest = 0
    """
    if dataset == "A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 14) & (h != 11) & (h != 10) & (h != 135) & (h != 159) & (h != 160)] = 0

    elif dataset == "Apolloscape_semantic":
        img = semantic_image

        binary = np.zeros_like(img)
        binary[:, :] = 0
        binary[img == 35] = 255
        binary[img == 163] = 255
        binary[img == 36] = 255
        binary[img == 164] = 255

    return binary


def filter_other(semantic_image, dataset):
    """

    :param semantic_image:
    :param dataset:
    :return:
    """
    if dataset == "A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        binary = np.zeros_like(semantic_image)

        binary[(h != 150) | (s != 255) & (s != 184) | (v != 255) & (v != 180)] = 255
        binary[(h != 0) & (h != 60)] = 255  # Cars & Small Vehicles
        binary[(h != 15) & (h != 19) & (h != 26) & (h != 120) & (h != 30) & (h != 15)] = 255
        binary[(h != 14) & (h != 11) & (h != 10) & (h != 135) & (h != 159) & (h != 160)] = 255

    elif dataset == "Apolloscape_semantic":
        img = semantic_image

        binary = np.zeros_like(img)
        binary[:, :] = 0
        binary[img != 35] = 255
        binary[img != 163] = 255
        binary[img != 36] = 255
        binary[img != 164] = 255
        binary[img != 38] = 255
        binary[img != 166] = 255
        binary[img != 39] = 255
        binary[img != 167] = 255
        binary[img != 33] = 255
        binary[img != 161] = 255
        binary[img != 49] = 255

    return binary


def get_road_pixels(lane_img, img, dataset):
    """

    :param lane_img: result of det_line_1lane_init function (for Apolloscape_stereo, KITTI, and Oxford).
    Semantic image for A2D2 and Apolloscape_semantic.
    :param img: Corresponding RGB image.
    :param dataset: "A2D2", "Apolloscape_stereo", "Apolloscape_semantic", "KITTI", or "Oxford
    :return: Pixel coordinates of every road pixel.
    """
    binary = filter_road(lane_img, dataset)
    [u, v] = np.where(binary == 255)

    if dataset == "KITTI":
        roi_factor_y = 0.45  # For the lane detection, we only considered the bottom 50% of the image
        roi_factor_x = 0.3
    elif dataset == "Apolloscape_stereo":
        roi_factor_y = 0.39  # For the lane detection, we only considered the bottom 60% of the image
        roi_factor_x = 0.3
    elif dataset == "Oxford":
        roi_factor_y = 0.45
        roi_factor_x = 0.3
    elif dataset == "A2D2":
        roi_factor_y = 0
        roi_factor_x = 0
    elif dataset == "Apolloscape_semantic":
        roi_factor_y = 0
        roi_factor_x = 0

    factor_y = int(roi_factor_y * img.shape[0])
    factor_x = int(roi_factor_x * img.shape[1])
    u = u + factor_y
    v = v + factor_x

    return np.transpose(np.vstack((u, v)))


def get_road_boundaries(lane_img, img, dataset="Apolloscape_stereo"):
    """

    :param dataset: "Apolloscape_stereo", "Apolloscape_semantic", "KITTI", "A2D2", or "Oxford"
    :param lane_img: result of det_line_1lane_init function (for Apolloscape_stereo, KITTI, and Oxford).
    Semantic image for A2D2 and Apolloscape_semantic
    :return: the left and right most pixels (in pixel coordinates) of the road for every row of the image
    """
    binary = filter_road(lane_img, dataset)

    l = np.empty(shape=(0, 2), dtype=int)
    r = np.empty(shape=(0, 2), dtype=int)

    for i in range(binary.shape[0] - 1):
        for j in range(binary.shape[1] - 1):
            if binary[i, j] == 255:
                l = np.append(l, np.array([[i, j]]), axis=0)
                break  # We are only interested in the height of the pixels which are on the road

    for i in range(binary.shape[0] - 1):
        for j in range(binary.shape[1] - 1, 0, -1):
            if binary[i, j] == 255:
                r = np.append(r, np.array([[i, j]]), axis=0)
                break  # We are only interested in the height of the pixels which are on the road

    # Transform from cropped roi coordinates to global image coordinates
    if dataset == "KITTI":
        roi_factor_y = 0.45  # For the lane detection, we only considered the bottom 50% of the image
        roi_factor_x = 0.3
    elif dataset == "Apolloscape_stereo":
        roi_factor_y = 0.39  # For the lane detection, we only considered the bottom 60% of the image
        roi_factor_x = 0.3
    elif dataset == "Oxford":
        roi_factor_y = 0.45
        roi_factor_x = 0.3
    elif dataset == "A2D2":
        roi_factor_y = 0
        roi_factor_x = 0
    elif dataset == "Apolloscape_semantic":
        roi_factor_y = 0
        roi_factor_x = 0

    factor_y = int(roi_factor_y * img.shape[0])
    factor_x = int(roi_factor_x * img.shape[1])
    l[:, 0] = l[:, 0] + factor_y
    r[:, 0] = r[:, 0] + factor_y
    l[:, 1] = l[:, 1] + factor_x
    r[:, 1] = r[:, 1] + factor_x
    l = np.asarray(l)
    r = np.asarray(r)
    l = l[0:len(l) - 1]
    r = r[0:len(r) - 1]

    return l, r


def get_car_pixels(binary):
    """

    :param binary: binary car image produced by filter_cars()
    :return: pixel coordinates of car pixels
    """
    [u, v] = np.where(binary == 255)[:2]

    return np.transpose(np.vstack((u, v)))


def get_truck_pixels(binary):
    """

    :param binary: binary car image produced by filter_trucks()
    :return: pixel coordinates of truck and bus pixels
    """
    [u, v] = np.where(binary == 255)

    return np.transpose(np.vstack((u, v)))


def get_ped_pixels(binary):
    """

    :param binary: binary car image produced by filter_ped_bc()
    :return: pixel coordinates of pedestrian and bicycle pixels
    """
    [u, v] = np.where(binary == 255)

    return np.transpose(np.vstack((u, v)))


def get_other_pixels(binary):
    """

    :param binary:
    :return:
    """
    [u, v] = np.where(binary == 255)

    return np.transpose(np.vstack((u, v)))


def blocking_factors(semantic_image, img, dataset, fg_mask=None):
    """
    Measures the visibility obstructions for every frame and also classifies the obstructions
    :param fg_mask: for Apolloscape_stereo datasets
    :param semantic_image: semantic image for A2D2 and Apolloscape_segmented, foreground mask for Apolloscape_stereo
    :param img: corresponding rgb image
    :return:
    """
    road_pixels_l, road_pixels_r = get_road_boundaries(semantic_image, img, dataset)
    road_pixels = get_road_pixels(semantic_image, img, dataset)
    nbr_road_pixels = road_pixels.shape[0]
    thresholds = [0.3, 0.4, 0.5]
    result = []
    if dataset == "A2D2":
        bin_cars = filter_cars(semantic_image, dataset)
        bin_trucks = filter_trucks(semantic_image, dataset)
        bin_ped_bc = filter_ped_bc(semantic_image, dataset)
        bin_road = filter_road(semantic_image, dataset)
        bin_other = filter_other(semantic_image, dataset)
        car_pixels = get_car_pixels(bin_cars)
        truck_pixels = get_truck_pixels(bin_trucks)
        ped_pixels = get_ped_pixels(bin_ped_bc)
        other_pixels = get_other_pixels(bin_other)
        cars = []
        trucks = []
        ped_bc = []
        other = []

        for i in range(road_pixels_l.shape[0] - 1):
            car_pixels_of_int = car_pixels
            truck_pixels_of_int = truck_pixels
            ped_pixels_of_int = ped_pixels
            other_pixels_of_int = other_pixels
            row = road_pixels_l[i, 0]
            l = road_pixels_l[i, 1]
            r = road_pixels_r[i, 1]
            l = int(round(l + (r - l) / 3.5))  # only interested in ego-lane
            car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 0] == row]
            car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 1] <= r]
            car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 1] >= l]
            truck_pixels_of_int = truck_pixels_of_int[truck_pixels_of_int[:, 0] == row]
            truck_pixels_of_int = truck_pixels_of_int[truck_pixels_of_int[:, 1] <= r]
            truck_pixels_of_int = truck_pixels_of_int[truck_pixels_of_int[:, 1] >= l]
            ped_pixels_of_int = ped_pixels_of_int[ped_pixels_of_int[:, 0] == row]
            ped_pixels_of_int = ped_pixels_of_int[ped_pixels_of_int[:, 1] <= r]
            ped_pixels_of_int = ped_pixels_of_int[ped_pixels_of_int[:, 1] >= l]
            other_pixels_of_int = other_pixels_of_int[other_pixels_of_int[:, 0] == row]
            other_pixels_of_int = other_pixels_of_int[other_pixels_of_int[:, 1] <= r]
            other_pixels_of_int = other_pixels_of_int[other_pixels_of_int[:, 1] >= l]
            cars.append(car_pixels_of_int.shape[0])
            trucks.append(truck_pixels_of_int.shape[0])
            ped_bc.append(ped_pixels_of_int.shape[0])
            other.append(other_pixels_of_int.shape[0])
            i = i + 1

        # return mildly occluded (0.3), partly occluded (0.4), fully blocked (0.5)
        try:
            for threshold in thresholds:
                if np.sum(cars) / nbr_road_pixels > threshold:
                    if np.sum(trucks) / nbr_road_pixels > threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 1, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 1, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 1, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 1, 0, 0])
                    elif np.sum(trucks) / nbr_road_pixels <= threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 0, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 0, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 0, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 0, 0, 0])
                elif np.sum(cars) / nbr_road_pixels <= threshold:
                    if np.sum(trucks) / nbr_road_pixels > threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 1, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 1, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 1, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 1, 0, 0])
                    elif np.sum(trucks) / nbr_road_pixels <= threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 0, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 0, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 0, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 0, 0, 0])
        except:
            result.append([0, 0, 0, 0])

        result = np.array(result)
        result = np.reshape(result, [1, 12])

    elif dataset == "Apolloscape_stereo":
        car_pixels = get_car_pixels(fg_mask)
        cars = []
        for i in range(road_pixels_l.shape[0] - 1):
            car_pixels_of_int = car_pixels
            row = road_pixels_l[i, 0]
            l = road_pixels_l[i, 1]
            r = road_pixels_r[i, 1]
            l = int(round(l + (r - l) / 3.5))  # only interested in ego-lane
            car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 0] == row]
            car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 1] <= r]
            car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 1] >= l]
            cars.append(car_pixels_of_int.shape[0])

        for threshold in thresholds:
            if np.sum(cars) / nbr_road_pixels > threshold:
                result.append([1, 0, 0, 0])
            else:
                result.append([0, 0, 0, 0])

        result = np.array(result)
        result = np.reshape(result, [1, 12])

    elif dataset == "Apolloscape_semantic":
        bin_cars = filter_cars(semantic_image, dataset)
        bin_trucks = filter_trucks(semantic_image, dataset)
        bin_ped_bc = filter_ped_bc(semantic_image, dataset)
        bin_road = filter_road(semantic_image, dataset)
        bin_other = filter_other(semantic_image, dataset)
        car_pixels = get_car_pixels(bin_cars)
        truck_pixels = get_truck_pixels(bin_trucks)
        ped_pixels = get_ped_pixels(bin_ped_bc)
        other_pixels = get_other_pixels(bin_other)
        cars = []
        trucks = []
        ped_bc = []
        other = []
        try:
            for i in range(road_pixels_l.shape[0] - 1):
                car_pixels_of_int = car_pixels
                truck_pixels_of_int = truck_pixels
                ped_pixels_of_int = ped_pixels
                other_pixels_of_int = other_pixels
                row = road_pixels_l[i, 0]
                l = road_pixels_l[i, 1]
                r = road_pixels_r[i, 1]
                l = int(round(l + (r - l) / 3.5))  # only interested in ego-lane
                car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 0] == row]
                car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 1] <= r]
                car_pixels_of_int = car_pixels_of_int[car_pixels_of_int[:, 1] >= l]
                truck_pixels_of_int = truck_pixels_of_int[truck_pixels_of_int[:, 0] == row]
                truck_pixels_of_int = truck_pixels_of_int[truck_pixels_of_int[:, 1] <= r]
                truck_pixels_of_int = truck_pixels_of_int[truck_pixels_of_int[:, 1] >= l]
                ped_pixels_of_int = ped_pixels_of_int[ped_pixels_of_int[:, 0] == row]
                ped_pixels_of_int = ped_pixels_of_int[ped_pixels_of_int[:, 1] <= r]
                ped_pixels_of_int = ped_pixels_of_int[ped_pixels_of_int[:, 1] >= l]
                other_pixels_of_int = other_pixels_of_int[other_pixels_of_int[:, 0] == row]
                other_pixels_of_int = other_pixels_of_int[other_pixels_of_int[:, 1] <= r]
                other_pixels_of_int = other_pixels_of_int[other_pixels_of_int[:, 1] >= l]
                cars.append(car_pixels_of_int.shape[0])
                trucks.append(truck_pixels_of_int.shape[0])
                ped_bc.append(ped_pixels_of_int.shape[0])
                other.append(other_pixels_of_int.shape[0])
                i = i + 1

            # return mildly occluded (0.3), partly occluded (0.4), fully blocked (0.5)

            for threshold in thresholds:
                if np.sum(cars) / nbr_road_pixels > threshold:
                    if np.sum(trucks) / nbr_road_pixels > threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 1, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 1, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 1, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 1, 0, 0])
                    elif np.sum(trucks) / nbr_road_pixels <= threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 0, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 0, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([1, 0, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([1, 0, 0, 0])
                elif np.sum(cars) / nbr_road_pixels <= threshold:
                    if np.sum(trucks) / nbr_road_pixels > threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 1, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 1, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 1, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 1, 0, 0])
                    elif np.sum(trucks) / nbr_road_pixels <= threshold:
                        if np.sum(ped_bc) / nbr_road_pixels > threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 0, 1, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 0, 1, 0])
                        elif np.sum(ped_bc) / nbr_road_pixels <= threshold:
                            if np.sum(other) / nbr_road_pixels > threshold:
                                result.append([0, 0, 0, 1])
                            elif np.sum(other) / nbr_road_pixels <= threshold:
                                result.append([0, 0, 0, 0])
        except:
            result.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        result = np.array(result)
        result = np.reshape(result, [1, 12])

    elif dataset == "KITTI":
        result = np.zeros([1, 12])
        result = np.array(result)
        result = np.reshape(result, [1, 12])


    elif dataset == "Oxford":
        result = np.zeros([1, 12])
        result = np.array(result)
        result = np.reshape(result, [1, 12])

    return result


def blocking(semantic_image, img, dataset, fg_mask=None):
    """

    :param semantic_image:
    :param img:
    :param dataset:
    :param fg_mask:
    :return:
    """
    l, r = get_road_boundaries(semantic_image, img, dataset)
    l = np.array(sorted(l, key=lambda x: x[0], reverse=True))
    r = np.array(sorted(r, key=lambda x: x[0], reverse=True))
    if dataset in ["A2D2", "Apolloscape_semantic"]:
        bin_cars = filter_cars(semantic_image, dataset)
        bin_trucks = filter_trucks(semantic_image, dataset)
        bin_ped_bc = filter_ped_bc(semantic_image, dataset)
    elif dataset=="Apolloscape_stereo":
        binary_foreground = semantic_image

    # Traverse image from bottom to top until blocking factor has been reached
    for i in range(l.shape[0]):
        left = l[i, 1]  # Interested in Ego Lane
        right = r[i, 1]
        road_width = right - left
        row = l[i, 0]
        if dataset in ["A2D2", "Apolloscape_semantic"]:
            car = np.sum(np.array(bin_cars[row, left:right] == 255))
            truck = np.sum(np.array(bin_trucks[row, left:right] == 255))
            ped = np.sum(np.array(bin_ped_bc[row, left:right] == 255))
        elif dataset == "Apolloscape_stereo":
            foreground = np.sum(np.array(binary_foreground[row, left:right] == 255))
        if road_width > 0:
            if dataset == "A2D2":
                if car / road_width > 0.5:
                    return [1, 0, 0], row
                    break
                elif truck / road_width > 0.5:
                    return [0, 1, 0], row
                    break
                elif ped / road_width > 0.5:
                    return [0, 0, 1], row
                    break
                else:
                    if i == l.shape[0] - 1:
                        return [0, 0, 0], 0, 0
                    else:
                        continue

            elif dataset == "Apolloscape_semantic":
                if car / road_width > 0.5:
                    return [1, 0, 0], row, int(road_width / 2)
                    break
                elif truck / road_width > 0.5:
                    return [0, 1, 0], row, int(road_width / 2)
                    break
                elif ped / road_width > 0.5:
                    return [0, 0, 1], row, int(road_width / 2)
                    break
                else:
                    if i == l.shape[0] - 1:
                        return [0, 0, 0], 0, 0
                    else:
                        continue

            elif dataset=="Apolloscape_stereo":
                if foreground / road_width > 0.4:
                    return [1, 0, 0], row, int(road_width / 2)
                    break
                else:
                    if i == l.shape[0] - 1:
                        return [0, 0, 0], 0, 0
                    else:
                        continue
        else:
            continue


def roi_r(img, dataset="KITTI"):
    '''RECTANGULAR ROI.: Reads in the image. Then generates a rectangular ROI.
    Any other pixels are cut off. border is the percentage/100 of the height of the image,
    that marks the upper end of the ROI.'''

    if dataset in ["KITTI", "Oxford"]:
        roi_factor_y = 0.45
    elif dataset=="Apolloscape_stereo":
        roi_factor_y = 0.39
    roi_factor_x = 0.3

    yborder = int(img.shape[0] * roi_factor_y)
    xborder = int(img.shape[1] * roi_factor_x)

    img = img[yborder:, :, :]
    img = img[:, xborder:img.shape[1] - xborder, :]

    return img


def gauss(image, kernel_size=(5, 5), sigma=0, dataset="Apolloscape_stereo"):
    """
    Apply Gaussian Blur --> Reduce noise and smoothen image
    :param image: RGB image
    :param kernel_size:
    :param sigma:
    :param dataset:
    :return:
    """
    if dataset == "Apolloscape_stereo":
        # Because of the higher resolution.
        kernel_size = (9, 9)

    return cv2.GaussianBlur(image, kernel_size, sigma)


def canny(image):
    """
    Apllies canny algorithm on the image
    :param image:
    :return:
    """
    edges = cv2.Canny(image, 50, 150)
    return edges


def edge_sobel(img, dx, dy, ksize=7):
    """
    Applies Sobel edge algorithm on the image
    :param img:
    :param dx: 1 if sobelx is required + dy = 0
    :param dy: 1 if sobely is required + dx = 0
    :param ksize: is the kernelsize of the smoothing kernel
    :return:
    """

    sobel = cv2.Sobel(img, cv2.CV_64F, dx, dy, ksize=ksize)

    # calculate absolute value of the derivative
    abs_sobel = np.absolute(sobel)

    # convert the absolute value to 8bit
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    return scaled_sobel


def edge_laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # calculate absolute value of the derivative
    abs_laplacian = np.absolute(laplacian)

    # convert the absolute value to 8bit
    scaled_laplacian = np.uint8(255 * abs_laplacian / np.max(abs_laplacian))

    return scaled_laplacian


def filter_yellow(img):
    """
    Reads in a rgb color image and filters the yellow pixels.
    It uses the hsv colorspace.
    :param img: bgr color image
    :return: RGB color image with only yellow pixels
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # get copy of img
    img_copy = np.copy(img)

    img_copy[(h <= 91) | (h >= 102) | (s < 80) | (v < 200)] = 0

    return img_copy


def filter_yellow_binary(img):
    """
    Reads in a bgr color image and filters the yellow pixels.
    It uses the hsv colorspace.
    :param img: rgb color image
    :return: binary image (0 or 255) where yellow pixels are in RGB
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # set binary image
    binary = np.zeros_like(h)

    # get copy of img
    # img_copy = np.copy(img)

    # binary[(h<=91)|(h>=102)|(s<80)|(v<200)] = 0
    # binary[(h <= 81) | (h >= 112) | (s < 80) | (v < 200)] = 0

    binary[(h >= 27) & (h <= 42)] = 255

    return binary


def filter_yellow_hls_binary(img):
    """
    Reads in a rgb color image and filters the yellow pixels.
    It uses the hls colorspace.
    :param img: rgb color image
    :return: binary image (0 or 255) where yellow pixels are
    """

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    # set binary image
    binary = np.zeros_like(h)
    # binary[:, :] = 255

    # get copy of img
    # img_copy = np.copy(img)

    # binary[(h<=91)|(h>=102)|(s<80)|(v<200)] = 0
    # binary[(h <= 81) | (h >= 112) | (s < 80) | (v < 200)] = 0

    binary[(h >= 27) & (h <= 42)] = 255

    return binary


def filter_white_from_grayscale(img):
    '''

    :param img: Grayscale img
    :return:
    '''
    binary = np.zeros_like(img)
    binary[(img > 230)] = 255

    return binary


def filter_white_hls_binary(img):
    '''
    Reads in a bgr color image and filters the  white pixels.
    It uses the hls colorspace.
    :param img: bgr color image
    :return: binary grayscale image (0 or 255) where white pixels are
    '''

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    # set binary image
    binary = np.zeros_like(h)
    # binary[:, :] = 255

    # get copy of img
    # img_copy = np.copy(img)

    # binary[(h<=91)|(h>=102)|(s<80)|(v<200)] = 0
    # binary[(h <= 81) | (h >= 112) | (s < 80) | (v < 200)] = 0

    # binary[ (l>190) & (s < 200)  ] = 255
    binary[(l > 200) & (s < 150)] = 255

    return binary
