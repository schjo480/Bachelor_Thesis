import json
import pprint
import urllib
import numpy as np
import numpy.linalg as la
import open3d as o3
from os.path import join
import glob
import matplotlib.pylab as pt
import cv2
import io
import timeit

with open('/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/cams_lidars.json', 'r') as f:
    config = json.load(f)


def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + 'camera_' + file_name_image[2] + '_' + file_name_image[3] + '.png'

    return file_name_image


def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)


'''def load_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    return objects'''


def load_image(img_filename):
    return cv2.imread(img_filename)


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def project_velo_to_cam2(calib):
    P_velo2cam_ref = calib['R_T'].reshape(4, 4)  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R_rect_00'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P_rect_02'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def prepare_kitti(velo_filename, img_filename, calib):
    '''
    Prepare the Lidar data from the KITTI dataset to be able to reuse the functions defined for the A2D2 dataset.
    :param velo_filename:
    :param img:
    :param calib:
    :return: Dictionary containing the different lidar data
    '''
    #lidar_points = nx4 (x, y, z, reflectance)
    lidar = load_velo_scan(velo_filename)
    img = load_image(img_filename)

    img_height = img.shape[0]
    img_width = img.shape[1]
    points = lidar[:, 0:3]
    depth = points[:, 0]
    distance = np.linalg.norm(points, axis=1)
    reflectance = lidar[:, 3]

    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)
    pixel_coord = project_to_image(points.transpose(), proj_velo2cam2)

    #Filter out the Lidar Points on the image
    inds = np.where((pixel_coord[0, :] < img_width) & (pixel_coord[0, :] >= 0) &
                    (pixel_coord[1, :] < img_height) & (pixel_coord[1, :] >= 0) &
                    (points[:, 0] > 0)
                    )[0]

    pixel_coord = pixel_coord[:, inds]
    pixel_coord = np.transpose(pixel_coord)
    points = points[inds, :]
    depth = depth[inds]
    distance = distance[inds]
    reflectance = reflectance[inds]

    row = pixel_coord[:, 0]
    col = pixel_coord[:, 1]
    lidar_dict = {'points': points, 'reflectance': reflectance, 'row': row, 'col': col, 'depth': depth, 'distance': distance}
    return lidar_dict


#Create a 3D pointcloud and obtain the lidar points (their coordinates in 3D) which are on the free road
def create_open3d_pc(lidar, cam_image = None):
    #create opend3d point cloud pcd
    pcd = o3.geometry.PointCloud()

    #assign points to the cloud
    pcd.points = o3.utility.Vector3dVector(lidar['points'])

    #assign clolors
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance'])/(median_reflectance*5)

        #clip colors for visualisation on white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :]/255.0

    road_color = np.array([1, 0, 1])
    indices = []
    road_pts = np.zeros((100000, 3))
    pcd.colors = o3.utility.Vector3dVector(colours)
    for i in range(len(pcd.colors)):
        if (np.asarray(pcd.colors[i]) == road_color).all():
            #access pcd points in lidar coord
            road_pts[i, ] = pcd.points[i]
            indices.append(i)

    indices = np.array(indices)
    road_pts = road_pts[np.all(road_pts != 0, axis=1)]
    return pcd, indices


def undistort_image(image, cam_name):
    if cam_name in ['front_left', 'front_right', 'front_center', 'side_right', 'side_left', 'rear_center']:
        intr_mat_undist = np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


#Get segmentation information for each image
def read_image_info(file_name):
    with open(file_name, 'r') as g:
        image_info = json.load(g)
    return image_info


#Convert HSV to RGB values
def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0 - f))
    i = i%6

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


#Visualize the mapping of point cloud
def map_lidar_points_onto_image(image_orig, lidar, dataset, pixel_size = 3, pixel_opacity = 1):
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
            image[pixel_rows, pixel_cols, :] = (1. - pixel_opacity) * np.multiply(image[pixel_rows, pixel_cols, :], colours[i]) + pixel_opacity * 255 * colours[i]
    elif dataset == "KITTI":
        for i in range (len(rows)):
            cv2.circle(image, (rows[i], cols[i]), 2, color=(255*colours[i, 0], 255*colours[i, 1], 255*colours[i, 2]), thickness=-1)
    return image.astype(np.uint8)


def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + 'label_' +  file_name_semantic_label[2] + '_' + file_name_semantic_label[3] + '.png'

    return file_name_semantic_label


def filter_road(semantic_image_front_center_undistorted_bgr):
    '''
    Takes in the semantic image in bgr format, converts it to hsv and filters out the road pixels
    :param semantic_image_front_center_undistorted_bgr:
    :return: binary image with road pixel value = 0, rest = 255
    '''
    hsv = cv2.cvtColor(semantic_image_front_center_undistorted_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    binary = np.zeros_like(h)
    binary[:, :] = 255
    binary[(h != 150) | (s != 255) & (s != 184) | (v != 255) & (v != 180)] = 0

    return binary


def filter_cars(semantic_image_front_center_undistorted_bgr):
    '''

    :param semantic_image_front_center_undistorted_bgr:
    :return:
    '''
    hsv = cv2.cvtColor(semantic_image_front_center_undistorted_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    binary = np.zeros_like(h)
    binary[:, :] = 255
    binary[(h != 0) & (h != 60)] = 0 #Cars & Small Vehicles

    return binary


def filter_trucks(semantic_image_front_center_undistorted_bgr):
    '''

    :param semantic_image_front_center_undistorted_bgr:
    :return:
    '''
    hsv = cv2.cvtColor(semantic_image_front_center_undistorted_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    binary = np.zeros_like(h)
    binary[:, :] = 255
    binary[(h != 15) & (h != 19) & (h != 26) & (h != 120) & (h != 30) & (h != 15)] = 0

    return binary


def filter_ped_bc(semantic_image_front_center_undistorted_bgr):
    '''

    :param semantic_image_front_center_undistorted_bgr:
    :return:
    '''

    hsv = cv2.cvtColor(semantic_image_front_center_undistorted_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    binary = np.zeros_like(h)
    binary[:, :] = 255
    binary[(h != 14) & (h != 11) & (h != 10) & (h != 135) & (h != 159) & (h != 160)] = 0

    return binary


def get_road_pixels(semantic_image_front_center_undistorted_bgr):
    '''
    :param semantic_image_front_center_undistorted_bgr:
    :return: Pixel coordinates of road pixels
    '''

    binary = filter_road(semantic_image_front_center_undistorted_bgr)

    road_pixels_l = np.empty(shape=(0, 2), dtype=int)
    road_pixels_r = np.empty(shape=(0, 2), dtype=int)
    for i in range(620, binary.shape[0]):
        for j in range(binary.shape[1]):
            if binary[i, j] == 255:
                road_pixels_l = np.append(road_pixels_l, np.array([[i, j]]), axis=0)
                i =+ 1 #We are only interested in the height of the pixels which are on the road

    for i in range(620, binary.shape[0]):
        for j in range(binary.shape[1]-1, 0, -1):
            if binary[i, j] == 255:
                road_pixels_r = np.append(road_pixels_r, np.array([[i, j]]), axis=0)
                i =+ 1 #We are only interested in the height of the pixels which are on the road

    road_pixels_l = np.asarray(road_pixels_l)
    road_pixels_r = np.asarray(road_pixels_r)
    return road_pixels_l, road_pixels_r


def blocking_factors(semantic_image_front_center_undistorted_bgr):
    '''

    :param semantic_image_front_center_undistorted_bgr:
    :return:
    '''
    road_pixels_l, road_pixels_r = get_road_pixels(semantic_image_front_center_undistorted_bgr)
    #Cars
    bin_cars = filter_cars(semantic_image_front_center_undistorted_bgr)
    bin_trucks = filter_trucks(semantic_image_front_center_undistorted_bgr)
    bin_ped_bc = filter_ped_bc(semantic_image_front_center_undistorted_bgr)
    cars = []
    trucks = []
    ped_bc = []
    nbr_road_pixel_cols = []
    for i in range(road_pixels_l.shape[0]):
        row = road_pixels_l[i, 0]
        l = road_pixels_l[i, 1]
        r = road_pixels_r[i, 1]
        l = int(round(l + (r - l)/2.5)) #only interested in ego-lane
        nbr_road_pixel_cols.append(r - l)
        for j in range(l, r):
            if bin_cars[row, j] == 255:
                cars.append(1)
            if bin_trucks[row, j] == 255:
                trucks.append(1)
            if bin_ped_bc[row, j] == 255:
                ped_bc.append(1)

    nbr_road_pixels = np.sum(nbr_road_pixel_cols) #Used to calculate what PROPORTION of the image is occupied by cars/trucks/pedestrians

    if np.sum(cars)/nbr_road_pixels > 0.4:
        if np.sum(trucks)/nbr_road_pixels > 0.022:
            if np.sum(ped_bc)/nbr_road_pixels > 0.04:
                return 1, 1, 1
            elif np.sum(ped_bc)/nbr_road_pixels <= 0.04:
                return 1, 1, 0
        elif np.sum(trucks)/nbr_road_pixels <= 0.022:
            if np.sum(ped_bc) / nbr_road_pixels > 0.04:
                return 1, 0, 1
            elif np.sum(ped_bc)/nbr_road_pixels <= 0.04:
                return 1, 0, 0
    elif np.sum(cars)/nbr_road_pixels <= 0.4:
        if np.sum(trucks)/nbr_road_pixels > 0.022:
            if np.sum(ped_bc)/nbr_road_pixels > 0.04:
                return 0, 1, 1
            elif np.sum(ped_bc)/nbr_road_pixels <= 0.04:
                return 0, 1, 0
        elif np.sum(trucks)/nbr_road_pixels <= 0.022:
            if np.sum(ped_bc) / nbr_road_pixels > 0.04:
                return 0, 0, 1
            elif np.sum(ped_bc)/nbr_road_pixels <= 0.04:
                return 0, 0, 0


#TAKES TOO LONG ==> IMPROVEMENT NEEDED maybe not up to the bottom as only interested in farther away pixels
'''def select_road_pixels(semantic_image_front_center_undistorted):
    road_color = np.array([255, 0, 255])
    road_pixels = np.empty(shape=(0,2), dtype=int)
    for i in range(600, semantic_image_front_center_undistorted.shape[0]):
        for j in range(400, 1500):
            if (semantic_image_front_center_undistorted[i, j] == road_color).all():
                road_pixels = np.append(road_pixels, np.array([[i, j]]), axis=0)
    road_pixels = np.asarray(road_pixels)
    return road_pixels
'''

'''def get_y_value_of_furthes_lidar_road_pt(road_pts, lidar):
    #Compute the maximal depth of detected road points in m
    max_distance_road = np.max(np.linalg.norm(road_pts, axis=1))
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)
    distance = lidar['distance']

    #Get row index where distance == max_distance_road
    index = np.where(np.around(distance, 10) == np.round(max_distance_road, 10))[0] #CHECK THIS BECAUSE WEIRD PIXEL/METER VALUES IN EVALUATION ==> Changed rounding numbers from 3 to 10
    x_value = cols[index]
    y_value = rows[index]

    return y_value, max_distance_road
'''

def roi_p5(img):
    '''ROI in a polygonal shape with 5 corners. Reads in an image and the outer points of the polygon, which shape the ROI.
    There is a preset polygon, which can be used by not giving any input points.
    Then it generates a new image out of the ROI, which gets returned.'''

    # preset corner points
    points = ([img.shape[0], 0], [img.shape[0] * 0.6, 1], [img.shape[0] * 0.5, img.shape[1] * 0.5],
              [img.shape[0] * 0.6, img.shape[1] - 1], [img.shape[0], img.shape[1] - 0])
    # Definition of the points [y,x]
    lb = points[0]  # lb = left bottom
    lt = points[1]  # lt = left top
    apx = points[2]  # apx = apex
    rt = points[3]  # rt = right top
    rb = points[4]  # rb = right bottom

    # define the lines between the points
    lb_lt = np.polyfit((lb[1], lt[1]), (lb[0], lt[0]), 1)  # from lb to lt
    lt_apx = np.polyfit((lt[1], apx[1]), (lt[0], apx[0]), 1)
    apx_rt = np.polyfit((apx[1], rt[1]), (apx[0], rt[0]), 1)
    rt_rb = np.polyfit((rt[1], rb[1]), (rt[0], rb[0]), 1)
    rb_lb = np.polyfit((rb[1], lb[1]), (rb[0], lb[0]), 1)
    print(lb_lt)
    # find region inside the lines
    xsize = img.shape[1]  # ysize of img -> needed for np.meshgrid
    ysize = img.shape[0]

    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * lb_lt[0] + lb_lt[1])) & \
                        (YY > (XX * lt_apx[0] + lt_apx[1])) & \
                        (YY > (XX * apx_rt[0] + apx_rt[1])) & \
                        (YY > (XX * rt_rb[0] + rt_rb[1])) & \
                        (YY < (XX * rb_lb[0] + rb_lb[1]))
    img[~region_thresholds] = [0, 0, 0]  # set pixels outside the region to [0,0,0]

    return img

def roi_p3(img):
    '''
    ROI in a polygonal shape with 3 points + left and right bottom points of the image.
    Reads in an image and the 3 upper points of the polygon, which shape the ROI.
    Then it generates a new image out of the ROI, which gets returned.
    '''

    # preset corner points
    points = ([img.shape[0] * 0.6 + 30, 1],  # left (top)
              [img.shape[0] * 0.5 + 40, img.shape[1] * 0.5],  # apex
              [img.shape[0] * 0.6 + 30, img.shape[1]])  # right (top)
    # Definition of the points [y,x]

    lt = points[0]  # lt = left top
    apx = points[1]  # apx = apex
    rt = points[2]  # rt = right top

    # define the lines between the points

    lt_apx = np.polyfit((lt[1], apx[1]), (lt[0], apx[0]), 1)
    apx_rt = np.polyfit((apx[1], rt[1]), (apx[0], rt[0]), 1)

    # find region inside the lines
    xsize = img.shape[1]  # ysize of img -> needed for np.meshgrid
    ysize = img.shape[0]

    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * lt_apx[0] + lt_apx[1])) & \
                        (YY > (XX * apx_rt[0] + apx_rt[1]))

    img[~region_thresholds] = [0, 0, 0]  # set pixels outside the region to [0,0,0]

    return img


def roi_r(img, border=0.5):
    '''RECTANGULAR ROI.: Reads in the image. Then generates a rectangular ROI.
    Any other pixels are cut off. border is the percentage/100 of the height of the image,
    that marks the upper end of the ROI.'''

    yborder = int(img.shape[0] * border)
    img = img[yborder:, :, :]
    return img


# Blurring (s=standard; m=median; g=gaussian; b=bilateral)
def blur_s(img, kernelsize=(5, 5)):
    '''
    Blurs a image with a normalized kernel of the size "kernelsize".
    Input: Image and the size of the kernel (by default 5x5)
    Output: Blurred image.
    '''

    img = cv2.blur(img, kernelsize)

    return img


def blur_m(img, kernelsize=5):
    '''
    Blurs a image with a median filter.
    Pixel value gets the average of the surrounding "in kernel pixels".
    Good for Salt and Pepper noise.
    Input: Image and the size of the quadratic kernel (by default 5)
    Output: Blurred image.
    '''

    img = cv2.medianBlur(img, kernelsize)

    return img


def blur_g(img, kernelsize=(5, 5), sigmaX=0):
    '''
    Blurs a image with a gaussian kernel of the size "kernelsize".
    Input: Image and the kernel size (by default 5x5), sigmaX (by default 0)
    Output: Blurred image.
    '''

    img = cv2.GaussianBlur(img, kernelsize, sigmaX)

    return img


def blur_b(img, kernelradius=10, sigmaColor=50, sigmaSpace=50):
    '''
    Bilateral Filtering.
    Blurs an image. Does not blur edges like the other methods. But therefore needs more calculation time.
    Input: image and kernelradius, sigmaColor, sigmaSpace with default settings
    Output: blurred Image
    '''

    img = cv2.bilateralFilter(img, kernelradius, sigmaColor, sigmaSpace)

    return img


def colorspace(img, destination, origin='bgr'):
    '''
    Converts Images into different colorspaces.
    Reads in a image, as well as the destination colorspace and origin colorspace.
    Covers only BGR/RGB/Gray/HlS/HSV Colorspace transformations.
    '''

    # Origin BGR
    if origin == 'bgr':
        # BGR 2 Gray
        if destination == 'gray' or destination == 'grey':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # BGR 2 HLS
        elif destination == 'hls':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # BGR 2 HSV
        elif destination == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # BGR 2 RGB
        elif destination == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        else:
            print('WRONG DESTINATION COLORSPACE INPUT!')

    # Origin HLS
    elif origin == 'hls':
        # HLS 2 BGR
        if destination == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)

        # HLS 2 RGB
        elif destination == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)

        else:
            print('WRONG DESTINATION COLORSPACE INPUT!')

    # Origin HSV
    elif origin == 'hsv':
        # HSV 2 BGR
        if destination == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # HSV 2 RGB
        elif destination == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        else:
            print('WRONG DESTINATION COLORSPACE INPUT!')

    # Origin RGB
    elif origin == 'rgb':
        # RGB 2 BGR
        if destination == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # RBG 2 Gray
        elif destination == 'gray' or destination == 'grey':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # RGB 2 HLS
        elif destination == 'hls':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        # RGB 2 HSV
        elif destination == 'hsv':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        else:
            print('WRONG DESTINATION COLORSPACE INPUT!')

    # Origin Gray
    elif origin == 'gray' or origin == 'grey':  # ATTENTION STAYS A GRAY IMAGE BUT HAS 3CHANNELS AFTERWARDS WITH THE SAME VALUE
        # Gray 2 BGR
        if destination == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Gray 2 RGB
        elif destination == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        else:
            print('WRONG DESTINATION COLORSPACE INPUT!')

    else:
        print('WRONG ORIGIN COLORSPACE INPUT!')

    return img


def filter_yellow(img):
    '''
    Reads in a bgr color image and filters the yellow pixels.
    It uses the hsv colorspace.
    :param img: bgr color image
    :return: bgr color image with only yellow pixels
    '''

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # get copy of img
    img_copy = np.copy(img)

    img_copy[(h <= 91) | (h >= 102) | (s < 80) | (v < 200)] = 0

    return img_copy


def filter_yellow_binary(img):
    '''
    Reads in a bgr color image and filters the yellow pixels.
    It uses the hsv colorspace.
    :param img: bgr color image
    :return: binary grayscale image (0 or 255) where yellow pixels where
    '''

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # set binary image
    binary = np.zeros_like(h)
    binary[:, :] = 255

    # get copy of img
    # img_copy = np.copy(img)

    # binary[(h<=91)|(h>=102)|(s<80)|(v<200)] = 0
    # binary[(h <= 81) | (h >= 112) | (s < 80) | (v < 200)] = 0

    binary[(h <= 81) | (h >= 102)] = 0

    return binary


def filter_yellow_hls_binary(img):
    '''
    Reads in a bgr color image and filters the yellow pixels.
    It uses the hls colorspace.
    :param img: bgr color image
    :return: binary grayscale image (0 or 255) where yellow pixels are
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

    binary[(h >= 91) & (h <= 102)] = 255

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
    binary[(l > 190)] = 255

    return binary


def adaptive_filter_white_hls_binary(img, percentage=0.01):
    '''
    Reads in a bgr color image and filters the  white pixels with an adaptive threshold.
    It uses the hls colorspace and filters in the Luminance channel.
    The threshold is performed by setting a percentage of the brightest pixels, that should go through.
    :param img: bgr color image
    :return: binary grayscale image (0 or 255) where white pixels are
    '''

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    l = hls[:, :, 1]

    # set binary image
    binary = np.zeros_like(l)

    # get histogram
    hist, bins = np.histogram(l, 256, [0, 256])

    summe = 0  # initialise variable summe
    maximum = sum(hist) * percentage  # amount of pixels that are included in the percentage

    # get brightness border for x-percent of brightest pixels
    for i in range(1, 257):  # go from the brightest to darkest
        summe += hist[-i]
        if summe >= maximum:
            border = 256 - i
            break

    binary[l > border] = 255  # binary[(l > 190) ] = 255
    return binary


# Treshold (bgr=only lower threshold, region=has both lower and upper threshold)
def thresh(img, threshold=(200, 200, 200)):
    '''
    BGR or 1channel Threshold.
    Takes in a rgb or 1channel image and sets every pixel value to 0 if it is smaller (1 if bigger) than the threshold.
    If image has only 1 channel threshold[0] is used.
    Thresholds are preset.
    '''

    if len(img.shape) == 3:  # >1 channel image
        b_channel = img[:, :, 0]
        g_channel = img[:, :, 1]
        r_channel = img[:, :, 2]

        binary = np.zeros_like(b_channel)

        binary[(b_channel > threshold[0]) & (g_channel > threshold[1]) & (r_channel > threshold[2])] = 255
    elif len(img.shape) == 2:  # 1 channel image

        binary = np.zeros_like(img)
        binary[img > threshold[0]] = 255

    return binary


def thresh_region(img, lower_threshold=(0, 130, 130), upper_threshold=(100, 255, 255)):
    '''
    Takes in a 1 or 3 channel image.
    Checks if the pixel values are in between the lower and upper threshold.
    If the values in any channel of the pixel is out of bounds, the binary value of the pixel is set to 0.
    if the image has only 1 channel lower_-/upper_threshold[0] is used
    Gives out a binary image.
    '''

    if len(img.shape) == 3:  # >1 channel image

        b_channel = img[:, :, 0]
        g_channel = img[:, :, 1]
        r_channel = img[:, :, 2]

        binary = np.zeros_like(b_channel)

        binary[(b_channel > lower_threshold[0]) & (b_channel < upper_threshold[0]) &
               (g_channel > lower_threshold[1]) & (g_channel < upper_threshold[1]) &
               (r_channel > lower_threshold[2]) & (r_channel < upper_threshold[2])] = 255
    elif len(img.shape) == 2:  # 1 channel image

        binary = np.zeros_like(img)
        binary[(img > lower_threshold[0]) & (img < upper_threshold[0])] = 255

    return binary


# Edge Detection
def edge_canny(img, low_threshold=50, high_threshold=150):
    '''
    Performs the canny edge detection with the cv2 method.
    :param img: Can also be a color image.
    :param low_threshold: Pixels with gradient values lower than this threshold are dumped anyways(always).
    :param high_threshold: Pixels with gradient values higher than this threshold are considered a edges(always).
    ==> pixels with gradient values in between those thresholds are considered a edge if they touch a pixel that is a edge
    by means of high_threshold. Otherwise it is dumped.
    :return img: Returns a 8bit edge image with values of 0 or 255.
    '''

    img = cv2.Canny(img, low_threshold, high_threshold)

    return img


def edge_sobel(img, dx, dy, ksize=7):
    '''

    :param img:
    :param dx: 1 if sobelx is required + dy = 0
    :param dy: 1 if sobely is required + dx = 0
    :param ksize: is the kernelsize of the smoothing kernel
    :return:
    '''

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


# Logic Operation
def logic_and(img1, img2, img3=None, img4=None):
    '''
    Performs a logic AND with 2 to 4 images of the same size.
    The images mustn't have more than one channel.
    Usually Binary images are used.
    Output is a single one channel image with 0 or 1 as values.
    So only if the pixelvalue is true (>0) in every of the input images, the output image pixel will be true.
    '''

    # set up image of the same size
    binary = np.zeros_like(img1)

    if type(img3) is not np.ndarray and type(img4) is not np.ndarray:  # only img1 and img2 are there as input
        # compare pixel values ( >0 ??? )
        binary[(img1 > 0) & (img2 > 0)] = 255

    elif type(img4) is not np.ndarray:  # img1/2&3 are there as input
        binary[(img1 > 0) & (img2 > 0) & (img3 > 0)] = 255

    elif type(img1) is np.ndarray and \
                    type(img2) is np.ndarray and \
                    type(img3) is np.ndarray and \
                    type(img4) is np.ndarray:  # there are 4 image inputs
        binary[(img1 > 0) & (img2 > 0) & (img3 > 0) & (img4 > 0)] = 255

    else:  # wrong input
        print('WRONG INPUT! CHECK THE IMAGE INPUT!')

    return binary


def logic_or(img1, img2, img3=None, img4=None):
    '''
    Performs a logic OR with 2 to 4 images of the same size.
    The images mustn't have more than one channel.
    Usually Binary images are used.
    Output is a single one channel image with 0 or 1 as values.
    So only if the pixelvalue is true (>0) in at least one of the input images, the output image pixel will be true.
    '''
    # set up image of the same size
    binary = np.zeros_like(img1)

    if type(img3) is not np.ndarray and type(img4) is not np.ndarray:  # only img1 and img2 are there as input
        # compare pixel values ( >0 ??? )
        binary[(img1 > 0) | (img2 > 0)] = 255

    elif type(img4) is not np.ndarray:  # img1/2&3 are there as input
        binary[(img1 > 0) | (img2 > 0) | (img3 > 0)] = 255

    elif type(img1) is np.ndarray and \
                    type(img2) is np.ndarray and \
                    type(img3) is np.ndarray and \
                    type(img4) is np.ndarray:  # there are 4 image inputs
        binary[(img1 > 0) | (img2 > 0) | (img3 > 0) | (img4 > 0)] = 255

    else:  # wrong input
        print('WRONG INPUT! CHECK THE IMAGE INPUT!')

    return binary


# postprocessing
def det_line_steps(img1, img2, step=5):
    '''
    Detects lines by canny and line image comparison (if canny-edge and next to it a colored line ->detected).
    Checks only certain image rows.
    :param img1: Canny image 1channel
    :param img2: line image 1channel
    :param step: distance between the checked rows
    :return: semi-binary image (pixelvalues of 0 or 255) 1channel
    '''
    binary = np.zeros((img1.shape[0], img1.shape[1], 1), np.uint8)
    logic_and = np.zeros_like(img1)
    logic_and[(img1 > 0) & (img2 > 0)] = 255
    for i in range(0, img1.shape[0], step):  # certain rows

        j = 0  # initialization of j

        # draw red lines to show wich lines where checked
        cv2.line(binary, (0, i), (binary.shape[1], i), 0)

        # check if pixel values are greater than 0 in both pictures
        while j < img1.shape[1] - 1:  # columns
            if img1[i, j] > 0 and img2[i, j + 1] > 0 and logic_and[i, j] > 0:
                binary[i, j, :] = 255
                j += 0  # nahe doppelwertungen verhindern durch erhöhung des spaltenindices
            elif img1[i, j] > 0 and img2[i, j - 1] > 0 and logic_and[i, j] > 0:  # rechte kante der linie
                binary[i, j, :] = 255
            j += 1
    return binary


def get_road_pixels_KIT(line_img, img):
    '''

    :param line_img: result of det_line_step function
    :return: the left and right most pixels of road
    '''
    road_pixels_l = np.empty(shape=(0, 2), dtype=int)
    road_pixels_r = np.empty(shape=(0, 2), dtype=int)

    for i in range(line_img.shape[0]-1):
        control = []
        for j in range(line_img.shape[1]):
            if line_img[i, j] == 255:
                for k in range(j+1, j+40):
                    control.append(np.int(line_img[i, k]))
            if np.sum(control) > 3*255:
                j =+ 1
            else:
                road_pixels_l = np.append(road_pixels_l, np.array([[i, j]]), axis=0)
                i =+ 1

    for i in range(line_img.shape[0]-1):
        for j in range(line_img.shape[1]-1, 0, -1):
            if line_img[i, j] == 255:
                road_pixels_r = np.append(road_pixels_r, np.array([[i, j]]), axis=0)
                i =+ 1

                '''k = 0
                counter = 0
                while k > -20:
                    if line_img[i, k] == 255:
                        counter =+ 1
                    k =- 1
                if counter < 6:
                    road_pixels_r = np.append(road_pixels_r, np.array([[i, j+k]]), axis=0)
                    i =+ 1'''
    #Transform from cropped roi coordinates to total image coordinates
    factor = int(0.5*img.shape[0])
    '''road_pixels_l[:, 0] += factor
    road_pixels_r[:, 0] += factor'''
    road_pixels_l = np.asarray(road_pixels_l)
    road_pixels_r = np.asarray(road_pixels_r)
    #CALCULATE THE PERCENTILES TO GET ROAD ENDS

    return road_pixels_l, road_pixels_r

'''def detect_road(line_img, rgb_img):
    width_treshold = 300 #decreases for every increase in img row, because the road gets narrower
    img_height = line_img.shape[0]
    img_width = line_img.shape[1]
    left = np.min(np.where(line_img[img_height-1, :] == 255)[0])
    right = np.max(np.where(line_img[img_height-1, :] == 255)[0])
    mid_point = int((right-left)/2)
    print(left, right, mid_point)'''

def det_line_1lane_init(canny, lineimg1, lineimg2, step=8):
    '''
    Detects lines on moving stripes (more than one striperow with 2 lines to detect)
    The stripes have a various width (linear from max and min width over y)->inside the stripe list (2nd input of every row).
    Initialises the starting position of the stripes.(not manually set)
    '''
    output = np.zeros((canny.shape[0], canny.shape[1], 3), np.uint8)

    def frame_counter():
        '''Defines the frame counter if not existing already. And increases it every frame.'''
        if 'frame_nbr' not in globals():  # frame number counter / definition
            global frame_nbr
            frame_nbr = 1
        else:
            frame_nbr += 1
        return

    # "points" list's handling function
    def enter_new_midpoint(value, spot, nmax=25):
        '''
        Enters a new midpoint(value) into the list.
        Also shifts the old points to the "right" -> so the newest point is in the first listplace.
        :param value: value that shall be entered as the last detected midpoint
        :param spot: spot is the points[x][1-z]
        :param nmax: maximum of the allowed spaces for former midpoints
        :return:
        '''

        # append a new space if less than the allowed spaces exist until sufficient
        while len(spot) < nmax:
            spot.append(None)

        # shift old points one to the right
        for i in range(nmax - 1, 0, -1):  # counts down from the last space index to 1
            spot[i] = spot[i - 1]  # shift the value to the space one on the right

        spot[0] = value

        return

    def get_lastvalid_midpoint(spot):
        '''
        returns the last value of the midpoints that is not None.
        spot is the points[x][1-z] and asks for a special midpoint
        :param self:
        :return:
        '''
        for point in spot:
            if point is not None:
                break

        return point

    def setup():
        '''sets up all lists.'''

        def stripe_width(y, ymax=canny.shape[0], wmin=60, wmax=120):
            '''
            calculates the linear width of the stripes. Depending on the row hight y.
            Smaller width on the top of the image.
            -> width = m*y+t m=(wmax-wmin)/ymax  t=wmin ymax=img.shape[1]
            :param y:
            :param wmin: minimum width of stripes
            :param wmax: maximum width
            :return: width of the row
            '''
            width = int(((wmax - wmin) / ymax) * y + wmin)
            return width

        if "stripe" not in globals():
            global stripe  # list for checking stripe's midpoints
            stripe = []
            global points  # list for detected lane markings points (None if not detected)
            points = []

            global det_counter  # list which has info if stripe ever detected a point (True/False)
            det_counter = []

            # set stripe default values for all rows
            for row in range(0, canny.shape[0] - 1, step):  # iteration through rows
                stripe_mid = int(
                    canny.shape[1] / 2)  # stripes are in the middle of the width (going to move outwards)
                width = stripe_width(row)
                stripe.append([row, width, stripe_mid, stripe_mid])  # stripe = [[row, width, x_left, x_right], ...]

                det_counter.append([None, None])  # set the counter to None

                points.append([row, [], []])  # points = [[row, [last points left], [last points right]], ...]
                # set last values of points to none
                for i in range(1, 3):  # left and right iteration
                    enter_new_midpoint(None, points[int(row / step)][i])

    def draw_stripes():
        # def output image (NOT NECESSARY ONLY FOR DEVELOPMENT AND VISUALISATION)
        output = np.zeros((canny.shape[0], canny.shape[1], 3), np.uint8)

        for row in stripe:
            y = row[0]
            w = row[1]
            mid_left = row[2]
            mid_right = row[3]
            output[y, int(mid_left - w / 2):int(mid_left + w / 2), 2] = 255
            output[y, int(mid_right - w / 2):int(mid_right + w / 2), 2] = 255

        return output

    def outmoving_stripe(pixelspeed=9):
        '''Moves the stripes to the left / right until they have found a marking.'''
        for i, row in enumerate(stripe):  # iterate through all stripes
            if get_lastvalid_midpoint(
                    points[i][1]) is None:  # if point is None(not detected)->move left stripe to the left
                if det_counter[i][0] is None:  # move only if stripe never detected something
                    stripe[i][2] -= pixelspeed
            if get_lastvalid_midpoint(points[i][2]) is None:  # right point(right lane marking)
                if det_counter[i][1] is None:
                    stripe[i][3] += pixelspeed

        return

    def end_outmoving_stripe():
        '''Sets all stripes counter to True that have a lower and higher stripe that already detected something'''
        # get highest and lowest stripe that already detected something for both sides
        global end_initialisation_framenbr
        end_initialisation_framenbr = 53

        highest_left = None
        lowest_left = None
        highest_right = None
        lowest_right = None
        for i, info in enumerate(det_counter):
            if info[0] == True:  # if left stripe in this row ever detected anything
                if lowest_left is None:
                    lowest_left = i
                highest_left = i
            if info[1] == True:  # right stripe
                if lowest_right is None:
                    lowest_right = i
                highest_right = i
        # set all spaces of det_counter to True between the highest and lowest already detected points
        if lowest_right and highest_right and lowest_left and highest_left:
            for i in range(lowest_left, highest_left + 1):
                det_counter[i][0] = True
            for i in range(lowest_right, highest_right + 1):
                det_counter[i][1] = True

        # set all the det_counter values to True after specific amount of frames
        if frame_nbr == end_initialisation_framenbr:  # canny.shape[1]/8/2:
            for i, row in enumerate(det_counter):  # all rows
                for it in range(0, 2):  # left and right
                    det_counter[i][it] = True
        return

    def runnaway_stripe(maxdist=100):
        if frame_nbr > end_initialisation_framenbr:  # check for wrong stripes after initialisation
            # iterate all rows of stripe

            # mid to top
            for i in range(int(len(stripe) / 2), 0, -1):
                for it in range(2, 4):  # left and right
                    # above too far away/below ok->top stripe is wrong
                    if abs(stripe[i][it] - stripe[i - 1][it]) > maxdist and abs(
                                    stripe[i][it] - stripe[i + 1][it]) < maxdist:
                        # set stripe above to same value

                        stripe[i - 1][it] = stripe[i][it]
            # mid+1 to bottom
            for i in range(int(len(stripe) / 2) + 1, len(stripe) - 1):
                for it in range(2, 4):  # left and right
                    # below too faar away/above ok->top stripe is wrong
                    if abs(stripe[i][it] - stripe[i + 1][it]) > maxdist and abs(
                                    stripe[i][it] - stripe[i - 1][it]) < maxdist:
                        # set stripe above to same value
                        stripe[i + 1][it] = stripe[i][it]

        return

    def detector():
        lines = lineimg1 + lineimg2
        for idx1, entry in enumerate(stripe):  # iterate over all rows (idx1 ist der Index des Eintrags in stripe)

            # Stripes: draw and calculate
            for idx2, mittelpkt in enumerate(
                    entry[2:]):  # iterate over all midpoints(stripes) of the row (therefore skip y and w)


                # calculate stripe points
                pkt_l = int(
                    mittelpkt - 0.5 * entry[1])  # left/right point of the stripe (entry[1] = width of stripe)
                pkt_r = int(mittelpkt + 0.5 * entry[1])

                # limitation of the stripe position (has to be in the image boundaries)
                if pkt_r >= lineimg1.shape[
                    1]:  # if the stripe points are out of the image set to left or right max position
                    pkt_r = lineimg1.shape[1] - 1
                    pkt_l = pkt_r - entry[1]
                elif pkt_l < 0:
                    pkt_l = 0
                    pkt_r = entry[1]

                y = entry[0]

                # draw stripe (visualisation)
                # cv2.line(output, (pkt_l, y), (pkt_r, y), (0, 0, 255))  # entry[0]=y

                # checking
                list_hits = []  # list for all the hits for one loop

                # check all pixels on the stripe for lane marking points
                for x in range(pkt_l, pkt_r):  # iterate over x

                    if canny[y, x] > 0 and (lines[y, x + 1] > 0 or lines[y, x - 1] > 0):  # check left & right edge
                        # output[y, x, :] = 255
                        list_hits.append(x)  # append new hit to list

                # calculate new midpoint & save in stripe list
                if len(list_hits):  # if entries are in list -> else division thru 0
                    midpoint = int(sum(list_hits) / len(list_hits))
                    output[y, midpoint, :] = 255
                    stripe[idx1][
                        idx2 + 2] = midpoint  # (idx2+2 ,because first 2 entries of stripe entries are y & width)
                    # detected -> put point into points list
                    enter_new_midpoint(midpoint, points[idx1][idx2 + 1])

                    # set the detection counter to True (stops the moving process of the stripe)
                    det_counter[idx1][idx2] = True
                else:  # if no point on stripe detected set points value to None

                    enter_new_midpoint(None, points[idx1][idx2 + 1])

    # Call all functions
    frame_counter()  # increase frame nbr
    setup()  # setup the lists...
    output = draw_stripes()
    detector()
    outmoving_stripe()
    end_outmoving_stripe()
    runnaway_stripe()

    return output

def connect_points(img):
    '''
    connects the detected points of each line marking with lines inbetween
    :param img:
    :return:
    '''
    global points_graph

    for iter1 in range(1, len(points[0])):  # perform on all line markings

        liste = []  # list to temporaryly store the selected points into

        # put all points into the list "liste" if they are not None
        for row in points:
            if row[iter1]:  # only if not None
                liste.append([row[iter1], row[0]])

        points_graph = np.array([liste], np.int32)

        # connect the points
        cv2.polylines(img, [points_graph], False, (0, 0, 255), 8)

    return img

def connect_points_v2(img):
    '''
    connects the detected points of each line marking with lines inbetween
    (
    v2: supports the new list size of points, which contains former midpoints.
        ->blinking lines are reduced because if the value of a lane marking point is None the last valid point is used
    :param img:
    :return:
    '''

    # "points" list's handling function
    def get_lastvalid_midpoint(spot):
        '''
        returns the last value of the midpoints that is not None.
        spot is the points[x][1-z] and asks for a special midpoint
        :param self:
        :return:
        '''
        for point in spot:
            if point is not None:
                break

        return point

    global points_graph

    for iter1 in range(1, len(points[0])):  # perform on all line markings

        liste = []  # list to temporaryly store the selected points into

        # put all points into the list "liste" if they are not None
        for row in points:
            if get_lastvalid_midpoint(row[iter1]):  # only if one of the former values is not None
                # liste.append([row[iter1],row[0]])
                liste.append([get_lastvalid_midpoint(row[iter1]), row[0]])

        points_graph = np.array([liste], np.int32)

        # connect the points
        cv2.polylines(img, [points_graph], False, (0, 0, 255), 5)

    return img

def connect_points_v3(img):
    '''
    connects the area between the detected points of the neighboring line markings
    (
    v2: supports the new list size of points, which contains former midpoints.
        ->blinking lines are reduced because if the value of a lane marking point is None the last valid point is used
    v3: connects the left and right lane marking to a trapeze & fills the area
    :param img:
    :return:
    '''

    # "points" list's handling function
    def get_lastvalid_midpoint(spot):
        '''
        returns the last value of the midpoints that is not None.
        spot is the points[x][1-z] and asks for a special midpoint
        :param self:
        :return:
        '''
        for point in spot:
            if point is not None:
                break

        return point

    global points_graph

    for iter1 in range(1,
                       len(points[0]) - 1):  # perform on all line markings (stop at the next-to-last lane marking)

        liste = []  # list to temporaryly store the selected points into

        # put all points into the list "liste" if they are not None
        for row in points:  # append points of left lane marking
            if get_lastvalid_midpoint(row[iter1]):  # only if one of the former values is not None

                liste.append([get_lastvalid_midpoint(row[iter1]), row[0]])

        for row in points[::-1]:  # append points of right lane marking
            if get_lastvalid_midpoint(row[iter1 + 1]):  # only if one of the former values is not None

                liste.append([get_lastvalid_midpoint(row[iter1 + 1]), row[0]])

        points_graph = np.array([liste], np.int32)

        # connect and fill the points
        img2 = np.zeros_like(img)
        cv2.fillConvexPoly(img2, points_graph, (0, 50, 0))
        cv2.polylines(img, [points_graph], True, (0, 0, 255), 5)
        img = img + img2

    return img

def regression(img, step=10, degree=2):
    '''
    Regression of the detected line points. Degree of the regression function is a input(1-4allowed).
    Searches for the best fitting function for the given line points.
    Uses the global list "points" and generates new points of the function on a special y axis step width.
    Then draws the graph onto the image by connecting the points with lines.
    The entries of the points list have to be sorted to the belonging line marking
    (so from the 2nd content of the row lists in points of all rows have to belong to one line marking and so on)
    (
    v2: supports the new list size of points, which contains former midpoints.
        ->blinking or twirling lines are reduced because if the value of a lane marking point is None the last valid point is used
    v3: connects the left and right lane marking to a trapeze & fills the area
    )
    :return: img
    '''

    # "points" list's handling function
    def get_lastvalid_midpoint(spot):
        '''
        returns the last value of the midpoints that is not None.
        spot is the points[x][1-z] and asks for a special midpoint
        :param self:
        :return:
        '''
        for point in spot:
            if point is not None:
                break

        return point

    global points_graph

    for iter1 in range(1, len(
            points[0]) - 1):  # do cubic regression on all line markings (stop at the next-to-last lane marking)

        # define all needed lists (inside the for loop otherwise the list wont be reset for next line markings)
        liste = []

        for iter2 in range(0, 2):  # iterate thru left and right marking of the lane
            x = []
            y = []

            # Put points in lists
            for row in points:
                if get_lastvalid_midpoint(row[iter1 + iter2]):  # only if one of the former values is not None
                    y.append(row[0])
                    x.append(get_lastvalid_midpoint(row[iter1 + iter2]))

            # Calculate the coefficients of the polynomial function
            z = np.polyfit(y, x, degree)

            # define regression function
            if degree == 1:  # linear regression
                def f(y, a=z[0], b=z[1]):
                    x = a * y + b
                    return int(x)
            elif degree == 2:  # quadratic regression
                def f(y, a=z[0], b=z[1], c=z[2]):
                    x = a * y ** 2 + b * y + c
                    return int(x)
            elif degree == 3:  # cubic regression
                def f(y, a=z[0], b=z[1], c=z[2], d=z[3]):
                    x = a * y ** 3 + b * y ** 2 + c * y + d
                    return int(x)
            elif degree == 4:
                def f(y, a=z[0], b=z[1], c=z[2], d=z[3], e=z[4]):
                    x = a * y ** 4 + b * y ** 3 + c * y ** 2 + d * y + e
                    return int(x)
            else:  # not supported degree
                print("Degree not supported. Change Input: 'degree'!")

            # define the starting row of the to be drawn marking line
            yborder = y[
                0]  # yborder = y[0] is the smallest y -> prevents that the graph is drawn in reagions where nothing is detected

            # calculate the xs of the first marking function and put them into "liste" list
            if iter2 == 0:  # append left marking's points
                for i in range(yborder, img.shape[0],
                               step):  # y[0] is the smallest y -> prevents that the graph is bad over the last detected point
                    liste.append([f(i), i])
            elif iter2 == 1:  # append right markingspoints but from the bottom up ->otherwise connection of the points will cross
                for i in range(img.shape[0], yborder, -step):
                    liste.append([f(i), i])

        points_graph = np.array([liste], np.int32)  # global numpy array -> polylines needs a np.array

        # connect and fill the points
        img2 = np.zeros_like(img)
        cv2.fillConvexPoly(img2, points_graph, (0, 50, 0))
        cv2.polylines(img, [points_graph], True, (0, 0, 255), 5)
        img = img + img2

        '''for eintrag in liste:#not necessary
            row = eintrag[1]
            pkt1 = eintrag[0]

            if pkt1 != None:
                cv2.circle(img, (pkt1, row), 2, (0, 0, 255), -1)'''

    return img