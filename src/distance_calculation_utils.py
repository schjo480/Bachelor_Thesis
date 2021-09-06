from file_utils import *
from lidar_utils import *
from image_utils import *
from road_detection import *

def calculate_road_distance(semantic_img, depth_img, dataset):
    """

    :param semantic_img: for Apolloscape_stereo: result of det_line_1lane_init function. For Apolloscape_semantic:
    semantic image
    :param depth_img: img attributing a disparity or depth value to every pixel of the original img
    :param dataset:
    :return: distance of the road in meters
    """
    if dataset == "Apolloscape_stereo":
        # Intrinsic Camera Parameters
        baseline = 0.36  # baseline is the distance between the 2 stereo cameras
        f = 2301.3147  # focal length in pixels

        road_pixels = get_road_pixels(semantic_img, depth_img, dataset)
        u = road_pixels[:, 0]
        v = road_pixels[:, 1]
        distance = []
        depth = []
        for i in range(len(u) - 1):
            z = ((f * baseline) / depth_img[u[i], v[i]])
            x = (z * u[i])/f
            y = (z * v[i])/f
            dist = np.sqrt(z**2 + x**2 + y**2)
            depth.append(z)
            distance.append(dist)
        depth = np.asarray(depth)
        distance = np.asarray(distance)

        return depth, distance, road_pixels

    elif dataset=="Apolloscape_semantic":
        road_pixels = get_road_pixels(semantic_img, depth_img, dataset)
        u = road_pixels[:, 0]
        v = road_pixels[:, 1]
        distance = []
        for i in range(len(u)-1):
            dist = 255.0*(depth_img[u[i], v[i]]/200.0)
            distance.append(dist)

        return 0, distance, road_pixels


def distance_regression(road_rows, road_distances, degree=5):
    """
    Fit a regression model to road_distances ~ road_rows, detected from LIDAR
    :param road_rows:
    :param road_distances:
    :param degree:
    :return:
    """
    x = road_rows
    y = road_distances
    model = np.polyfit(x, y, degree)

    return model


def calculate_distance_from_regression(model, u_values_road):
    """
    Calculate the road distance using the road pixels from the camera image and the regression model.
    :param model:
    :param u_values_road: 5th, 10th, 15th, 20th and 25th percentile
    :return: Extrapolated distance for pixel heights u_values_road
    """
    if len(u_values_road) > 1:
        extrapolated_distance_0 = model[0] * u_values_road[0] ** 5 + model[1] * u_values_road[0] ** 4 + model[
            2] * u_values_road[0] ** 3 + model[3] * u_values_road[0] ** 2 + model[4] * u_values_road[0] + model[5]
        extrapolated_distance_1 = model[0] * u_values_road[1] ** 5 + model[1] * u_values_road[1] ** 4 + model[
            2] * u_values_road[1] ** 3 + model[3] * u_values_road[1] ** 2 + model[4] * u_values_road[1] + model[5]
        extrapolated_distance_5 = model[0] * u_values_road[2] ** 5 + model[1] * u_values_road[2] ** 4 + model[
            2] * u_values_road[2] ** 3 + model[3] * u_values_road[2] ** 2 + model[4] * u_values_road[2] + model[5]
        extrapolated_distance_10 = model[0] * u_values_road[3] ** 5 + model[1] * u_values_road[3] ** 4 + model[
            2] * u_values_road[3] ** 3 + model[3] * u_values_road[3] ** 2 + model[4] * u_values_road[3] + model[5]
        extrapolated_distance_15 = model[0] * u_values_road[4] ** 5 + model[1] * u_values_road[4] ** 4 + model[
            2] * u_values_road[4] ** 3 + model[3] * u_values_road[4] ** 2 + model[4] * u_values_road[4] + model[5]
        extrapolated_distance_20 = model[0] * u_values_road[5] ** 5 + model[1] * u_values_road[5] ** 4 + model[
            2] * u_values_road[5] ** 3 + model[3] * u_values_road[5] ** 2 + model[4] * u_values_road[5] + model[5]
        extrapolated_distance_25 = model[0] * u_values_road[5] ** 5 + model[1] * u_values_road[5] ** 4 + model[
            2] * u_values_road[5] ** 3 + model[3] * u_values_road[5] ** 2 + model[4] * u_values_road[5] + model[5]

        return extrapolated_distance_0, extrapolated_distance_1, extrapolated_distance_5, extrapolated_distance_10, \
               extrapolated_distance_15, extrapolated_distance_20, extrapolated_distance_25
    else:
        return model[0] * u_values_road[0] ** 5 + model[1] * u_values_road[0] ** 4 + model[
            2] * u_values_road[0] ** 3 + model[3] * u_values_road[0] ** 2 + model[4] * u_values_road[0] + model[5]
