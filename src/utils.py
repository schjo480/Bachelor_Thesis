import json
import pprint
from urllib.request import urlopen
import numpy as np
import numpy.linalg as la
import open3d as o3
from os.path import join
import glob
import matplotlib.pylab as pt
import cv2
import io
import timeit
import matplotlib.pylab as plt
from PIL import Image
import os
from typing import AnyStr
import ast
import pickle
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import operator
import webbrowser
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import re
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt
import bisect
import csv
from scipy.ndimage import map_coordinates
import datetime
import seaborn as sns
from matplotlib.ticker import PercentFormatter

#Open the calibration file for A2D2
with open('/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/cams_lidars.json', 'r') as f:
    config = json.load(f)

#Camera models for the Oxford RobotCar Dataset
camera_models_directory = "/Volumes/Extreme SSD/Bachelorarbeit/Oxford/camera-models"

'''
Getting filenames
'''
def get_depth_files(dataset, date=None):
    """

    :param date: for KITTI, date must be in ["09_26", "09_28", "09_29", "09_30", "10_03"]
    :param dataset:
    :return: the filenames providing depth information in an ordered and iterable array
    """
    if dataset=="A2D2":
        root_path = "/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/A2D2/camera_lidar_semantic_bboxes"
        depth_files = sorted(glob.glob(join(root_path, '*/lidar/cam_front_center/*.npz')))
    elif dataset=="Apolloscape_stereo":
        root_path = "/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Apolloscape"
        depth_files = sorted(glob.glob(join(root_path, '*/disparity/*.png')))
    elif dataset=="Apolloscape_semantic":
        root_path = "/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/Apolloscape/road02_depth"
        depth_files = sorted(glob.glob(join(root_path, '*/Camera 5/*.png')))
    elif dataset=="KITTI":
        if date in ["09_26", "09_28", "09_29", "09_30", "10_03"]:
            root_path = "/Volumes/Extreme SSD/Bachelorarbeit/KITTI/Raw Data/2011_" + date
            depth_files = sorted(glob.glob(join(root_path, '*/velodyne_points/data/*.bin')))
        else:
            print("Enter a correct date value!")
    elif dataset=="Oxford":
        root_path = "/Volumes/Extreme SSD/Bachelorarbeit/Oxford/Lidar"
        depth_files = sorted(glob.glob(join(root_path, '*/ldmrs/*.bin')))
    else:
        print("Enter a correct dataset!")

    return depth_files


def extract_image_file_name_from_depth_file_name(file_name_depth, dataset = "A2D2"):
    """

    :param file_name_depth: Filename of the file that provides depth information: Lidar files for A2D2, KITTI, and
    Oxford and disparity or depth image for Apolloscape_stereo and Apolloscape_semantic, respectively
    :param dataset: name of the dataset, options: "A2D2", "KITTI", "Apolloscape_stereo", "Apolloscape_semantic, or
    "Oxford"
    :return: filename of the corresponding RGB image
    """
    if dataset == "A2D2":
        file_name_image = file_name_depth.split('/')
        tstamp = file_name_image[10].split('_')[0] + file_name_image[10].split('_')[1]
        file_name_nbr = file_name_image[-1].split('.')[0]
        file_name_nbr = file_name_nbr.split('_')[3]
        file_name_image = '/' + file_name_image[1] + '/' + file_name_image[2] + '/' + file_name_image[3] + '/' + \
                          file_name_image[4] + '/' + file_name_image[5] + '/' + file_name_image[6] + '/' + \
                          file_name_image[7] + '/' + file_name_image[8] + '/' + file_name_image[9] + '/' + \
                          file_name_image[10] + '/camera/cam_front_center/' + tstamp + '_camera_frontcenter_' + \
                          file_name_nbr + '.png'
    elif dataset == "KITTI":
        file_name_image = file_name_depth.split('/')
        file_name_nbr = file_name_image[-1].split('.')[0]
        file_name_image = '/' + file_name_image[1] + '/' + file_name_image[2] + '/' + file_name_image[3] + '/' + \
                          file_name_image[4] + '/' + file_name_image[5] + '/' + file_name_image[6] + '/' + \
                          file_name_image[7] + '/image_02/data/' + file_name_nbr + '.png'
    elif dataset == "Apolloscape_stereo":
        file_name_image = file_name_depth.split('/')
        file_name_nbr = file_name_image[-1].split('.')[0]
        file_name_image = '/' + file_name_image[1] + '/' + file_name_image[2] + '/' + file_name_image[3] + '/' + \
                          file_name_image[4] + '/' + file_name_image[5] + '/' + file_name_image[6] + '/' + \
                          file_name_image[7] + '/' + file_name_image[8] + '/' + file_name_image[9] + '/camera_5/' \
                          + file_name_nbr + '.jpg'
    elif dataset == "Apolloscape_semantic":
        file_name_image = file_name_depth.split('/')
        record_nbr = file_name_image[10]
        file_name_nbr = file_name_image[-1].split('.')[0]
        file_name_image = '/' + file_name_image[1] + '/' + file_name_image[2] + '/' + file_name_image[3] + '/' + \
                          file_name_image[4] + '/' + file_name_image[5] + '/' + file_name_image[6] + '/' + \
                          file_name_image[7] + '/' + file_name_image[8] + '/' + 'road02_seg/ColorImage/' + record_nbr +\
                          '/' + file_name_image[11] + '/' + file_name_nbr + '.jpg'
    elif dataset == "Oxford":
        file_name_image = file_name_depth.split('/')
        lidar_timestamp = file_name_image[-1].split('.')[0]
        lidar_timestamp = np.asarray(lidar_timestamp, dtype=np.int64)
        date = file_name_image[6]
        camera_timestamp_file = '/' + file_name_image[1] + '/' + file_name_image[2] + '/' + file_name_image[3] + '/' + \
                          file_name_image[4] + '/Camera/' + date + '/stereo.csv'
        camera_timestamp_df = pd.read_csv(camera_timestamp_file, header=None)
        camera_timestamp_df.iloc[:, [0]] = camera_timestamp_df.iloc[:, [0]]
        camera_timestamp_arr = camera_timestamp_df.to_numpy()
        camera_timestamp = []
        for i in range(camera_timestamp_arr.shape[0]):
            nbr = str(camera_timestamp_arr[i])
            nbr = nbr.split(' ')
            nbr = nbr[0]
            nbr = nbr[2:]
            camera_timestamp.append(nbr)

        camera_timestamp = np.asarray(camera_timestamp, dtype=np.int64)
        index = (np.abs(camera_timestamp - lidar_timestamp)).argmin()
        camera_timestamp = camera_timestamp[index]
        camera_timestamp = str(camera_timestamp)
        file_name_image = '/' + file_name_image[1] + '/' + file_name_image[2] + '/' + file_name_image[3] + '/' + \
                          file_name_image[4] + '/Camera/' + date + '/stereo/centre/' + camera_timestamp + '.png'

    return file_name_image


def extract_semantic_file_name_from_file_name_depth(file_name_depth, dataset="A2D2"):
    """

    :param file_name_depth: Filename of the file that provides depth information: Lidar files for A2D2, KITTI, and
    Oxford and disparity or depth image for Apolloscape_stereo and Apolloscape_semantic, respectively
    :param dataset: "A2D2", "Apolloscape_stereo", "Apolloscape_semantic"
    :return: filename of the corresponding semantic image
    """
    if dataset == "A2D2":
        file_name_semantic_label = file_name_depth.split('/')
        tstamp = file_name_semantic_label[10].split('_')[0] + file_name_semantic_label[10].split('_')[1]
        file_name_nbr = file_name_semantic_label[-1].split('.')[0]
        file_name_nbr = file_name_nbr.split('_')[3]
        file_name_semantic_label = '/' + file_name_semantic_label[1] + '/' + file_name_semantic_label[2] + '/' + \
                                   file_name_semantic_label[3] + '/' + \
                                   file_name_semantic_label[4] + '/' + file_name_semantic_label[5] + '/' + \
                                   file_name_semantic_label[6] + '/' + \
                                   file_name_semantic_label[7] + '/' + file_name_semantic_label[8] + '/' + \
                                   file_name_semantic_label[9] + '/' + \
                                   file_name_semantic_label[
                                       10] + '/label/cam_front_center/' + tstamp + '_label_frontcenter_' + \
                                   file_name_nbr + '.png'
    elif dataset == "Apolloscape_stereo":
        file_name_semantic_label = file_name_depth.split('/')
        tstamp = file_name_semantic_label[-1].split('.')[0]
        file_name_semantic_label = '/' + file_name_semantic_label[1] + '/' + file_name_semantic_label[2] + '/' + \
                                   file_name_semantic_label[3] + '/' + file_name_semantic_label[4] + '/' + \
                                   file_name_semantic_label[5] + '/' + file_name_semantic_label[6] + '/' + \
                                   file_name_semantic_label[7] + '/' + file_name_semantic_label[8] + '/' + \
                                   file_name_semantic_label[9] + '/fg_mask/' + tstamp + '.png'
    elif dataset == "Apolloscape_semantic":
        file_name_semantic_label = file_name_depth.split('/')
        record_nbr = file_name_semantic_label[10]
        tstamp = file_name_semantic_label[-1].split('.')[0]
        tstamp = tstamp.split('_')
        tstamp = tstamp[0] + '_' + tstamp[1] + '_' + 'Camera_5_bin'
        file_name_semantic_label = '/' + file_name_semantic_label[1] + '/' + file_name_semantic_label[2] + '/' + \
                                   file_name_semantic_label[3] + '/' + file_name_semantic_label[4] + '/' + \
                                   file_name_semantic_label[5] + '/' + file_name_semantic_label[6] + '/' + \
                                   file_name_semantic_label[7] + '/' + file_name_semantic_label[8] + \
                                   '/road02_seg/Label/' + record_nbr + '/Camera 5/' + tstamp + '.png'

    return file_name_semantic_label


def extract_bus_folder_name_from_depth_file_name(file_name_depth, dataset = "A2D2"):
    """

    :param file_name_depth: Filename of the file that provides depth information: Lidar files for A2D2, KITTI, and
    Oxford and disparity or depth image for Apolloscape_stereo and Apolloscape_semantic, respectively
    :param dataset: Name of the dataset, options: "A2D2", "KITTI", "Apolloscape_stereo", "Apolloscape_semantic, or
    "Oxford"
    :return: filename of the corresponding IMU data
    """
    if dataset == "A2D2":
        folder_name_bus = file_name_depth.split('/')
        folder_name_bus_nbr = folder_name_bus[10]
        folder_name_bus = '/' + folder_name_bus[1] + '/' + folder_name_bus[2] + '/' + folder_name_bus[3] + '/' + \
                          folder_name_bus[4] + '/' + folder_name_bus[5] + '/' + folder_name_bus[6] + '/' + \
                          folder_name_bus[7] + '/' + folder_name_bus[8] + '/camera_lidar_semantic_bus/' + \
                          folder_name_bus_nbr
    elif dataset == "KITTI":
        folder_name_bus = file_name_depth.split('/')
        folder_name_bus_nbr = folder_name_bus[10]
        folder_name_bus_nbr = folder_name_bus_nbr.split('.')[0]
        folder_name_bus = '/' + folder_name_bus[1] + '/' + folder_name_bus[2] + '/' + folder_name_bus[3] + '/' + \
                          folder_name_bus[4] + '/' + folder_name_bus[5] + '/' + folder_name_bus[6] + '/' + \
                          folder_name_bus[7] + '/oxts/data/' + folder_name_bus_nbr + '.txt'

    elif dataset == "Apolloscape_stereo":
        return
    elif dataset == "Oxford":
        folder_name_bus = file_name_depth.split('/')
        date = folder_name_bus[6]
        date = date.split(' ')[0]
        folder_name_bus = '/' + folder_name_bus[1] + '/' + folder_name_bus[2] + '/' + folder_name_bus[3] + '/' + \
                          folder_name_bus[4] + '/GPS/' + date + '/gps/ins.csv'

    return folder_name_bus


def extract_gps_folder_name_from_depth_file_name(file_name_depth, dataset="Oxford"):
    """

    :param file_name_depth: Filename of the file that provides depth information: Lidar files for A2D2, KITTI, and
    Oxford and disparity or depth image for Apolloscape_stereo and Apolloscape_semantic, respectively
    :param dataset: "Oxford"
    :return: The name of the folder storing the GPS information for corresponding Oxford frames
    """
    if dataset == "Oxford":
        folder_name_gps = file_name_depth.split('/')
        date = folder_name_gps[6]
        date = date.split(' ')[0]
        folder_name_gps = '/' + folder_name_gps[1] + '/' + folder_name_gps[2] + '/' + folder_name_gps[3] + '/' + \
                          folder_name_gps[4] + '/GPS/' + date + '/gps/gps.csv'
        return folder_name_gps


def extract_ins_folder_name_from_depth_file_name(file_name_depth, dataset="Oxford"):
    """

    :param file_name_depth: Filename of the file that provides depth information: Lidar files for Oxford
    :param dataset: "Oxford"
    :return: The name of the folder storing the INS information for corresponding Oxford frames
    """
    if dataset == "Oxford":
        folder_name_ins = file_name_depth.split('/')
        date = folder_name_ins[6]
        date = date.split(' ')[0]
        folder_name_ins = '/' + folder_name_ins[1] + '/' + folder_name_ins[2] + '/' + folder_name_ins[3] + '/' + \
                          folder_name_ins[4] + '/GPS/' + date + '/gps/ins.csv'
        return folder_name_ins


def extract_vo_folder_name_from_depth_file_name(file_name_depth, dataset="Oxford"):
    """

    :param file_name_depth: Filename of the file that provides depth information: Lidar files for Oxford
    :param dataset: "Oxford"
    :return: The name of the folder storing the Visual Odometry information for corresponding Oxford frames
    """
    if dataset == "Oxford":
        folder_name_vo = file_name_depth.split('/')
        date = folder_name_vo[6]
        folder_name_vo = '/' + folder_name_vo[1] + '/' + folder_name_vo[2] + '/' + folder_name_vo[3] + '/' + \
                          folder_name_vo[4] + '/GPS/' + date + '/gps/vo.csv'
        return folder_name_vo


def read_image_info(file_name_image, dataset="A2D2"):
    """

    :param dataset:
    :param file_name_image:
    :return: A2D2: returns a dictionary containing the timestamp and camera name, Oxford: Returns a dataframe
    """
    if dataset == "A2D2":
        file_name_info = file_name_image.replace(".png", ".json")
        with open(file_name_info, 'r') as g:
            image_info = json.load(g)
    elif dataset == "Oxford":
        file_name = extract_bus_folder_name_from_depth_file_name(file_name_image, dataset)
        image_info = pd.read_csv(file_name)

    return image_info


def get_img_directory(img_filename, dataset):
    """

    :param img_filename:
    :param dataset: "Oxford"
    :return: The directory in which the image is stored. This directory name is later used to build point clouds.
    """
    if dataset == "Oxford":
        image_directory = img_filename.split('/')
        image_directory = image_directory[:len(image_directory) - 1]
        image_directory = '/'.join(image_directory)
        return image_directory


def get_timestamp(file_name_image, dataset):
    """

    :param file_name_image: File name of the RGB image
    :param dataset: "A2D2", "Apolloscape_stereo", "Apolloscape_semantic", "KITTI", "Oxford"
    :return: The timestamp of every frame.
    """
    if dataset == "A2D2":
        img_info = read_image_info(file_name_image) #UNIX format
        return img_info["cam_tstamp"]
    elif dataset in ["Apolloscape_stereo", "Apolloscape_semantic"]:
        file_name_image = file_name_image.split('/')
        tstamp = file_name_image[-1].split('.')[0]
        tstamp = tstamp.split('_')
        tstamp = tstamp[0] + ' ' + tstamp[1] # YY:MM:DD hh:mm:ss:ms
        return tstamp
    elif dataset == "KITTI":
        img_nbr = file_name_image.split('/')[-1]
        img_nbr = int(img_nbr.split('.')[0])
        file_name_timestamps = file_name_image.split('/')
        file_name_timestamps = file_name_timestamps[:len(file_name_timestamps) - 2]
        file_name_timestamps = '/'.join(file_name_timestamps) + '/timestamps.txt'
        with open(file_name_timestamps) as f:
            lines = f.readlines()
        tstamp = lines[img_nbr]  # YYYY-MM-DD hh:mm:ss:ns
        tstamp = tstamp.split(' ')[1]
        tstamp = tstamp.split('.')[0]  # hh:mm:ss
        return tstamp
    elif dataset == "Oxford":
        file_name_image = file_name_image.split('/')
        file_name_image = file_name_image[9]
        tstamp = file_name_image.split('.')[0] #UNIX format
        return tstamp


def get_meta_parameters(file_name_depth, image_nbr=None, dataset="A2D2"):
    """

    :param file_name_depth:
    :param image_nbr: For the Oxford dataset. Timestamps of images, LIDAR scans, and INS data are not synced.
    Therefore, search INS info with image number, rather than timestamp.
    :param dataset:
    :return: Array of the available Meta Parameters for each Image/Lidar pair. Here: Speed, Acceleration, Longitiude,
    Latitude
    """
    if dataset=="A2D2":
        file_nbr = file_name_depth.split('/')
        file_nbr = file_nbr[10].replace('_', '')
        file_name_image = extract_image_file_name_from_depth_file_name(file_name_depth)
        bus_data_file = extract_bus_folder_name_from_depth_file_name(file_name_depth, dataset) + '/bus/' + file_nbr + \
                        '_bus_signals.json'
        f = open(bus_data_file)
        bus = json.load(f)
        speed = np.mean(bus[image_nbr]['flexray']['vehicle_speed']['values']) #vehicle speed in km/h
        steering_angle = np.mean(bus[image_nbr]['flexray']['steering_angle_calculated']['values']) #steering angle in \
                                                                                                   # degree of arc
        roll_angle = np.mean(bus[image_nbr]['flexray']['roll_angle']['values']) #roll angle in degree of arc
        pitch_angle = np.mean(bus[image_nbr]['flexray']['pitch_angle']['values']) #pitch angle in degree of arc
        brake_pressure = np.mean(bus[image_nbr]['flexray']['brake_pressure']['values']) #brake pressure in bar
        acceleration_x = np.mean(bus[image_nbr]['flexray']['acceleration_x']['values'])
        #acceleration in x-direction in m/s^2
        acceleration_y = np.mean(bus[image_nbr]['flexray']['acceleration_y']['values'])
        #acceleration in y-direction in m/s^2
        acceleration_z = np.mean(bus[image_nbr]['flexray']['acceleration_z']['values'])
        #acceleration in z-direction in m/s^2
        acceleration = np.sqrt(np.float(acceleration_x)**2 + np.float(acceleration_y)**2)
        latitude = np.mean(bus[image_nbr]['flexray']['latitude_degree']['values'])
        longitude = np.mean(bus[image_nbr]['flexray']['longitude_degree']['values'])

    elif dataset in ["Apolloscape_stereo", "Apolloscape_semantic"]:
        speed = 0
        longitude = 0
        latitude = 0
        acceleration = 0

    elif dataset=="Oxford":
        file_name_image = extract_image_file_name_from_depth_file_name(file_name_depth, dataset)
        lidar_time_stamp = file_name_depth.split('/')
        lidar_time_stamp = lidar_time_stamp[-1].split('.')[0]
        lidar_time_stamp = np.asarray(lidar_time_stamp, dtype=np.int64)
        file_name_ins = extract_bus_folder_name_from_depth_file_name(file_name_depth, dataset)
        ins_dataframe = pd.read_csv(file_name_ins)
        ins_timestamps = ins_dataframe["timestamp"].to_numpy()
        index = (np.abs(ins_timestamps - lidar_time_stamp)).argmin()
        ins_timestamp = ins_timestamps[index]
        ins_parameters = ins_dataframe[ins_dataframe["timestamp"] == ins_timestamp]
        speed = np.sqrt((np.float(ins_parameters["velocity_north"]))**2 + \
                        np.float((ins_parameters["velocity_east"]))**2)
        acceleration = 0
        roll = int(ins_parameters["roll"])
        pitch = int(ins_parameters["pitch"])
        yaw = int(ins_parameters["yaw"])
        altitude = int(ins_parameters["altitude"])

        file_name_gps = extract_gps_folder_name_from_depth_file_name(file_name_depth, dataset)
        gps_dataframe = pd.read_csv(file_name_gps)
        gps_timestamps = gps_dataframe["timestamp"].to_numpy()
        index = (np.abs(gps_timestamps - lidar_time_stamp)).argmin()
        gps_timestamp = gps_timestamps[index]
        gps_parameters = gps_dataframe[gps_dataframe["timestamp"] == gps_timestamp]
        longitude = gps_parameters["longitude"]
        latitude = gps_parameters["latitude"]

    elif dataset=="KITTI":
        file_name_bus = extract_bus_folder_name_from_depth_file_name(file_name_depth, "KITTI")
        file = open(file_name_bus, "r")
        content = file.read()
        content = content.split(' ')
        latitude = content[0] #degree
        longitude = content[1] #degree
        altitude = content[2] #m
        roll = content[3] #rad
        pitch = content[4] #rad
        yaw = content[5] #rad
        speed = np.sqrt(np.float(content[6])**2 + np.float(content[7])**2) #m/s
        acceleration = np.sqrt(np.float(content[11])**2 + np.float(content[12])**2) #m/s^2

    meta_parameters = [speed, longitude, latitude, acceleration]

    return meta_parameters


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    Retrieved on: 05.05.2021
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


'''
LIDAR
'''
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


'''
Image processing
'''
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

        if lens=='Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens=='Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    elif dataset in ["Apolloscape_stereo", "Apolloscape_semantic"]:
        image = cv2.imread(img_filename)
        type = re.search('(camera_5|fg_mask|ColorImage|Label|depth)', img_filename).group(0)
        if type in ["camera_5", "ColorImage"]:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # undistorted image is provided
        elif type=="fg_mask":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif type=="Label":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif type=="depth":
            image = cv2.imread(img_filename)
        return image
    elif dataset == "KITTI":
        image = cv2.imread(img_filename) #undistorted image is provided
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

        if lens=='Fisheye':
            return cv2.fisheye.undistortImage(image, intr_mat_dist, D=dist_parms, Knew=intr_mat_undist)
        elif lens=='Telecam':
            return cv2.undistort(image, intr_mat_dist, distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


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

    elif dataset=="Apolloscape_semantic":
        img = semantic_image

        binary = np.zeros_like(img)
        binary[:, :] = 0
        binary[img==49] = 255

    return binary


def filter_cars(semantic_image, dataset):
    """

    :param semantic_image:
    :param dataset:
    :return: Binary image with car pixel value = 255, rest = 0
    """
    if dataset=="A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 0) & (h != 60)] = 0 #Cars & Small Vehicles

    elif dataset=="Apolloscape_semantic":
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
    if dataset=="A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 15) & (h != 19) & (h != 26) & (h != 120) & (h != 30) & (h != 15)] = 0

    elif dataset=="Apolloscape_semantic":
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
    if dataset=="A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        binary = np.zeros_like(h)
        binary[:, :] = 255
        binary[(h != 14) & (h != 11) & (h != 10) & (h != 135) & (h != 159) & (h != 160)] = 0

    elif dataset=="Apolloscape_semantic":
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
    if dataset=="A2D2":
        hsv = cv2.cvtColor(semantic_image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        binary = np.zeros_like(semantic_image)

        binary[(h != 150) | (s != 255) & (s != 184) | (v != 255) & (v != 180)] = 255
        binary[(h != 0) & (h != 60)] = 255 #Cars & Small Vehicles
        binary[(h != 15) & (h != 19) & (h != 26) & (h != 120) & (h != 30) & (h != 15)] = 255
        binary[(h != 14) & (h != 11) & (h != 10) & (h != 135) & (h != 159) & (h != 160)] = 255

    elif dataset=="Apolloscape_semantic":
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
    [u, v] = np.where(binary==255)

    if dataset == "KITTI":
        roi_factor_y = 0.5  # For the lane detection, we only considered the bottom 50% of the image
        roi_factor_x = 0.3
    elif dataset == "Apolloscape_stereo":
        roi_factor_y = 0.5  # For the lane detection, we only considered the bottom 60% of the image
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
        roi_factor_y = 0.5  # For the lane detection, we only considered the bottom 50% of the image
        roi_factor_x = 0.3
    elif dataset == "Apolloscape_stereo":
        roi_factor_y = 0.5  # For the lane detection, we only considered the bottom 60% of the image
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
    if dataset=="A2D2":
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

        #return mildly occluded (0.3), partly occluded (0.4), fully blocked (0.5)
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

    elif dataset=="Apolloscape_stereo":
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
            if np.sum(cars)/nbr_road_pixels > threshold:
                result.append([1, 0, 0, 0])
            else:
                result.append([0, 0, 0, 0])

        result = np.array(result)
        result = np.reshape(result, [1, 12])

    elif dataset=="Apolloscape_semantic":
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

    elif dataset=="KITTI":
        result = np.zeros([1, 12])
        result = np.array(result)
        result = np.reshape(result, [1, 12])

        
    elif dataset=="Oxford":
        result = np.zeros([1, 12])
        result = np.array(result)
        result = np.reshape(result, [1, 12])

    
    return result


def roi_r(img, dataset="KITTI"):
    '''RECTANGULAR ROI.: Reads in the image. Then generates a rectangular ROI.
    Any other pixels are cut off. border is the percentage/100 of the height of the image,
    that marks the upper end of the ROI.'''

    roi_factor_y = 0.45
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
    if dataset=="Apolloscape_stereo":
        #Because of the higher resolution.
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


'''
Road detection
'''

################################################################################
#
# Title: Fahrspurerkennung.py
# Authors: Girstl, Felix
# Date: 2018
#
################################################################################


# Logic Operation
def logic_and(img1, img2, img3=None, img4=None):
    """
    Performs a logic AND with 2 to 4 images of the same size.
    The images mustn't have more than one channel.
    Usually Binary images are used.
    Output is a single one channel image with 0 or 1 as values.
    So only if the pixel value is true (>0) in every of the input images, the output image pixel will be true.
    """

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
    """
    Performs a logic OR with 2 to 4 images of the same size.
    The images mustn't have more than one channel.
    Usually Binary images are used.
    Output is a single one channel image with 0 or 1 as values.
    So only if the pixelvalue is true (>0) in at least one of the input images, the output image pixel will be true.
    """
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


def det_line_steps(img1, img2, step=1):
    """
    Detects lines by canny and line image comparison (if canny-edge and next to it a colored line ->detected).
    Checks only certain image rows.
    :param img1: Canny image 1channel
    :param img2: line image 1channel
    :param step: distance between the checked rows
    :return: semi-binary image (pixelvalues of 0 or 255) 1channel
    """
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


def det_line_1lane_init(canny, lineimg1, lineimg2, dataset, step=1):
    """
    Detects lines on moving stripes (more than one striperow with 2 lines to detect)
    The stripes have a various width (linear from max and min width over y)->inside the stripe list (2nd input of every
    row).
    Initialises the starting position of the stripes.(not manually set)
    :param canny: result of canny_edge()
    :param lineimg1: result of color filter (e.g. white)
    :param lineimg2: result of alternative color filter (e.g. yellow)
    :param dataset:
    :param step:
    :return:
    """
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
    def enter_new_midpoint(value, spot, nmax=10): #maybe decrease nmax from the original 25
        """
        Enters a new midpoint(value) into the list.
        Also shifts the old points to the "right" -> so the newest point is in the first listplace.
        :param value: value that shall be entered as the last detected midpoint
        :param spot: spot is the points[x][1-z]
        :param nmax: maximum of the allowed spaces for former midpoints
        :return:
        """

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
        """
        Sets up all lists
        :return:
        """

        def stripe_width(y, ymax=canny.shape[0], wmin=50, wmax=120):
            """
            calculates the linear width of the stripes. Depending on the row hight y.
            Smaller width on the top of the image.
            -> width = m*y+t m=(wmax-wmin)/ymax  t=wmin ymax=img.shape[1]
            :param y:
            :param wmin: minimum width of stripes
            :param wmax: maximum width
            :return: width of the row
            """
            if dataset == "KITTI":
                wmin = 30
                wmax = 380
            elif dataset == "Apolloscape_stereo":
                wmin = 140
                wmax = 1100
            elif dataset == "Oxford":
                wmin = 60
                wmax = 700
            width = int(((wmax - wmin) / ymax) * y + wmin)
            return width

        if "stripe" not in globals():
            global stripe  # list for checking stripe's midpoints
            stripe = []
            global points  # list for detected lane markings points (None if not detected)
            points = []

            global det_counter  # list which has info if stripe ever detected a point (True/False)
            det_counter = []
            if dataset=="Apolloscape_stereo":
                mid = 2.25
            else:
                mid = 2.0
            # set stripe default values for all rows
            for row in range(0, canny.shape[0] - 1, 1):  # iteration through rows
                stripe_mid = int(canny.shape[1] / mid)# stripes are in the middle of the width (going to move outwards)
                width = stripe_width(row)
                stripe.append([row, width, stripe_mid, stripe_mid])  # stripe = [[row, width, x_left, x_right], ...]

                det_counter.append([None, None])  # set the counter to None

                points.append([row, [], []])  # points = [[row, [last points left], [last points right]], ...]
                # set last values of points to none
                for i in range(1, 3):  # left and right iteration
                    enter_new_midpoint(None, points[int(row / 1)][i])

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
            output[y, int(mid_left - w / 2):int(mid_left + w / 2), 0] = 255
            output[y, int(mid_right - w / 2):int(mid_right + w / 2), 0] = 255

        return output

    def outmoving_stripe(pixelspeed=4): #decreased pixel speed from 9 to 4
        '''Moves the stripes to the left / right until they have found a marking.'''
        for i, row in enumerate(stripe):  # iterate through all stripes
            if get_lastvalid_midpoint(points[i][1]) is None:  # if point is None(not detected)->move left stripe to
                # the left
                if det_counter[i][0] is None:  # move only if stripe never detected something
                    stripe[i][2] -= pixelspeed

            if get_lastvalid_midpoint(points[i][2]) is None:  # right point(right lane marking)
                if det_counter[i][1] is None:
                    stripe[i][3] += pixelspeed

        return

    def end_outmoving_stripe():
        """
        Sets all stripes counter to True that have a lower and higher stripe that already detected something
        :return:
        """
        # get highest and lowest stripe that already detected something for both sides
        global end_initialisation_framenbr
        end_initialisation_framenbr = 3 #Maybe change this!! Original 53

        highest_left = None
        lowest_left = None
        highest_right = None
        lowest_right = None
        for i, info in enumerate(det_counter):
            if info[0]==True:  # if left stripe in this row ever detected anything
                if lowest_left is None:
                    lowest_left = i
                highest_left = i
            if info[1]==True:  # right stripe
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

    def runnaway_stripe(maxdist=20): #changed maxdist from originally 100 to 10
        if frame_nbr > end_initialisation_framenbr:  # check for wrong stripes after initialisation
            # iterate all rows of stripe

            # stripe position from top to bottom
            for i in range(len(stripe) - 1):
                for it in range(2, 4): #left and right
                    if abs(stripe[i][it] - stripe[i + 1][it]) > maxdist and abs(stripe[i][it] - stripe[i - 1][it]) < \
                            maxdist:
                        # set stripe above to same value
                        stripe[i + 1][it] = stripe[i][it]

            # is the stripe too far to the left or too far to the right (focus on bottom half, because less curvature
            # has occurred here from bottom to middle
            for i in range(len(stripe) - 2, int(len(stripe)/2), -1):
                for it in range(2, 4): #left and right
                    if stripe[i][2] < 300:
                        stripe[i][2] = stripe[i + 1][2]
                    if stripe[i][3] > (canny.shape[1] - 300):
                        stripe[i][3] = stripe[i + 1][3]

        return

    def detector():
        lines = lineimg1 + lineimg2
        for idx1, entry in enumerate(stripe):  # iterate over all rows (idx1 ist der Index des Eintrags in stripe)

            # Stripes: draw and calculate
            for idx2, mittelpkt in enumerate(entry[2:]):  # iterate over all midpoints(stripes) of the row
                # (therefore skip y and w)

                # calculate stripe points
                pkt_l = int(mittelpkt - 0.5 * entry[1])  # left/right point of the stripe (entry[1] = width of stripe)
                pkt_r = int(mittelpkt + 0.5 * entry[1])

                # limitation of the stripe position (has to be in the image boundaries)
                if pkt_r >= lineimg1.shape[1]:  # if the stripe points are out of the image set to left or right
                    # max position
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
                for x in range(pkt_l, np.min([pkt_r, canny.shape[1] - 2])):  # iterate over x
                    if x < canny.shape[1]:
                        if canny[y, x] > 0 and (lines[y, x + 1] > 0 or lines[y, x - 1] > 0):  # check left & right edge
                            # output[y, x, :] = 255
                            list_hits.append(x)  # append new hit to list

                # calculate new midpoint & save in stripe list
                if len(list_hits):  # if entries are in list -> else division thru 0
                    midpoint = int(sum(list_hits) / len(list_hits))
                    output[y, midpoint, :] = 255
                    stripe[idx1][idx2 + 2] = midpoint  # (idx2+2 ,because first 2 entries of stripe entries are
                    # y & width)
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


def overlay_road_on_image(img, dataset):
    """

    :param img: RGB image
    :param dataset:
    :return: RGB image with the detected road by det_line_1lane_init() overlaid on top of it.
    """
    output = img
    roi = roi_r(img, dataset)
    roi = gauss(roi, dataset)
    canny_edge = canny(roi)
    white_line = filter_white_hls_binary(roi)
    yellow_line = filter_yellow_hls_binary(roi)
    semantic_image = det_line_1lane_init(canny_edge, white_line, yellow_line, dataset)

    if dataset=="KITTI":
        roi_factor_y = 0.5
        roi_factor_x = 0.3
    elif dataset == "Apolloscape_stereo":
        roi_factor_y = 0.5
        roi_factor_x = 0.3
    elif dataset=="Oxford":
        roi_factor_y = 0.45
        roi_factor_x = 0.3

    factor_y = int(roi_factor_y * img.shape[0])
    factor_x = int(roi_factor_x * img.shape[1])

    output[factor_y:, factor_x:output.shape[1]-factor_x, :] = 0.6 * img[factor_y:, factor_x:img.shape[1]-factor_x, :] +\
                                                              0.4 * semantic_image

    return output


'''
Distance Calculation
'''
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
    extrapolated_distance_5 = model[0] * u_values_road[0] ** 5 + model[1] * u_values_road[0] ** 4 + model[
        2] * u_values_road[0] ** 3 + model[3] * u_values_road[0] ** 2 + model[4] * u_values_road[0] + model[5]
    extrapolated_distance_10 = model[0] * u_values_road[1] ** 5 + model[1] * u_values_road[1] ** 4 + model[
        2] * u_values_road[1] ** 3 + model[3] * u_values_road[1] ** 2 + model[4] * u_values_road[1] + model[5]
    extrapolated_distance_15 = model[0] * u_values_road[2] ** 5 + model[1] * u_values_road[2] ** 4 + model[
        2] * u_values_road[2] ** 3 + model[3] * u_values_road[2] ** 2 + model[4] * u_values_road[2] + model[5]
    extrapolated_distance_20 = model[0] * u_values_road[3] ** 5 + model[1] * u_values_road[3] ** 4 + model[
        2] * u_values_road[3] ** 3 + model[3] * u_values_road[3] ** 2 + model[4] * u_values_road[3] + model[5]
    extrapolated_distance_25 = model[0] * u_values_road[4] ** 5 + model[1] * u_values_road[4] ** 4 + model[
        2] * u_values_road[4] ** 3 + model[3] * u_values_road[4] ** 2 + model[4] * u_values_road[4] + model[5]

    return extrapolated_distance_5, extrapolated_distance_10, extrapolated_distance_15, extrapolated_distance_15, \
            extrapolated_distance_20, extrapolated_distance_25


'''
Analysis
'''

def get_road_category(longitude, latitude):
    """
    The api takes the format: longitude, latitude in decimal format.
    Connection to TUM network required. (e.g. by using the cisco anyconnect VPN)
    :param longitude:
    :param latitude:
    :return: road category and type as defined by OSM
    """
    url = "http://gis.ftm.mw.tum.de/reverse?coordinates=[" + str(longitude) + "," + str(latitude) + "]"
    response = urlopen(url)
    data = json.loads(response.read())
    #addresstype = data['features'][0]['properties']['addresstype']
    category = data['features'][0]['properties']['category']
    location = data['features'][0]['properties']['display_name']
    type = data['features'][0]['properties']["type"]

    return category, location, type


'''
Oxford specials
'''
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
