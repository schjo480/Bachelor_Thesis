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

###################################################################################################################
######################Building and getting filenames for loading image, lidar, and INS data.#######################
###################################################################################################################

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