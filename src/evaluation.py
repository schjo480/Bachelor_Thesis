from image_utils import *
from lidar_utils import *
from file_utils import *
from distance_calculation_utils import *
from road_detection import *

#enter the name of one of the following datasets: A2D2, Apolloscape_stereo, Apolloscape_semantic, KITTI, Oxford
dataset = "Apolloscape_semantic"

#get an ordered array of the filenames
#enter one of the following dates for KITTI: ["09_26", "09_28", "09_29", "09_30", "10_03"]
if dataset=="KITTI":
    date = input("Please enter the date for KITTI: ")
    calib = input("Enter calibration filename (calib_cam_to_cam.txt): ")
    calib = read_calib_file(calib)
    depth_files = get_depth_files(dataset, date=date)
else:
    depth_files = get_depth_files(dataset, date=None)
    calib = None

if dataset=="A2D2":
    # In the following .json file, all bus signals from the A2D2 are combined.
    f = open("/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/A2D2/camera_lidar_semantic_bus/"
             "20180807145028_bus_signals Kopie.json")
    bus = json.load(f)

number_of_frames = len(depth_files) - 1
frame_jumper = 1

#initialize arrays
max_road_distance_detected_75 = []
max_road_distance_detected_80 = []
max_road_distance_detected_85 = []
max_road_distance_detected_90 = []
max_road_distance_detected_95 = []
max_road_distance_detected_99 = []
max_road_distance_detected = []
max_road_distance_extrapolated = []
max_road_distance_extrapolated_99 = []
max_road_distance_extrapolated_95 = []
max_road_distance_extrapolated_90 = []
max_road_distance_extrapolated_85 = []
max_road_distance_extrapolated_80 = []
max_road_distance_extrapolated_75 = []
blocked_dist = []

cars03 = []
cars04 = []
cars05 = []
trucks03 = []
trucks04 = []
trucks05 = []
ped_bc03 = []
ped_bc04 = []
ped_bc05 = []
other03 = []
other04 = []
other05 = []

timestamps = []
meta_parameters = []
speed = []
accel = []
longitude = []
latitude = []

folders = []
j = 0  # Counter for image number Oxford
folder_counter = 0

for i in range(0, number_of_frames, frame_jumper):
    print("Frame:", i, "of", number_of_frames)

    if dataset in ["A2D2", "KITTI"]:
        img_number = None
    else:
        lidar_folder = depth_filename.split('/')[6]
        folders.append(lidar_folder)
        if folders[folder_counter] != folders[folder_counter - 1]:
            j = 0
        img_number = j

    depth_filename = depth_files[i]
    print(depth_filename)
    img_filename = extract_image_file_name_from_depth_file_name(depth_filename, dataset, img_number)
    img = load_image(img_filename, dataset)
    timestamp = get_timestamp(img_filename, dataset)

    if dataset in ["A2D2", "KITTI", "Oxford"]:
        #returns a dict containing all the relevant information about lidar points
        lidar = get_lidar_data(depth_filename, dataset=dataset, calib=calib, img_number=j)
        # Lidar data
        points = lidar['points']
        distances = lidar['distance']
        depth = lidar['depth']
        rows = lidar['row']
        cols = lidar['col']

    elif dataset=="Apolloscape_stereo":
        depth_img = get_disparity_map(depth_filename)
        fg_mask_filename = extract_semantic_file_name_from_file_name_depth(depth_filename, dataset)
        semantic_mask = load_image(fg_mask_filename, dataset)

    elif dataset=="Apolloscape_semantic":
        depth_img = load_image(depth_filename, dataset)

    try:
        if dataset in ["Apolloscape_stereo", "KITTI", "Oxford"]:
            #lane detection
            roi = roi_r(img, dataset=dataset)
            roi = gauss(roi)
            canny_edge = canny(roi)
            white_line = filter_white_hls_binary(roi)
            yellow_line = filter_yellow_hls_binary(roi)
            semantic_image = det_line_1lane_init(canny_edge, white_line, yellow_line, dataset)
            parameters = get_meta_parameters(depth_filename, dataset=dataset)

        elif dataset in ["A2D2", "Apolloscape_semantic"]:
            semantic_filename = extract_semantic_file_name_from_file_name_depth(depth_filename, dataset)
            semantic_mask = load_image(semantic_filename, dataset)
            semantic_image = semantic_mask

        else:
            print("Wrong dataset name entered.")

        #get depth information
        if dataset in ["A2D2", "KITTI", "Oxford"]:
            #get all the road points which were captured by the LIDAR scanner
            ptcld_colored, road_indcs = create_open3d_pc(lidar, semantic_image, dataset)
            road_indcs = road_indcs[road_indcs < len(distances)]
            road_distances_lidar = distances[road_indcs]
            road_depth_lidar = depth[road_indcs]
            road_points_lidar = points[road_indcs]
            road_rows_lidar = rows[road_indcs]
            road_cols_lidar = cols[road_indcs]

            if dataset in ["A2D2", "KITTI"]:
                #get all the road pixels in the image
                road_pixels = get_road_pixels(semantic_image, img, dataset) #u are columns, v are rows
                u_value_road_0 = int(np.min(road_pixels[:, 0]))
                u_value_road_1 = int(np.percentile(road_pixels[:, 0], 1))
                u_value_road_5 = int(np.percentile(road_pixels[:, 0], 5))
                u_value_road_10 = int(np.percentile(road_pixels[:, 0], 10))
                u_value_road_15 = int(np.percentile(road_pixels[:, 0], 15))
                u_value_road_20 = int(np.percentile(road_pixels[:, 0], 20))
                u_value_road_25 = int(np.percentile(road_pixels[:, 0], 25))
                u_values_road = [u_value_road_0, u_value_road_1, u_value_road_5, u_value_road_10, u_value_road_15, \
                                 u_value_road_20, u_value_road_25]

                #regression
                model = distance_regression(road_rows_lidar, road_distances_lidar)
                max_extrap_dist, max_extrap_dist_99, max_extrap_dist_95, max_extrap_dist_90, max_extrap_dist_85, \
                max_extrap_dist_80, max_extrap_dist_75 = calculate_distance_from_regression(model, u_values_road)

                if dataset=="A2D2":
                    block, row = blocking(semantic_image, img, dataset)
                    row = [row]
                    calculated_dist_until_blocked = calculate_distance_from_regression(model, row)
                    if (row[0] < 800) & (row[0] > 720):
                        cars03.append(block[0])
                        cars04.append(0)
                        cars05.append(0)
                        trucks03.append(block[1])
                        trucks04.append(0)
                        trucks05.append(0)
                        ped_bc03.append(block[2])
                        ped_bc04.append(0)
                        ped_bc05.append(0)
                        blocked_dist.append(calculated_dist_until_blocked)
                    elif (row[0] < 1000) & (row[0] >= 800):
                        cars03.append(0)
                        cars04.append(block[0])
                        cars05.append(0)
                        trucks03.append(0)
                        trucks04.append(block[1])
                        trucks05.append(0)
                        ped_bc03.append(0)
                        ped_bc04.append(block[2])
                        ped_bc05.append(0)
                        blocked_dist.append(calculated_dist_until_blocked)
                    elif row[0] >= 1000:
                        cars03.append(0)
                        cars04.append(0)
                        cars05.append(block[0])
                        trucks03.append(0)
                        trucks04.append(0)
                        trucks05.append(block[1])
                        ped_bc03.append(0)
                        ped_bc04.append(0)
                        ped_bc05.append(block[2])
                        blocked_dist.append(calculated_dist_until_blocked)
                    else:
                        cars03.append(0)
                        cars04.append(0)
                        cars05.append(0)
                        trucks03.append(0)
                        trucks04.append(0)
                        trucks05.append(0)
                        ped_bc03.append(0)
                        ped_bc04.append(0)
                        ped_bc05.append(0)
                        blocked_dist.append(max_extrap_dist_99)

                    filename_lidar_split = depth_filename.split('/')
                    tstamp_meta_param = filename_lidar_split[10].split('_')[0] + filename_lidar_split[10].split('_')[1]
                    for j in bus:
                        if j['timestamp'] == timestamp:
                            speed.append(np.mean(j['flexray']['vehicle_speed']['values']))
                            acceleration_x = np.mean(j['flexray']['acceleration_x']['values'])  # acceleration in
                            # x-direction in m/s^2
                            acceleration_y = np.mean(j['flexray']['acceleration_y']['values'])  # acceleration in
                            # y-direction in m/s^2
                            accel.append(np.sqrt(np.float(acceleration_x) ** 2 + np.float(acceleration_y) ** 2))
                            longitude.append(np.mean(j['flexray']['longitude_degree']['values']))
                            latitude.append(np.mean(j['flexray']['latitude_degree']['values']))
                            continue

            max_road_distance_extrapolated.append(max_extrap_dist)
            max_road_distance_extrapolated_99.append(max_extrap_dist_99)
            max_road_distance_extrapolated_95.append(max_extrap_dist_95)
            max_road_distance_extrapolated_90.append(max_extrap_dist_90)
            max_road_distance_extrapolated_85.append(max_extrap_dist_85)
            max_road_distance_extrapolated_80.append(max_extrap_dist_80)
            max_road_distance_extrapolated_75.append(max_extrap_dist_75)

            max_road_distance_detected_75.append(np.percentile(road_distances_lidar, 75))
            max_road_distance_detected_80.append(np.percentile(road_distances_lidar, 80))
            max_road_distance_detected_85.append(np.percentile(road_distances_lidar, 85))
            max_road_distance_detected_90.append(np.percentile(road_distances_lidar, 90))
            max_road_distance_detected_95.append(np.percentile(road_distances_lidar, 95))
            max_road_distance_detected_99.append(np.percentile(road_distances_lidar, 99))
            max_road_distance_detected.append(np.max(road_distances_lidar))

        elif dataset in ["Apolloscape_stereo", "Apolloscape_semantic"]:
            road_depth, road_distances, road_pixels = calculate_road_distance(semantic_image, depth_img, dataset)

            max_road_distance_detected_75.append(np.percentile(road_distances, 75))
            max_road_distance_detected_80.append(np.percentile(road_distances, 80))
            max_road_distance_detected_85.append(np.percentile(road_distances, 85))
            max_road_distance_detected_90.append(np.percentile(road_distances, 90))
            max_road_distance_detected_95.append(np.percentile(road_distances, 95))
            max_road_distance_detected_99.append(np.percentile(road_distances, 99))
            max_road_distance_detected.append(np.max(road_distances))

            if dataset=="Apolloscape_semantic":
                block, row, col = blocking(semantic_image, img, dataset)
                calculated_dist_until_blocked = 255.0*(depth_img[row, col]/200.0)
                if (row[0] < 1720) & (row[0] > 1810):
                    cars03.append(block[0])
                    cars04.append(0)
                    cars05.append(0)
                    trucks03.append(block[1])
                    trucks04.append(0)
                    trucks05.append(0)
                    ped_bc03.append(block[2])
                    ped_bc04.append(0)
                    ped_bc05.append(0)
                    blocked_dist.append(calculated_dist_until_blocked)
                elif (row[0] < 2050) & (row[0] >= 1810):
                    cars03.append(0)
                    cars04.append(block[0])
                    cars05.append(0)
                    trucks03.append(0)
                    trucks04.append(block[1])
                    trucks05.append(0)
                    ped_bc03.append(0)
                    ped_bc04.append(block[2])
                    ped_bc05.append(0)
                    blocked_dist.append(calculated_dist_until_blocked)
                elif row[0] >= 2050:
                    cars03.append(0)
                    cars04.append(0)
                    cars05.append(block[0])
                    trucks03.append(0)
                    trucks04.append(0)
                    trucks05.append(block[1])
                    ped_bc03.append(0)
                    ped_bc04.append(0)
                    ped_bc05.append(block[2])
                    blocked_dist.append(calculated_dist_until_blocked)
                else:
                    cars03.append(0)
                    cars04.append(0)
                    cars05.append(0)
                    trucks03.append(0)
                    trucks04.append(0)
                    trucks05.append(0)
                    ped_bc03.append(0)
                    ped_bc04.append(0)
                    ped_bc05.append(0)
                    blocked_dist.append(np.percentile(road_distances, 99))

        timestamps.append(timestamp)
        j = j + 1
        folder_counter = folder_counter + 1

    except:
        print("Problem encountered at frame", i, "with timestamp", timestamp)
        i = i + 1
        j = j + 1
        folder_counter = folder_counter + 1


d = {'timestamp': timestamps, 'dist_until_blocked': blocked_dist,'max_detected': max_road_distance_detected, \
     '99% detected': max_road_distance_detected_99, '95% detected': max_road_distance_detected_95, '90% detected': \
    max_road_distance_detected_90, '85% detected': max_road_distance_detected_85, '80% detected': \
    max_road_distance_detected_80, '75% detected': max_road_distance_detected_75, 'speed': speed, \
    'acceleration': accel, 'longitude': longitude, 'latitude': latitude, 'cars 50%': cars05, 'trucks 50%': \
    trucks05, 'pedestrian 50%': ped_bc05, 'cars 40%': cars04, 'trucks 40%': trucks04, \
    'pedestrian 40%': ped_bc04, 'cars 30%': cars03, 'trucks 30%': trucks03, 'pedestrian 30%': ped_bc03}

final_result = pd.DataFrame(data=d)
final_result.to_csv('Results/Apolloscape_semantic.csv', index=False)
