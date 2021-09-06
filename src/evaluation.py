from utils import *

#enter the name of one of the following datasets: A2D2, Apolloscape_stereo, Apolloscape_semantic, KITTI, Oxford
dataset = "A2D2"

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

    depth_filename = depth_files[i]
    print(depth_filename)
    img_filename = extract_image_file_name_from_depth_file_name(depth_filename, dataset)
    img = load_image(img_filename, dataset)
    timestamp = get_timestamp(img_filename, dataset)

    if dataset in ["A2D2", "KITTI", "Oxford"]:
        #returns a dict containing all the relevant information about lidar points
        if dataset in ["A2D2", "KITTI"]:
            img_number = None
        else:
            img_folder = img_filename.split('/')[6]
            folders.append(img_folder)
            if folders[folder_counter] != folders[folder_counter-1]:
                j = 0
            img_number = j

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
                max_extrap_dist, max_extrap_dist_99, max_extrap_dist_95, max_extrap_dist_90, max_extrap_dist_85, max_extrap_dist_80, max_extrap_dist_75 = calculate_distance_from_regression(model, u_values_road)

                if dataset=="A2D2":
                    block, row = blocking2(semantic_image, img, dataset)
                    row = [row]
                    calculated_dist_until_blocked = calculate_distance_from_regression(model, row)
                    if (row[0] < 800) & (row[0] > 720):
                        cars03.append(block[0])
                        trucks03.append(block[1])
                        ped_bc03.append(block[2])
                        blocked_dist.append(calculated_dist_until_blocked)
                    elif (row[0] < 1000) & (row[0] >= 800):
                        cars04.append(block[0])
                        trucks04.append(block[1])
                        ped_bc04.append(block[2])
                        blocked_dist.append(calculated_dist_until_blocked)
                    elif row[0] >= 1000:
                        cars05.append(block[0])
                        trucks05.append(block[1])
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
                block, row, col = blocking2(semantic_image, img, dataset)
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

        '''if dataset in ["A2D2", "Apolloscape_stereo", "Apolloscape_semantic"]:
            blocking = blocking_factors(semantic_image, img, dataset, fg_mask=semantic_mask)
            cars03.append(blocking[0][0])
            cars04.append(blocking[0][1])
            cars05.append(blocking[0][2])
            trucks03.append(blocking[0][3])
            trucks04.append(blocking[0][4])
            trucks05.append(blocking[0][5])
            ped_bc03.append(blocking[0][6])
            ped_bc04.append(blocking[0][7])
            ped_bc05.append(blocking[0][8])
            other03.append(blocking[0][9])
            other04.append(blocking[0][10])
            other05.append(blocking[0][11])'''

        timestamps.append(timestamp)
        #j = j + frame_jumper
        #folder_counter = folder_counter + 1

    except:
        print("Problem encountered at frame", i, "with timestamp", timestamp)
        i = i + 1
        #j = j + 1
        #folder_counter = folder_counter + 1

timestamps = np.array(timestamps)
np.savetxt("Results/A2D2_timestamps", timestamps, delimiter=',', fmt="%s")

blocked_dist = np.array(blocked_dist)
np.savetxt("Results/A2D2_blocked_dist", blocked_dist, fmt="%1.5f")

max_road_distance_detected = np.array(max_road_distance_detected)
max_road_distance_detected_99 = np.array(max_road_distance_detected_99)
max_road_distance_detected_95 = np.array(max_road_distance_detected_95)
max_road_distance_detected_90 = np.array(max_road_distance_detected_90)
max_road_distance_detected_85 = np.array(max_road_distance_detected_85)
max_road_distance_detected_80 = np.array(max_road_distance_detected_80)
max_road_distance_detected_75 = np.array(max_road_distance_detected_75)
np.savetxt("Results/A2D2_detected 75", max_road_distance_detected_75, fmt="%1.5f")
np.savetxt("Results/A2D2_detected 80", max_road_distance_detected_80, fmt="%1.5f")
np.savetxt("Results/A2D2_detected 85", max_road_distance_detected_85, fmt="%1.5f")
np.savetxt("Results/A2D2_detected 90", max_road_distance_detected_90, fmt="%1.5f")
np.savetxt("Results/A2D2_detected 95", max_road_distance_detected_95, fmt="%1.5f")
np.savetxt("Results/A2D2_detected 99", max_road_distance_detected_99, fmt="%1.5f")
np.savetxt("Results/A2D2_detected 100", max_road_distance_detected, fmt="%1.5f")

max_road_distance_extrapolated = np.array(max_road_distance_extrapolated)
max_road_distance_extrapolated_99 = np.array(max_road_distance_extrapolated_99)
max_road_distance_extrapolated_95 = np.array(max_road_distance_extrapolated_95)
max_road_distance_extrapolated_90 = np.array(max_road_distance_extrapolated_90)
max_road_distance_extrapolated_85 = np.array(max_road_distance_extrapolated_85)
max_road_distance_extrapolated_80 = np.array(max_road_distance_extrapolated_80)
max_road_distance_extrapolated_75 = np.array(max_road_distance_extrapolated_75)
np.savetxt("Results/A2D2_extrapolated 75", max_road_distance_extrapolated_75, fmt="%1.5f")
np.savetxt("Results/A2D2_extrapolated 80", max_road_distance_extrapolated_80, fmt="%1.5f")
np.savetxt("Results/A2D2_extrapolated 85", max_road_distance_extrapolated_85, fmt="%1.5f")
np.savetxt("Results/A2D2_extrapolated 90", max_road_distance_extrapolated_90, fmt="%1.5f")
np.savetxt("Results/A2D2_extrapolated 95", max_road_distance_extrapolated_95, fmt="%1.5f")
np.savetxt("Results/A2D2_extrapolated 99", max_road_distance_extrapolated_99, fmt="%1.5f")
np.savetxt("Results/A2D2_extrapolated 100", max_road_distance_extrapolated, fmt="%1.5f")


cars03 = np.array(cars03)
np.savetxt("Results/A2D2_cars03", cars03, fmt="%1.5f")
cars04 = np.array(cars04)
np.savetxt("Results/A2D2_cars04", cars04, fmt="%1.5f")
cars05 = np.array(cars05)
np.savetxt("Results/A2D2_cars05", cars05, fmt="%1.5f")
trucks03 = np.array(trucks03)
np.savetxt("Results/A2D2_trucks03", trucks03, fmt="%1.5f")
trucks04 = np.array(trucks04)
np.savetxt("Results/A2D2_trucks04", trucks04, fmt="%1.5f")
trucks05 = np.array(trucks05)
np.savetxt("Results/A2D2_trucks05", trucks05, fmt="%1.5f")
ped_bc03 = np.array(ped_bc03)
np.savetxt("Results/A2D2_ped03", ped_bc03, fmt="%1.5f")
ped_bc04 = np.array(ped_bc04)
np.savetxt("Results/A2D2_ped04", ped_bc04, fmt="%1.5f")
ped_bc05 = np.array(ped_bc05)
np.savetxt("Results/A2D2_ped05", ped_bc05, fmt="%1.5f")
'''other03 = np.array(other03)
np.savetxt("Results/AA2D2_other03", other03, fmt="%1.5f")
other04 = np.array(other04)
np.savetxt("Results/A2D2_other04", other04, fmt="%1.5f")
other05 = np.array(other05)
np.savetxt("Results/A2D2_other05", other05, fmt="%1.5f")'''



np.savetxt("Results/A2D2_speed", speed, fmt="%1.5f")
np.savetxt("Results/A2D2_longitude", longitude, fmt="%1.5f")
np.savetxt("Results/A2D2_latitude", latitude, fmt="%1.5f")
np.savetxt("Results/A2D2_acceleration", accel, fmt="%1.5f")

d = {'timestamp': timestamps, 'dist_until_blocked': blocked_dist,'max_detected': max_road_distance_detected, '99% detected': \
    max_road_distance_detected_99, '95% detected': max_road_distance_detected_95, '90% detected': \
    max_road_distance_detected_90, '85% detected': max_road_distance_detected_85, '80% detected': \
    max_road_distance_detected_80, '75% detected': max_road_distance_detected_75, 'max_extrapolated': max_road_distance_extrapolated, '99% extrapolated':
    max_road_distance_extrapolated_99, '95% extrapolated': max_road_distance_extrapolated_95, '90% extrapolated': \
    max_road_distance_extrapolated_90, '85% extrapolated': max_road_distance_extrapolated_85, '80% extrapolated': \
    max_road_distance_extrapolated_80, '75% extrapolated': max_road_distance_extrapolated_75, 'speed': speed, \
    'acceleration': accel, 'longitude': longitude, 'latitude': latitude, 'cars 50%': cars05, 'trucks 50%': \
    trucks05, 'pedestrian 50%': ped_bc05, 'cars 40%': cars04, 'trucks 40%': trucks04, \
    'pedestrian 40%': ped_bc04, 'cars 30%': cars03, 'trucks 30%': trucks03, 'pedestrian 30%': \
    ped_bc03}

final_result = pd.DataFrame(data=d)
final_result.to_csv('Results/A2D2_new.csv', index=False)
