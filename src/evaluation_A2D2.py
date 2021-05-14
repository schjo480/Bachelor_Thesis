from utils_A2D2 import *
import matplotlib.pyplot as plt

with open('/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/cams_lidars.json', 'r') as f:
    config = json.load(f)

root_path = "/Users/joeschmit/Dokumente/TUM/SS21/Bachelorarbeit/Code/A2D2/camera_lidar_semantic_bboxes"
file_names = sorted(glob.glob(join(root_path, '*/lidar/cam_front_center/*.npz')))

'''
Image coordinates: ------>x
                   |
                   |
                   |
                   v
                   y
'''


#Initialize array holding the euclidian distances (rows) lidar pts from the road for every image (cols) in 20180807_145028
nbr_of_pics = 5621
max_road_distance_detected = []
dist_furthest_free_road_pixel = []
cars = []
trucks = []
ped_bc = []

#Compute detected distances on road from lidar points for 942 images (20180807_145028)
for i in range(nbr_of_pics):
    #Lidar
    file_name_lidar = file_names[i]
    seg_name = file_name_lidar.split('/')[10]
    lidar_front_center = np.load(file_name_lidar)
    points = lidar_front_center['points']
    distances = lidar_front_center['distance']
    rows = lidar_front_center['row']
    cols = lidar_front_center['col']

    #Corresponding Image
    file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
    file_name_image = join(root_path, seg_name, 'camera/cam_front_center/', file_name_image)
    image_front_center = cv2.imread(file_name_image)

    #Undistort Image
    undist_image_front_center = undistort_image(image_front_center, 'front_center')

    #Json information utilised for segmentation
    file_name_image_info = file_name_image.replace('.png', '.json')

    #Corresponding semantic Image
    file_name_semantic_label = extract_semantic_file_name_from_image_file_name(file_name_image)
    file_name_semantic_label = join(root_path, seg_name, 'label/cam_front_center/', file_name_semantic_label)
    semantic_image_front_center = cv2.imread(file_name_semantic_label)
    semantic_image_front_center_undistorted = undistort_image(semantic_image_front_center, 'front_center')
    semantic_image_front_center_undistorted_rgb = cv2.cvtColor(semantic_image_front_center_undistorted, cv2.COLOR_BGR2RGB)


    ###Visualizations###

    #Image with lidar points
    '''image_with_lidar_pts = map_lidar_points_onto_image(undist_image_front_center, lidar_front_center)
    pt.fig = pt.figure(figsize=(20, 20))
    pt.imshow(image_with_lidar_pts)
    pt.axis('off')
    pt.show()
    '''

    #Semantic Image
    '''pt.fig = pt.figure(figsize=(15, 15))
    pt.imshow(semantic_image_front_center_undistorted)
    pt.axis('off')
    pt.title('label front center')
    pt.show()
    '''

    # Colorised (from semantic information) Pointcloud
    pcd_lidar_colored, road_indices = create_open3d_pc(lidar_front_center, semantic_image_front_center_undistorted)
    if any(road_indices):
        road_points = points[road_indices]
        road_distances = distances[road_indices]
        rows = (lidar_front_center['row'] + 0.5).astype(np.int)
        road_rows = rows[road_indices]
        cols = (lidar_front_center['col'] + 0.5).astype(np.int)
        road_cols = cols[road_indices]
        road_pixels_new = np.column_stack((rows, cols))
    else:
        continue
    '''o3.visualization.draw_geometries([pcd_lidar_colored])'''


    ###Computations###

    # Select furthest away (minimum y-value of) road pixel. We select the 5th percentile for better robustness and in
    # order to exclude very narrow road segments.
    if any(road_pixels_new[:, 0]):
        min_y_value_road_new = np.percentile(road_pixels_new[:, 0], 10)
        # Get y-value of furthest away lidar road point
        if any(road_points[:, 0]):
            rand_indices = np.random.choice(road_indices.shape[0], 100)
            rand_distances = road_distances[rand_indices]
            rand_rows = road_pixels_new[rand_indices, 0]

            m_per_pixel_new = np.mean(road_distances / (1208 - road_rows))
            print(i)

            max_road_distance_detected.append(np.percentile(road_distances, 90))
            dist_furthest_free_road_pixel.append(m_per_pixel_new * (1208 - min_y_value_road_new))  # distance of furthest free road pixel in meters
            c, t, pb = blocking_factors(semantic_image_front_center_undistorted)
            cars.append(c)
            trucks.append(t)
            ped_bc.append(pb)
        else:
            max_road_distance_detected.append(0)
            dist_furthest_free_road_pixel.append(0)
            c, t, pb = blocking_factors(semantic_image_front_center_undistorted)
            cars.append(c)
            trucks.append(t)
            ped_bc.append(pb)
    else:
        dist_furthest_free_road_pixel.append(np.mean(dist_furthest_free_road_pixel))
        continue



'''x = range(0, nbr_of_pics)
plt.scatter(x=x, y=dist_furthest_free_road_pixel, alpha=0.25)
plt.xlabel("Picture")
plt.ylabel("Maximal extrapolated road distance")
plt.show()

plt.scatter(x=x, y=max_road_distance_detected, alpha=0.25)
plt.xlabel("Picture")
plt.ylabel("Maximal detected road distance")
plt.show()'''


dist_furthest_free_road_pixel = np.array(dist_furthest_free_road_pixel)
np.savetxt('images_upto5621_extrapolated_distance1', dist_furthest_free_road_pixel, fmt='%1.5f')

max_road_distance_detected = np.array(max_road_distance_detected)
np.savetxt('images_upto5621_detected_distance1', max_road_distance_detected, fmt='%1.5f')

cars = np.array(cars)
np.savetxt('images_upto5621_detected_cars04', cars, fmt='%d')

trucks = np.array(trucks)
np.savetxt('images_upto5621_detected_trucks01', trucks, fmt='%d')

ped_bc = np.array(ped_bc)
np.savetxt('images_upto5621_detected_ped_bc01', ped_bc, fmt='%d')