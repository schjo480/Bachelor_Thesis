from utils_A2D2 import *

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
dist = np.zeros((10000, 942))
dist_furthest_free_road_pixel = np.zeros((1, 942))
m_per_pixel = np.zeros(942)
min_y_value_road = np.zeros(942)
min_y_value_road_lidar = np.zeros(942)
max_road_distance_lidar = np.zeros(942)


#Compute detected distances on road from lidar points for 942 images (20180807_145028)
for i in range(942):
    #Lidar
    file_name_lidar = file_names[i]
    seg_name = file_name_lidar.split('/')[10]
    lidar_front_center = np.load(file_name_lidar)

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
    pcd_lidar_colored, road_points = create_open3d_pc(lidar_front_center, semantic_image_front_center_undistorted)
    '''o3.visualization.draw_geometries([pcd_lidar_colored])'''


    ###Computations###
    for j in range(road_points.shape[0]):
        dist[j, i] = np.sqrt(road_points[j, 0]**2 + road_points[j, 1]**2 + road_points[j, 2]**2)

    # Select furthest away (minimum y-value of) road pixel
    min_y_value_road[i] = select_furthest_road_pixel(semantic_image_front_center_undistorted)

    # Get y-value of furthest away lidar road point
    min_y_value_road_lidar[i], max_road_distance_lidar[i] = get_y_value_of_furthes_lidar_road_pt(road_pts=road_points, lidar=lidar_front_center)

    #Conversion from pixel value of min_y_value_road to distance in m, using min_y_value_road_lidar and its
    # corresponding distance
    m_per_pixel[i] = (max_road_distance_lidar[i] / (1208 - min_y_value_road_lidar[i]))
    dist_furthest_free_road_pixel[0, i] = m_per_pixel[i]*(1208 - min_y_value_road[i]) #distance of furthest free road pixel in meters

dist = dist[np.all(dist != 0, axis=1)]
print(dist)

print(dist_furthest_free_road_pixel)