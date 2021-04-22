import os
import matplotlib.pyplot as plt
import numpy as np
from projection_utils import *


'''def render_image_with_boxes(img, objects, calib):
    """
    Show image with 3D boxes
    """
    # projection matrix
    P_rect2cam2 = calib['P2'].reshape((3, 4))

    img1 = np.copy(img)
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pixelcoord = map_box_to_image(obj, P_rect2cam2)
        img1 = draw_projected_box3d(img1, box3d_pixelcoord)

    plt.imshow(img1)
    plt.yticks([])
    plt.xticks([])
    plt.show()
'''

'''def render_lidar_with_boxes(pc_velo, objects, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pc_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pc_velo[:, 0] > 0)
                    )[0]
    imgfov_pc_velo = pc_velo[inds, :]

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 500))

    draw_lidar(imgfov_pc_velo, fig=fig)

    # Projection matrix
    proj_cam2_2_velo = project_cam2_to_velo(calib)

    # Draw objects on lidar
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # Project boxes from camera to lidar coordinate
        boxes3d_pts = project_camera_to_lidar(obj.in_camera_coordinate(), proj_cam2_2_velo)

        # Draw boxes
        draw_gt_boxes3d(boxes3d_pts, fig=fig)
    mlab.show()'''


def render_lidar_on_image(pts_velo, img, calib, img_width, img_height):
    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_velo.transpose()
    depth_of_pts_in_img = imgfov_pc_velo[:, 0]

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=-1)
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    return img, depth_of_pts_in_img


if __name__ == '__main__':
    # Load image, calibration file, label bbox
    rgb = cv2.cvtColor(cv2.imread("/Volumes/Extreme SSD/Bachelorarbeit/KITTI/Raw Data/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000001.png"), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    rgb2 = cv2.cvtColor(cv2.imread("/Volumes/Extreme SSD/Bachelorarbeit/KITTI/Raw Data/2011_09_29/2011_09_29_drive_0004_sync/image_02/data/0000000148.png"), cv2.COLOR_BGR2RGB)


    # Load calibration
    calib = read_calib_file("/Volumes/Extreme SSD/Bachelorarbeit/KITTI/Raw Data/2011_09_26/calib_cam_to_cam.txt")

    # Load labels
    #labels = load_label('data/000114_label.txt')

    # Load Lidar PC
    pc_velo = load_velo_scan("/Volumes/Extreme SSD/Bachelorarbeit/KITTI/Raw Data/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000001.bin")[:, :3]

    pc_velo2 = load_velo_scan("/Volumes/Extreme SSD/Bachelorarbeit/KITTI/Raw Data/2011_09_29/2011_09_29_drive_0004_sync/velodyne_points/data/0000000148.bin")[:, :3]
    #render_image_with_boxes(rgb, labels, calib)
    #render_lidar_with_boxes(pc_velo, labels, calib, img_width=img_width, img_height=img_height)
    lid_on_image, depth = render_lidar_on_image(pc_velo2, rgb2, calib, img_width, img_height)
    #cv2.imshow(lid_on_image)
    print(max(depth))
    print(min(depth))
    print(np.median(depth))