# Bachelor_Thesis
Optical Sensor Availability in Autonomous Driving

Different Python programs covering image recognition, computer vision and image segmentation in the context of optical sensor data in autonomous driving.


Data Organization and Structure

/Evaluation/
	|
	|——Code/
	|	  |
	|	  |
	|	  |——evaluation.py
	|	  |
	|	  |——file_utils.py
	|	  |
	|	  |——lidar_utils.py
	|	  |
	|	  |——image_utils.py
	|	  |
	|	  |——distance_calculation_utils.py
	|	  |
	|	  |——post_processing_utils.py
	|	  |
	|	  |——road_detection.py
	|	  |
	|	  |——visualizations.py
	|
	|
	|——Datasets/
			|
			|
			|——A2D2/
			|	  |
			|	  |
			|	  |——camera_lidar_semantic_bboxes/
			|	  |				|
			|	  |				|
			|	  |				|——20180810_142822/
			|	  |				.			|
			|	  |				.			|
			|	  |				.			|——camera/
			|	  |				.			|
			|	  |				.			|——label/
			|	  |				.			|
			|	  |				.			|——lidar/
			|	  |				.
			|	  |				|——20181204_170238/
			|	  |
			|	  |
			|	  |——camera_lidar_semantic_bus/
			|						|
			|						|
			|						|——20180810_142822/				
			|						.		  |
			|						.		  |
			|						.		  |——bus/
			|					  .
			|						|——20181204_170238/
			|						|
			|						|
			|						|——bus_signals_combined.json
			|
			|
			|——Apolloscape/
			|		  |
			|		  |
			|		  |——Apolloscape_semantic/
			|		  |				|
			|		  |				|
			|		  |				|——road02_depth/
			|		  |				|			|
			|		  |				|			|
			|		  |				|			|——Record022/
			|		  |				|			.		|
			|		  |				|			.		|
			|		  |				|			.		|——Camera5/
			|		  |				|			.
			|		  |				|			|——Record048/
			|		  |				|
			|		  |				|
			|		  |				|——road02_seg/
			|		  |							|
			|		  |							|
			|		  |							|——ColorImage/
			|		  |							|		|
			|		  |							|		|
			|		  |							|		|——Record022/
			|		  |							|		.         |
			|		  |							|		.         |
			|		  |							|		.         |——Camera5/			
			|		  |							|		.
			|		  |							|		|——Record048/
			|		  |							|
			|		  |							|
			|		  |							|——Label/
			|		  |									|
			|		  |									|
			|		  |									|——Record022/
			|		  |									.         |
			|		  |									.         |
			|		  |									.         |——Camera5/			
			|		  |									.
			|		  |									|——Record048/
			|		  |				
			|		  |
			|		  |——Apolloscape_stereo/
			|						  |
			|						  |
			|						  |——stereo_train_001/
			|						  .			|
			|						  .			|
			|						  .			|——camera_5/
			|						  .			|
			|						  .			|——disparity/
			|						  .			|
			|						  .			|——fg_mask/
			|						  .
			|						  |——stereo_train_003/
			|
			|
			|——KITTI/
			|		  |
			|		  |
			|		  |—-Raw Data/
			|			    |
			|			    |	
			|			    |——2011_09_26/
			|			    .			|
			|			    .			|
			|			    .			|——2011_09_26_drive_0001_sync/
			|			    .			.			|
			|			    .			.			|
			|			    .			.			|——image_02/
			|			    .			.			|	  |
			|			    .			.			|	  |
			|			    .			.			|	  |——data/			
			|			    .			.			|	
			|			    .			.			|
			|			    .			.			|——oxts/
			|			    .			.			|	  |
			|			    .			.			|	  |
			|			    .			.			|	  |——data/
			|			    .			.			| 
			|			    .			.			|——velodyne_points/
			|			    .			.				  |
			|			    .			.				  |
			|			    .			.				  |——data/
			|			    .			.
			|			    .			|——2011_09_26_drive_0117_sync/
			|			    .			|
			|			    .			|
			|			    .			|——calib_cam_to_cam.txt
			|			    .			|
			|			    .			|
			|			    .			|——calib_velo_to_cam.txt
			|			    .
			|			    |——2011_10_03/
			|		
			|		
			|——Oxford/
					|
					|
					|——Camera/
					|	  |
					|	  |
					|	  |——2014-05-06-12-54-54/
					|	  .			|
					|	  .			|
					|	  .			|——stereo/
					|	  .			|		|
					| 	.			|		|
					|	  .			|		|——centre/
					|	  .			|
					|	  .			|——stereo.csv
					|	  .			
					|	  |——2014-11-21-16-07-03 8/
					|
					|
					|——LIDAR/
					|	  |
					|	  |
					|	  |——2014-05-06-12-54-54/
					|	  .			|
					|	  .			|
					|	  .			|——ldmrs/
					|	  .			|
					|	  .			|
					|	  .			|——ldmrs.timestamps
					|	  .
					|	  |——2014-11-21-16-07-03 8/
					|
					|
					|——GPS/
					|	  |
					|	  |
					|	  |——2014-05-06-12-54-54/
					|	  .			|
					|	  .			|
					|	  .			|——gps/
					|	  .				|
					|	  .				|
					|	  .				|——gps.csv
					|	  .				|
					|	  .				|——ins.csv
					|	  .
					|	  |——2014-11-21-16-07-03/
					|
					|
					|——camera-models/
					|
					|
					|——extrinsics/
