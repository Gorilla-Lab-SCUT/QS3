import os
import cv2
import numpy as np
import torch
import ipdb


def get_K(K_file):
    """[To get focal_length and camera_center from intrinsic matrix]

    Args:
        K_file ([string]): [We save camera intrinsic matrix as txt file (e.g. Camera_view_0_K.txt) and in our settings the left and right camera are the same ]

    """
    camera_K = np.genfromtxt(K_file, delimiter=' ', skip_header=1)
    focal_length = camera_K[0,0]
    camera_center = camera_K[0,2]
    return focal_length, camera_center

def get_RT(R1_file, T1_file, R2_file, T2_file):
    """[To get R and T for cam2world from extrinsics matrix]

    Args:
        R1_file ([string]): [We save left camera to World R as txt file (e.g. Camera_view_0_R.txt)]
        T1_file ([string]): [We save left camera to World T as txt file (e.g. Camera_view_0_T.txt)]
        R2_file ([string]): [We save right camera to World R as txt file (e.g. Camera_view_1_R.txt)]
        T2_file ([string]): [We save right camera to World T as txt file (e.g. Camera_view_1_T.txt)]

    """
    R1 = np.genfromtxt(R1_file, delimiter=' ', skip_header=1)
    T1 = np.genfromtxt(T1_file, delimiter=' ', skip_header=1)
    R2 = np.genfromtxt(R2_file, delimiter=' ', skip_header=1)
    T2 = np.genfromtxt(T2_file, delimiter=' ', skip_header=1)
    return R1, T1, R2, T2

def calculate_base_line(R1, T1, R2, T2):
    """[To calculate R T and base_line between the left and right camera from their cam2world extrinsics matrix]

    Args:
        R1 ([np.array]): [left camera cam2world R]
        T1 ([np.array]): [left camera cam2world T]
        R2 ([np.array]): [right camera cam2world R]
        T2 ([np.array]): [right camera cam2world T]

    """
    R2_inv = np.linalg.inv(R2)
    R = np.dot(R1, R2_inv)
    T = T1- np.dot(R, T2)
    b = np.linalg.norm(T)
    return b

def raw_disparity_calculate(left_cam, right_cam, max_disparities = 160, block_size = 19):
    """
    calculate the disparity map
    input: 
    left_cam: the left rendered image
    right_cam: the right rendered image
    """
    stereo = cv2.StereoBM_create(numDisparities=max_disparities, blockSize=block_size)

    raw_disparity = stereo.compute(left_cam, right_cam) / 16.0
    
    return raw_disparity

def raw_disparity_mask_calculate(raw_disparity):
    """
    calculate the raw mask
    input:
    raw_disparity: the disparity map
    """
    mask_index = np.where(raw_disparity<=0)
    mask_index_x = mask_index[0]
    mask_index_y = mask_index[1]
    raw_mask = np.ones_like(raw_disparity)
    raw_mask[mask_index_x,mask_index_y]=0

    return raw_mask

def raw_depth_map_generate(raw_disparity, raw_mask, image_size, base_line, focal_length, max_distance = 6, min_distance=1):
    """
    get the calculate depth map
    input:
    raw_disparity: the calculated disparity map
    raw_mask: the calculated mask
    image_size: the image size
    base line: the calculated base line
    focal_length: the focal length calculated from K matrix
    max_distance: the farthest point
    min_distance: the nearest point

    output:
    raw_Z: the point cloud 
    raw_mask: the new raw mask
    """

    raw_Z = np.zeros_like(raw_disparity)
    for ix in range(int(image_size)):
        for iy in range(int(image_size)):
            raw_d = raw_disparity[iy,ix]
            if raw_mask[iy,ix] == 1:
                raw_z = base_line* focal_length/ raw_d
                if raw_z>min_distance and raw_z<max_distance:
                    raw_Z[iy,ix] = raw_z
                else:
                    raw_mask[iy,ix] = 0

    return raw_Z, raw_mask
    
def raw_depth_and_camera_resize(depth, mask, focal_length, camera_center, image_size, scale):
    """
    calculate resized depth, mask and camera parameters
    input:
    depth: depth before resized
    mask: mask before resized
    focal_length: focal_length before resized
    camera_center: camera_center before resized
    image_size: image size
    scale: the scale factor
    """

    depth = torch.from_numpy(depth).float()
    depth = depth.unsqueeze(0).unsqueeze(0)
    depth = torch.nn.functional.interpolate(depth, scale_factor=scale)
    depth = depth.squeeze(0).squeeze(0).numpy()

    mask = torch.from_numpy(mask).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask, scale_factor=scale)
    mask = mask.squeeze(0).squeeze(0).numpy()

    focal_length *= scale
    camera_center *= scale
    image_size *= scale

    return depth, mask, focal_length, camera_center, image_size

def clean_depth_resize(depth, focal_length, camera_center, image_size, scale):
    """
    calculate resized depth, mask and camera parameters
    input:
    depth: depth before resized
    mask: mask before resized
    focal_length: focal_length before resized
    camera_center: camera_center before resized
    image_size: image size
    scale: the scale factor
    """

    # Resizing Raw Depth and Camera
    depth = torch.from_numpy(depth).float()
    depth = depth.unsqueeze(0).unsqueeze(0)
    depth = torch.nn.functional.interpolate(depth, scale_factor=scale)
    depth = depth.squeeze(0).squeeze(0).numpy()

    focal_length *= scale
    camera_center *= scale
    image_size *= scale

    return depth, focal_length, camera_center, image_size

def raw_point_cloud_rotate_generate(raw_depth, raw_mask, image_size, camera_center, focal_length, R, T):
    """
    calculate the raw point cloud
    input:
    raw_depth: the calculated depth
    raw_mask: the calculated mask
    image_size: the image size
    camera center: the camera center
    focal_length: the focal_length
    R: rotation matrix
    T: transform matrix
    output:
    canonical_pointcloud: the point cloud
    """
    canonical_pointcloud = []
    for ix in range(int(image_size)):
        for iy in range(int(image_size)):
            if raw_mask[iy,ix] == 1:
                raw_z = raw_depth[iy,ix]
                nx = (ix - camera_center ) / focal_length
                ny = (iy - camera_center ) / focal_length
                raw_x = raw_z*nx
                raw_y = raw_z*ny
                raw_point = np.matrix([raw_x, raw_y, raw_z])
                raw_point = raw_point - T
                canonical_point = np.matrix(np.linalg.inv(R))*(raw_point.T)
                canonical_point = canonical_point.T
                canonical_pointcloud.append(canonical_point)
    
    canonical_pointcloud = np.array(canonical_pointcloud)
    canonical_pointcloud = np.squeeze(canonical_pointcloud, axis=1)

    return canonical_pointcloud

def farthest_point_sample_np(xyz, npoint):
    """[FPS Sampling for original PointCloud]

    Args:
        xyz ([type]): [description]
        npoint ([type]): [description]

    Returns:
        [type]: [description]
    """
    xyz = xyz.transpose(1,0)
    xyz = np.expand_dims(xyz, axis=0)
    B, C, N = xyz.shape
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,), dtype=np.int64)
    batch_indices = np.arange(B, dtype=np.int64)
    centroids_vals = np.zeros((B, C, npoint))
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].reshape(B, C, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].copy()
        dist = np.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = np.argmax(distance, axis=1)  # get the index of the point farthest away
    fps_pcd = centroids_vals.squeeze(0).transpose(1,0)

    return fps_pcd


def clean_point_cloud_generate(clean_depth, image_size, camera_center, focal_length, R, T):
    """
    generate clean point cloud

    input:
    clean_depth: the clean depth map
    image_size: the image size
    camera_center, focal_length: the camera parameters from K intinsic matrix
    R: Rotaion matrix
    T: Transform matrix

    output: 
    clean_pointcloud: the clean point cloud
    """
    clean_pointcloud = []
    for ix in range(int(image_size)):
        for iy in range (int(image_size)):
            clean_z = clean_depth[iy, ix]
            nx = (ix - camera_center) / focal_length
            ny = (iy - camera_center) / focal_length
            clean_x = clean_z*nx
            clean_y = clean_z*ny
            clean_point = np.matrix([clean_x, clean_y, clean_z])
            clean_point = clean_point - T
            clean_point = np.matrix(np.linalg.inv(R))*(clean_point.T)
            clean_point = clean_point.T
            clean_pointcloud.append(clean_point)
    
    clean_pointcloud = np.array(clean_pointcloud).squeeze(axis=1)

    return clean_pointcloud

def crop_by_bboxes_dict(pointcloud, bbox, delta=0):
    """
    use bbox to crop the point cloud
    input:
    pointcloud: the point cloud needed to be cropped
    bbox: the bounding box of object
    output:
    crop_pointcloud: the cropped point cloud
    """
    crop_index = np.where((pointcloud[:,0]>=(np.min(bbox[:,0])+delta)) & (pointcloud[:,0]<=(np.max(bbox[:,0])-delta)) \
        & (pointcloud[:,1]>=(np.min(bbox[:,1])+delta)) & (pointcloud[:,1]<=(np.max(bbox[:,1])-delta))\
            & (pointcloud[:,2]>=(np.min(bbox[:,2])+delta)) & (pointcloud[:,2]<=(np.max(bbox[:,2])-delta)))
    crop_pointcloud = pointcloud[crop_index]
        
    return crop_pointcloud


def generate_noisy_pc(save_path, view):
    """
    generate noisy point cloud from rendered images

    input:
    save_path: the path that contain rendered images, camera parameters and the generated point clouds
    """
    raw_pc = []
    clean_pc_list = []
    for i in range(view):
        focal_length, camera_center = get_K(os.path.join(save_path, "Camera_K_left_view_{}.txt".format(i)))
        R_left, T_left, R_right, T_right = get_RT(os.path.join(save_path, "Camera_R_left_view_{}.txt".format(i)), os.path.join(save_path, "Camera_T_left_view_{}.txt".format(i)), os.path.join(save_path, "Camera_R_right_view_{}.txt".format(i)), os.path.join(save_path, "Camera_T_right_view_{}.txt".format(i)))
        

        baseline = calculate_base_line(R_left, T_left, R_right, T_right)
        left_cam = cv2.imread(os.path.join(save_path, "view_{}_f0.png".format(i)), 0)
        right_cam = cv2.imread(os.path.join(save_path, "view_{}_f1.png".format(i)), 0)
        raw_disparity = raw_disparity_calculate(left_cam, right_cam)
        raw_mask = raw_disparity_mask_calculate(raw_disparity)
        raw_z, raw_mask = raw_depth_map_generate(raw_disparity, raw_mask, 1080, baseline, focal_length)
        rraw_Z, rraw_mask, rfocal_length, rcamera_center, rimage_size = raw_depth_and_camera_resize(raw_z, raw_mask, focal_length, camera_center, 1080, 0.25)
        canonical_pointcloud = raw_point_cloud_rotate_generate(rraw_Z, rraw_mask, rimage_size, rcamera_center, rfocal_length, R_left, T_left)
        bbox = np.load(save_path + '/bbox.npy', allow_pickle=True)
        bbox = bbox[()]["bbox"]
        canonical_pointcloud = crop_by_bboxes_dict(canonical_pointcloud, bbox, delta=0)
        raw_pc.append(canonical_pointcloud)
        np.savetxt(os.path.join(save_path,  'raw_view_{}.xyz'.format(i)), canonical_pointcloud)


        #clean point cloud
        clean_depth = cv2.imread(os.path.join(save_path,'view_{}_f0_depth0001.exr'.format(i)), cv2.IMREAD_UNCHANGED)[:,:,0]
        rclean_depth,  rfocal_length, rcamera_center, rimage_size = clean_depth_resize(clean_depth, focal_length, camera_center, 1080, 0.25)
        clean_pc = clean_point_cloud_generate(rclean_depth, 270, rcamera_center, rfocal_length, R_left, T_left)
        crop_pc = crop_by_bboxes_dict(clean_pc, bbox, delta=0)
        clean_pc_list.append(crop_pc)
        np.savetxt(os.path.join(save_path,  'clean_view_{}.xyz'.format(i)), crop_pc)  
    for j in range(view ):
        if j == 0:
            pc = raw_pc[j]
        else:
            pc =  np.concatenate((raw_pc[j], pc))
    fps_pc = farthest_point_sample_np(pc, 2048)
    np.savetxt(os.path.join(save_path,  'raw_pc.xyz'.format(i)), fps_pc)

    for j in range(view ):
        if j == 0:
            pc = clean_pc_list[j]
        else:
            pc =  np.concatenate((clean_pc_list[j], pc))
    fps_pc = farthest_point_sample_np(pc, 2048)
    np.savetxt(os.path.join(save_path,  'clean_pc.xyz'.format(i)), fps_pc)

