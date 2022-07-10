"""
render:
read the spot.blend file, import the obj file, and then set the camera and projector pointing at the object, render image
compute depth:
read the rendered images and camera parameters, then compute the depth

usage:
cd /data2/lab-chen.yongwei/SpeckleNet/
screen -R render2
CUDA_VISIBLE_DEVICES=0,1,2 /home/chen/blender/blender-2.93.0-linux-x64/blender ./blend_file/spot.blend -b --python toy_data_multiview.py -- --save_dir=/data3/lab-chen.yongwei/datasets/multiview_image_1_sample_5_view_16_disparaty --view=5 --category_list
"""


import math
import bpy
import os
import argparse
import sys
import argparse, sys, os
import random
import numpy as np
from mathutils import Matrix, Vector
import ipdb
import cv2
from random import choice
import glob
import torch

def parse_arg():
    parser = argparse.ArgumentParser(description='render blend file')
    parser.add_argument("--blend_file", dest="blend_file", type=str, default="./spot.blend", help="specify the blend file that contain the spot light")
    parser.add_argument("--modelnet_dir", dest="modelnet_dir", type=str, default="/data3/lab-chen.yongwei/datasets/ModelNet40/", help="the directory that contain the shapenet objects")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="./multiview_image_1021_5_view", help="specify the save path of images")
    parser.add_argument("--category_list", dest="category_list", type=str, nargs = '+', default=["Bed"], help="the category list that contain the interested objects")
    parser.add_argument("--view", type=int)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    return args

#create a new directory 
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")

def read_obj_file(modelnet_dir, category_list):
    """
    return the obj file path in dir
    input:
    shapenet_dir: the path that contains shapenet objects
    output:
    the shapenet obj file paths(dict: key(category):value(path list))
    """

    train_obj_path = {}
    test_obj_path = {}
    
    for category in  category_list:
        if category.lower() =="cabinet":
            category = "night_stand"
        train_path = modelnet_dir + category.lower() + "/train"
        test_path = modelnet_dir + category.lower() + "/test"
        train_obj_path[category] = glob.glob(train_path + '/*.obj')
        test_obj_path[category] = glob.glob(test_path + '/*.obj')
        # ipdb.set_trace()
    
    return train_obj_path, test_obj_path

def scale_and_rotate(obj):
    """
    scale object and rotate it the suite the blender coordinate
    input:
    obj: the object need to be processed
    output:
    BBOX: the bounding box of imported object
    """
    # Get min/max along x/y-axis from bounding box of room
    bounding_box = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    min_corner = np.min(bounding_box, axis=0)
    max_corner = np.max(bounding_box, axis=0)
    # ipdb.set_trace()

    scale_factor = 1 / np.max(np.array((max_corner[0] - min_corner[0], max_corner[1] - min_corner[1], max_corner[2] - min_corner[2])))

    obj.scale = [scale_factor, scale_factor, scale_factor]
    obj.rotation_euler = [0, 0, np.random.uniform(0,2 * np.pi)]

    bpy.ops.object.visual_transform_apply()
    move_2_center(obj)
    bpy.ops.object.visual_transform_apply()
    BBOX = {}
    bounding_box = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    bbox_corners = np.array(bounding_box)
    BBOX["bbox"] = bbox_corners
    ipdb.set_trace()

    return BBOX


def move_2_center(obj):
    """
    move the obj to the center of the world
    input:
    obj: the object needed to br moved
    """
    bbox = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    x_middle = (np.min(bbox[:, 0]) + np.max(bbox[:, 0])) / 2
    y_middle = (np.min(bbox[:, 1]) + np.max(bbox[:, 1])) / 2
    z_middle = (np.min(bbox[:, 2]) + np.max(bbox[:, 2])) / 2

    obj.location[0] -= x_middle
    obj.location[1] -= y_middle
    obj.location[2] -= z_middle






def import_obj_file(obj_file_path):
    """
    import the off file
    input:
    obj_file_path: the obj file path
    """
    old_objs = set(bpy.data.objects)
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    imported_obj = list(set(bpy.data.objects) - old_objs)[0]
    bbox = scale_and_rotate(imported_obj)

    return imported_obj, bbox

def init_render_setting(resolution):
    """
    initialize the render parameters in blender
    input:
    resolution: render resolution 
    """
    bpy.context.scene.render.resolution_x   =  resolution
    bpy.context.scene.render.resolution_y   =  resolution

    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(1.10871, 0.0132652, 1.14827), scale=(1, 1, 1))
    bpy.context.scene.camera = bpy.context.object

    bpy.context.scene.cycles.device = 'GPU'
    cpref = bpy.context.preferences.addons['cycles'].preferences
    cpref.compute_device_type = 'CUDA'
    # Use GPU devices only
    cpref.get_devices()
    for device in cpref.devices:
        device.use = True if device.type == 'CUDA' else False

def add_depth_node():
    """
    add depth node to output the depth map
    """
    bpy.context.scene.use_nodes = True
    #cerate depth output nodes
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # render_layers = nodes.new('CompositorNodeRLayers')
    for n in bpy.context.scene.node_tree.nodes:
        bpy.context.scene.node_tree.nodes.remove(n)
    bpy.context.scene.view_layers["View Layer"].use_pass_normal = True
    render_layers = nodes.new('CompositorNodeRLayers')

    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.format.color_depth = '16'
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

    return depth_file_output

def camera_point_at_obj(camera, obj):
    """
    sample a camera location around interested objects, and set the camera pointing at obj
    input:
    camera: the camera in blender
    obj: the interested objects
    """
    # bounding_box = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])
    # min_corner = np.min(bounding_box, axis=0)
    # max_corner = np.max(bounding_box, axis=0)
    # # ipdb.set_trace()
    # # center = np.array(((max_corner[0] + min_corner[0])/2, (max_corner[1] - min_corner[1])/2, (max_corner[2] - min_corner[2])/2))

    o = bpy.data.objects.new( "empty", None )
    o.location = [0,0,0]

    radius = np.random.uniform(3,5)
    elevation = np.random.uniform(0,180)
    azimuth = np.random.uniform(20,50)
    #set camera locations
    camera.location[0] = radius * np.cos(azimuth * np.pi/180) * np.sin(elevation * np.pi/180)
    camera.location[1] = radius * np.cos(azimuth * np.pi/180) * np.cos(elevation * np.pi/180)
    camera.location[2] = radius * np.sin(azimuth * np.pi/180)
    camera.data.sensor_height = camera.data.sensor_width
    camera_constraint = camera.constraints.new(type="TRACK_TO")
    camera_constraint.target = o
    camera_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    camera_constraint.up_axis = 'UP_Y'
    bpy.ops.object.select_all(action='DESELECT')
    camera.select_set(state=True)
    bpy.ops.object.visual_transform_apply()
    camera.constraints.remove(camera_constraint) 

    bpy.data.objects.remove(o)


def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))


    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraint
    #ipdb.set_trace()
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    #print(location)
    #print(Vector((np.matrix(np.array(R_world2bcam)).I @ T_world2bcam * -1)[0].T))
    
    

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam

    #print(Vector((np.matrix(np.array(R_bcam2cv)).I @ T_world2cv)[0].T))
    #print(T_world2bcam)

    # change the camera location in cv 
    T_world2cv_transform = T_world2cv - Vector((0.1,0,0))
    location_transform = Vector((np.matrix(np.array(R_world2bcam)).I @ Vector((np.matrix(np.array(R_bcam2cv)).I @ T_world2cv_transform)[0].T) * -1)[0].T)


    
    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT, location_transform

def set_projector(camera):
    """
    let the projector to have the same rotation matrix as camera, and locate in the center of stereo camera
    input:
    camera: the bpy context camera 
    """
    _, c2location =  get_3x4_RT_matrix_from_blender(camera)

    bpy.data.objects['Spot'].location[0] = ((c2location[0] - camera.location[0]) / 2 + camera.location[0])
    bpy.data.objects['Spot'].location[1] = ((c2location[1] - camera.location[1]) / 2 + camera.location[1])
    bpy.data.objects['Spot'].location[2] = ((c2location[2] - camera.location[2]) / 2 + camera.location[2])
    bpy.data.objects['Spot'].rotation_euler = camera.rotation_euler
    bpy.data.lights["Spot"].node_tree.nodes["Mapping"].inputs[3].default_value = [0.8, 0.8, 0.8]   

    return c2location

def render_image(render_view, save_dir, obj, depth_file_output):
    """
    render spotted images
    input:
    render_view: the view number needed to be rendered
    save_dir: the depth and rendered image save dir
    obj: the interested object
    depth_file_output: depth render node
    """
    camera = bpy.data.objects['Camera']
    bpy.context.scene.camera = camera
    camera_point_at_obj(camera, obj)

    camera_0_location = Vector([camera.location[0], camera.location[1], camera.location[2]])
    camera_0_rotation_euler = Vector([camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]])
    # bpy.ops.wm.save_as_mainfile(filepath="/data2/lab-chen.yongwei/BlenderProc/test.blend")
    # ipdb.set_trace()
    
    for i in range(render_view):
        if i == 0:
            pass
        else:
            # ipdb.set_trace()
            camera.location = camera_0_location + Vector(np.clip(0.05 * ((i+1) / render_view)* np.random.randn(3, 1), -1 * 0.1, 0.1))
            # print(camera_0_location)
            # ipdb.set_trace()
            camera.rotation_euler[0] = camera_0_rotation_euler[0] + float(np.clip(0.1 * ((i+1) / render_view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
            camera.rotation_euler[1] = camera_0_rotation_euler[1] + float(np.clip(0.1 * ((i+1) / render_view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
            camera.rotation_euler[2] = camera_0_rotation_euler[2] + float(np.clip(0.1 * ((i+1) / render_view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
            bpy.ops.object.select_all(action='DESELECT')
            camera.select_set(state=True)
            bpy.ops.object.visual_transform_apply()
        c2location = set_projector(camera)
        bpy.data.scenes["Scene"].render.filepath = save_dir+'/view_{}_f0'.format(i)
        depth_file_output.file_slots[0].path = bpy.data.scenes["Scene"].render.filepath + "_depth"
        bpy.ops.render.render(write_still=True)
        #get camera parameters
        K_0 = get_calibration_matrix_K_from_blender(camera.data)
        RT_0, _ = get_3x4_RT_matrix_from_blender(camera)
        save_K_RT(K_0 , RT_0, save_dir, "left", i)


        camera.location[0] = c2location[0] 
        camera.location[1] = c2location[1] 
        camera.location[2] = c2location[2] 
        camera.rotation_euler = camera.rotation_euler
        bpy.ops.object.visual_transform_apply()

        bpy.data.scenes["Scene"].render.filepath = save_dir+'/view_{}_f1'.format(i)
        depth_file_output.file_slots[0].path = bpy.data.scenes["Scene"].render.filepath + "_depth"
        bpy.ops.render.render(write_still=True)
        K_1 = get_calibration_matrix_K_from_blender(camera.data)
        RT_1, _ = get_3x4_RT_matrix_from_blender(camera)
        save_K_RT(K_1 , RT_1, save_dir, "right", i)

def save_blend(save_path):
    """
    save current blend file in the path
    input:
    save_path: the saving path of the blend file
    """
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath = save_path + "/scene.blend")

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    """
    get the camera sensor size
    """
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    """
    get the camera sensor fit
    """
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_calibration_matrix_K_from_blender(camd):
    """
    get the K matrix from camera data
    """
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

def save_K_RT(K , RT, save_dir, side, view):
    """
    write K, R, T file to save directory
    input:
    K: the intrinsic matrix
    R: the rotation matrix
    T: the transform matrix
    side: the left side or right side
    view: the view number
    """
    K = np.matrix(K)
    RT = np.matrix(RT)
    # ipdb.set_trace()

    camera_file_R = '{}/Camera_R_{}_view_{}.txt'.format(save_dir, side, view)
    file = open(camera_file_R,'w')
    file.write('R')
    file.write('\n')
    for i in range(0,3):
        for j in range(0,3):
            file.write(str(RT[i,j]))
            file.write(' ')
        file.write('\n')
    file.close()

    camera_file_T = '{}/Camera_T_{}_view_{}.txt'.format(save_dir, side, view)
    file = open(camera_file_T,'w')
    file.write('T')
    file.write('\n')
    for i in range(0,3):
        file.write(str(RT[i,3]))
        file.write(' ')
        file.write('\n')
    file.close()

    camera_file_K = '{}/Camera_K_{}_view_{}.txt'.format(save_dir, side, view)
    file = open(camera_file_K,'w')
    file.write('K')
    file.write('\n')
    for i in range(0,3):
        for j in range(0,3):
            file.write(str(K[i,j]))
            file.write(' ')
        file.write('\n')
    file.write('\n')
    
    file.close()


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

    # raw_disparity = stereo.compute(left_cam, right_cam) / 16.0
    # ipdb.set_trace()
    raw_disparity = stereo.compute(left_cam, right_cam) / 2.0
    raw_disparity = np.rint(raw_disparity)
    raw_disparity /= 8.0
    
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

    # print("Resizing Raw Depth and Camera")
    depth = torch.from_numpy(depth).float()
    depth = depth.unsqueeze(0).unsqueeze(0)
    # ipdb.set_trace()
    depth = torch.nn.functional.interpolate(depth, scale_factor=scale)
    depth = depth.squeeze(0).squeeze(0).numpy()

    mask = torch.from_numpy(mask).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = torch.nn.functional.interpolate(mask, scale_factor=scale)
    mask = mask.squeeze(0).squeeze(0).numpy()

    # print("Nums where mask equals 1:", np.array(np.where(mask==1)).shape[1])

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

    # print("Resizing Raw Depth and Camera")
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
    # print("Current PointCloud Nums:", N)
    # print("Propose DownSampling PointCloud Nums:", npoint)
    centroids = np.zeros((B, npoint), dtype=np.int64)
    distance = np.ones((B, N)) * 1e10
    # ipdb.set_trace()
    # print('B',B)
    # print("N",N)
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

    # print("FPS DONE")

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
        # ipdb.set_trace()
        left_cam = cv2.imread(os.path.join(save_path, "view_{}_f0.png".format(i)), 0)
        right_cam = cv2.imread(os.path.join(save_path, "view_{}_f1.png".format(i)), 0)
        # ipdb.set_trace()
        raw_disparity = raw_disparity_calculate(left_cam, right_cam)
        raw_mask = raw_disparity_mask_calculate(raw_disparity)
        raw_z, raw_mask = raw_depth_map_generate(raw_disparity, raw_mask, 1080, baseline, focal_length)
        # ipdb.set_trace()
        rraw_Z, rraw_mask, rfocal_length, rcamera_center, rimage_size = raw_depth_and_camera_resize(raw_z, raw_mask, focal_length, camera_center, 1080, 0.25)
        canonical_pointcloud = raw_point_cloud_rotate_generate(rraw_Z, rraw_mask, rimage_size, rcamera_center, rfocal_length, R_left, T_left)
        # ipdb.set_trace()
        bbox = np.load(save_path + '/bbox.npy', allow_pickle=True)
        # ipdb.set_trace()
        bbox = bbox[()]["bbox"]
        canonical_pointcloud = crop_by_bboxes_dict(canonical_pointcloud, bbox, delta=0)
        raw_pc.append(canonical_pointcloud)
        np.savetxt(os.path.join(save_path,  'raw_view_{}.xyz'.format(i)), canonical_pointcloud)
        # fps_pc = farthest_point_sample_np(canonical_pointcloud, 2048)


        # np.savetxt(os.path.join(save_path,  'raw.xyz'), fps_pc)

        #clean point cloud
        clean_depth = cv2.imread(os.path.join(save_path,'view_{}_f0_depth0001.exr'.format(i)), cv2.IMREAD_UNCHANGED)[:,:,0]
        rclean_depth,  rfocal_length, rcamera_center, rimage_size = clean_depth_resize(clean_depth, focal_length, camera_center, 1080, 0.25)
        clean_pc = clean_point_cloud_generate(rclean_depth, 270, rcamera_center, rfocal_length, R_left, T_left)
        crop_pc = crop_by_bboxes_dict(clean_pc, bbox, delta=0)
        # clean_pc_crop = farthest_point_sample_np(crop_pc, 2048)
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

def get_empty_dir(data_path, category, type, model_dir):
    """
    get the empty path that need to generate point cloud 

    input:
    data_path: the generated point clouds path
    category: the category need to be checked
    type: train or test
    model_dir: the modelnet40 path

    return:
    empty_list: the empty obj file name
    """
    dir = os.path.join(data_path, category, type)
    # names = os.listdir(dir)
    model_path = os.path.join(model_dir, category, type)
    import glob
    names = glob.glob(model_path + "/*.obj")
    names = [name.split('/')[-1].split('.')[0] for name in names]
    obj_list = []
    # ipdb.set_trace()
    for name in names:
        if os.path.exists(os.path.join(dir, name, "raw_pc.xyz")):
            pass
        else:
            obj_list.append(os.path.join(model_path, name + ".obj"))
    
    return obj_list


args = parse_arg()    
save_dir = args.save_dir

if args.category_list == "all":
    data_path = "/data3/lab-chen.yongwei/datasets/ModelNet40/"
    category_list = set(os.listdir(data_path))
    category_list = category_list - set(["bed", "bathtub", "night_stand", "table", "monitor", "chair", "plant", 'shape_names.txt', 'bookshelf', 'lamp', 'sofa'])
    args.category_list = category_list 
else:
    pass

train_path_list, test_path_list = read_obj_file(args.modelnet_dir, args.category_list)
# read_blend(args.blend_file)
init_render_setting(1080)
depth_file_output = add_depth_node()

#create new directory to save images
mkdir(args.save_dir)
# ipdb.set_trace()
for category in args.category_list:
    mkdir(args.save_dir +  '/' + category )
    mkdir(args.save_dir +  '/' + category + '/train')
    mkdir(args.save_dir +  '/' + category + '/test')

for category in args.category_list:
    train_obj_files = train_path_list[category]


    #fix the ungenerated objs
    train_obj_files = get_empty_dir(args.save_dir, category, "train", "/data3/lab-chen.yongwei/datasets/ModelNet40/")
    # ipdb.set_trace()
    # train_obj_files = ["/data3/lab-chen.yongwei/datasets/ModelNet40/lamp/train/lamp_0027.obj"]
    

    old_objs = set(bpy.data.objects)
    for obj_file in train_obj_files:
        obj_name = obj_file.split('/')[-1].split('.')[0]
        save_path = args.save_dir +  '/' + category + '/train/' + obj_name
        mkdir(save_path)
        obj, bbox = import_obj_file(obj_file)
        np.save(os.path.join(save_path, "bbox.npy"), bbox)
        # ipdb.set_trace()
        render_image(render_view = args.view, save_dir = save_path, obj = obj, depth_file_output = depth_file_output)
        # ipdb.set_trace()

        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

        #generate noisy point cloud
        generate_noisy_pc(save_path, args.view)

        
        
    
    test_obj_files = test_path_list[category]

    #fix empty bugs
    test_obj_files = get_empty_dir( args.save_dir, category, "test", "/data3/lab-chen.yongwei/datasets/ModelNet40/")
    for obj_file in test_obj_files:
        obj_name = obj_file.split('/')[-1].split('.')[0]
        save_path = args.save_dir + '/' + category + '/test/' + obj_name
        mkdir(save_path)
        obj, bbox = import_obj_file(obj_file)
        np.save(os.path.join(save_path, "bbox.npy"), bbox)
        render_image(render_view = args.view, save_dir = save_path, obj = obj, depth_file_output = depth_file_output)

        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

        #generate noisy point cloud
        generate_noisy_pc(save_path, args.view)
        
        # ipdb.set_trace()
    
    



