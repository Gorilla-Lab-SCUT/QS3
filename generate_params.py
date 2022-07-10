"""generate object and camera parameters, save in given directory

usage:
cd /data2/lab-chen.yongwei/SpeckleNet/
screen -R render2
/home/chen/blender/blender-2.93.0-linux-x64/blender -b --python generate_params.py -- --pose_dir=./pose_params/ --view=1
"""


import sys
import bpy
sys.path.append("./utils/")
import utils
import argparse
import os
import numpy as np 
from mathutils import Vector
import ipdb

def parse_arg():
    """receive the input parameters and return a parser

    Returns:
        parser: parser that contain input parameters
    """
    parser = argparse.ArgumentParser(description='generate object poses and camera poses')
    parser.add_argument("--modelnet_dir", dest="modelnet_dir", type=str, default="/data3/lab-chen.yongwei/datasets/ModelNet40/", help="the directory that contain the shapenet objects")
    parser.add_argument("--pose_dir", dest="save_dir", type=str, default="./SpeckleNet_poses", help="specify the save path of images")
    parser.add_argument("--view", type=int)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_arg()
    modelnet_dir = args.modelnet_dir
    view = args.view
    pose_dir = args.save_dir + "/{}_view".format(view)

    """get object list
    """
    category_list = ["bed", "bathtub", "night_stand", "table", "monitor", "chair", "plant", 'bookshelf', 'lamp', 'sofa']
    # category_list = ["bed"]
    train_path_list, test_path_list = utils.read_obj_file(modelnet_dir, category_list)

    """create diectory to save the pose files
    """
    utils.mkdir(pose_dir)
    for category in category_list:
        utils.mkdir(pose_dir +  '/' + category )
        utils.mkdir(pose_dir +  '/' + category + '/train')
        utils.mkdir(pose_dir +  '/' + category + '/test')

    for category in category_list:
        train_obj_files = train_path_list[category]
        for obj_file in train_obj_files:
            param_dict = {}
            obj_name = obj_file.split('/')[-1].split('.')[0]
            # get the imported object scale, rotation and location
            obj,_, scale, rotation, location = utils.import_obj_file(obj_file)
            param_dict["obj_scale"] = scale
            param_dict["obj_rotation"] = rotation
            param_dict["obj_location"] = location

            """get camera parameters
            """
            camera = bpy.data.objects['Camera']
            bpy.context.scene.camera = camera
            utils.camera_point_at_origin(camera)
            camera_location = []
            camera_rotation = []
            camera_0_location = np.array([camera.location[0], camera.location[1], camera.location[2]])
            camera_0_rotation_euler = np.array([camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]])
            camera_location.append(camera_0_location)
            camera_rotation.append(camera_0_rotation_euler)
        
            for i in range(view):
                if i == 0:
                    pass
                else:
                    camera.location = camera_0_location + Vector(np.clip(0.05 * ((i+1) / view)* np.random.randn(3, 1), -1 * 0.1, 0.1))
                    camera.rotation_euler[0] = camera_0_rotation_euler[0] + float(np.clip(0.1 * ((i+1) / view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
                    camera.rotation_euler[1] = camera_0_rotation_euler[1] + float(np.clip(0.1 * ((i+1) / view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
                    camera.rotation_euler[2] = camera_0_rotation_euler[2] + float(np.clip(0.1 * ((i+1) / view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
                    bpy.ops.object.select_all(action='DESELECT')
                    camera.select_set(state=True)
                    bpy.ops.object.visual_transform_apply()
                    camera_location.append(np.array([camera.location[0], camera.location[1], camera.location[2]]))
                    camera_rotation.append(np.array([camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]]))
            
            param_dict["camera_location"] = camera_location
            param_dict["camera_rotation"] = camera_rotation
            param_dict["camera_lens"] = camera.data.lens
            param_dict["camera_sensor_fit"] = camera.data.sensor_fit
            param_dict["camera_sensor_width"] = camera.data.sensor_width
            param_dict["camera_sensor_height"] = camera.data.sensor_height
            param_dict["camera_shift_x"] = camera.data.shift_x
            param_dict["camera_shift_y"] = camera.data.shift_y

            """save parameters in an npy file
            """
            # ipdb.set_trace()
            json_file_path = pose_dir +  '/' + category + '/train/' + obj_name + ".npy"
            np.save(json_file_path, param_dict)

            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()


    for category in category_list:
        test_obj_files = test_path_list[category]
        for obj_file in test_obj_files:
            param_dict = {}
            obj_name = obj_file.split('/')[-1].split('.')[0]
            # get the imported object scale, rotation and location
            obj,_, scale, rotation, location = utils.import_obj_file(obj_file)
            param_dict["obj_scale"] = scale
            param_dict["obj_rotation"] = rotation
            param_dict["obj_location"] = location

            """get camera parameters
            """
            camera = bpy.data.objects['Camera']
            bpy.context.scene.camera = camera
            utils.camera_point_at_origin(camera)
            camera_location = []
            camera_rotation = []
            camera_0_location = np.array([camera.location[0], camera.location[1], camera.location[2]])
            camera_0_rotation_euler = np.array([camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]])
            camera_location.append(camera_0_location)
            camera_rotation.append(camera_0_rotation_euler)
        
            for i in range(view):
                if i == 0:
                    pass
                else:
                    camera.location = camera_0_location + Vector(np.clip(0.05 * ((i+1) / view)* np.random.randn(3, 1), -1 * 0.1, 0.1))
                    camera.rotation_euler[0] = camera_0_rotation_euler[0] + float(np.clip(0.1 * ((i+1) / view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
                    camera.rotation_euler[1] = camera_0_rotation_euler[1] + float(np.clip(0.1 * ((i+1) / view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
                    camera.rotation_euler[2] = camera_0_rotation_euler[2] + float(np.clip(0.1 * ((i+1) / view) * np.random.randn(1, 1), -1 * 0.1, 0.1))
                    bpy.ops.object.select_all(action='DESELECT')
                    camera.select_set(state=True)
                    bpy.ops.object.visual_transform_apply()
                    camera_location.append(np.array([camera.location[0], camera.location[1], camera.location[2]]))
                    camera_rotation.append(np.array([camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]]))
            
            param_dict["camera_location"] = camera_location
            param_dict["camera_rotation"] = camera_rotation
            param_dict["camera_lens"] = camera.data.lens
            param_dict["camera_sensor_fit"] = camera.data.sensor_fit
            param_dict["camera_sensor_width"] = camera.data.sensor_width
            param_dict["camera_sensor_height"] = camera.data.sensor_height
            param_dict["camera_shift_x"] = camera.data.shift_x
            param_dict["camera_shift_y"] = camera.data.shift_y

            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()

            """save parameters in an npy file
            """
            json_file_path = pose_dir +  '/' + category + '/test/' + obj_name + ".npy"
            np.save(json_file_path, param_dict)







        







    


