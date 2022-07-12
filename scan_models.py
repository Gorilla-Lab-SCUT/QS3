"""
workflow:
render phase:
read the spot.blend file, import the obj file, and then set the camera and projector pointing at the object, render image
compute depth phase:
read the rendered images and camera parameters, then compute the depth
"""

import bpy
import os
import argparse
import sys
sys.path.append("./")
import argparse, sys, os 
import numpy as np
from utils.utils import mkdir, read_obj_file, import_obj_file
from utils.pc_utils import generate_noisy_pc
from utils.scan_utils import get_empty_dir, init_render_setting, add_depth_node, render_image
import shutil

def parse_arg():
    parser = argparse.ArgumentParser(description='render blend file')
    parser.add_argument("--blend_file", dest="blend_file", type=str, default="./spot.blend", help="specify the blend file that contain the spot light")
    parser.add_argument("--modelnet_dir", dest="modelnet_dir", type=str, default=None, help="the directory that contain the modelnet 3d objects")
    parser.add_argument("--raw_data_dir", dest="raw_data_dir", type=str, default="./raw_data", help="specify the save path of images")
    parser.add_argument("--noisy_pc_save_path", dest="noisy_pc_save_path", type=str, default="./noisy_point_clouds", help="specify the save path of noisy point clouds")
    parser.add_argument("--clean_pc_save_path", dest="clean_pc_save_path", type=str, default="./clean_point_clouds", help="specify the save path of clean point clouds")
    parser.add_argument("--category_list", dest="category_list", type=str, nargs = '+', default=["Bed"], help="the category list that contain the interested objects")
    parser.add_argument("--view", type=int)
    parser.add_argument("--keep_raw_data", type=bool, default=False, help="if False, delete the raw data directory after finishing generation")

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    return args

def save_blend(save_path):
    """
    save current blend file in the path
    input:
    save_path: the saving path of the blend file
    """
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath = save_path + "/scene.blend")


args = parse_arg()    
raw_data_dir = args.raw_data_dir
np.random.seed(0)

train_path_list, test_path_list = read_obj_file(args.modelnet_dir, args.category_list)
init_render_setting(1080)
depth_file_output = add_depth_node()

# create new directory to save images
mkdir(args.raw_data_dir)
for category in args.category_list:
    mkdir(args.raw_data_dir +  '/' + category )
    mkdir(args.raw_data_dir +  '/' + category + '/train')
    mkdir(args.raw_data_dir +  '/' + category + '/test')

for category in args.category_list:
    train_obj_files = train_path_list[category]


    #fix the ungenerated objs
    train_obj_files = get_empty_dir(args.raw_data_dir, category, "train", args.modelnet_dir)
    
    old_objs = set(bpy.data.objects)
    for obj_file in train_obj_files:
        obj_name = obj_file.split('/')[-1].split('.')[0]
        save_path = args.raw_data_dir +  '/' + category + '/train/' + obj_name
        mkdir(save_path)
        obj, bbox = import_obj_file(obj_file)
        np.save(os.path.join(save_path, "bbox.npy"), bbox)
        render_image(render_view = args.view, save_dir = save_path, depth_file_output = depth_file_output)

        # Deselect all
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.delete()

        #generate noisy point cloud
        generate_noisy_pc(save_path, args.view)
    
    test_obj_files = test_path_list[category]

    #fix empty bugs
    test_obj_files = get_empty_dir( args.raw_data_dir, category, "test", args.modelnet_dir)
    for obj_file in test_obj_files:
        obj_name = obj_file.split('/')[-1].split('.')[0]
        save_path = args.raw_data_dir + '/' + category + '/test/' + obj_name
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
        

# generate two directories that save clean pont clouds and noisy point clouds respectively
mkdir(args.noisy_pc_save_path)
mkdir(args.clean_pc_save_path)
category = os.listdir(args.raw_data_dir)

for type in ["train", "test"]:
    for cat in category:
        train_path =os.path.join(args.raw_data_dir, cat + "/{}".format(type))
        obj_path = os.listdir(train_path)
        for obj_name in obj_path:
            mkdir(os.path.join(args.noisy_pc_save_path, cat + "/{}".format(type)))
            mkdir(os.path.join(args.clean_pc_save_path, cat + "/{}".format(type)))
            try:
                shutil.copyfile(os.path.join(train_path, obj_name+"/raw_pc.xyz"), os.path.join(args.noisy_pc_save_path, cat + "/{}".format(type), obj_name + '.xyz'))
                shutil.copyfile(os.path.join(train_path, obj_name+"/clean_pc.xyz"), os.path.join(args.clean_pc_save_path, cat + "/{}".format(type), obj_name + '.xyz'))
            except:
                pass

if args.keep_raw_data == False:
    shutil.rmtree(args.raw_data_dir)


