"""scan objects from given pose parameters

usage:
cd 
CUDA_VISIBLE_DEVICES=2,3 /home/chen/blender/blender-2.93.0-linux-x64/blender ./blend_file/spot.blend -b --python scan_from_params.py -- --save_dir=/data3/lab-chen.yongwei/datasets/sim2real_datasets/from_given_pose/ --pose_dir=./pose_params/5_view/  --category_list bed
"""

import argparse
import sys
sys.path.append("./utils")
import scan_utils
import utils
import pc_utils
import numpy as np
import os
import bpy
import ipdb




def parse_arg():
    parser = argparse.ArgumentParser(description='generate object poses and camera poses')
    parser.add_argument("--blend_file", dest="blend_file", type=str, default="./blend_file/spot.blend", help="specify the blend file that contain the spot light")
    parser.add_argument("--modelnet_dir", dest="modelnet_dir", type=str, default="/data3/lab-chen.yongwei/datasets/ModelNet40/", help="the directory that contain the shapenet objects")
    parser.add_argument("--save_dir", dest="save_dir", type=str, default="./multiview_image_1021_5_view", help="specify the save path of images")
    parser.add_argument("--category_list", dest="category_list", type=str, nargs = '+', default=["Bed"], help="the category list that contain the interested objects")
    parser.add_argument("--pose_dir", type = str)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    args = parse_arg()
    category_list = args.category_list
    modelnet_dir = args.modelnet_dir
    view = args.pose_dir.split('/')[-2]
    save_dir = args.save_dir + '/' + str(view)
    pose_dir = args.pose_dir

    # traing_objs, test_objs = scan_utils.get_obj_name(modelnet_dir)
    scan_utils.init_render_setting(200)
    depth_file_output = scan_utils.add_depth_node()
    utils.mkdir(save_dir)
    for category in category_list:
        utils.mkdir(save_dir +  '/' + category )
        utils.mkdir(save_dir +  '/' + category + '/train')
        utils.mkdir(save_dir +  '/' + category + '/test')

    # training objects
    for category in category_list:
        train_objs = scan_utils.get_empty_dir(save_dir, category, "train", modelnet_dir)
        print(len(train_objs))
        for obj_name in train_objs:
            save_path = save_dir +  '/' + category + '/train/' + obj_name
            model_path = modelnet_dir +  '/' + category + '/train/' + obj_name + ".obj"
            pose_file_path = pose_dir + '/' + category + '/train/' + obj_name + ".npy"
            pose_file = np.load(pose_file_path, allow_pickle = True)[()]
            utils.mkdir(save_path)
            obj, bbox = scan_utils.import_obj_in_specific_pose(model_path, pose_file)
            np.save(os.path.join(save_path, "bbox.npy"), bbox)
            scan_utils.render_image(save_dir = save_path, depth_file_output = depth_file_output, pose_file = pose_file)

            # Deselect all
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            # ipdb.set_trace()
            bpy.ops.object.delete()

            #generate noisy point cloud
            view_num = len(pose_file["camera_location"])
            # pc_utils.generate_noisy_pc(save_path, view_num, 16)
            # try:
            #     pc_utils.generate_noisy_pc(save_path, view_num, 16)
            #     # pc_utils.generate_noisy_pc(save_path, view_num, 8)
            #     # pc_utils.generate_noisy_pc(save_path, view_num, 4)
            #     # pc_utils.generate_noisy_pc(save_path, view_num, 2)
            #     # pc_utils.generate_noisy_pc(save_path, view_num, 1)
            # except:
            #     pass
            # # ipdb.set_trace()

    # testing objects
    for category in category_list:
        test_objs = scan_utils.get_empty_dir(save_dir, category, "test", modelnet_dir)
        for obj_name in test_objs:
            save_path = save_dir +  '/' + category + '/test/' + obj_name
            model_path = modelnet_dir +  '/' + category + '/test/' + obj_name + ".obj"
            pose_file_path = pose_dir + '/' + category + '/test/' + obj_name + ".npy"
            pose_file = np.load(pose_file_path, allow_pickle = True)[()]
            utils.mkdir(save_path)
            obj, bbox = scan_utils.import_obj_in_specific_pose(model_path, pose_file)
            np.save(os.path.join(save_path, "bbox.npy"), bbox)
            scan_utils.render_image(save_dir = save_path, depth_file_output = depth_file_output, pose_file = pose_file)

            # Deselect all
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()

            #generate noisy point cloud
            view_num = len(pose_file["camera_location"])
            # pc_utils.generate_noisy_pc(save_path, view_num)
            pc_utils.generate_noisy_pc(save_path, view_num, 16)
            # ipdb.set_trace()








