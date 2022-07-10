import os 
import glob
import shutil
import ipdb

#create a new directory 
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)            
		print("---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")


data_path = "/data3/lab-chen.yongwei/datasets/sim2real_datasets/from_given_pose/5_view/"
raw_save_path = "/data3/lab-chen.yongwei/datasets/sim2real_datasets/eccv_specklenet_given_pose/raw_pc_5_view_16_disparity"
# clean_save_path = "/data3/lab-chen.yongwei/datasets/Modelnet40_pc/clean_pc_multiview_point_at/"
mkdir(raw_save_path)
clean_save_path = "/data3/lab-chen.yongwei/datasets/sim2real_datasets/eccv_specklenet_given_pose/clean_pc_5_view_16_disparity"
mkdir(clean_save_path)
# jitter_save_path = "/data3/lab-chen.yongwei/datasets/Modelnet40_pc/jitter_pc/"
category = os.listdir(data_path)

for type in ["train", "test"]:
    for cat in category:
        # ipdb.set_trace()
        train_path =os.path.join(data_path, cat + "/{}".format(type))
        obj_path = os.listdir(train_path)
        # ipdb.set_trace()
        for obj_name in obj_path:
            # ipdb.set_trace()
            mkdir(os.path.join(raw_save_path, cat + "/{}".format(type)))
            mkdir(os.path.join(clean_save_path, cat + "/{}".format(type)))
            # mkdir(os.path.join(raw_save_path_25, cat + "/{}".format(type)))
            # mkdir(os.path.join(clean_save_path, "{}/".format(type) + cat))
            # mkdir(os.path.join(jitter_save_path, "{}/".format(type) + cat))
            #shutil.copyfile(os.path.join(train_path, obj_name+"/clean_pc.xyz"), os.path.join(clean_save_path,cat + "/{}".format(type), obj_name + '.xyz'))
            try:
                # ipdb.set_trace()
                shutil.copyfile(os.path.join(train_path, obj_name+"/raw_pc_fps_disparity_16.xyz"), os.path.join(raw_save_path,cat + "/{}".format(type), obj_name + '.xyz'))
                shutil.copyfile(os.path.join(train_path, obj_name+"/clean_pc_fps_disparity_16.xyz"), os.path.join(clean_save_path,cat + "/{}".format(type), obj_name + '.xyz'))
                # shutil.copyfile(os.path.join(train_path, obj_name+"/raw_pc_sample_0.xyz"), os.path.join(raw_save_path,cat + "/{}".format(type), obj_name + '_sample_0.xyz'))
                # shutil.copyfile(os.path.join(train_path, obj_name+"/clean_pc_sample_0.xyz"), os.path.join(clean_save_path,cat + "/{}".format(type), obj_name + '_sample_0.xyz'))
                # shutil.copyfile(os.path.join(train_path, obj_name+"/raw_pc_sample_1.xyz"), os.path.join(raw_save_path,cat + "/{}".format(type), obj_name + '_sample_1.xyz'))
                # shutil.copyfile(os.path.join(train_path, obj_name+"/clean_pc_sample_1.xyz"), os.path.join(clean_save_path,cat + "/{}".format(type), obj_name + '_sample_1.xyz'))
            except:
                pass
            # shutil.copyfile(os.path.join(train_path, obj_name+"/raw_denoise_0.25.xyz"), os.path.join(raw_save_path_25, cat + "/{}".format(type), obj_name + '.xyz'))
            # shutil.copyfile(os.path.join(train_path, obj_name+"/clean.xyz"), os.path.join(clean_save_path, "{}/".format(type) + cat, obj_name + '.xyz'))
            # shutil.copyfile(os.path.join(train_path, obj_name+"/Guassian.xyz"), os.path.join(jitter_save_path, "{}/".format(type) + cat, obj_name + '.xyz'))
            # ipdb.set_trace()
