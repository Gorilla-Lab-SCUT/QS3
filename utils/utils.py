import numpy as np
import glob
import os
import bpy
from mathutils import Matrix, Vector
import ipdb


def read_obj_file(modelnet_dir: str, category_list: list) -> [list, list]:
    """return obj file path list, including training objects and testing objects

    Args:
        modelnet_dir (str): the modelnet path
        category_list (list): the category that we need to use

    Returns:
        [list, list]: training object list, testin object list
    """


    train_obj_path = {}
    test_obj_path = {}
    
    for category in  category_list:
        # same to PointDAN, the cabinet category in ModelNet is renamed as night_stand category
        if category.lower() =="cabinet":
            category = "night_stand"
        train_path = modelnet_dir + category.lower() + "/train"
        test_path = modelnet_dir + category.lower() + "/test"
        train_obj_path[category] = glob.glob(train_path + '/*.obj')
        test_obj_path[category] = glob.glob(test_path + '/*.obj')
    
    return train_obj_path, test_obj_path



#create a new directory 
def mkdir(path: str) -> None:
    """check if given path exists, if not, create one

    Args:
        path (str): the input path
    """
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)
        print("---  new folder {}  ---".format(path))
    else:
        print("---  folder already exists!  ---")
    

def import_obj_file(obj_file_path: str):
    """import an object in blender, then scale it in a unit cube and rotate

    Args:
        obj_file_path (str): the path to the object

    Returns:
        bpy.data.object, np.array, float, float: imported object, bounding box, scale, z-rotation
    """
    old_objs = set(bpy.data.objects)
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    imported_obj = list(set(bpy.data.objects) - old_objs)[0]
    bbox = scale_and_rotate(imported_obj)

    return imported_obj, bbox


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

def camera_point_at_origin(camera):
    """
    sample a camera location around interested objects, and set the camera pointing at obj
    input:
    camera: the camera in blender
    obj: the interested objects
    """
    o = bpy.data.objects.new( "empty", None )
    o.location = [0,0,0]

    radius = np.random.uniform(3, 5)
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
    model_path = os.path.join(model_dir, category, type)
    import glob
    names = glob.glob(model_path + "/*.obj")
    names = [name.split('/')[-1].split('.')[0] for name in names]
    obj_list = []
    for name in names:
        if os.path.exists(os.path.join(dir, name, "raw_pc.xyz")):
            pass
        else:
            obj_list.append(os.path.join(model_path, name + ".obj"))
    
    return obj_list
