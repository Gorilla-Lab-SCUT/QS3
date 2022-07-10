import os
import bpy
import numpy as np
from mathutils import Vector, Matrix
import ipdb

def get_obj_name(modelnet_dir: str, category_list: list):
    """get all the training and test objects in modelnet directory

    Args:
        modelnet_dir (str): modelnet directory
    """
    train_objs = {}
    test_objs = {}
    
    for category in  category_list:
        if category.lower() =="cabinet":
            category = "night_stand"
        train_path = modelnet_dir + category.lower() + "/train"
        test_path = modelnet_dir + category.lower() + "/test"
        train_objs[category] = [obj.split(".")[0] for obj in os.listdir(train_path)]
        test_objs[category] = [obj.split(".")[0] for obj in os.listdir(test_path)]
    
    return train_objs, test_objs

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
        if os.path.exists(os.path.join(dir, name, "clean_pc_fps_disparity_16.xyz")):
            pass
        else:
            obj_list.append(name)
    obj_list.sort()
    # ipdb.set_trace()
    
    return obj_list


def import_obj_in_specific_pose(obj_file_path: str, pose: dict):
    """import an object in blender, then scale it in a unit cube and rotate

    Args:
        obj_file_path (str): the path to the object
        pose (dict): the pose parameters

    Returns:
        bpy.data.object, np.array: imported object, bounding box
    """

    old_objs = set(bpy.data.objects)
    bpy.ops.import_scene.obj(filepath=obj_file_path)
    imported_obj = list(set(bpy.data.objects) - old_objs)[0]
    # ipdb.set_trace()
    imported_obj.scale = pose["obj_scale"]
    imported_obj.rotation_euler = pose["obj_rotation"]
    imported_obj.location = pose["obj_location"]
    bpy.ops.object.visual_transform_apply()

    BBOX = {}
    bounding_box = np.array([imported_obj.matrix_world @ Vector(corner) for corner in imported_obj.bound_box])
    bbox_corners = np.array(bounding_box)
    BBOX["bbox"] = bbox_corners
    # bbox = scale_and_rotate(imported_obj)

    return imported_obj, BBOX

def render_image(save_dir: str, depth_file_output, pose_file: dict):
    """
    render spotted images
    input:
    save_dir: the depth and rendered image save dir
    obj: the interested object
    depth_file_output: depth render node
    """
    camera = bpy.data.objects['Camera']
    bpy.context.scene.camera = camera
    # camera_point_at_obj(camera, obj

    camera_location = pose_file["camera_location"]
    camera_rotation = pose_file["camera_rotation"]

    render_view = len(camera_location)

    # camera_0_location = Vector([camera.location[0], camera.location[1], camera.location[2]])
    # camera_0_rotation_euler = Vector([camera.rotation_euler[0], camera.rotation_euler[1], camera.rotation_euler[2]])
    # # bpy.ops.wm.save_as_mainfile(filepath="/data2/lab-chen.yongwei/BlenderProc/test.blend")
    # # ipdb.set_trace()

    # set camera intrinsic parameters
    camera.data.lens = pose_file["camera_lens"] 
    camera.data.sensor_fit = pose_file["camera_sensor_fit"]
    camera.data.sensor_width = pose_file["camera_sensor_width"] 
    camera.data.sensor_height = pose_file["camera_sensor_height"]
    camera.data.shift_x = pose_file["camera_shift_x"]
    camera.data.shift_y = pose_file["camera_shift_y"]

    
    for i in range(render_view):
        # left camera
        camera.location = camera_location[i]
        camera.rotation_euler = Vector(camera_rotation[i])

        bpy.ops.object.select_all(action='DESELECT')
        camera.select_set(state=True)
        bpy.ops.object.visual_transform_apply()

        c2location = set_projector(camera)
        bpy.data.scenes["Scene"].render.filepath = save_dir+'/view_{}_f0'.format(i)
        depth_file_output.file_slots[0].path = bpy.data.scenes["Scene"].render.filepath + "_depth"
        bpy.ops.render.render(write_still=True)
        K_0 = get_calibration_matrix_K_from_blender(camera.data)
        RT_0, _ = get_3x4_RT_matrix_from_blender(camera)
        save_K_RT(K_0 , RT_0, save_dir, "left", i)

        # right camera
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

