import numpy as np
import cv2
import trimesh
import pyrender
from rotations import mat_rx, mat_ry, mat_rz


def rotate_object(obj_rot_matrix, angle, axis):
    result = None
    if axis == 'x':
        result = np.matmul(obj_rot_matrix, mat_rx(angle))
    elif axis == 'y':
        result = np.matmul(obj_rot_matrix, mat_ry(angle))
    elif axis == 'z':
        result = np.matmul(obj_rot_matrix, mat_rz(angle))
    return result


def calculate_poses(objectron_obj_rot, objectron_obj_trans):
    object_pose = np.zeros((4, 4), dtype=np.float64)
    object_pose[3, 3] = 1  # 1 for homogeneous matrix
    object_pose[0:3, 0:3] = objectron_obj_rot
    object_pose[0:3, 3] = objectron_obj_trans
    object_pose = rotate_object(object_pose, -90/180*np.pi, 'x')

    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return object_pose, camera_pose


def make_scene(mesh_path, object_pose, camera_pose):
    # read mesh
    fuze_trimesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    # create scene objects
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0,
                               outerConeAngle=np.pi / 6.0)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=640/480)

    # create scene
    scene = pyrender.Scene()

    # add nodes to scene
    nm = pyrender.Node(mesh=mesh, matrix=object_pose)
    nl = pyrender.Node(light=light, matrix=camera_pose)
    nc = pyrender.Node(camera=cam, matrix=camera_pose)
    scene.add_node(nc)
    scene.add_node(nl)
    scene.add_node(nm, parent_node=nc)

    return scene


def get_render(mesh_path, objectron_obj_rot, objectron_obj_trans):
    """Prepares the rendered image for given pose of object

    Args:
        mesh_path (str): Path to the mesh file
        objectron_obj_rot (numpy.ndarray): Rotation matrix from Objectron
        objectron_obj_trans (numpy.ndarray): Translation vector from Objectron

    Returns:
        numpy.ndarray: rendered image on white background
    """
    object_pose, camera_pose = calculate_poses(
        objectron_obj_rot, objectron_obj_trans)
    scene = make_scene(mesh_path, object_pose, camera_pose)

    # do rendering
    r = pyrender.OffscreenRenderer(640, 480)
    color, depth = r.render(scene)

    # display
    color_cv = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    return color_cv


if __name__ == "__main__":
    # below matrices were extracted from mug.png image by objectron
    objectron_obj_rot = np.array([[0.9987607, 0.02022175, 0.14083396],
                                  [0.04595965, 0.98727977, -0.86985153],
                                  [-0.01909464, 0.15770334, 0.47278368]])
    objectron_obj_trans = np.array([0.03409538, 0.01351676, -0.4729079])
    rendered_img = get_render('examples/models/fuze.obj',
               objectron_obj_rot, objectron_obj_trans)
    cv2.imshow('Render', rendered_img)
    cv2.waitKey(0)
