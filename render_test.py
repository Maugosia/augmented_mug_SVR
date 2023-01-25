import numpy as np
import cv2
import trimesh
import pyrender
import matplotlib.pyplot as plt
from rotations import *


def rendering():
    s = np.sqrt(2) / 2

    RX = mat_rx(np.pi/2)
    RZ = mat_rz(-np.pi/2)
    print("RX", RX)
    print("RZ", RZ)
    camera_pose = np.array([
        [0.0, -s, s, 0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, s, s, 0.45],
        [0.0, 0.0, 0.0, 1.0],
    ])
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    object_pose = np.matmul(RX, RZ)+np.array([
        [0.0, 0, 0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0, 0.0, -0.65],
        [0.0, 0.0, 0.0, 0.0],
    ])
    print("object pose", object_pose)
    # read data
    fuze_trimesh = trimesh.load('examples/models/fuze.obj')
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

    # do rendering
    r = pyrender.OffscreenRenderer(640, 480)
    color, depth = r.render(scene)

    # display
    color_cv = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    tf = scene.get_pose(nm)
    print("get mesh pose", tf)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    cv2.imshow('MediaPipe Objectron', color_cv)

    plt.show()


if __name__ == "__main__":
    rendering()
