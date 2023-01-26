import trimesh
import pyrender
import matplotlib.pyplot as plt
from rotations import *
import mediapipe as mp
import numpy as np
from render_on_image import put_render_on_image
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron


def get_render(dx, dy, dz, rx, ry, rz):
    RX = mat_rx(rx)
    RY = mat_ry(ry)
    RZ = mat_rz(rz)

    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    light_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.05],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    rotation = np.matmul(np.matmul(RX, RY), RZ)
    object_pose = rotation + np.array([
        [0.0, 0, 0, dx],
        [0.0, 0.0, 0.0, dy],
        [0.0, 0, 0.0, dz],
        [0.0, 0.0, 0.0, 0.0],
    ])
    # print("object pose", object_pose)
    # read data
    fuze_trimesh = trimesh.load('examples/models/fuze.obj')
    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

    # create scene objects
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 10.0,
                               outerConeAngle=np.pi / 3.0)
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=640 / 480)

    # create scene
    scene = pyrender.Scene()

    # add nodes to scene
    nm = pyrender.Node(mesh=mesh, matrix=object_pose)
    nl = pyrender.Node(light=light, matrix=light_pose)
    nc = pyrender.Node(camera=cam, matrix=camera_pose)
    scene.add_node(nc)
    scene.add_node(nl, parent_node=nc)
    scene.add_node(nm, parent_node=nc)

    # do rendering
    r = pyrender.OffscreenRenderer(640, 480)
    color, depth = r.render(scene)
    return color


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=5,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.2,
                                model_name='Cup') as objectron:
        rz_curr = 0 # zmienna w czasie rotacja
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = objectron.process(image)

            # Draw the box landmarks on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detected_objects:
                for detected_object in results.detected_objects:

                    mp_drawing.draw_landmarks(
                        image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                    mp_drawing.draw_axis(image, detected_object.rotation,
                                         detected_object.translation)
            # Flip the image horizontally for a selfie-view display.
            rz_curr = rz_curr + 0.1
            render = get_render(dx=-0.1, dy=-0.1, dz=-0.35,
                                rx = -np.pi/2, ry = 0, rz=rz_curr)
            image_with_render = put_render_on_image(render, cv2.flip(image, 1))

            cv2.imshow('Look at him go', image_with_render)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()