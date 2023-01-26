import trimesh
import pyrender
import matplotlib.pyplot as plt
from rotations import *
import mediapipe as mp
import numpy as np
from render_on_image import put_render_on_image
import cv2
from rendering import get_render

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron
nescafe_path = 'examples/models/nescafe_mug.obj'
fuze_path = 'examples/models/fuze.obj'
coffe_path = 'examples/models/final/model.obj'
final_path = 'examples/models/final/model.obj'

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_objectron.Objectron(static_image_mode=False,
                                max_num_objects=1,
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
                    render = get_render(final_path,
                                        detected_object.rotation, detected_object.translation)
                    image = put_render_on_image(render, image)

            cv2.imshow('Cup rendering', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()