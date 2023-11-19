import os
import cv2
import time
import torch
import argparse
import numpy as np
import streamlit as st
import simpleaudio as sa

# Workaround to avoid modifying "human_falling_detect_tracks" code
import sys
sys.path.append("./human_falling_detect_tracks")

from human_falling_detect_tracks.Detection.Utils import ResizePadding
from human_falling_detect_tracks.CameraLoader import CamLoader, CamLoader_Q
from human_falling_detect_tracks.DetectorLoader import TinyYOLOv3_onecls

from human_falling_detect_tracks.PoseEstimateLoader import SPPE_FastPose
from human_falling_detect_tracks.fn import draw_single

from human_falling_detect_tracks.Track.Tracker import Detection, Tracker
from human_falling_detect_tracks.ActionsEstLoader import TSSTG


def play_alert_sound():
    ALERT_SOUND_PATH = "./assets/alert.wav"
    wave_obj = sa.WaveObject.from_wave_file(ALERT_SOUND_PATH)
    wave_obj.play()

def preproc(image):
    """preprocess function for CameraLoader.
    """
    global resize_fn
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def yolo_detect(source : str = '0'):
    # Streamlit utils
    frame_placeholder, status_placeholder = st.empty(), st.empty()

    # DETECTION MODEL.
    inp_dets = 384
    detect_model = TinyYOLOv3_onecls(inp_dets, device='cuda', config_file="human_falling_detect_tracks/Models/yolo-tiny-onecls/yolov3-tiny-onecls.cfg", weight_file="human_falling_detect_tracks/Models/yolo-tiny-onecls/best-model.pth")

    # POSE MODEL.
    pose_input_size = '224x160'
    inp_pose = pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    
    # hack to load pose_model
    os.chdir("human_falling_detect_tracks")
    pose_model = SPPE_FastPose('resnet50', inp_pose[0], inp_pose[1], device='cuda')
    os.chdir("..")
    
    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG(weight_file="human_falling_detect_tracks/Models/TSSTG/tsstg-model.pth")

    global resize_fn
    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = source
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    outvid = False
    save_out = ''

    fps_time = 0
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]


        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            # VISUALIZE
            show_skeleton = True
            if track.time_since_update == 0:
                if show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                
                if action.split(':')[0] == 'Fall Down':
                    status_placeholder.error('⚠️ Alert: A fall has been detected!')
                    play_alert_sound()
                    
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame)