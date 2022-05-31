from numpy.lib.twodim_base import eye
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import mediapipe as mp
from CvFpsCalc import CvFpsCalc
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m

class Align_Depth_Eye_Track():
    def __init__(self):
        self.list_alpha = np.linspace(-87/2,87/2,640)*m.pi/180 # camera 55.47
        self.list_beta = np.linspace(-58/2,58/2,480)*m.pi/180 # camera 69.4
        self.alpha, self.beta = np.meshgrid(self.list_alpha, self.list_beta)
        self.alpha_beta = np.stack((self.alpha, self.beta), axis=2)
        self.eye = np.zeros((3,1))
        self.debug_image = None
        self.color_image = None

    def starting_camera(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # device 불러오기
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break
        if not self.found_rgb:
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Pipeline 시작
        self.profile = self.pipeline.start(self.config)

        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = self.depth_sensor.get_depth_scale()

        clipping_distance_in_meters = 1 # 1m
        clipping_distance = clipping_distance_in_meters / depth_scale

        # align object 생성
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        

    def starting_mediapipe(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--model_selection", type=int, default=0)
        parser.add_argument("--min_detection_confidence",
                            help='min_detection_confidence',
                            type=float,
                            default=0.7)

        args = parser.parse_args()

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=args.model_selection,
            min_detection_confidence=args.min_detection_confidence,
        )

        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
    
    def get_align_depth(self):
        frames = self.pipeline.wait_for_frames()
        # 640x360 depth

        # depth frame -> color frame
        self.aligned_frames = self.align.process(frames)

        self.aligned_depth_frame = self.aligned_frames.get_depth_frame()
        self.color_frame = self.aligned_frames.get_color_frame()

        # depth와 color frame의 validity 확인
        if not self.aligned_depth_frame or not self.color_frame:
            return False, False

        self.depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        self.distort_correction_depth = self.depth_image/(np.cos(self.alpha)*np.cos(self.beta))
        self.color_image = np.asanyarray(self.color_frame.get_data())
    
    def face_detect(self):
        display_fps = self.cvFpsCalc.get()
        image = self.color_image
        self.debug_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)

        self.left_eye, self.right_eye = np.array([0,0]), np.array([0,0])

        if results.detections is not None:
            for detection in results.detections:
                self.debug_image, self.left_eye, self.right_eye = self.draw_detection(self.debug_image, detection)

        cv2.putText(self.debug_image, "FPS:" + str(display_fps), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.waitKey(1)        
    
    def main(self):
        x_Coord, y_Coord, z_Coord = self.cvt2XYZ_total(self.distort_correction_depth)
        left_eye_x, left_eye_y, left_eye_z = self.cvt2XYZ(self.distort_correction_depth,self.left_eye[0], self.left_eye[1])
        right_eye_x, right_eye_y, right_eye_z = self.cvt2XYZ(self.distort_correction_depth, self.right_eye[0], self.right_eye[1])

        Eye_coord_x = np.vstack([left_eye_x,right_eye_x])
        Eye_coord_y = np.vstack([left_eye_y,right_eye_y])
        Eye_coord_z = np.vstack([left_eye_z,right_eye_z])

        self.eye = np.array([Eye_coord_x,Eye_coord_y,Eye_coord_z]).reshape(3,2)

        self.total, self.eye = np.array([x_Coord,y_Coord,z_Coord]), self.eye.mean(axis=1)

        self.total = np.array([x_Coord.reshape(-1),y_Coord.reshape(-1),z_Coord.reshape(-1)])
        
        # 눈 좌표 calibration (mm -> cm)
        self.eye[0] /= 10
        self.eye[1] /= 10 
        self.eye[2] /= 10

    def cvt2XYZ(self, data, a, b):
        x = data[a,b]*np.sin(self.alpha[a,b])/np.sqrt(1+np.tan(self.beta[a,b])*np.tan(self.beta[a,b])*np.cos(self.alpha[a,b])*np.cos(self.alpha[a,b]))
        y = x*np.tan(self.beta[a,b])/np.tan(self.alpha[a,b])
        z = x/np.tan(self.alpha[a,b])
        return x, y, z

    def cvt2XYZ_total(self,data):
        x = data*np.sin(self.alpha)/np.sqrt(1+np.tan(self.beta)*np.tan(self.beta)*np.cos(self.alpha)*np.cos(self.alpha))
        y = x*np.tan(self.beta)/np.tan(self.alpha)
        z = x/np.tan(self.alpha)
        return x, y, z   
        