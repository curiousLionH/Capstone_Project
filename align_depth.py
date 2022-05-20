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
        # self.A, self.B, self.C = [0, 0, 0]
        self.debug_image = None
        self.color_image = None

    def starting_camera(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        # print(self.pipeline)

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        # print(self.device)
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break
        if not self.found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = self.depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
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
        print("get_align_depth!!!!")
        # Streaming loop
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        self.aligned_frames = self.align.process(frames)

        # Get aligned frames
        self.aligned_depth_frame = self.aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        self.color_frame = self.aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not self.aligned_depth_frame or not self.color_frame:
            return False, False

        self.depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        self.distort_correction_depth = self.depth_image/(np.cos(self.alpha)*np.cos(self.beta))
        self.color_image = np.asanyarray(self.color_frame.get_data())
        # print(f"color_image: {self.color_image}")
        # cv2.imshow("colorimg", self.color_image)
    
    def face_detect(self):
        display_fps = self.cvFpsCalc.get()
        image = self.color_image
        self.debug_image = copy.deepcopy(image)
        # print(f"image: {self.debug_image}")
        # cv2.imshow("dbimg", self.debug_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image)

        self.left_eye, self.right_eye = np.array([0,0]), np.array([0,0])

        if results.detections is not None:
            # print("detection success")
            # print(results.detections)
            for detection in results.detections:
                # print(detection)
                # 描画
                self.debug_image, self.left_eye, self.right_eye = self.draw_detection(self.debug_image, detection)

        cv2.putText(self.debug_image, "FPS:" + str(display_fps), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        # キー処理(ESC：終了) #################################################
        cv2.waitKey(1)

        # 画面反映 #############################################################
        cv2.imshow('MediaPipe Face Detection Demo', self.debug_image)
        
    
    def draw_detection(self,image,detection):
        image_width, image_height = image.shape[1], image.shape[0]
        # キーポイント：右目
        keypoint0 = detection.location_data.relative_keypoints[0]
        keypoint0.x = int(keypoint0.x * image_width)
        keypoint0.y = int(keypoint0.y * image_height)

        cv2.circle(image, (int(keypoint0.x), int(keypoint0.y)), 5, (0, 255, 0), 2)

        # キーポイント：左目
        keypoint1 = detection.location_data.relative_keypoints[1]
        keypoint1.x = int(keypoint1.x * image_width)
        keypoint1.y = int(keypoint1.y * image_height)

        cv2.circle(image, (int(keypoint1.x), int(keypoint1.y)), 5, (0, 255, 0), 2)

        return image, np.array([int(keypoint0.y), int(keypoint0.x)]), np.array([int(keypoint1.y), int(keypoint1.x)])

    def main(self):

        rot_x_20 = np.array([[1, 0,            0],
                            [0, m.cos(m.pi/9), m.sin(m.pi/9)],
                            [0, -m.sin(m.pi/9), m.cos(m.pi/9)]])
        
        rot_y_20 = np.array([[m.cos(m.pi/9), 0, -m.sin(m.pi/9)],
                            [0,              1, 0],
                            [m.sin(m.pi/9), 0, m.cos(m.pi/9)]])

        x_Coord, y_Coord, z_Coord = self.cvt2XYZ_total(self.distort_correction_depth)

        left_eye_x, left_eye_y, left_eye_z = self.cvt2XYZ(self.distort_correction_depth,self.left_eye[0], self.left_eye[1])

        right_eye_x, right_eye_y, right_eye_z = self.cvt2XYZ(self.distort_correction_depth, self.right_eye[0], self.right_eye[1])

        Eye_coord_x = np.vstack([left_eye_x,right_eye_x])
        Eye_coord_y = np.vstack([left_eye_y,right_eye_y])
        Eye_coord_z = np.vstack([left_eye_z,right_eye_z])

        self.eye = np.array([Eye_coord_x,Eye_coord_y,Eye_coord_z]).reshape(3,2)

        self.total, self.eye = np.array([x_Coord,y_Coord,z_Coord]), self.eye.mean(axis=1)

        # self.total = np.dot(rot_y_20,np.dot(rot_x_20,np.array([x_Coord.reshape(-1),y_Coord.reshape(-1),z_Coord.reshape(-1)])))
        # self.eye = np.dot(rot_y_20,np.dot(rot_x_20,self.eye))
        self.total = np.dot(rot_x_20,np.array([x_Coord.reshape(-1),y_Coord.reshape(-1),z_Coord.reshape(-1)]))
        self.eye = np.dot(rot_x_20,self.eye)
        self.x_cal = 44.69871786340485
        self.y_cal = 101.03948852156007
        self.z_cal = 0

        # desk height
        self.eye[0] = self.eye[0] - self.x_cal
        self.eye[1] = self.eye[1] - self.y_cal
        self.eye[2] = self.eye[2] - self.z_cal

        # eye coord calibration


        self.eye[0] /= 10
        self.eye[1] /= 10 
        self.eye[2] /= 10

        print(self.eye)
        # 3D plot
        # self.plotCoord(self.total,self.eye)

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

    def plotCoord(self, total, eye):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-500,500) # x_coord
        ax.set_ylim3d(0,1000) # z_coord
        ax.set_zlim3d(-500,500) # y_coord

        plt.xlabel('x')
        plt.ylabel('y')
        
        ax.plot(total[0,:].reshape(-1,),total[1,:].reshape(-1,),total[2,:].reshape(-1,),'yo',markersize=0.1)
        ax.plot(eye[0].reshape(-1,),eye[1].reshape(-1,),eye[2].reshape(-1,),'bo',markersize=1)
        plt.show()

    def crop_screen_horizontal(self,LCD_to_pic_z):
        x_d, y_d, z_d = 0,0,0
        [x_e, y_e, z_e] = self.eye
        x_s, y_s, z_s = 0,0,-35       # LCD screen
        w = 136       # LCD width
        horizontal_angle = 70*m.pi/180
        
        eye_to_LCD_z = abs(z_s-z_e)
        LCD_to_pic_z = LCD_to_pic_z+z_s        # LCD(Webcam)으로 부터 맺힌 상(?)까지의 거리
        
        left_slope, right_slope = -eye_to_LCD_z / (x_s+w/2-x_e), -eye_to_LCD_z / (x_s-w/2-x_e)
        
        H = LCD_to_pic_z*m.tan(horizontal_angle/2)*2
        x_1, x_2 = LCD_to_pic_z*m.tan(horizontal_angle/2), -LCD_to_pic_z*m.tan(horizontal_angle/2)
        x_A = -(LCD_to_pic_z+z_e)/left_slope+x_e
        x_C = -(LCD_to_pic_z+z_e)/right_slope+x_e
        A = x_1 - x_A
        C = x_C - x_2
        B = w*(eye_to_LCD_z + LCD_to_pic_z)/eye_to_LCD_z # or x_A - x_C
        
        if A < 0:
            return round(0.5*B/H*1440)
            
        elif C < 0: # C = 0
            A = H - B/2
            return round(A/H*1440)
                        
        else:
            return round((A+B/2)/(A+B+C)*1440)
            

    def crop_screen_vertical(self, LCD_to_pic_z):
        x_d, y_d, z_d = 0,0,0       # depth camera
        [x_e, y_e, z_e] = self.eye
        x_s, y_s, z_s = 0,0,-35       # LCD screen
        l = 365       # LCD length (mm)
        vertical_angle = 102.44*m.pi/180

        eye_to_LCD_z = abs(z_s-z_e)

        LCD_to_pic_z = LCD_to_pic_z+z_s        # LCD(Webcam)으로 부터 맺힌 상(?)까지의 거리
        
        left_slope, right_slope = -eye_to_LCD_z / (y_s+l/2-y_e), -eye_to_LCD_z / (y_s-l/2-y_e)
        
        H = LCD_to_pic_z*m.tan(vertical_angle/2)*2
        y_1, y_2 = LCD_to_pic_z*m.tan(vertical_angle/2), -LCD_to_pic_z*m.tan(vertical_angle/2)
        y_A = -(LCD_to_pic_z+z_e)/left_slope+y_e
        y_C = -(LCD_to_pic_z+z_e)/right_slope+y_e
        A = y_1 - y_A
        C = y_C - y_2
        B = l*(eye_to_LCD_z + LCD_to_pic_z)/eye_to_LCD_z

        if A < 0:
            return round(0.5*B/H*2560)
            
        elif C < 0: # C = 0
            A = H - B/2
            return round(A/H*2560)
            
        else:
            return round((A+B/2)/(A+B+C)*2560)
            


        
        
if __name__ == "__main__":
    Eye = Align_Depth_Eye_Track()
    Eye.starting_camera()
    Eye.starting_mediapipe()

    vidcap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
    vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

    try:
        filtered_xy = np.zeros((0,2))
        # f = open("average.csv", "w")
        count = 0
        while(1): 
            ret, image = vidcap.read()
            # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

            Eye.get_align_depth()
            Eye.face_detect()
            Eye.main()
                    
        Eye.pipeline.stop()

    finally:
        Eye.pipeline.stop()