# -*- coding: utf-8 -*-
from time import time, sleep
import math

# import matplotlib.pyplot as plt
import argparse
from align_depth import Align_Depth_Eye_Track

import face_recognition
import cv2
import numpy as np
from tkinter import *
import threading
import ctypes
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
import os


# UI파일 가져오기
form_class = uic.loadUiType("pyqt_SSS.ui")[0]

# GUI 클래스 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 유저 얼굴 확인 됐으면 True, 아니면 False
        self.identify_user_token = 0
        self.user_dic = {0: "jiwon", 1: "junhwan"}
        self.cap = cv2.VideoCapture(0)

        # faceID랑 faceID_alcohol이랑 겹치는 부분 뺐음 (사진 임베딩 하는 부분)
        self.known_face_encodings = []
        self.known_face_names = []

        img_files = os.listdir(".\image")
        for img_file in img_files:
            img = face_recognition.load_image_file(img_file)
            self.known_face_encodings.append(face_recognition.face_encodings(img)[0])
            self.known_face_names.append(img_file[:-4])

        # 알코올 패스 함?
        self.alcohol_pass = False
        self.alcohol_restart = False

        # 블루투스통신값(음주측정)
        self.alcohol_value = None

        # 얘가 음주측정 안내, 성공 실패, 얼굴인식 다 여기서 멘트 안내.
        # 가이드 라벨 invisible
        self.user_guide_label.setVisible(False)

        # 카메라 시작 버튼 연결
        self.camera_start_button.clicked.connect(self.camera_start)
        self.camera_start_button.clicked.connect(lambda: self.faceID_start("jiwon"))

        self.time = time()
        self.Eye_Track = Align_Depth_Eye_Track()

    # 카메라 보여주기
    def camera_show(self):

        # 라벨들 가져오기
        camera_label = self.camera_show_label
        guide_label = self.user_guide_label

        # 카메라 가져오기
        # 내 생각엔 이거 얼굴인식, 눈 인식, gui 표시까지
        # 다 하나의 사진으로 처리해야 되는 부분이 추가되어야 함
        # 이렇게 다 카메라 불러오면 너무 느려
        # cap = cv2.VideoCapture(0)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        camera_label.resize(width, height)
        guide_label.setVisible(True)
        guide_label.setText("얼굴 인식이 진행중입니다. 잠시만 기다려주세요.")
        while True:
            ret, img = self.cap.read()
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                self.Eye_Track.debug_image
                pixmap = QtGui.QPixmap.fromImage(qImg)
                camera_label.setPixmap(pixmap)
            else:
                guide_label.setText("카메라를 불러올 수 없음")
                break
        self.cap.release()
        print("Thread end.")

    # 카메라 끝
    def camera_finish(self):
        camera_label = self.camera_show_label
        # 이 자리에 뭘 보여줘야 할까...

    def camera_start(self):
        th = threading.Thread(target=self.camera_show)
        th.start()
        print("started..")

    def faceID_start(self, user):
        th = threading.Thread(target=self.faceID, args=(user,))
        th.start()
        print("started..")

    def faceID_alcohol_start(self, user):
        th = threading.Thread(target=self.faceID_alcohol, args=(user,))
        th.start()
        print("started..")

    # 음주측정 가이드 라벨 보여주기
    def guide_alcohol_check(self):
        guide_label = self.user_guide_label
        guide_label.setText("음주 측정을 진행해주세요")
        guide_label.setVisible(True)

    # 얼굴 재인식 가이드 라벨 보여주기
    def guide_face_check_again(self):
        guide_label = self.user_guide_label
        guide_label.setText("얼굴인식이 실패함. 재측정 요망")
        guide_label.setVisible(True)

    def faceID(self, user="jiwon"):
        flag = True
        print("let's start faceID")

        video_capture = self.cap

        data_path = ".\image"

        data_list_all = os.listdir(data_path)

        for data_list in data_list_all:
            path = "%s.jpg" % (user)
            img = face_recognition.load_image_file(path)
            face_encoding = face_recognition.face_encodings(img)[0]

        known_face_encodings = [face_encoding]
        known_face_names = [user]

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        while self.identify_user_token <= 1 and flag:
            try:
                # Grab a single frame of video
                ret, frame = video_capture.read()
                print(f"video_capture ret = {ret}")
                self.face_frame = frame

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(
                            known_face_encodings, face_encoding
                        )
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)

                        if name == "Unknown":
                            print("who are you")
                            self.identify_user_token += 1
                        else:
                            print("hello jiwon")
                            flag = False
                            break

                process_this_frame = not process_this_frame
            except:
                pass
        self.camera_start_button.setVisible(True)
        self.camera_start_button.setText("음주 측정 시작")
        self.user_guide_label.setText("음주 측정을 시작해주세요")
        self.camera_start_button.clicked.disconnect()
        # self.camera_start_button.clicked.disconnect(lambda: self.faceID_start("jiwon"))
        # self.camera_start_button.clicked.disconnect(self.faceID_start)

        self.camera_start_button.clicked.connect(
            lambda: self.faceID_alcohol_start("jiwon")
        )
        return

    def faceID_alcohol(self, user="jiwon"):
        print("let's start faceID")

        video_capture = self.cap

        path = "%s.jpg" % (user)
        img = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(img)[0]

        known_face_encodings = [face_encoding]
        known_face_names = [user]

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        while not self.alcohol_pass:
            try:
                # Grab a single frame of video
                ret, frame = video_capture.read()
                print(f"video_capture ret = {ret}")
                self.face_frame = frame

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(
                            known_face_encodings, face_encoding
                        )
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)

                        if name == "Unknown":
                            print("who are you")
                            self.alcohol_restart = True
                        else:
                            print("hello jiwon")
                            if self.alcohol_value:  # 음주 통과
                                self.alcohol_pass = True

                process_this_frame = not process_this_frame
            except:
                pass

    def eye_track(self):
        t = time()
        while time() - t < 1:
            self.Eye_Track.starting_camera()
            self.Eye_Track.starting_mediapipe()

            vidcap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
            vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

            try:
                filtered_xy = np.zeros((0, 2))
                # f = open("average.csv", "w")
                count = 0
                eyex_list = []
                eyey_list = []
                start_time = time()
                time_limit = 100
                flag = True
                # while(flag or time() - start_time < time_limit):
                while flag:
                    ret, image = vidcap.read()
                    # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

                    self.Eye_Track.get_align_depth()
                    self.Eye_Track.face_detect()
                    self.Eye_Track.main()
                    # if flag and self.Eye_Track.eye[0] > 0:
                    #     print("Eye detected!!!!")
                    #     start_time = time()
                    #     time_limit = 5
                    #     flag = False
                    if count > 1000:
                        flag = False
                    eyex_list.append(self.Eye_Track.eye[0])
                    eyey_list.append(self.Eye_Track.eye[1])
                    count += 1

                self.Eye_Track.pipeline.stop()
                print(f"Eye x coordinate average : {sum(eyex_list)/len(eyex_list)}")
                print(f"Eye y coordinate average : {sum(eyey_list)/len(eyey_list)}")

            finally:
                self.Eye_Track.pipeline.stop()

            # self.Eye_Track.main()

    def TK_Face(self):
        self.Tkinter = Tk()
        self.label = Label(self.Tkinter, text="Face Recognition Start")
        self.label.pack()
        self.Tkinter.mainloop()
        print("-----GUI running-----")

    def Threading_manager(self):
        self.t1 = threading.Thread(target=self.TK_Face, args=())
        user = "jiwon"
        self.t2 = threading.Thread(target=self.faceID, args=())

        self.t1.start()
        self.t2.start()
        self.t1.join()
        self.t2.join()

    def get_id(self):

        # returns id of the respective thread
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")


if __name__ == "__main__":
    print("start")
    # 실행
    app = QApplication(sys.argv)
    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()
    # 프로그램 화면을 보여주는 코드
    myWindow.show()
    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()

    # Eye = Align_Depth_Eye_Track()
    # result = GUI.faceID('jiwon')
    # print(result)
    myWindow.eye_track()
