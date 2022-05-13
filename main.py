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
import json

# UI파일 가져오기
form_class = uic.loadUiType("pyqt_SSS.ui")[0]

# GUI 클래스 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 유저들 (조원들) 키 정보 불러오기
        with open("./info.json", "r", encoding="UTF8") as file:
            self.users_height_data = json.load(file)

        # 유저 얼굴 확인 됐으면 True, 아니면 False
        self.identify_user_token = 0
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
        # 0:다시불어, 1:정상, 2:음주상태
        self.alcohol_value = None

        # 얘가 음주측정 안내, 성공 실패, 얼굴인식 다 여기서 멘트 안내.
        # 가이드 라벨 invisible
        self.user_guide_label.setVisible(False)

        # 카메라 시작 버튼 연결
        # 기존에 버튼에 2개의 함수 연결하던것을 check_ID_Password로 합침
        self.camera_start_button.clicked.connect(self.check_ID_Password)

        self.time = time()
        self.Eye_Track = Align_Depth_Eye_Track()

        # 사용자 눈 좌표
        self.eye_pos = [0, 0, 0]

    def camera_start(self):
        th = threading.Thread(target=self.camera_show)
        th.start()
        print("camera_start")

    def faceID_start(self, user):
        th = threading.Thread(target=self.faceID, args=(user,))
        th.start()
        print("faceID_start")

    def alcohol_value_update_start(self):
        th = threading.Thread(target=self.alcohol_value_update)
        th.start()
        print("alcohol_value_update_start")

    def faceID_alcohol_start(self, user):
        th = threading.Thread(target=self.faceID_alcohol, args=(user,))
        th.start()
        print("faceID_alcohol_start")

    def eye_track_start(self):
        th = threading.Thread(target=self.eye_track)
        th.start()
        print("eye_track_start")

    # 기존에 button 하나에 얼굴인식이랑 사진 보여주기를 모두 시행했던 부분을
    # ID랑 비밀번호 체크 후에 valid 할 때에만 진행될 수 있도록 변경.
    def check_ID_Password(self):

        # 유저 이름 가져오기
        self.user_id = self.IDTextField.text()
        self.user_password = self.passwordTextField.text()
        # 입력 칸 다시 비우기
        self.IDTextField.setText("")
        self.passwordTextField.setText("")

        # 1. 유저 이름이 info.json 그니까 정보에 있는지 확인
        # 2. 패스워드가 일치하는지 확인
        if (self.user_id not in self.users_height_data) or (
            self.user_password != self.users_height_data[self.user_id]["password"]
        ):
            self.user_guide_label.setVisible(True)
            self.user_guide_label.setText("ID 혹은 Password를 다시 확인 해 주세요")
            return

        # 위 두 조건을 모두 패스하면 이름, 키 정보 저장하고 camera_start랑 faceID_start 실행 (쓰레드 실행)
        self.user_name = self.users_height_data[self.user_id]["name"]
        self.user_height = self.users_height_data[self.user_id]["height"]
        self.camera_start()
        self.faceID_start(self.user_name)

    # 카메라 보여주기
    def camera_show(self):

        # 라벨들 가져오기
        camera_label = self.camera_show_label
        guide_label = self.user_guide_label
        camera_label.resize(width, height)
        guide_label.setVisible(True)

        # 카메라 가져오기
        # 내 생각엔 이거 얼굴인식, 눈 인식, gui 표시까지
        # 다 하나의 사진으로 처리해야 되는 부분이 추가되어야 함
        # 이렇게 다 카메라 불러오면 너무 느려
        # cap = cv2.VideoCapture(0)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
                guide_label.setText("카메라를 불러올 수 없습니다. 잠시 후에 다시 시도해주세요. ")
                break
        self.cap.release()
        print("Thread end.")

    def faceID(self, user="jiwon"):
        flag = True
        print("let's start faceID")

        video_capture = self.cap

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
                            self.known_face_encodings, face_encoding
                        )
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]

                        face_names.append(name)

                        if name == "Unknown":
                            print("who are you")
                            self.identify_user_token += 1
                        else:
                            print("hello " + user)
                            flag = False
                            break

                process_this_frame = not process_this_frame
            except:
                pass
        self.camera_start_button.setVisible(True)
        self.camera_start_button.setText("음주 측정 시작")
        self.user_guide_label.setText("사용자 인식이 완료되었습니다. 음주 측정을 시작해주세요")
        self.camera_start_button.clicked.disconnect()
        # self.camera_start_button.clicked.disconnect(lambda: self.faceID_start("jiwon"))
        # self.camera_start_button.clicked.disconnect(self.faceID_start)

        self.camera_start_button.clicked.connect(
            lambda: self.faceID_alcohol_start(user)
        )
        return

    # 아두이노로 부터 값을 지속적으로 갱신하는 함수
    def alcohol_value_update(self):
        self.alcohol_value  # = 블루투스 어쩌구 저쩌구...

    def faceID_alcohol(self, user="jiwon"):
        print("let's start faceID_alcohol")

        video_capture = self.cap

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
                            self.known_face_encodings, face_encoding
                        )
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]

                        face_names.append(name)

                        if name == "Unknown":
                            print("who are you")
                            self.alcohol_restart = True
                        else:
                            print("알콜측정 얼굴인식 hello " + user)
                            if self.alcohol_value == 1:  # 음주 통과
                                self.alcohol_pass = True

                process_this_frame = not process_this_frame
            except:
                pass

        print("좌석 조절 시작 (눈 위치 인식 시작)")
        self.camera_start_button.setVisible(True)
        self.camera_start_button.setText("눈 위치 인식 시작")
        self.user_guide_label.setText(
            "사용자의 안전을 위하여 좌석 및 사이드미러 조절이 진행될 예정입니다. 자연스럽게 정면을 바라봐주세요. "
        )
        self.camera_start_button.clicked.disconnect()

        self.camera_start_button.clicked.connect(self.eye_track_start)
        return

    def eye_track(self):
        t = time()
        while time() - t < 1:
            self.Eye_Track.starting_camera()
            self.Eye_Track.starting_mediapipe()

            # 이부분도 faceID 합친것 처럼 하나로 합쳐야 할 것 같아요
            vidcap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
            vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

            try:
                filtered_xy = np.zeros((0, 2))
                # f = open("average.csv", "w")
                count = 0
                eyex_list = []
                eyey_list = []
                eyez_list = []
                eye_pos_average = [0, 0, 0]

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
                    eyez_list.append(self.Eye_Track.eye[2])

                    # 어차피 평균 낼꺼면 이런식으로 하는게 더 효율적일 것 같아요!
                    eye_pos_average[0] += self.Eye_Track.eye[0] / 1000
                    eye_pos_average[1] += self.Eye_Track.eye[1] / 1000
                    eye_pos_average[2] += self.Eye_Track.eye[2] / 1000

                    count += 1

                self.Eye_Track.pipeline.stop()
                self.eye_pos = eye_pos_average.copy()

            finally:
                self.Eye_Track.pipeline.stop()

            # self.Eye_Track.main()

    def calc_seat_pos(self):  # 처음 위치로부터 얼마나 이동해야 하는지 계산
        # https://www.physiomed.co.uk/uploads/guide/file/21/Physiomed_Sitting_Guide_-_Driving_Digital.pdf
        # 근거

        # 1. 무릎 뒤쪽이 의자랑 닿지 않도록.
        # 손가락 2개 정도의 틈이 생기면 좋다.

        # 2. 무릎이 최소 20~30도 정도는 굽혀지도록

        # 페달의 위치를 원점으로 (0, 0, 0)

        CHAIR_HEIGHT = 45  # 의자엉덩이 높이
        CAMERA_HEIGHT = 136  # 정확한 측정 후 수정 예정
        INIT_SEAT_POS = 110  # 초기 의자의 위치

        CONSTANT_FOR_EYE_HEAD = 0.07  # 사람 키에 대한 눈~정수리 거리
        user_distance_between_eye_head = self.user_height * CONSTANT_FOR_EYE_HEAD

        # 사용자의 앉은 키 user_sit_height = camera_height - y
        user_sit_height = (
            CAMERA_HEIGHT
            - self.eye_pos[1]
            + user_distance_between_eye_head
            - CHAIR_HEIGHT
        )

        user_leg_length = self.user_height - user_sit_height + 3  # 사용자의 다리길이 + 신발 두께

        # 자세한 설명은 정준환 노트북에 있음.
        adjusted_seat_pos = math.sqrt(
            (user_leg_length * math.cos(math.radians(20 / 2))) ** 2 - CHAIR_HEIGHT**2
        )

        distance_to_move = INIT_SEAT_POS - adjusted_seat_pos
        return distance_to_move


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
