# -*- coding: utf-8 -*-
from time import time, sleep
import math
import serial

# import matplotlib.pyplot as plt
import argparse
from align_depth import Align_Depth_Eye_Track

import face_recognition
import cv2
import numpy as np
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

        self.Eye_Track = Align_Depth_Eye_Track()
        # 유저들 (조원들) 키 정보 불러오기
        with open("./info.json", "r", encoding="UTF8") as file:
            self.users_height_data = json.load(file)

        # 유저 얼굴 확인 됐으면 True, 아니면 False
        self.identify_user_token = 0
        # self.cap = cv2.VideoCapture(6)      # 0 : webcam 4: depth (d435i) 6: rgb(d435i)

        # 사진 임베딩 하는 부분
        self.known_face_encodings = []
        self.known_face_names = []

        img_files = os.listdir("image")
        for img_file in img_files:
            img = face_recognition.load_image_file("image/" + img_file)
            self.known_face_encodings.append(face_recognition.face_encodings(img)[0])
            self.known_face_names.append(img_file[:-4])

        # 아두이노랑 시리얼 포트 연결
        self.ser = serial.Serial(
            # port='/dev/cu.HC-06-DevB',
            port='/dev/ttyACM0',
            baudrate=9600,
        )

        # 알코올 패스 함?
        self.alcohol_pass = False

        # 블루투스통신값(음주측정)
        # 0:다시불어, 1:정상, 2:음주상태
        self.alcohol_value = None

        # 얘가 음주측정 안내, 성공 실패, 얼굴인식 다 여기서 멘트 안내.
        # 가이드 라벨 invisible
        self.user_guide_label.setVisible(False)

        # 카메라 시작 버튼 연결
        # 기존에 버튼에 2개의 함수 연결하던것을 check_ID_Password로 합침
        self.camera_start_button.clicked.connect(self.check_ID_Password)

        # 비밀번호 입력창 숫자 숨겨주기
        self.passwordTextField.setEchoMode(QtGui.QLineEdit.Password)

        # 사용자 눈 좌표
        self.eye_pos = [0, 0, 0]

    def align_depth_repeat(self):
        while True:
            self.Eye_Track.get_align_depth()
            self.camera_show()

    def camera_start(self):
        self.Eye_Track.starting_camera()
        self.Eye_Track.starting_mediapipe()
        print("camera_start")
        th = threading.Thread(target=self.align_depth_repeat)
        th.start()
        # while True:
        #     self.Eye_Track.get_align_depth()
        #     self.cap = self.Eye_Track.color_image

    # def camera_start(self):
    #     th = threading.Thread(target=self.camera_show)
    #     th.start()
    #     print("camera_start")

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

        # 아이디 비밀번호 입력 칸, 로그인 버튼 숨기기
        self.IDLabel.setVisible(False)
        self.IDTextField.setVisible(False)
        self.passwordLabel.setVisible(False)
        self.passwordTextField.setVisible(False)
        self.camera_start_button.setVisible(False)

    # 카메라 보여주기
    def camera_show(self):
        print("Camera show")

        # width = self.Eye_Track.color_image.get(cv2.CAP_PROP_FRAME_WIDTH)
        # height = self.Eye_Track.color_image.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 라벨들 가져오기
        camera_label = self.camera_show_label
        guide_label = self.user_guide_label
        camera_label.resize(1080, 720)
        guide_label.setVisible(True)

        guide_label.setText("얼굴 인식이 진행중입니다. 잠시만 기다려주세요.")

        # 카메라 가져오기
        # cap = cv2.VideoCapture(0)
        while True:
            # ret, img = self.cap.read()
            try:
                # img = cv2.cvtColor(self.Eye_Track.color_image, cv2.COLOR_BGR2RGB)
                img = self.Eye_Track.color_image
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                camera_label.setPixmap(pixmap)
            except:
                guide_label.setText("카메라를 불러올 수 없습니다. 잠시 후에 다시 시도해주세요. ")
                break
        self.Eye_Track.color_image.release()
        print("Thread end.")

    def faceID(self, user="jiwon"):

        # self.user_guide_label.setText("사용자 인식이 완료되었습니다. 음주 측정을 시작해주세요")
        # self.faceID_alcohol_start(user)
        # return

        flag = True
        print("let's start faceID")

        # video_capture = self.cap

        face_locations = []
        face_encodings = []
        process_this_frame = True
        while self.identify_user_token <= 1 and flag:
            try:
                # Grab a single frame of video
                # ret, frame = video_capture.read()
                # print(f"video_capture ret = {ret}")
                self.face_frame = self.Eye_Track.color_image

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(
                    self.Eye_Track.color_image, (0, 0), fx=0.25, fy=0.25
                )

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time
                if process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    # print(f"rgb_small_frame: {rgb_small_frame}, face_locations: {face_locations}")
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )

                    # print(face_encodings)
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
        # 얼굴 인식이 성공하든 실패하든 이쪽으로 넘어오게 됨
        # face_names[-1] == user
        # 이런식의 코드로 성공한지 실패한지 판단해야 할듯?

        # self.camera_start_button.setVisible(True)
        # self.camera_start_button.setText("음주 측정 시작")

        # 이 부분도 그냥 버튼으로 하지 말고 시간초로 주는게 좋지 않을까 싶어요?
        self.user_guide_label.setText("사용자 인식이 완료되었습니다. 10초 이내에 음주 측정을 완료해주세요")
        self.faceID_alcohol_start(user)
        return

    # 아두이노로 부터 값을 지속적으로 갱신하는 함수
    def alcohol_value_update(self):
        start_time = time()
        # 조금 여유롭게 15초동안 실행
        while time() - start_time < 15:
            temp_list = []
            while self.ser.readable():
                a = self.ser.read().decode()
                if a == "\n":
                    break
                temp_list.append(a)
            print(temp_list)
            try:
                # 기존 코드대로 짜면 아두이노에서 값을 읽자마자 브레이크 되어버림
                # A1이 나올때까지 값을 읽거나 시간제한을 두거나 하는방식으로 바꿔봄

                # if temp_list[0] == "A":
                #     self.alcohol_value = int(temp_list[1])
                #     print(f"Alcohol state : {self.alcohol_value}")
                #     break

                if temp_list[0] == "A":
                    self.alcohol_value = int(temp_list[1])
                    if self.alcohol_value == 0:
                        self.user_guide_label.setText("다시 불어줘요")
                    elif self.alcohol_value == 2:
                        self.user_guide_label.setText("술마시면 안돼용")
                        # 프로그램 자체 종료해버리는 무언가...
                        sys.exit()
                    else:
                        self.user_guide_label.setText("통과!")
                        self.alcohol_pass = True

                        # 더이상 값을 읽어올 필요가 없다.
                        break
                    pass
            except:
                pass

    def faceID_alcohol(self, user="jiwon"):
        print("let's start faceID_alcohol")
        start_time = time()
        while not self.alcohol_pass:
            # alcohol_value_update method에서 값 처리까지 다 하니까 이거만 해도 됨
            self.alcohol_value_update_start()

            # 10초안에 음주측정 통과 못할시 프로그램 종료
            if time() - start_time > 10:
                sys.exit()

        print("좌석 조절 시작 (눈 위치 인식 시작)")
        self.user_guide_label.setText(
            "사용자의 안전을 위하여 좌석 및 사이드미러 조절이 진행될 예정입니다. 자연스럽게 정면을 바라봐주세요. "
        )

        # time.sleep(2)
        self.eye_track_start()
        return

    def eye_track(self):
        t = time()
        while time() - t < 1:
            # self.Eye_Track.starting_camera()
            # self.Eye_Track.starting_mediapipe()

            # 이부분도 faceID 합친것 처럼 하나로 합쳐야 할 것 같아요
            # vidcap = cv2.VideoCapture(cv2.CAP_DSHOW + 0)
            # vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            # vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

            # vidcap은 이미지가 아니라 카메라라서 이렇게 하면 틀리지 않나요??
            vidcap = self.Eye_Track.color_image

            try:
                filtered_xy = np.zeros((0, 2))
                # f = open("average.csv", "w")
                count = 0
                eye_pos_average = [0, 0, 0]

                start_time = time()
                time_limit = 100
                flag = True
                # while(flag or time() - start_time < time_limit):
                while flag:
                    ret, image = vidcap.read()
                    # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)

                    # self.Eye_Track.get_align_depth()
                    self.Eye_Track.face_detect()
                    self.Eye_Track.main()
                    # if flag and self.Eye_Track.eye[0] > 0:
                    #     print("Eye detected!!!!")
                    #     start_time = time()
                    #     time_limit = 5
                    #     flag = False
                    if count > 1000:
                        flag = False

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
        # 움직여야 하는 거리를 기어비를 통해 회전수로 변환
        distance_to_move = INIT_SEAT_POS - adjusted_seat_pos
        number_of_rev = int((32 * distance_to_move) / 20.72)

        return number_of_rev

    def calc_side_angle(self):

        # 단위: cm
        l = 200
        l_1 = 10
        l_s = 15
        m = l_1 + l_s / 5
        k = 45
        z = self.eye_pos[2]

        y_axis = 136 - 90
        y = self.eye_pos[1]

        best_angle = 360
        for i in range(90):
            error = i / 180 * math.pi * 2 - (
                np.arctan(
                    (l - m * np.cos(i / 180 * math.pi))
                    / (m * np.sin(i / 180 * math.pi))
                )
                + np.arctan(
                    (z - m * np.cos(i / 180 * math.pi))
                    / (k + m * np.sin(i / 180 * math.pi))
                )
            )
            best_angle = min(best_angle, np.absolute(error))

        theta1 = int(best_angle)  # degree
        theta2 = int(
            np.arctan((y_axis - y) / (z - m * math.cos(theta1))) / 2 * 180 / math.pi
        )  # degree
        return theta1, theta2

    def send_data(self):
        A = str(self.calc_seat_pos())
        B, C = map(str, self.calc_side_angle())

        Trans = "Q" + A + B + C
        Trans = Trans.encode("utf-8")

        starttime = time()

        while (time() - starttime) <= 2:
            self.ser.write(Trans)


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
