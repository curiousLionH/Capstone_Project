# -*- coding: utf-8 -*-
import time
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

        # 사진 임베딩
        self.known_face_encodings = []
        self.known_face_names = []

        img_files = os.listdir("image")
        for img_file in img_files:
            img = face_recognition.load_image_file("image/" + img_file)
            self.known_face_encodings.append(face_recognition.face_encodings(img)[0])
            self.known_face_names.append(img_file[:-4])

        # 아두이노랑 시리얼 포트 연결 (ser : alcohol / ser2 : seat, mirror)
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',
            baudrate=9600,
        )
        self.ser2 = serial.Serial(
            port='/dev/ttyACM0',
            baudrate=9600,
        )

        # 알코올 테스트 통과
        self.alcohol_pass = False

        # 블루투스통신값(음주측정)
        # 0:재측정요구, 1:정상, 2:음주상태
        self.alcohol_value = None

        # 가이드 라벨 숨기기
        self.user_guide_label.setVisible(False)

        # 로그인 버튼 연결
        self.login_button.clicked.connect(self.check_ID_Password)

        # 사용자 눈 좌표
        self.eye_pos = [0, 0, 0]

    def camera_start(self):
        self.Eye_Track.starting_camera()
        self.Eye_Track.starting_mediapipe()
        self.Eye_Track.get_align_depth()
        th = threading.Thread(target=self.camera_show)
        th.start()

    def faceID_start(self, user):
        th = threading.Thread(target=self.faceID, args=(user,))
        th.start()

    def alcohol_value_update_start(self):
        th = threading.Thread(target=self.alcohol_value_update)
        th.start()

    def alcohol_test_start(self, user):
        th = threading.Thread(target=self.alcohol_test)
        th.start()

    def eye_track_start(self):
        th = threading.Thread(target=self.eye_track)
        th.start()

    def send_data_start(self):
        th = threading.Thread(target=self.send_data)
        th.start()

    # 시스템 종료 (로그인 화면으로 돌아감)
    def reset_program(self):
        self.IDLabel.setVisible(True)
        self.IDTextField.setVisible(True)
        self.passwordLabel.setVisible(True)
        self.passwordTextField.setVisible(True)
        self.login_button.setVisible(True)
        self.user_guide_label.setVisible(False)
        self.camera_show_label.setVisible(False)

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
            self.user_password != self.users_height_data[self.user_id]["password"]):
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
        self.login_button.setVisible(False)

    # 카메라 보여주기
    def camera_show(self):
        camera_label = self.camera_show_label
        guide_label = self.user_guide_label
        camera_label.resize(1080, 720)
        guide_label.setVisible(True)
        # 얼굴인식 시작 안내
        guide_label.setText("얼굴 인식이 진행중입니다. 잠시만 기다려주세요.")
        camera_label.setVisible(True)

        # 카메라 가져오기
        while True:
            try:
                self.Eye_Track.get_align_depth()
                img = self.Eye_Track.color_image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                camera_label.setPixmap(pixmap)
            except:     # 카메라 불러올 수 없는 경우
                guide_label.setText("카메라를 불러올 수 없습니다. 잠시 후에 다시 시도해주세요. ")
                break

    def faceID(self, user):

        flag = True

        face_locations = []
        face_encodings = []
        process_this_frame = True
        while self.identify_user_token <= 1 and flag:

            try:
                self.face_frame = self.Eye_Track.color_image
                self.face_frame = cv2.cvtColor(self.face_frame, cv2.COLOR_BGR2RGB)

                # 인식 속도 개선을 위한 프레임 크기 조절
                rgb_small_frame = cv2.resize(
                    self.Eye_Track.color_image, (0, 0), fx=0.25, fy=0.25
                )

                if process_this_frame:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )

                    for face_encoding in face_encodings:
                        # 매칭 여부 확인
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, face_encoding
                        )
                        name = "Unknown"

                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]

                        if name == "Unknown":
                            self.identify_user_token += 1
                        else:
                            flag = False
                            break
                process_this_frame = not process_this_frame
            except:
                pass


        # 얼굴인식 실패
        if self.identify_user_token >= 2:
            self.user_guide_label.setText("얼굴인식에 실패했습니다. 시스템이 종료됩니다.")
            time.sleep(3)
            self.reset_program()

        # 얼굴인식 성공
        else:
            self.user_guide_label.setText(f"{self.user_name}님 안녕하세요.\n사용자 인식이 완료되었습니다. 10초 이내에 음주 측정을 완료해주세요")
            time.sleep(3)
            self.alcohol_test_start()

        return

    def alcohol_test(self):
        start_time = time.time()
        # 음주측정 제한 시간 10초
        while time.time()-start_time < 10 and not self.alcohol_pass:
            try:
                if self.ser.readable():
                    a = self.ser.read().decode()
                    if a == "0":
                        self.user_guide_label.setText("알코올 측정이 제대로 되지 않았습니다. 센서를 다시 불어주세요.")
                    elif a == "2":
                        self.user_guide_label.setText("음주 상태입니다. 시스템을 종료합니다.")
                        self.reset_program()
                        return
                    elif a == "1":
                        self.user_guide_label.setText("알코올 측정 통과!")
                        self.alcohol_pass = True
                        break
                else:
                    print("alcohol not readable")
            except:
                pass
        # 10초 내에 음주측정 실패 시 시스템 종료
        if (not self.alcohol_pass):
            self.reset_program()
            return

        print("좌석 조절 시작 (눈 위치 인식 시작)")
        self.user_guide_label.setText(
            "사용자의 안전을 위하여 좌석 및 사이드미러 조절이 진행될 예정입니다. 자연스럽게 정면을 바라봐주세요. "
        )
        time.sleep(3)
        self.eye_track_start()
        return

    def eye_track(self):
        vidcap = self.Eye_Track.color_image
        vidcap = cv2.cvtColor(vidcap, cv2.COLOR_BGR2RGB)
        eye_pos_average = [0, 0, 0]

        for i in range(50):
            self.Eye_Track.face_detect()
            self.Eye_Track.main()

            # 50회 tracking한 평균값 저장
            eye_pos_average[0] += self.Eye_Track.eye[0] / 50
            eye_pos_average[1] += self.Eye_Track.eye[1] / 50
            eye_pos_average[2] += self.Eye_Track.eye[2] / 50

        self.eye_pos = eye_pos_average.copy()

        print("좌석 조절 데이터 전송 시작")
        self.user_guide_label.setText(
            "좌석 및 사이드미러 조절이 시작됩니다."
        )

        # 좌석, 사이드미러 조절값 전송
        self.send_data_start()
        self.user_guide_label.setText(
            "이제 시동이 걸립니다.\n즐겁고 안전한 주행 되세요!"
        )
        return

    def calc_seat_pos(self):  
        # 1. 무릎 뒤쪽이 의자랑 닿지 않도록.
        # 손가락 2개 정도의 틈이 생기면 좋다.
        # 2. 무릎이 최소 20~30도 정도는 굽혀지도록
        # 페달의 위치를 원점으로 (0, 0, 0)

        CHAIR_HEIGHT = 45  # 의자엉덩이 높이
        CAMERA_HEIGHT = 136  # 정확한 측정 후 수정 예정
        INIT_SEAT_POS = 100  # 초기 의자의 위치

        CONSTANT_FOR_EYE_HEAD = 0.07  # 사람 키에 대한 눈~정수리 거리
        user_distance_between_eye_head = self.user_height * CONSTANT_FOR_EYE_HEAD

        # 사용자의 앉은 키
        user_sit_height = (
            CAMERA_HEIGHT
            - self.eye_pos[1]
            + user_distance_between_eye_head
            - CHAIR_HEIGHT
        )

        user_leg_length = self.user_height - user_sit_height + 3  # 사용자의 다리길이 + 신발 두께

        adjusted_seat_pos = math.sqrt(
            (user_leg_length * math.cos(math.radians(20 / 2))) ** 2 - CHAIR_HEIGHT**2
        )

        # 움직여야 하는 거리를 기어비를 통해 회전수로 변환
        distance_to_move = INIT_SEAT_POS - adjusted_seat_pos
        number_of_rev = int((32 * distance_to_move) / 20.72)

        return number_of_rev, distance_to_move

    def calc_side_angle(self, D):
    
        # 단위: cm
        l = 200 # 카메라부터 차체 끝까지의 길이
        l_1 = 15 # 차체에서 사이드미러까지의 길이
        l_s = 13.8 # 사이드 미러의 총 길이
        m = (l_1+l_s/5)
        k = 20.7 # 카메라의 중앙에서 차제 시작점까지의 거리
        z = self.eye_pos[2] - D # 카메라에서 눈까지의 거리 (깊이)
        
        y_axis = 136 - 90 - 24.3 # 카메라에서 눈까지의 거리 (높이)
        y = self.eye_pos[1]
        
        # 수식 상 정확한 값을 찾기 어려워, 결과값과 가장 유사한 (오차가 적은) 값을 사용
        temp_min = 100
        best_angle = 0
        for i in range(90):
            error = 2*i*math.pi/180 - (math.atan2(l-m*np.cos(i*math.pi/180), m*np.sin(i*math.pi/180))+math.atan2(z-m*np.cos(i*math.pi/180), k+m*np.sin(i/math.pi*180)))
            if np.absolute(error) < temp_min:
                temp_min =np.absolute(error)
                best_angle = i
            print(temp_min)
            
        theta1 = int(best_angle) # degree
        theta2 = int(np.arctan((y_axis-y)/(z-m*math.cos(theta1)))/2*180/math.pi) # degree
        return theta1, theta2

    def send_data(self):
        A, D = str(self.calc_seat_pos())
        B, C = map(str, self.calc_side_angle(D))

        A = A.zfill(2)
        B = B.zfill(2)
        C = C.zfill(2)
        Trans = "Q" + A + B + C
        print(f"sending data {Trans} ...")
        Trans = Trans.encode("utf-8")

        starttime = time.time()

        # while True:
        #     self.ser2.write(Trans)
        #     time.sleep(1)

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
