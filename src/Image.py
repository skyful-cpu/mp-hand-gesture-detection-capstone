'''
    이미지 생성, 불러오기 관련 코드
'''

from unittest import result
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import csv
import os
import sys

class Image:
    def Image(self):
        pass

    def create_new_image(self, filename):
        '''
            OpenCV를 사용해 새로운 연사 이미지 파일을 생성하고 지정한 파일 이름 + 인덱스 순으론 저장
        '''
        cap = cv.VideoCapture(0)
        filename_idx = 0

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                print("empty camera frame")
                continue

            cv.imshow('''saving frames, press 'q' to quit''', frame)
            cv.imwrite(f'assets/img/{filename}{str(filename_idx)}.jpg', frame)
            filename_idx += 1
            time.sleep(0.1)

            if cv.waitKey(1) == ord('q'):
                break
        
        cv.destroyAllWindows()
        cap.release()

    def save_image(self, filename):
        cv.imwrite(f'assets/img/{filename}.jpg')

    def save_landmarks(self, hand_landmarks):
        with open('landmarks_data.csv', 'w') as writer:
            csv_out_writer = csv.writer(writer)
            landmark_list = []

            for idx in range(21):
                list1 = [
                    hand_landmarks.landmark[idx].x,
                    hand_landmarks.landmark[idx].y,
                    hand_landmarks.landmark[idx].z
                ]

                landmark_list.append(list1)

            csv_out_writer.writerows(landmark_list)

    def load_image(self):
        pass

    def get_landmarks(self):
        with open('landmarks_data.csv', 'w') as writer:
            csv_out_writer = csv.writer(writer)
            landmark_list = []
            
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            cap = cv.VideoCapture(0)

            with mp_hands.Hands(
                model_complexity=0,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

                while cap.isOpened():
                    success, frame = cap.read()

                    if not success:
                        print("empty frame")
                        continue

                    frame.flags.writeable = False
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    results = hands.process(frame)

                    if results.multi_hand_landmarks:

                        for hand_landmarks in results.multi_hand_landmarks:

                            for idx in range(21):

                                landmark_list.append(hand_landmarks.landmark[idx].x)
                                landmark_list.append(hand_landmarks.landmark[idx].y)
                                landmark_list.append(hand_landmarks.landmark[idx].z)

                            csv_out_writer.writerow(landmark_list)
                            landmark_list = []

                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )

                    cv.imshow('landmark', frame)
                    time.sleep(1)

                    if cv.waitKey(1) == ord('q'):
                        break

                cv.destroyAllWindows()
                cap.release()