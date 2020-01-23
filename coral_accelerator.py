
import statistics

import argparse
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
import cv2
import time
import os
import numpy as np
import RPi.GPIO as GPIO

from pyfirmata import Arduino, SERVO
import sys
from demo import lane, map, map_camera

GPIO.cleanup()
GPIO.setmode(GPIO.BCM)

TRIG = 17
ECHO = 4

port = '/dev/ttyACM0'
board = Arduino(port)

sys.path.append("/home/pi/Desktop/Accelerator/python-tflite-source/edgetpu/")

pin = 8
servo2 = 9
servo3 = 10
servo4 = 11
servo5 = 12
servo6 = 13


ENA = 13  # //L298ʹ��A
ENB = 20  # //L298ʹ��B
IN1 = 19  # //����ӿ�1
IN2 = 16  # //����ӿ�2
IN3 = 21  # //����ӿ�3
IN4 = 26  # //����ӿ�4


def Motor_Setup():
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENB, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)


def Motor_Forward():
    print('motor forward')
    GPIO.output(ENA, True)
    GPIO.output(ENB, True)
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)
    print("time.sleep")
    time.sleep(0.1)
    print("Prepare to read it")
    Motor_Stop()

    print("Read it")

def Motor_Backward():
    print('motor_backward')
    GPIO.output(ENA, True)
    GPIO.output(ENB, True)
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, False)
    GPIO.output(IN4, True)
    time.sleep(0.01)

def Motor_TurnLeft():
    print('motor_turnleft')
    GPIO.output(ENA, True)
    GPIO.output(ENB, True)
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, True)
    time.sleep(0.01)

def Motor_TurnRight():
    print('motor_turnright')
    GPIO.output(ENA, True)
    GPIO.output(ENB, True)
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, True)
    GPIO.output(IN4, False)
    time.sleep(0.01)

def Motor_Stop():
    print('motor stop')
    GPIO.output(ENA, True)
    GPIO.output(ENB, True)
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)
    time.sleep(0.01)

def distance_estimation():
    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    GPIO.output(TRIG, True)
    time.sleep(0.0001)
    GPIO.output(TRIG, False)
    start = 0
    end = 0
    while GPIO.input(ECHO) == False:
        start = time.time()

    while GPIO.input(ECHO) == True:
        end = time.time()
    if start is not None:
        sig_time = end - start
    else:
        start = 0

    distance = sig_time / 0.000058

    print(distance)

    print("Disance : {} cm".format(distance))

    return distance


def setServoAngle(pin, angle):
    board.digital[pin].write(angle)


def setServo2Angle(servo2, angle):
    board.digital[servo2].write(angle)


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def ReadLabelFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = line.strip().split(maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret

def main():
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='Path of the detection model.', required=True)
    parser.add_argument(
        '--label', help='Path of the labels file.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    change_res(cap, 640, 480)
    degree_revised = 34
    count = []
    count_stop = []
    save_distance = []
    img_count = 0

    
    while True:
        t1 = cv2.getTickCount()
        ret, image = cap.read()

        cv2.imwrite("../images/frame.jpg", image)
        img = Image.open("../images/frame.jpg")

        engine = DetectionEngine(args.model)
        labels = ReadLabelFile(args.label) if args.label else None

        ans = engine.DetectWithImage(img, threshold=0.05, 
            keep_aspect_ratio=True,relative_coord=False, top_k=10)

        def camera_left(degree):
            board.digital[pin].mode = SERVO

            setServoAngle(pin, degree)
            degree_revised = degree + 3
            return degree_revised

        def camera_right(degree):
            board.digital[pin].mode = SERVO

            setServoAngle(pin, degree)
            degree_revised = degree - 3
            return degree_revised

        def camera_wideleft(degree):
            board.digital[pin].mode = SERVO

            for i in range(degree, degree + 15):
                setServoAngle(pin, degree)

            return degree_revised

        def camera_wideright(degree):
            board.digital[pin].mode = SERVO

            for i in range(degree, degree - 15):
                setServoAngle(pin, degree)

            return degree_revised

        def camera_stop(degree):
            board.digital[pin].mode = SERVO
            setServoAngle(pin, degree)
            degree_revised = degree
            return degree_revised

        def camera1_left(degree):
            board.digital[servo3].mode = SERVO

            setServoAngle(servo3, degree)
            degree_revised = degree + 3
            return degree_revised

        def camera1_right(degree):
            board.digital[servo3].mode = SERVO

            setServoAngle(servo3, degree)
            degree_revised = degree - 3
            return degree_revised

        def camera1_wideleft(degree):
            board.digital[servo3].mode = SERVO

            for i in range(degree, degree + 10):
                setServoAngle(servo3, degree)

            return degree_revised

        def camera1_wideright(degree):
            board.digital[servo3].mode = SERVO

            for i in range(degree, degree - 10):
                setServoAngle(servo3, degree)

            return degree_revised

        def camera1_stop(degree):
            board.digital[servo3].mode = SERVO
            setServoAngle(servo3, degree)
            degree_revised = degree
            return degree_revised

        if ans:
            for obj in ans:
                if obj.score > 0.7:
                    print('-----------------------------------------')
                    if labels:
                        print("label = ", labels[obj.label_id])
                    print('score = ', obj.score)
                    box = obj.bounding_box.flatten().tolist()
                    print('box = ', box)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, "{}".format(labels[obj.label_id]), (int(x1), int(y1)), font, 1, (255, 255, 255),
                                1, cv2.LINE_AA)

                    if "bottle" in labels[obj.label_id]:
                        number = 1
                        count.append(number)                    
                    
                        print("len(count):", len(count))
                        if len(count) == 10:
                            if (x2 + x1) / 2 >= 500:
                                camera1_wideright(30)
                                camera_wideright(30)
                                print("Rotate 30 degrees to the right side")
                            elif (x2 + x1) / 2 <= 180:
                                camera1_wideleft(30)
                                camera_wideleft(30)
                                print("Rotate 30 degrees to the left side")
                    
                        if 10 < len(count) and (len(count) / 5).is_integer() == True:
                    
                            if degree_revised > 64:
                                degree_revised = 64
                            elif degree_revised <= 4:
                                degree_revised = 4
                    
                            if (x2 + x1) / 2 < 280:
                                camera_left(degree_revised)
                                degree_revised = camera_left(degree_revised)
                                print("left:", degree_revised)
            
                    
                            elif 280 <= (x2 + x1) / 2 < 400:
                                img_count += 1
                                cv2.imwrite("../object_images/object" + "." + str(img_count) + "." + str(t1) + ".jpg",
                                            image[int(y1)-50:int(y2)+50, int(x1)-50:int(x2)+50])
                                print("Take", img_count, "th photo!")
                                camera_stop(degree_revised)
                                degree_revised = camera_stop(degree_revised)
                                print("stop:", degree_revised)
                    
                                if distance_estimation() is not None:
                                    Y, P = str(distance_estimation()).split(".")
                                    Y = int(Y)
                                    print("Y:", Y)
                                    save_distance.append(Y)
                                    if Y > 10:
                                        number_stop = 1
                                        count_stop.append(number_stop)
                                        print("number_stop:", number_stop)

                    
                            elif 400 <= (x2 + x1) / 2:
                                camera_right(degree_revised)
                                degree_revised = camera_right(degree_revised)
                                print("right:", degree_revised)

                    
                        print("degree_revised:", degree_revised)
                    
                    
                    if len(count_stop) >= 3 and (len(count)/70).is_integer() == True:
                        print("save_distance:", save_distance)
                        median_distance = statistics.median(save_distance)
                        print("median_distance:", median_distance)
                        if median_distance <= 100:
                            if median_distance >= 10:
                                print("Go Forward")
                                # Motor_Setup()
                                # Motor_Forward()
                                # Motor_Stop()
                                print("Moved 7cm to the front side")
                                # revise pls from here
                    
                    print("count_stop:", len(count_stop))
                    t2 = cv2.getTickCount()
                    time = (t2 - t1) / freq
                    time = 1 / time
                    print("FPS =", time)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, "FPS: {0:.2f}".format(time), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    if len(count) == 100:
                        print("count is {} break pls".format(len(count)))
                        return
        else:
            print('No object detected!')

        cv2.imshow('Object detector', image)
        if cv2.waitKey(30) == ord('q'):
            break
        elif img_count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    return


if __name__ == '__main__':

    main()


    
        

        

    

