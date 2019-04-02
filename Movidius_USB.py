import sys

if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    sys.exit(1)
import numpy as np
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading

import statistics

import RPi.GPIO as GPIO

#
# import argparse
# import platform
# import subprocess
# from PIL import Image
# from PIL import ImageDraw
# import cv2
# import time
# import os
# import numpy as np
#
#
#
# import sys
#
# #from edgetpu.detection.engine import DetectionEngine
import RPi.GPIO as GPIO
from pyfirmata import Arduino, SERVO
# from demo import lane, map, map_camera, total, motor


GPIO.setmode(GPIO.BCM)

TRIG = 17
ECHO = 4

# Setting up the Arduino board
port = '/dev/ttyACM0'
board = Arduino(port)
# Need to give some time to pyFirmata and Arduino to synchronize
# Set mode of the pin 13 as SERVO
pin = 8

servo2 = 9

servo3 = 10

servo4 = 11

servo5 = 12

servo6 = 13

lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
camera_width = 320
camera_height = 240
window_name = ""
ssd_detection_mode = 1
face_detection_mode = 0
elapsedtime = 0.0

LABELS = [['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor'],
          ['background', 'face']]

def setServoAngle(pin, angle):
    board.digital[pin].write(angle)


def setServo2Angle(servo2, angle):
    board.digital[servo2].write(angle)


def camThread(LABELS, results, frameBuffer, camera_width, camera_height, vidfps, number_of_camera):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name

    cam = cv2.VideoCapture(0)
    if cam.isOpened() != True:
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    window_name = "USB Camera"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    degree_revised = 34
    count = []
    count_stop = []
    save_distance = []
    img_count = 0
    bottle_token = 0
    person_token = 0
    bottle_location = []
    person_location = []
    left_bottle_token = 0
    middle_bottle_token = 0
    right_bottle_token = 0
    normal_count = []
    normal_number = 0
    bottle_x1, bottle_x2, bottle_y1, bottle_y2 = 0, 0, 0, 0

    while True:
        number = 1
        normal_count.append(normal_number)
        print("normal_count:", len(normal_count))

        t1 = time.perf_counter()

        # USB Camera Stream Read
        s, color_image = cam.read()

        if not s:
            continue
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        height = color_image.shape[0]
        width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        res = None

        def camera_left(degree):
            board.digital[pin].mode = SERVO

            setServoAngle(pin, degree)
            degree_revised = degree + 3
            print("degree:",degree_revised)
            return degree_revised

        def camera_right(degree):
            board.digital[pin].mode = SERVO

            setServoAngle(pin, degree)
            degree_revised = degree - 3
            print("degree:", degree_revised)
            return degree_revised

        def camera_stop(degree):
            board.digital[pin].mode = SERVO
            setServoAngle(pin, degree)
            degree_revised = degree
            print("degree:", degree_revised)
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



        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            imdraw, bottle_token, bottle_location, person_token, person_location = overlay_on_image(frames, res, LABELS,
                                                                                                    count, number,
                                                                                                    bottle_token,
                                                                                                    bottle_location,
                                                                                                    person_token,
                                                                                                    person_location)


            print("bottle, person:", bottle_location, person_location)


            try:
                print("2")
                bottle_location = np.array(bottle_location)
                bottle_location1 = bottle_location.flatten().tolist()

                bottle_location_x1, bottle_location_y1, bottle_location_x2, bottle_location_y2 = bottle_location1
                print("bottle:",bottle_location)
                if len(normal_count) % 13 == 0:
                    print("working1")
                    #bottle_location.append(bottle_location1)
                    if (int(bottle_location_x1) + int(bottle_location_x2))/2 >= 130:
                        print("1")
                        camera_right(degree_revised)
                        degree_revised = camera_right(degree_revised)
                        print("right:", degree_revised)
                    elif 110 <= (int(bottle_location_x1) + int(bottle_location_x2))/2 <= 130:
                        print("1")
                        camera_stop(degree_revised)
                        degree_revised = camera_stop(degree_revised)
                        print("stop:", degree_revised)

                    elif (int(bottle_location_x1) + int(bottle_location_x2))/2 <= 110:
                        print("1")
                        camera_left(degree_revised)
                        degree_revised = camera_left(degree_revised)
                        print("left:", degree_revised)




                # person_location = np.array(person_location)
                # person_location1 = person_location.flatten().tolist()
                # person_location_x1, person_location_y1, person_location_x2, person_location_y2 = person_location1
                # print("person:", person_location)
                #
                # if len(normal_count) % 13 == 0:
                #     #person_location.append(person_location1)
                #     print("working2", person_location_x1, person_location_x2)
                #     if (int(person_location_x1) + int(person_location_x2))/2 >= 130:
                #         print("right1:", degree_revised)
                #         camera_right(degree_revised)
                #         print("left2:", degree_revised)
                #         degree_revised = camera_right(degree_revised)
                #         print("right3:", degree_revised)
                #     elif 110 <= (int(person_location_x1) + int(person_location_x2))/2 <= 130:
                #         print("stop1:", degree_revised)
                #         camera_stop(degree_revised)
                #         print("left2:", degree_revised)
                #         degree_revised = camera_stop(degree_revised)
                #         print("stop3:", degree_revised)
                #
                #     elif (int(person_location_x1) + int(person_location_x2))/2 <= 110:
                #         print("left1:", degree_revised)
                #         camera_left(degree_revised)
                #         print("left2:", degree_revised)
                #         degree_revised = camera_left(degree_revised)
                #         print("left3:", degree_revised)

            except Exception as e:
                pass

            lastresults = res











        else:
            imdraw = frames

        cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps = "(Playback) {:.1f} FPS".format(time1 / 15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount / time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2 - t1
        time1 += 1 / elapsedTime
        time2 += elapsedTime

        if len(normal_count) == 200:
            print("bottle_location, person_location", bottle_location, person_location)

            print("The number of len is 200")



    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):
    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, frameBuffer, results, camera_width, camera_height, number_of_ncs):
        self.devid = devid
        self.frameBuffer = frameBuffer
        self.model_xml = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.xml"
        self.model_bin = "./lrmodel/MobileNetSSD/MobileNetSSD_deploy.bin"
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device="MYRIAD")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs

    def image_preprocessing(self, color_image):

        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg - 127.5
        prepimg = prepimg * 0.007843
        prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg

    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            prepimg = self.image_preprocessing(self.frameBuffer.get())
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1
                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            cnt, dev = heapq.heappop(self.heap_request)

            if self.exec_net.requests[dev].wait(0) == 0:
                self.exec_net.requests[dev].wait(-1)
                out = self.exec_net.requests[dev].outputs["detection_out"].flatten()
                self.results.put([out])
                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev))

        except:
            import traceback
            traceback.print_exc()


def inferencer(results, frameBuffer, ssd_detection_mode, face_detection_mode, camera_width, camera_height,
               number_of_ncs):
    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(
        NcsWorker(devid, frameBuffer, results, camera_width, camera_height, number_of_ncs),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()


def overlay_on_image(frames, object_infos, LABELS, count, number, bottle_token, bottle_location, person_token,
                     person_location):
    try:

        color_image = frames

        if isinstance(object_infos, type(None)):
            return color_image

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        img_cp = color_image.copy()

        for (object_info, LABEL) in zip(object_infos, LABELS):

            drawing_initial_flag = True

            for box_index in range(100):
                if object_info[box_index + 1] == 0.0:
                    break
                base_index = box_index * 7
                if (not np.isfinite(object_info[base_index]) or
                        not np.isfinite(object_info[base_index + 1]) or
                        not np.isfinite(object_info[base_index + 2]) or
                        not np.isfinite(object_info[base_index + 3]) or
                        not np.isfinite(object_info[base_index + 4]) or
                        not np.isfinite(object_info[base_index + 5]) or
                        not np.isfinite(object_info[base_index + 6])):
                    continue

                x1 = max(0, int(object_info[base_index + 3] * height))
                y1 = max(0, int(object_info[base_index + 4] * width))
                x2 = min(height, int(object_info[base_index + 5] * height))
                y2 = min(width, int(object_info[base_index + 6] * width))
                print("x1,y1,x2,y2:", x1, y1, x2, y2)

                object_info_overlay = object_info[base_index:base_index + 7]

                min_score_percent = 60

                source_image_width = width
                source_image_height = height

                base_index = 0
                class_id = object_info_overlay[base_index + 1]
                percentage = int(object_info_overlay[base_index + 2] * 100)
                if (percentage <= min_score_percent):
                    continue

                box_left = int(object_info_overlay[base_index + 3] * source_image_width)
                box_top = int(object_info_overlay[base_index + 4] * source_image_height)
                box_right = int(object_info_overlay[base_index + 5] * source_image_width)
                box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

                label_text = LABEL[int(class_id)] + " (" + str(percentage) + "%)"
                print("label_text:", label_text)

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                              label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            label_text_color, 1)

                if "bottle" in LABEL[int(class_id)]:
                    count.append(number)
                    bottle_token += 1
                    # cv2.imwrite("../object_images/object" + "." + str(img_count) + "." + str(t1) + ".jpg",
                    #             image[int(y1) - 150:int(y2) + 150, int(x1) - 150:int(x2) + 150])
                    try:
                        bottle_location = [x1, y1, x2, y2]
                    except:
                        pass

                elif "person" in LABEL[int(class_id)]:
                    count.append(number)
                    person_token += 1
                    # cv2.imwrite("../object_images/person" + "." + str(img_count) + "." + str(t1) + ".jpg",
                    #             image[int(y1) - 150:int(y2) + 150, int(x1) - 150:int(x2) + 150])
                    try:
                        person_location = [x1,y1,x2,y2]
                    except:
                        pass

        cv2.putText(img_cp, fps, (width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        return img_cp, bottle_token, bottle_location, person_token, person_location

    except:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':

    forward_count = 0
    backward_count = 0
    rightward_count = 0
    leftward_count = 0
    left_count = 0
    middle_count = 0
    right_count = 0
    object_array = []
    bottle_array = []
    person_array = []
    origin = [340, 260, 300, 220]
    direction_count = 100


    parser = argparse.ArgumentParser()
    parser.add_argument('-cn','--numberofcamera',dest='number_of_camera',type=int,default=0,help='USB camera number. (Default=0)')
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=320,help='Width of the frames in the video stream. (Default=320)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=240,help='Height of the frames in the video stream. (Default=240)')
    parser.add_argument('-sd','--ssddetection',dest='ssd_detection_mode',type=int,default=1,help='[Future functions] SSDDetectionMode. (0:=Disabled, 1:=Enabled Default=1)')
    parser.add_argument('-fd','--facedetection',dest='face_detection_mode',type=int,default=0,help='[Future functions] FaceDetectionMode. (0:=Disabled, 1:=Full, 2:=Short Default=0)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (Default=30)')

    args = parser.parse_args()

    number_of_camera = args.number_of_camera
    camera_width  = args.camera_width
    camera_height = args.camera_height
    ssd_detection_mode = args.ssd_detection_mode
    face_detection_mode = args.face_detection_mode
    number_of_ncs = args.number_of_ncs
    vidfps = args.fps_of_video

    if ssd_detection_mode == 0 and face_detection_mode != 0:
        del(LABELS[0])

    try:

        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(LABELS, results, frameBuffer, camera_width, camera_height, vidfps, number_of_camera),
                       daemon=True)
        p.start()
        processes.append(p)

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, ssd_detection_mode, face_detection_mode, camera_width, camera_height, number_of_ncs),
                       daemon=True)
        p.start()
        processes.append(p)


        while True:
            sleep(1)


    except:
        import traceback
        traceback.print_exc()



