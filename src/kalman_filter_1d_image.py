#!/usr/bin/env python

import sys
import signal
import numpy as np
from numpy.linalg import inv
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from yolov5_pytorch_ros.msg import BoundingBoxes


def signal_handler(signal, frame):
    print('pressed ctrl + c!!!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class KalmanFilter1D:
    def __init__(self):
        print("KalmanFilter1D init!")
        rospy.init_node('KalmanFilter1D', anonymous=True)
        rospy.Subscriber('/yolov5/bounding_boxes', BoundingBoxes, self.bboxCB)
        rospy.Subscriber('/yolov5/image_output', Image, self.imgCB)
        self.imgPub = rospy.Publisher(
            '/kcy/image_output', Image, queue_size=10)
        self.kfPub = rospy.Publisher(
            '/kcy/kf_output', Float32MultiArray, queue_size=10)

        self.br = CvBridge()
        self.kfData = Float32MultiArray()

        self.A = np.array([[1, .1],
                           [0,  1]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[.1, 0],
                           [0, 50]])
        self.R = np.array([[.1]])

        self.prevTime = rospy.get_time()
        self.dt = 0.0

        self.px = 337.528163  # princial x
        self.py = 222.945560  # princial y

        self.x_esti = np.array([self.px, 0])  # x_0 [x, vx]
        self.P = 100 * np.eye(2)  # P_0
        self.z_meas = np.array([self.px])

    def bboxCB(self, msg):
        if len(msg.bounding_boxes) > 0:
            if msg.bounding_boxes[0].Class == "blueBall":
                curTime = rospy.get_time()
                self.dt = curTime - self.prevTime
                self.prevTime = curTime
                self.A = np.array([[1, self.dt],
                                   [0, 1]])

                xPos_px = (
                    msg.bounding_boxes[0].xmax + msg.bounding_boxes[0].xmin) / 2
                self.z_meas = np.array([xPos_px])
                (self.x_esti, self.P) = self.runKF(
                    self.z_meas, self.x_esti, self.P)

                self.kfData.data.clear()
                self.kfData.data.append(self.x_esti[0])
                self.kfData.data.append(self.x_esti[1])
                self.kfPub.publish(self.kfData)

    def imgCB(self, img):
        try:
            imgCV = self.br.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        cv2.line(imgCV, (int(self.x_esti[0]), int(self.py)), (int(
            self.x_esti[0]), int(self.py)), (0, 255, 0), thickness=30)
        cv2.arrowedLine(imgCV, (int(self.x_esti[0]), int(self.py)), (int(
            self.x_esti[0]) + int(self.x_esti[1]), int(self.py)), (0, 255, 0), thickness=10)

        try:
            self.imgPub.publish(self.br.cv2_to_imgmsg(imgCV, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def runKF(self, z_meas, x_esti, P):
        """Kalman Filter Algorithm."""
        # (1) Prediction.
        x_pred = self.A @ x_esti
        P_pred = self.A @ P @ self.A.T + self.Q

        # (2) Kalman Gain.
        K = P_pred @ self.H.T @ inv(self.H @ P_pred @ self.H.T + self.R)

        # (3) Estimation.
        x_esti = x_pred + K @ (z_meas - self.H @ x_pred)

        # (4) Error Covariance.
        P = P_pred - K @ self.H @ P_pred

        return x_esti, P

    def run(self):
        pass


if __name__ == "__main__":
    kalmanFilter1D = KalmanFilter1D()
    rospy.spin()
    print("KalmanFilter1D Shutdown!")
