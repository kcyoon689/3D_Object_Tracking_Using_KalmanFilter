#!/usr/bin/env python

import sys
import signal
import numpy as np
from numpy.linalg import inv
import cv2
from cv_bridge import CvBridge, CvBridgeError

import rospy
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker


def signal_handler(signal, frame):
    print('pressed ctrl + c!!!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class KalmanFilter2D:
    def __init__(self):
        print("KalmanFilter2D init!")
        rospy.init_node('KalmanFilter2D', anonymous=True)
        rospy.Subscriber('/kcy/pose', PoseStamped, self.poseCB)
        rospy.Subscriber('/yolov5/image_output', Image, self.imgCB)
        self.posPub = rospy.Publisher(
            '/kcy/kf/pos', PoseStamped, queue_size=10)
        self.velPub = rospy.Publisher(
            '/kcy/kf/vel', TwistStamped, queue_size=10)
        self.markerPosPub = rospy.Publisher(
            '/kcy/kf/marker_pos', Marker, queue_size=10)
        self.markerVelPub = rospy.Publisher(
            '/kcy/kf/marker_vel', Marker, queue_size=10)
        self.imgPub = rospy.Publisher(
            '/kcy/kf/image_output', Image, queue_size=10)

        self.br = CvBridge()
        self.imgCV = np.zeros(1)
        self.K = np.array([[639.012268, 0.000000, 337.528163],
                           [0.000000, 642.997107, 222.945560],
                           [0.000000, 0.000000, 1.000000]])

        self.posData = PoseStamped()
        self.seqPos = 0

        self.velData = TwistStamped()
        self.seqVel = 0

        self.markerPosData = Marker()
        self.markerPosMax = rospy.get_param("~marker_pos_max", 10)
        self.seqMarkerPos = 0

        self.markerVelData = Marker()
        self.seqMarkerVel = 0

        self.A = np.array([[1, .1, 0,  0, 0,  0],
                           [0,  1, 0,  0, 0,  0],
                           [0,  0, 1, .1, 0,  0],
                           [0,  0, 0,  1, 0,  0],
                           [0,  0, 0,  0, 1, .1],
                           [0,  0, 0,  0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0]])
        self.Q = np.array([[.1, 0, 0, 0, 0, 0],
                           [0, 50, 0, 0, 0, 0],
                           [0, 0, .1, 0, 0, 0],
                           [0, 0, 0, 50, 0, 0],
                           [0, 0, 0, 0, .1, 0],
                           [0, 0, 0, 0, 0, 50]])
        self.R = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self.prevTime = rospy.get_time()
        self.dt = 0.0

        # x_0 [x, vx, y, vy, z, vz]
        self.x_esti = np.array([0, 0, 0, 0, 0, 0]).T
        self.P = 100 * np.eye(6)  # P_0
        self.z_meas = np.array([0, 0, 0]).T

    def poseCB(self, msg):
        curTime = rospy.get_time()
        self.dt = curTime - self.prevTime
        self.prevTime = curTime
        self.A = np.array([[1, self.dt, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 1]])

        xPos_m = msg.pose.position.x
        yPos_m = msg.pose.position.y
        zPos_m = msg.pose.position.z
        self.z_meas = np.array([xPos_m, yPos_m, zPos_m]).T
        (self.x_esti, self.P) = self.runKF(self.z_meas, self.x_esti, self.P)

        self.posData.header.seq = self.seqPos
        self.posData.header.stamp = rospy.Time.now()
        self.posData.header.frame_id = "camera"
        self.posData.pose.position.x = self.x_esti[0]
        self.posData.pose.position.y = self.x_esti[2]
        self.posData.pose.position.z = self.x_esti[4]
        self.posPub.publish(self.posData)
        self.seqPos += 1

        self.velData.header.seq = self.seqVel
        self.velData.header.stamp = rospy.Time.now()
        self.velData.header.frame_id = "camera"
        self.velData.twist.linear.x = self.x_esti[1]
        self.velData.twist.linear.y = self.x_esti[3]
        self.velData.twist.linear.z = self.x_esti[5]
        self.velPub.publish(self.velData)
        self.seqVel += 1

        self.markerPosData.header.seq = self.seqMarkerPos
        self.markerPosData.header.stamp = rospy.Time.now()
        self.markerPosData.header.frame_id = "camera"
        self.markerPosData.type = Marker.LINE_STRIP
        self.markerPosData.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.markerPosData.scale = Vector3(0.01, 0.01, 0.01)
        self.markerPosData.points.append(
            Point(self.x_esti[0], self.x_esti[2], self.x_esti[4]))
        if len(self.markerPosData.points) > self.markerPosMax:
            self.markerPosData.points.pop(0)
        self.markerPosData.colors.append(ColorRGBA(0.2, 0.2, 1.0, 1.0))
        if len(self.markerPosData.colors) > self.markerPosMax:
            self.markerPosData.colors.pop(0)
        self.markerPosPub.publish(self.markerPosData)
        self.seqMarkerPos += 1

        self.markerVelData.header.seq = self.seqMarkerVel
        self.markerVelData.header.stamp = rospy.Time.now()
        self.markerVelData.header.frame_id = "camera"
        self.markerVelData.type = Marker.ARROW
        self.markerVelData.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.markerVelData.scale = Vector3(0.01, 0.01, 0.01)
        self.markerVelData.color = ColorRGBA(1.0, 0.2, 0.2, 1.0)
        self.markerVelData.points.clear()
        self.markerVelData.points.append(Point(self.x_esti[0],
                                               self.x_esti[2],
                                               self.x_esti[4]))
        self.markerVelData.points.append(Point(self.x_esti[0] + self.x_esti[1],
                                               self.x_esti[2] + self.x_esti[3],
                                               self.x_esti[4] + self.x_esti[5]))
        self.markerVelPub.publish(self.markerVelData)
        self.seqMarkerVel += 1

        dcmFLU2Camera = np.array([[0, -1,  0],
                                  [0,  0, -1],
                                  [1,  0,  0]])
        velStartCamera = dcmFLU2Camera @ np.array([self.x_esti[0],
                              self.x_esti[2],
                              self.x_esti[4]])
        velStartImage = self.K @ velStartCamera / velStartCamera[2]

        velEndCamera = dcmFLU2Camera @ np.array([self.x_esti[0] + self.x_esti[1],
                            self.x_esti[2] + self.x_esti[3],
                            self.x_esti[4] + self.x_esti[5]])
        velEndImage = self.K @ velEndCamera / velEndCamera[2]

        cv2.line(self.imgCV,
                 (int(velStartImage[0]), int(velStartImage[1])),
                 (int(velStartImage[0]), int(velStartImage[1])),
                 (0, 255, 0),
                 thickness=30)
        cv2.arrowedLine(self.imgCV,
                        (int(velStartImage[0]), int(velStartImage[1])),
                        (int(velEndImage[0]), int(velEndImage[1])),
                        (0, 255, 0),
                        thickness=10)

        try:
            self.imgPub.publish(self.br.cv2_to_imgmsg(self.imgCV, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def imgCB(self, img):
        try:
            self.imgCV = self.br.imgmsg_to_cv2(img, "bgr8")
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


if __name__ == "__main__":
    kalmanFilter2D = KalmanFilter2D()
    rospy.spin()
    print("KalmanFilter2D Shutdown!")
