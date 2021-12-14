#!/usr/bin/env python

import sys
import signal
import math
import numpy as np

import rospy
from std_msgs.msg import Float32
from std_msgs.msg import ColorRGBA
from yolov5_pytorch_ros.msg import BoundingBoxes
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from visualization_msgs.msg import Marker


def signal_handler(signal, frame):
    print('pressed ctrl + c!!!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Intrinsic K, object image 2pos(pixel), object distance(m) -> object world 3pos(m)
# hard coding, sub - /yolov5/bounding_boxes, /kcy/depth -> pub geometry_msgs/pose


class ImageToWorld:
    def __init__(self):
        print("ImageToWorld init!")
        rospy.init_node('ImageToWorld', anonymous=True)
        rospy.Subscriber('/yolov5/bounding_boxes', BoundingBoxes, self.bboxCB)
        rospy.Subscriber('/kcy/depth', Float32, self.depthCB)
        self.posePub = rospy.Publisher('/kcy/pose', PoseStamped, queue_size=10)
        self.markerPub = rospy.Publisher('/kcy/marker', Marker, queue_size=10)

        self.K = np.array([[639.012268, 0.000000, 337.528163],
                           [0.000000, 642.997107, 222.945560],
                           [0.000000, 0.000000, 1.000000]])

        self.depth = 0.0
        self.pos2d = np.array([337.528163, 222.945560, 1]).transpose()
        self.pos3d = np.array([0, 0, 0]).transpose()

        self.poseData = PoseStamped()
        self.seqPose = 0

        self.markerData = Marker()
        self.markerMax = rospy.get_param("~marker_pos_max", 10)
        self.seqMarker = 0

    def bboxCB(self, msg):
        if len(msg.bounding_boxes) > 0:
            if msg.bounding_boxes[0].Class == "blueBall":
                xPos_px = (
                    msg.bounding_boxes[0].xmax + msg.bounding_boxes[0].xmin) / 2
                yPos_px = (
                    msg.bounding_boxes[0].ymax + msg.bounding_boxes[0].ymin) / 2
                self.pos2d = np.array([xPos_px, yPos_px, 1]).transpose()

    def depthCB(self, msg):
        self.depth = msg.data
        self.pos3d = self.transformImage2World(self.K, self.depth, self.pos2d)
        self.dcmCamera2FLU = np.array([[ 0,  0, 1],
                                       [-1,  0, 0],
                                       [ 0, -1, 0]])
        self.pos3dFLU = self.dcmCamera2FLU @ self.pos3d

        self.poseData.header.seq = self.seqPose
        self.poseData.header.stamp = rospy.Time.now()
        self.poseData.header.frame_id = "camera"
        self.poseData.pose.position = Point(
            self.pos3dFLU[0], self.pos3dFLU[1], self.pos3dFLU[2])
        self.poseData.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.posePub.publish(self.poseData)
        self.seqPose += 1

        self.markerData.header.seq = self.seqMarker
        self.markerData.header.stamp = rospy.Time.now()
        self.markerData.header.frame_id = "camera"
        self.markerData.type = Marker.LINE_STRIP
        self.markerData.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.markerData.scale = Vector3(0.01, 0.01, 0.01)
        self.markerData.points.append(
            Point(self.pos3dFLU[0], self.pos3dFLU[1], self.pos3dFLU[2]))
        if len(self.markerData.points) > self.markerMax:
            self.markerData.points.pop(0)
        self.markerData.colors.append(ColorRGBA(0.2, 1.0, 0.2, 1.0))
        if len(self.markerData.colors) > self.markerMax:
            self.markerData.colors.pop(0)
        self.markerPub.publish(self.markerData)
        self.seqMarker += 1

    # 2dPosV = K * [R|t] * 3dPosV
    # where, 2dPosV = [u; v; 1], 3dPosV = [X; Y; Z; 1], [R|t] = [I|O] (3x4)
    # I_3x3 = [1 0 0; 0 1 0; 0 0 1]
    # O_3x1 = [0; 0; 0]
    #
    # 2dPosV = K * 3dPosV
    # Where, 3dPos = [X; Y; Z]
    def transformImage2World(self, K, distance, pos2d):
        K_inv = np.linalg.inv(K)
        pos3d = K_inv @ pos2d * distance
        return pos3d


if __name__ == "__main__":
    imageToWorld = ImageToWorld()
    rospy.spin()
    print("ImageToWorld Shutdown!")
