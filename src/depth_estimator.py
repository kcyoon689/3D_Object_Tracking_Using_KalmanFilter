#!/usr/bin/env python

import sys
import signal
import math

import rospy
from std_msgs.msg import Float32
from yolov5_pytorch_ros.msg import BoundingBoxes


def signal_handler(signal, frame):
    print('pressed ctrl + c!!!')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class DepthEstimator:
    def __init__(self):
        print("DepthEstimator init!")
        rospy.init_node('DepthEstimator', anonymous=True)
        rospy.Subscriber('/yolov5/bounding_boxes', BoundingBoxes, self.bboxCB)
        self.depthPub = rospy.Publisher(
            '/kcy/depth', Float32, queue_size=10)

        self.depthData = Float32()

        self.xSize = 0.0
        self.ySize = 0.0
        self.diagonalSize = 0.0
        self.depth = 0.0

    def bboxCB(self, msg):
        if len(msg.bounding_boxes) > 0:
            if msg.bounding_boxes[0].Class == "blueBall":
                self.xSize = msg.bounding_boxes[0].xmax - \
                    msg.bounding_boxes[0].xmin
                self.ySize = msg.bounding_boxes[0].ymax - \
                    msg.bounding_boxes[0].ymin
                # self.diagonalSize = math.sqrt(self.xSize**2 + self.ySize**2)
                # self.depth = self.pixelDiagonalSize2depth(self.diagonalSize)
                self.depth = self.pixelMaxSize2depth(max(self.xSize, self.ySize))
                self.depthData.data = self.depth
                self.depthPub.publish(self.depthData)

    def pixelDiagonalSize2depth(self, pixelSize):
        if pixelSize == 0 or pixelSize > 800:
            return 0.0
        else:
            return 0.06526501 + (954146.8 - 0.06526501)/(1 + (pixelSize/0.0004039717)**1.165955)

    def pixelMaxSize2depth(self, pixelSize):
        if pixelSize == 0 or pixelSize > 480:
            return 0.0
        else:
            return 0.09369008 + (3201486 - 0.09369008)/(1 + (pixelSize/0.0002647558)**1.259128)

if __name__ == "__main__":
    depthEstimator = DepthEstimator()
    rospy.spin()
    print("DepthEstimator Shutdown!")
