#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

import cv2
import numpy as np
import tf

import os

class GTGenerator:
    def __init__(self):
        rospy.init_node('gt_sync_filter_node', anonymous=True)

        bag_name = rospy.get_param("~bag_name", "A")
        self.save_dir = "/home/jairlab/gt_output_" + bag_name + "_FILTER"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_dir + "/images"):
            os.makedirs(self.save_dir + "/images")

        self.pose_file = open(self.save_dir + "/gt_poses.txt", "w")

        img_topic = "/rgb/image_raw/compressed"
        odom_topic = "/Odometry"

        img_sub = message_filters.Subscriber(img_topic, CompressedImage)
        odom_sub = message_filters.Subscriber(odom_topic, Odometry)

        # 슬롭 증가: 0.10 → 0.15
        ats = message_filters.ApproximateTimeSynchronizer(
            [img_sub, odom_sub],
            queue_size=100,
            slop=0.15,
            allow_headerless=True
        )
        ats.registerCallback(self.callback)

        self.count = 0
        rospy.loginfo("GT Sync Filter Node started")

    def callback(self, img_msg, odom_msg):
        cv_img = np.frombuffer(img_msg.data, np.uint8)
        cv_img = cv2.imdecode(cv_img, cv2.IMREAD_COLOR)

        img_name = "%06d.png" % self.count
        cv2.imwrite(self.save_dir + "/images/" + img_name, cv_img)

        # 위치
        pos = odom_msg.pose.pose.position

        # 오리엔테이션
        ori = odom_msg.pose.pose.orientation
        qx, qy, qz, qw = ori.x, ori.y, ori.z, ori.w

        # -------------------------------
        # 🔥 quaternion 보정 추가 (요청한 부분)
        # -------------------------------
        if qw == 0.0:
            roll = qx
            pitch = qy
            yaw = qz
            qx, qy, qz, qw = tf.transformations.quaternion_from_euler(
                roll, pitch, yaw
            )

        # GT 저장
        self.pose_file.write(
            "{} {} {} {} {} {} {} {}\n".format(
                img_name,
                pos.x, pos.y, pos.z,
                qx, qy, qz, qw
            )
        )

        self.count += 1
        if self.count % 50 == 0:
            rospy.loginfo("[SYNC] Save #{}: {}".format(self.count, img_name))

    def shutdown(self):
        self.pose_file.close()


if __name__ == '__main__':
    node = GTGenerator()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()

