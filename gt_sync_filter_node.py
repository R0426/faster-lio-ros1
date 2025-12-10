#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry

import message_filters
import cv2
import numpy as np
import tf
import os
from collections import deque


class GTGenerator(object):
    def __init__(self):
        rospy.init_node('gt_sync_filter_node', anonymous=True)

        # --- 저장 경로 설정 ---
        bag_name = rospy.get_param("~bag_name", "A")
        self.save_dir = "/home/jairlab/gt_output_" + bag_name + "_FILTER"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        img_dir = os.path.join(self.save_dir, "images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        self.img_dir = img_dir

        # GT 파일 (img_name px py pz qx qy qz qw)
        self.pose_file = open(os.path.join(self.save_dir, "gt_poses.txt"), "w")

        # --- 토픽 설정 ---
        img_topic = rospy.get_param("~img_topic", "/rgb/image_raw/compressed")
        odom_topic = rospy.get_param("~odom_topic", "/Odometry")

        # --- 버퍼 및 파라미터 ---
        self.odom_buffer = deque()  # (stamp_sec, Odometry)
        self.buffer_len_sec = rospy.get_param("~buffer_len_sec", 5.0)  # 최대 5초 유지
        self.count = 0

        # 기준 좌표계 (첫 포즈)
        self.have_ref_pose = False
        self.T_ref_inv = None  # 4x4 inverse of first pose

        # --- Subscriber 등록 ---
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                         self.odom_callback, queue_size=200)
        self.img_sub = rospy.Subscriber(img_topic, CompressedImage,
                                        self.image_callback, queue_size=50)

        rospy.loginfo("GT Sync Node started.")
        rospy.loginfo("Save dir: %s", self.save_dir)
        rospy.loginfo("Img topic: %s, Odom topic: %s", img_topic, odom_topic)

    # ---------------- Odometry 버퍼 콜백 ----------------
    def odom_callback(self, msg):
        t = msg.header.stamp.to_sec()
        self.odom_buffer.append((t, msg))

        # 오래된 것 제거
        while self.odom_buffer and (t - self.odom_buffer[0][0] > self.buffer_len_sec):
            self.odom_buffer.popleft()

    # ---------------- 보간 함수 ----------------
    def get_interpolated_odom(self, t_img):
        """
        이미지 시각 t_img에 대해, 버퍼에서 앞/뒤 오도메트리를 찾아
        선형 보간(위치) + quaternion slerp(자세) 수행.
        못 찾으면 None 반환.
        """
        if len(self.odom_buffer) == 0:
            return None

        # t_img보다 바로 앞/뒤에 있는 두 메시지 찾기
        before = None
        after = None

        for (t, msg) in self.odom_buffer:
            if t <= t_img:
                before = (t, msg)
            if t >= t_img:
                after = (t, msg)
                break

        # 케이스 분기
        if before is None and after is None:
            return None
        if before is None:
            # t_img 이전 오도메트리가 없으면 가장 첫 after 사용
            return after[1]
        if after is None:
            # t_img 이후 오도메트리가 없으면 가장 마지막 before 사용
            return before[1]

        t0, msg0 = before
        t1, msg1 = after

        if abs(t1 - t0) < 1e-6:
            # 너무 가까우면 그냥 하나 사용
            return msg0

        # 보간 비율
        alpha = (t_img - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))

        # 위치 보간
        p0 = msg0.pose.pose.position
        p1 = msg1.pose.pose.position
        px = (1 - alpha) * p0.x + alpha * p1.x
        py = (1 - alpha) * p0.y + alpha * p1.y
        pz = (1 - alpha) * p0.z + alpha * p1.z

        # quaternion slerp
        q0 = msg0.pose.pose.orientation
        q1 = msg1.pose.pose.orientation
        q0_arr = np.array([q0.x, q0.y, q0.z, q0.w])
        q1_arr = np.array([q1.x, q1.y, q1.z, q1.w])

        # tf의 quaternion_slerp 사용
        q_interp = tf.transformations.quaternion_slerp(q0_arr, q1_arr, alpha)

        # 새 Odometry 메시지 생성 (header는 t_img 사용)
        odom_interp = Odometry()
        odom_interp.header.stamp = rospy.Time.from_sec(t_img)
        odom_interp.header.frame_id = msg0.header.frame_id
        odom_interp.child_frame_id = msg0.child_frame_id

        odom_interp.pose.pose.position.x = px
        odom_interp.pose.pose.position.y = py
        odom_interp.pose.pose.position.z = pz

        odom_interp.pose.pose.orientation.x = q_interp[0]
        odom_interp.pose.pose.orientation.y = q_interp[1]
        odom_interp.pose.pose.orientation.z = q_interp[2]
        odom_interp.pose.pose.orientation.w = q_interp[3]

        return odom_interp

    # ---------------- 이미지 콜백 ----------------
    def image_callback(self, img_msg):
        # 이미지 시간
        t_img = img_msg.header.stamp.to_sec()

        # 해당 시간에 가까운 오도메트리 보간
        odom_msg = self.get_interpolated_odom(t_img)
        if odom_msg is None:
            rospy.logwarn("No odometry available for image time %.6f", t_img)
            return

        # 이미지 디코드 및 저장
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_name = "%06d.png" % self.count
        img_path = os.path.join(self.img_dir, img_name)
        cv2.imwrite(img_path, cv_img)

        # ---------------- 현재 포즈 가져오기 ----------------
        pos = odom_msg.pose.pose.position
        ori = odom_msg.pose.pose.orientation
        q = [ori.x, ori.y, ori.z, ori.w]

        # 4x4 변환행렬 생성
        T = tf.transformations.quaternion_matrix(q)
        T[0, 3] = pos.x
        T[1, 3] = pos.y
        T[2, 3] = pos.z

        # ---------------- 기준 좌표계로 변환 ----------------
        if not self.have_ref_pose:
            # 첫 번째 포즈를 기준으로
            self.T_ref_inv = tf.transformations.inverse_matrix(T)
            self.have_ref_pose = True
            rospy.loginfo("Reference pose fixed (frame of first image).")

        T_rel = np.dot(self.T_ref_inv, T)
        trans_rel = tf.transformations.translation_from_matrix(T_rel)
        quat_rel = tf.transformations.quaternion_from_matrix(T_rel)

        px, py, pz = trans_rel
        qx, qy, qz, qw = quat_rel

        # ---------------- GT 저장 ----------------
        self.pose_file.write(
            "{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n".format(
                img_name,
                px, py, pz,
                qx, qy, qz, qw
            )
        )
        self.pose_file.flush()

        self.count += 1
        if self.count % 50 == 0:
            rospy.loginfo("[GT] Saved %d images/poses", self.count)

    # ---------------- 종료 처리 ----------------
    def shutdown(self):
        rospy.loginfo("Shutting down GTGenerator.")
        if not self.pose_file.closed:
            self.pose_file.close()


if __name__ == "__main__":
    node = GTGenerator()
    rospy.on_shutdown(node.shutdown)
    rospy.spin()

