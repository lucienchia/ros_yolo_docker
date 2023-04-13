#!/usr/bin/env python3

import unittest
import rospy, rostest
import rostest
import sys
from sesto_msgs.srv import GetYoloDetector
import time

class TestYolo(unittest.TestCase):
        
   def test_detect_horse(self):
     rospy.wait_for_service('yolo_detection_sevice')
     yolo_trolley_detection_srv = rospy.ServiceProxy("yolo_detection_sevice",GetYoloDetector)
     image_path = rospy.get_param("image_path")
     image_path = image_path + "/"+"horses.jpg"
     yolo_trolley_detection_resp =yolo_trolley_detection_srv(image_path)
     time.sleep(2)
     self.assertTrue(yolo_trolley_detection_resp.success) 
     self.assertFalse(yolo_trolley_detection_resp.detection)
     
   def test_detect_trolley(self):
     rospy.wait_for_service('yolo_detection_sevice')
     yolo_trolley_detection_srv = rospy.ServiceProxy("yolo_detection_sevice",GetYoloDetector)
     image_path = rospy.get_param("image_path")
     image_path = image_path + "/"+"trolley.png"
     yolo_trolley_detection_resp =yolo_trolley_detection_srv(image_path)
     time.sleep(2)
     self.assertTrue(yolo_trolley_detection_resp.success) 
     self.assertTrue(yolo_trolley_detection_resp.detection)
     
   def test_empty_camera(self):
     rospy.wait_for_service('yolo_detection_sevice')
     yolo_trolley_detection_srv = rospy.ServiceProxy("yolo_detection_sevice",GetYoloDetector)
     yolo_trolley_detection_resp =yolo_trolley_detection_srv("/")
     time.sleep(2)
     self.assertFalse(yolo_trolley_detection_resp.success) 
     self.assertFalse(yolo_trolley_detection_resp.detection)
     
     
if __name__ == '__main__':
    rostest.rosrun('yolo', 'test_yolo',TestYolo)
