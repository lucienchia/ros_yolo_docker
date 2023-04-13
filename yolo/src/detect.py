#!/usr/bin/env python3
import time
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
from sesto_msgs.srv import GetYoloDetector,GetYoloDetectorResponse
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size,  non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging
#from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class yolo_dependencies:
    def __init__(self):
        self.weights_path = rospy.get_param('weights_path')
        self.weights = rospy.get_param('weights', 'best.pt')
        self.img_size = rospy.get_param('img-size', 640)
        self.conf_thresh = rospy.get_param('conf-thresh', 0.5)
        self.iou_thresh = rospy.get_param('iou-thresh', 0.5)
        self.device = rospy.get_param('device','cpu')
        self.weights = self.weights_path + '/' + self.weights 
        self.start_detection = False
        self.is_yolo_object_detected = False
        self.image_path  = "/"
        self.load_model()
        yolo_detection_service = rospy.Service("yolo_detection_sevice",GetYoloDetector, self.yolo_service_detector_cb)
        self.image_cb_spin = False

        rospy.Subscriber("/camera_front/color/image_raw",Image,self.image_cb)
        
    def load_model(self):
          self.device = select_device(self.device)
          self. model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
          trace = True
          self.half = self.device.type != 'cpu' 
          if trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
          if self.half:
            self.model.half()  # to FP16

    def yolo_service_detector_cb (self,req):
        self.image_cb_spin = False
        time.sleep(0.2) 
        self.image_path = req.path
        self.resp= GetYoloDetectorResponse()
        self.start_detection = True
        self.resp.success = True
        if self.image_path != "/":
            rospy.loginfo("[Yolo Detector] image path: %s",self.image_path)
            self.detect()
        else:
          if self.image_cb_spin == False:
            rospy.loginfo("[Yolo Detector] Front Camera not up")
            self.resp.detection = False
            self.resp.success = False
            return self.resp
        time.sleep(2)
        ## the 2 sec sleep is use to let YOLO to full classify the objects, therfore giving the correct results
        self.resp.detection = self.is_yolo_object_detected
        return self.resp

    def imgmsg_to_cv2(self,img_msg):
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        #if img_msg.is_bigendian == (sys.byteorder == 'little'):
         #   image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv

    def image_cb(self,Image):
      self.image_cb_spin = True
      #self.is_yolo_object_detected = False
      if self.start_detection == True:
        self.is_yolo_object_detected = False
        self.sources = self.imgmsg_to_cv2(Image)
        self.detect()
        
        
    def detect(self):
      if self.start_detection == True:
        t0 = time.time()
    # Initialize
        set_logging()
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size

    # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        
        # the following if function is only use for testing the code
        if self.image_path != "/":
          image = cv2.imread (self.image_path)
          self.sources = cv2.resize (image, (720,640), interpolation = cv2.INTER_LINEAR)
          self.image_path = "/"

        img0 = self.sources
        img = letterbox(img0, self.img_size, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
   

    # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        
        #for img, im0s in dataset:
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=True)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=True)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=0, agnostic=True)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
                s, im0 = '', img0
               
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  
                        name = f"{names[int(c)]}"
                        if name == "trolley":
                            self.is_yolo_object_detected = True
                        rospy.loginfo("[Yolo Detector]Detected:%s", name)
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  

                # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        if self.is_yolo_object_detected == False:
          rospy.loginfo("[Yolo Detector] Detected noting")
        time_spend = time.time() - t0
        rospy.loginfo("[Yolo Detector] Done detection in: %.3f", time_spend)
        self.start_detection = False



if __name__ == '__main__':
    dependencies = yolo_dependencies()
    rospy.init_node("yolo_detector_node",anonymous=True)
    try:
      rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
