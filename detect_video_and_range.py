# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""

import argparse
import time
import datetime
from DRV8825 import DRV8825
import RPi.GPIO as GPIO
import multiprocessing

from PIL import Image
from PIL import ImageDraw
import cv2

import detect
import tflite_runtime.interpreter as tflite
import platform

from numpy import *
import pyrealsense2 as rs


EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])


def draw_objects(draw, objs, labels, distance):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s score:%.2f' % (labels.get(obj.id, obj.id), obj.score) + "\nDISTANCE: " + str(distance),
              fill='red')
              
def fire_gun(pin):
    print("firing gun")
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(.25)
    GPIO.output(pin, GPIO.LOW)

def cease_fire(pin):
    print("cease fire")
    GPIO.output(pin, GPIO.LOW)
    
def aim_turret(object_Xcenter, object_Ycenter, Motor1, Motor2):
    channel = 23
    cease_fire(channel)
    pixels_from_centerX = 320 - object_Xcenter
    pixels_from_centerY = 240 - object_Ycenter
    steps_to_centerX = abs(int(pixels_from_centerX / 3))
    steps_to_centerY = abs(int(pixels_from_centerY / 3))
    print("steps from centerX", steps_to_centerX)
    def MotorX_forward(steps_to_centerX):
        Motor1.TurnStep(Dir='forward', steps=5, stepdelay=0.0004)
    def MotorX_backward(steps_to_centerX):
        Motor1.TurnStep(Dir='backward', steps=5, stepdelay=0.0004)
    def MotorY_forward(steps_to_centerY):
        Motor1.TurnStep(Dir='forward', steps=5, stepdelay=0.0004)
    def MotorY_backward(steps_to_centerY):
        Motor1.TurnStep(Dir='backward', steps=5, stepdelay=0.0004)
    if 320 - object_Xcenter < 0:
        #kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.DOUBLE) # right
        #process1 = multiprocessing.Process(target=MotorX_forward, args=(0.0004,))
        #process1.run()
        MotorX_forward(0.0004)
        object_Xcenter = 320
        fire_gun(channel)
    elif 320 - object_Xcenter > 0:
        #kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE) # left
        #process1 = multiprocessing.Process(target=MotorX_backward, args=(0.0004,))
        #process1.run()
        MotorX_backward(0.0004)
        object_Xcenter = 320
        fire_gun(channel)
    if 240 - object_Ycenter < 0:
        #kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.DOUBLE)
        #process1 = multiprocessing.Process(target=MotorY_backward, args=(0.0004,))
        object_Ycenter = 240
        #process2 = multiprocessing.Process(target=fire_gun, args=(channel,))
        #process1.start()
        #process2.start()
        #process1.join()
        #process2.join()
    elif 240 - object_Ycenter > 0:
        #kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
        #process1 = multiprocessing.Process(target=MotorY_forward, args=(0.0004,))
        object_Ycenter = 240
        #process2 = multiprocessing.Process(target=fire_gun, args=(channel,))
        #process1.start()
        #process2.start()
        #process1.join()
        #process2.join()
        


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', default="models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input',
                      help='File path of image to process.')
  parser.add_argument('-l', '--labels', default="models/coco_labels.txt",
                      help='File path of labels file.')
  parser.add_argument('-t', '--threshold', type=float, default=0.7,
                      help='Score threshold for detected objects.')
  parser.add_argument('-o', '--output',
                      help='File path for the result image with annotations')
  parser.add_argument('-c', '--count', type=int, default=1,
                      help='Number of times to run inference')
  args = parser.parse_args()

  labels = load_labels(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()
  
  #depth stuff
  config = rs.config()
  config.enable_stream(rs.stream.color, 640, 480)
  pc = rs.pointcloud()
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  pipeline = rs.pipeline()
  pipeline.start(config)
  
  #decimate = rs.decimation_filter()
  #decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
  
  # setup motors
  Motor1 = DRV8825(dir_pin=13, step_pin=19, enable_pin=12, mode_pins=(16, 17, 20))
  Motor2 = DRV8825(dir_pin=24, step_pin=18, enable_pin=4, mode_pins=(21, 22, 27))
  
  # setup relay
  channel = 23
  GPIO.setmode(GPIO.BCM)
  GPIO.setup(channel, GPIO.OUT)
  
  #Motor1.SetMicroStep('hardward','fullstep')
  #Motor2.SetMicroStep('hardward','fullstep')

  #print("Starting video capture")
  #cap = cv2.VideoCapture(0)
  print("Starting video capture")
  while(True):
    #try:
        frames = pipeline.wait_for_frames()
        #_, cv2_im = cap.read()
        color_frame = frames.get_color_frame()
        cv2_im = asanyarray(color_frame.get_data())
        #cv2_im = cv2.imread(cv2_im)
        #cv2_im = cv2.resize(cv2_im, (640, 480))
        cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv2_im)
        scale = detect.set_input(interpreter, image.size,
                               lambda size: image.resize(size, Image.ANTIALIAS))
                          

        #_,cv2_im = cap.read()
        #cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        
        interpreter.invoke()
        #inference_time = time.perf_counter() - start
        objs = detect.get_output(interpreter, args.threshold, scale)

        #image = image.convert('RGB')
        
        #image.save(args.output)
        #image.show()
        
        distance = 0
        object_id = 0
        object_Xcenter = 320
        object_Ycenter = 240
        for obj in objs:
            object_id = obj.id
            print("object id:", obj.id)
            if(object_id == 15 or object_id == 0):
              print("person or geese")
            else:
              break
            height = obj.bbox.ymax-obj.bbox.ymin
            width = obj.bbox.xmax-obj.bbox.xmin
            crop_img = cv2_im[obj.bbox.ymin:obj.bbox.ymax, obj.bbox.xmin:obj.bbox.xmax] # crop image around object
            try:
              cv2.imshow("cropped image", crop_img)
            except:
              pass
              #black_image = zeros((100,100))
              #cv2.imshow("cropped_image", black_image)
            
            object_Xcenter = int(obj.bbox.xmin + (width/2))
            object_Ycenter = int(obj.bbox.ymin + (height/3))        
            # depth stuff
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #depth_frame = decimate.process(depth_frame)

            # Grab new intrinsics (may be changed by decimation)
            #depth_intrinsics = rs.video_stream_profile(
            #    depth_frame.profile).get_intrinsics()
            #w, h = depth_intrinsics.width, depth_intrinsics.height
            depth_image = asanyarray(depth_frame.get_data())
            depth_image = depth_image[object_Ycenter:object_Ycenter+1, object_Xcenter:object_Xcenter+1] #[obj.bbox.ymin:obj.bbox.ymax, obj.bbox.xmin:obj.bbox.xmax] for the cropped bounding box
            print("depth image size:", len(depth_image))
            #depth_colormap = asanyarray(
            #    colorizer.colorize(depth_frame).get_data())
            #mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            #pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            #print("v:", asanyarray(v))
            verts = asanyarray(v).view(float32).reshape(-1, 3)  # xyz
            #print("verts size", verts.size)
            #cropped_verts = verts[obj.bbox.ymin:obj.bbox.ymax, obj.bbox.xmin:obj.bbox.xmax]
            texcoords = asanyarray(t).view(float32).reshape(-1, 2)  # uv
            #print("texcoords:", texcoords)
            #print("height:", obj.bbox.ymax-obj.bbox.ymin, "width:", obj.bbox.xmax-obj.bbox.xmin)
            #print("shape[0]:", cropped_verts.shape[0], "shape[1]:", cropped_verts.shape[1])
            #print("object_Xcenter:", object_Xcenter)
            #print("object_Ycenter:", object_Ycenter)
            try:
                distance = verts[0][0]
                print("distance:", distance)
                print("depth image:", depth_image)
            except Exception as e:
                print(e)
                pass
            
            # # calculating average distance is too computationally heavy
            # # try:
                # # sumOfVerts = 0
                # # numOfDistancePoints = 0
                # # for vert in verts:
                    # # if vert[2] != 0:
                        # # sumOfVerts += vert[2]
                        # # numOfDistancePoints += 1
                # # avg_distance = sumOfVerts/numOfDistancePoints
                
            # # except:
                # # pass
            # # print("AVERAGE DISTANCE:", avg_distance)
            
            
            # size = height*width*distance # I made up the these units to calculate size of object
            print("height:", height, "width:", width)
            #print("size:", size)
            #distance = str(distance)
            draw_objects(ImageDraw.Draw(image), objs, labels, distance)
            if(obj.id == 15): # geese
                date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                open_cv_image = array(image)
                #open_cv_image = open_cv_image[:, :, ::-1]
                cv2.imwrite("images/" + str(date_string) + ".jpg", open_cv_image)
                file = open("text_logs/geese.txt","a") 
                #file.write("TIME:" + str(date_string) + " SIZE: " + size + " DISTANCE: " + distance + "\n")
                file.close()
            
            break # this is to speed up code and only focus on one object
        #print("X center:", object_Xcenter, " Y center:", object_Ycenter)
        cease_fire(channel)
        if(object_id == 0):
            aim_turret(object_Xcenter, object_Ycenter, Motor1, Motor2)
        open_cv_image = array(image) 
        # Convert RGB to BGR 
        #open_cv_image = open_cv_image[:, :, ::-1]    
        cv2.imshow("image", open_cv_image)
    #except Exception as e:
        #print(e)
      
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
  Motor1.Stop()
  Motor2.Stop()
  print("done")


if __name__ == '__main__':
  main()
