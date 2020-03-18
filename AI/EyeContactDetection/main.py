# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:23:06 2020

@author: Hardik
"""

import dlib
import cv2
import time
import numpy as np




def return_biggest_contour_in_image(frame):
    contours,_=cv2.findContours(frame,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)            
    if len(contours)>0:
        return contours[0]
    else:
        return None
    

def extract_left_eye(shape_predictor,frame):
    x1_left=shape_predictor.part(36).x 
    x2_left=shape_predictor.part(39).x 
    y1_left=shape_predictor.part(37).y 
    y2_left=shape_predictor.part(40).y
    
    return frame[y1_left:y2_left,x1_left:x2_left]

def extract_right_eye(shape_predictor, frame):
    x1_right=shape_predictor.part(42).x 
    x2_right=shape_predictor.part(45).x
    y1_right=shape_predictor.part(43).y
    y2_right=shape_predictor.part(46).y
    return frame[y1_right:y2_right,x1_right:x2_right]


## this function find the midpoint of the cornea
def detect_contour_location(frame,cv2):
    (x,y,w,h)=cv2.boundingRect(frame)
    return float(x)+float(w/2), float(y)+float(h/2)



    
def detect_eye_contact(vertical_line,horizontal_line,ref_vertical_line,ref_horizontal_line):
    vertical_line_absolute=abs(vertical_line-ref_vertical_line)
    horizontal_line_absolute=abs(horizontal_line-ref_horizontal_line)
    
    if vertical_line_absolute>3 or horizontal_line_absolute>3:
        return False
    else:
        return True


def increase_contrast(frame,cv2):
    lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def determine_fps_of_video(cap,cv2):
    print('fps is')
    print(cap.get(cv2.CAP_PROP_FPS))


def main():
    
    detector=dlib.get_frontal_face_detector()
    eyes_calibrated=False
    total_frames_count=0
    eye_contact_frames=0
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    lefteye_vertical_line=0
    lefteye_horizontal_line=0
    righteye_vertical_line=0
    righteye_horizontal_line=0
    
    cap=cv2.VideoCapture('video2.mp4')
    while True:
        if total_frames_count%15==0:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                #frame=increase_contrast(frame,cv2)           --Don't Apply Contrast on the image because the contours won't be detected
                
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=detector(frame)
                
                shape=predictor(frame,faces[0])
                
                #Left Eye
                lefteye=extract_left_eye(shape,frame)
                _,lefteye=cv2.threshold(lefteye,0,255,cv2.THRESH_BINARY_INV)
                lefteye_contour=return_biggest_contour_in_image(lefteye)
    
                if lefteye_contour is None:
                    continue
                
                #Right Eye
                righteye=extract_right_eye(shape,frame)
                _,righteye=cv2.threshold(righteye,0,255,cv2.THRESH_BINARY_INV)
                righteye_contour=return_biggest_contour_in_image(righteye)
               
                if righteye_contour is None:
                    continue
                
                #calibrate eyes for finding the center poin
                left_v,left_h=detect_contour_location(lefteye_contour,cv2)
                right_v,right_h=detect_contour_location(righteye_contour,cv2)
                if eyes_calibrated is False:
                    print("Setting the eye midpoints")
                    lefteye_vertical_line,lefteye_horizontal_line=left_v,left_h
                    righteye_vertical_line,righteye_horizontal_line=right_v,right_h
                    print("Calibrated: left (%f, %f) Right(%f, %f) " % (left_v,left_h,right_v,right_h))
                    eyes_calibrated=True
                else:
                    #determine lefteyeContact
                    is_left_eye_contact_maintained=detect_eye_contact(left_v,left_h,lefteye_vertical_line,lefteye_horizontal_line)
                    is_right_eye_contact_maintained=detect_eye_contact(right_v,right_h,righteye_vertical_line,righteye_horizontal_line)
                    
                    if is_left_eye_contact_maintained and is_right_eye_contact_maintained:
                        eye_contact_frames=eye_contact_frames+15
                    else:
                        #path='C:\Hardik Personal Data\C Drive Data\EyeContactDetection\eyeContactGone\image{}.png'.format(count)
                        #cv2.imwrite(path,frame)
                        print("Eye Contact is gone: left (%f, %f) Right(%f, %f) " % (left_v,left_h,right_v,right_h))
                
                cv2.imshow('Video',frame)
                cv2.imshow('Left Eye',lefteye)
                cv2.imshow('Right Eye',righteye)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            else:
                break
        total_frames_count=total_frames_count+1

    print("Final Result:")
    print("Percentage of eyecontact maintained is:")
    print((eye_contact_frames/total_frames_count)*100)
    
    cap.release()
    cv2.destroyAllWindows()



main()