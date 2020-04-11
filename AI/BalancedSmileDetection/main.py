
import dlib
import cv2
import time
import numpy as np
import math



def extract_smile(shape,frame):
    x1=shape.part(48).x
    x2=shape.part(54).x
    y1=shape.part(50).y
    y2=shape.part(57).y
    
    return frame[y1-2:y2+2,x1-2:x2+2]


def calculate_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def classify_the_smile(shape,frame):
    x1=shape.part(50).x
    y1=shape.part(50).y
    x2=shape.part(58).x
    y2=shape.part(58).y
    
    L1=calculate_distance(x1,y1,x2,y2)
    
    x1=shape.part(51).x
    y1=shape.part(51).y
    x2=shape.part(57).x
    y2=shape.part(57).y
    
    L2=calculate_distance(x1,y1,x2,y2)
    
    x1=shape.part(52).x
    y1=shape.part(52).y
    x2=shape.part(56).x
    y2=shape.part(56).y
    
    L3=calculate_distance(x1,y1,x2,y2)
    
    x1=shape.part(48).x
    y1=shape.part(48).y
    x2=shape.part(54).x
    y2=shape.part(54).y
    
    D=calculate_distance(x1,y1,x2,y2)
    
    return (L1+L2+L3)/(3*D)


def main():
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap=cv2.VideoCapture('video5.mp4')
    
    total_frames_count=0
    balanced_smile_frames_count=0;
    
    while True:
        if total_frames_count%1==0:
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces=detector(frame)
                if len(faces)==0:
                    continue
                shape=predictor(frame,faces[0])
                
                smile=extract_smile(shape,frame)
                
                if smile is None:
                        continue
                
                cv2.imshow("Original Image",frame)
                cv2.imshow("Smile",smile)
                
                MAR=classify_the_smile(shape,frame)
                print(MAR)
                if(MAR<=0.35):
                    print('Balanced smile detected',MAR)
                    balanced_smile_frames_count=balanced_smile_frames_count+1
                else:
                    print('Exaggerated or timid Smile detected')
                   
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            else:
                cap.release()
                cv2.destroyAllWindows()
                break
        total_frames_count=total_frames_count+1
    print('Percentage Score:')
    print((balanced_smile_frames_count/total_frames_count)*100)


main()