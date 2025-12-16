
import cv2 as cv 
import numpy as np 
import mediapipe as mp 
from collections import deque



# use four color in our canvus:
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]



# index: told that which color is currently selected
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0



# kernel is used for dilation purpouse
kernel = np.ones(shape=(5,5),dtype=np.uint8)


# color: R G B Y
colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)] 
colorIndex = 0


# we will create two windows: one is static like white board, another 
# where our live video will be shown
whiteWindows = np.ones((471,636,3))

# # now for 4 color make 4 window + 1 clear window
# whiteWindows = cv.rectangle(whiteWindows,(40,1),(140,65),(0,0,0),2)
# whiteWindows = cv.rectangle(whiteWindows,(160,1),(255,65),(255,0,0),2)
# whiteWindows = cv.rectangle(whiteWindows,(275,1),(370,65),(0,255,0),2)
# whiteWindows = cv.rectangle(whiteWindows,(390,1),(485,65),(0,0,255),2)
# whiteWindows = cv.rectangle(whiteWindows,(505,1),(600,65),(0,255,255),2)


# # now write which rectagle what will be thir jobs:
# cv.putText(whiteWindows,"C L E A R",(49,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
# cv.putText(whiteWindows,"B L U E",(180,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2,cv.LINE_AA)
# cv.putText(whiteWindows,"G R E E N",(280,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,cv.LINE_AA)
# cv.putText(whiteWindows,"R E D",(410,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv.LINE_AA)
# cv.putText(whiteWindows,"YELLOW",(520,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv.LINE_AA)

# for testing purpose:)
# cv.imshow("demo",whiteWindows)
# if cv.waitKey(0) and 0xff==ord('q'):
#     cv.destroyAllWindows()

cv.namedWindow(winname="white_board")


# from mediapipe for hand detection: 
hands = mp.solutions.hands
nhands = hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)
# hand landmarks, (4point for each finger)
mpDraw = mp.solutions.drawing_utils


# take video input:
cap = cv.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height
while cap.isOpened():
    ret,frame = cap.read()
    frame = cv.flip(frame,1)
    h, w, _ = frame.shape
    rgbframe = cv.cvtColor(src=frame,code=cv.COLOR_BGR2RGB)
    
    # frame where the user will be seen:
    frame = cv.rectangle(frame,(40,1),(140,65),(0,0,0),2)
    frame = cv.rectangle(frame,(160,1),(255,65),(255,0,0),2)
    frame = cv.rectangle(frame,(275,1),(370,65),(0,255,0),2)
    frame = cv.rectangle(frame,(390,1),(485,65),(0,0,255),2)
    frame = cv.rectangle(frame,(505,1),(600,65),(0,255,255),2)

    # now write which rectagle what will be thir jobs:
    cv.putText(frame,"C L E A R",(49,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv.LINE_AA)
    cv.putText(frame,"B L U E",(180,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2,cv.LINE_AA)
    cv.putText(frame,"G R E E N",(280,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2,cv.LINE_AA)
    cv.putText(frame,"R E D",(410,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv.LINE_AA)
    cv.putText(frame,"YELLOW",(520,33),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2,cv.LINE_AA)
    
    # detect hands, must give rgbframe:
    result = nhands.process(rgbframe)
    
    # Draw landmarks if hands are detected
    if result.multi_hand_landmarks:
        landmarks = []
        for handLms in result.multi_hand_landmarks:
            # Draw hand landmarks
            mpDraw.draw_landmarks(frame, handLms, hands.HAND_CONNECTIONS)
            
            """
                - We want that, when our finger(index-finger) is near to our thumbs finger then we 
                will not write anything.
                
                - if coordinate of y of index-finger and thumbs < 30 then no draw:
            """
            # handLms have -> x,y,z cordiante value range(0~1)
            # Get index finger tip (landmark 8) and thumb tip (landmark 4)
            index_finger_tip = handLms.landmark[8]
            thumb_tip = handLms.landmark[4]
            
            # Convert to pixel coordinates
            ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            # Calculate distance between index finger and thumb
            distance = ((ix - tx) ** 2 + (iy - ty) ** 2) ** 0.5
        
            #print("differnece and index: y value")
            # print(((thumb_finger[1]-index_finger[1])))
            # print(index_finger[1])
            # no draw
            if (distance<30):
                rpoints.append(deque(maxlen=1024))
                red_index +=1
                
                gpoints.append(deque(maxlen=1024))
                green_index +=1
                
                bpoints.append(deque(maxlen=1024))
                blue_index +=1
                
                ypoints.append(deque(maxlen=1024))
                yellow_index +=1
            
            # if index-finger, y-coor value <=65 then draw:
            elif (iy<=65):
                # 1. =========== if our finger in button ===========
                # if the finger in clear button: reset all 
                if 40<=ix<=140:
                    
                    # empty all sotred point from dequeu
                    bpoints = [deque(maxlen=1024)]
                    gpoints = [deque(maxlen=1024)]
                    rpoints = [deque(maxlen=1024)]
                    ypoints = [deque(maxlen=1024)]
                    
                    # reset color index:
                    red_index = 0
                    green_index = 0
                    blue_index = 0
                    yellow_index = 0
                    
                    # again while the canvus: 67, remaining buttons
                    whiteWindows[:,:,:] = 255
                    
                # blue:
                elif 160<=ix<=255:
                    colorIndex = 0
                
                # green:
                elif 275<=ix<=370:
                    colorIndex = 1
                    
                # red:
                elif 390<=ix<=485:
                    colorIndex = 2 
                
                # yellow:
                elif 505<=ix<=600:
                    colorIndex = 3 
                    
                # 2. =========== if our finger not button and y vlaue<=65 then draw ===========
                # draw means add all point in the dequeue
            else:
                if colorIndex==0:
                    bpoints[blue_index].appendleft((ix,iy))
                elif colorIndex==1:
                    gpoints[green_index].appendleft((ix,iy))
                elif colorIndex==2:
                    rpoints[red_index].appendleft((ix,iy))
                elif colorIndex==3:
                    ypoints[yellow_index].appendleft((ix,iy))
                    
    # if hand are not detected then:
    else:
        rpoints.append(deque(maxlen=1024))
        red_index +=1
        
        gpoints.append(deque(maxlen=1024))
        green_index +=1
        
        bpoints.append(deque(maxlen=1024))
        blue_index +=1
        
        ypoints.append(deque(maxlen=1024))
        yellow_index +=1
        
    
    points  = [bpoints,gpoints,rpoints,ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv.line(whiteWindows,(points[i][j][k-1]),(points[i][j][k]),colors[i],2)
                cv.line(frame,(points[i][j][k-1]),(points[i][j][k]),colors[i],2)
                #print("painting:")
                
    cv.imshow("output",frame)
    cv.imshow("white_board",whiteWindows)
    
    if cv.waitKey(1) and 0xff == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()

