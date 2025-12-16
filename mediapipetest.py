import cv2
import numpy as np
import mediapipe as mp 

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands
nhands = hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils



while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgbframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    """ 
    result contains:
        i) multi_hand_landmarks → list of detected hand landmarks.
        ii) multi_handedness → left/right hand info.
    """
    result = nhands.process(rgbframe)
    
    # Draw landmarks if hands are detected
    if result.multi_hand_landmarks:
        """ 
        | Landmark | Name              | Finger |
        | -------- | ----------------- | ------ |
        | 0        | WRIST             | Base   |
        | 1        | THUMB_CMC         | Thumb  |
        | 2        | THUMB_MCP         | Thumb  |
        | 3        | THUMB_IP          | Thumb  |
        | 4        | THUMB_TIP         | Thumb  |
        | 5        | INDEX_FINGER_MCP  | Index  |
        | 6        | INDEX_FINGER_PIP  | Index  |
        | 7        | INDEX_FINGER_DIP  | Index  |
        | 8        | INDEX_FINGER_TIP  | Index  |
        | 9        | MIDDLE_FINGER_MCP | Middle |
        | 10       | MIDDLE_FINGER_PIP | Middle |
        | 11       | MIDDLE_FINGER_DIP | Middle |
        | 12       | MIDDLE_FINGER_TIP | Middle |
        | 13       | RING_FINGER_MCP   | Ring   |
        | 14       | RING_FINGER_PIP   | Ring   |
        | 15       | RING_FINGER_DIP   | Ring   |
        | 16       | RING_FINGER_TIP   | Ring   |
        | 17       | PINKY_MCP         | Pinky  |
        | 18       | PINKY_PIP         | Pinky  |
        | 19       | PINKY_DIP         | Pinky  |
        | 20       | PINKY_TIP         | Pinky  |
        """
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, hands.HAND_CONNECTIONS)
            # Thumb tip
            thumb_tip = handLms.landmark[4]
            
            # Index tip
            index_tip = handLms.landmark[8]
            
            # Middle tip
            middle_tip = handLms.landmark[12]
            
            # Ring tip
            ring_tip = handLms.landmark[16]
            
            # Pinky tip
            pinky_tip = handLms.landmark[20]


            # get image dimensions to convert normalized coords to pixels
            h, w, c = frame.shape 

            print("Thumb tip:", int(thumb_tip.x * w), int(thumb_tip.y * h))
            print("Index tip:", int(index_tip.x * w), int(index_tip.y * h))
            print("Middle tip:", int(middle_tip.x * w), int(middle_tip.y * h))
            print("Ring tip:", int(ring_tip.x * w), int(ring_tip.y * h))
            print("Pinky tip:", int(pinky_tip.x * w), int(pinky_tip.y * h))


    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
