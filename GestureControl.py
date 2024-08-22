import cv2
import mediapipe as mp
from mediapipe.tasks import python
import threading
import pyautogui

class GestureRecognizer:
    pyautogui.FAILSAFE = False
    quit_program = False
    def main(self):
        num_hands = 1
        model_path = "com_control.task"
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.lock = threading.Lock()
        self.current_gestures = []
        options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = num_hands,
            result_callback=self.__result_callback)
        recognizer = GestureRecognizer.create_from_options(options)

        timestamp = 0 
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame_height, frame_width, _ = frame.shape
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.quit_program:
                break

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1 # should be monotonically increasing, because in LIVE_STREAM mode

                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        # print(ids, landmrk)
                        if ids==9:
                            adjust_x=1920/640*1.25
                            adjust_y=1080/480*1.25
                            cx, cy = landmrk.x * frame_width, landmrk.y*frame_height
                            pyautogui.moveTo(cx*adjust_x,cy*adjust_y)
                            mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            # center_coor=(int(cx),int(cy))
                            # radius=100
                            # color = (0, 255, 0) 

                            # frame = cv2.circle(frame,center_coor,radius,color,2)
                    self.put_gestures(frame)
        
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()

    def put_gestures(self, frame):
        '''
        Places the name of the gesture at the top left of the window.
        '''
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        for hand_gesture_name in gestures:
            # show the prediction on the frame
            cv2.putText(frame, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0,0,255), 2, cv2.LINE_AA)
            y_pos += 50

    def __result_callback(self, result, output_image, timestamp_ms):
        #print(f'gesture recognition result: {result}')
        self.lock.acquire() # solves potential concurrency issues
        self.current_gestures = []
        if result is not None and any(result.gestures):
            print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                if "thumbs_down" in gesture_name:
                    pyautogui.scroll(-50)
                elif "thumbs_up" in gesture_name:
                    pyautogui.scroll(50)
                elif "fist" in gesture_name:
                    pyautogui.click()
                    pyautogui.PAUSE=2.0
                    pyautogui.PAUSE=0
                elif "copy" in gesture_name:
                    pyautogui.hotkey('ctrl', 'c')
                    pyautogui.PAUSE=0.5
                    pyautogui.PAUSE=0
                print(gesture_name) 
                self.current_gestures.append(gesture_name)
        self.lock.release()

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()
