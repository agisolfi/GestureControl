# Gesture Control
This project uses mediapipe's customizable gesture recognition and PyAutoGUI to track your hand through your webcam and control your mouse. 

## Use
Running the main file (**GestureControl.py**) will open your webcam. Hold your hand out as if you were high-fiving your computer and now move your hand anywhere in the webcam's vision to move your mouse. 

**Controls:**

-  Click: Make a fist with your hand
-  Scroll up: Make a thumbs up gesture
-  Scroll down: Make a thumbs down gesture
-  Copy: Make a "C" with your hand.

**Note**: If user is still in terminal window, the copy gesture can be used to terminate the program.

## Training
A dataset of custom hand gestures was created. This dataset was custom made using a webcam and can be easily recreated. To train your own model with custom gestures, use the  **train_task.py** file with your own data.

Included in this repo:
- **GestureControl.py** - Main File
- **com_control.task** - Pretrained model
- **train_task.py** - Mediapipe's method of training a model from images.
- **requirements.txt** - Requirements for python enviroment.
