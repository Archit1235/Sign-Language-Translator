import customtkinter as ctk
import csv
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
from model import KeyPointClassifier
import itertools
import copy
from datetime import datetime

# Function to calculate the landmark points from an image
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Iterate over each landmark and convert its coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to preprocess landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Load the KeyPointClassifier model
keypoint_classifier = KeyPointClassifier()

# Read labels from a CSV file
with open('model/keypoint_classifier/label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]


prev = ""

mphands = mp.solutions.hands
mpdrawing = mp.solutions.drawing_utils

# Function to open the camera and perform hand gesture recognition
def open_camera1():
    global prev

    with mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,static_image_mode=False) as hands:            
            _, frame = vid.read()
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            opencv_image = cv2.resize(opencv_image, (display_width, display_height))
                        
            processFrames = hands.process(opencv_image)
            if processFrames.multi_hand_landmarks:
                for lm in processFrames.multi_hand_landmarks:
                    mpdrawing.draw_landmarks(frame, lm, mphands.HAND_CONNECTIONS)

                    landmark_list = calc_landmark_list(frame, lm)

                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    cur = keypoint_classifier_labels[hand_sign_id]
                    if(cur == prev) : 
                        letter_label.configure(text=cur)
                    elif(cur):
                        prev = cur
                   
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame = cv2.flip(frame,1)
            captured_image = Image.fromarray(frame)
            my_image = ctk.CTkImage(dark_image=captured_image,size=(display_width, display_height))
            video_label.configure(image=my_image)
            video_label.after(10, open_camera1)

# Set the appearance mode and color theme for the custom tkinter library
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Create the main window
window = ctk.CTk()
window.geometry('720x1080')
window.title("SIGN LANGUAGE TRANSLATION")

# Initialize the video capture
vid = cv2.VideoCapture(0)
original_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = original_width / original_height
display_width = 700
display_height = int(display_width / aspect_ratio)

letter_font = ctk.CTkFont(family='Consolas', weight='bold', size=200)

main_frame = ctk.CTkFrame(window)
main_frame.pack(fill=ctk.BOTH, expand=True)

video_frame = ctk.CTkFrame(main_frame, corner_radius=12)
video_frame.pack(fill=ctk.BOTH, side=ctk.TOP, expand=True, padx=10, pady=10)

video_label = ctk.CTkLabel(video_frame, text='', height=display_height, width=display_width, justify=ctk.CENTER)
video_label.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

right_frame = ctk.CTkFrame(main_frame, corner_radius=12)
right_frame.pack(fill=ctk.BOTH, side=ctk.BOTTOM, expand=True, padx=10, pady=10)

letter_label = ctk.CTkLabel(right_frame, font=letter_font, justify=ctk.CENTER)
letter_label.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)
letter_label.configure(text='')

open_camera1()

window.mainloop()
