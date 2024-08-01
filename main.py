from tkinter import messagebox
import cv2
from tkinter import *
import tkinter as tk
import time
import pygame
from PIL import Image, ImageTk
import numpy as np

# Initialize the face and smile cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize variables
is_running = False
video = None
start_time = None
timer_running = False

def start_stop_program():
    global is_running, start_time, timer_running
    is_running = not is_running  # Toggle running state
    if is_running:
        start_button.config(text="Stop", bg="#FF5733", fg="white")
        start_time = time.time()
        timer_running = True
        update_timer()
        loop_program()
    else:
        start_button.config(text="Start", bg="#4CAF50", fg="white")
        timer_running = False
        # Release video capture if stopped
        if video:
            video.release()
        cv2.destroyAllWindows()

def play_alarm_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("soundalarm.wav")  # Load the alarm sound file
    pygame.mixer.music.play()
    pygame.time.wait(10000)  # Wait for 10 seconds (10000 milliseconds)
    pygame.mixer.music.stop()  # Stop the alarm sound after 10 seconds

def loop_program():
    if is_running:
        root.after(5000, play_alarm_and_video)  # Schedule alarm and video after 5 seconds

def play_alarm_and_video():
    if not is_running:
        return
    global start_time
    play_alarm_sound()
    video_player = cv2.VideoCapture("sport.mp4")
    start_time = time.time()  # Reset the timer when the video starts
    while video_player.isOpened():
        check, frame = video_player.read()
        if check:
            resized_frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480
            cv2.imshow('Video Player', resized_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    video_player.release()
    cv2.destroyAllWindows()
    open_webcam_for_smile_detection()

def open_webcam_for_smile_detection():
    global video
    video = cv2.VideoCapture(0)  # Open webcam
    if video.isOpened():
        update_gui()

def detect_smile(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        if len(smiles) > 0:
            return True
    return False

def show_success_message():
    global video, start_time, timer_running
    messagebox.showinfo("Success", "چه لبخند قشنگی داری :)")
    if video:
        video.release()
    cv2.destroyAllWindows()
    timer_running = True  # Start timer after closing the message box
    start_time = time.time()
    if is_running:
        root.after(10000, loop_program)  # Restart loop after 10 seconds

def update_gui():
    global video
    if is_running and video is not None:
        check, frame = video.read()
        if check:
            if detect_smile(frame):
                show_success_message()
                return
            cv2.imshow('Smile Detection', frame)
        cv2.waitKey(1)
    if is_running:
        root.after(1, update_gui)

def update_timer():
    if timer_running:
        elapsed_time = int(time.time() - start_time)
        timer_label.config(text=f"Elapsed Time: {elapsed_time} seconds")
    if is_running:
        root.after(1000, update_timer)  # Update timer every second

# Initialize Tkinter window
root = Tk()
root.title("Smile Detection Alarm")
root.configure(bg="#ECECEC")

# Create start/stop button
start_button = Button(root, text="Start", command=start_stop_program, bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), bd=0, padx=10, pady=5)
start_button.pack(pady=10)

# Create timer label
timer_label = Label(root, text="Elapsed Time: 0 seconds", bg="#ECECEC", font=("Arial", 12))
timer_label.pack(pady=10)

# Main loop for the program
root.mainloop()
