# import cv2
# import numpy as np
# import face_recognition
# import os
# import time
# import keyboard
# from datetime import datetime


# path = "Data_set/Face_recognition"

# images = []
# className = []

# myList = os.listdir(path)
# print(myList)

# for element in myList:
#     extension_type = element.split(".")[-1]
#     if extension_type == "jpg" or extension_type == "jpeg":
#         # print(element)
#         curImg = cv2.imread(f"{path}/{element}")
#         images.append(curImg)
#         name = element.split("_")[0]
#         # print(name)
#         className.append(name)


# def create_encode_list(images):
#     encode_list = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encode_list.append(encode)
#     return encode_list


# encoded_list = create_encode_list(images)
# print("Encoding is completed.")


# def render_distance(value):
#     if value == "close":
#         quality = 0.25
#         multiplier = 4
#     elif value == "mid":
#         quality = 0.5
#         multiplier = 2
#     else:
#         quality = 1
#         multiplier = 1
#     return quality, multiplier


# def attendance_entry(name):
#     now = datetime.now()
#     currentTime = now.strftime("%H:%M:%S")
#     currentDate = now.strftime("%d-%m-%Y")

#     path = "Attendance/"
#     os.makedirs(path, exist_ok=True)

#     myList = os.listdir(path)
#     file_name = [element.split(".")[0] for element in myList if element.endswith(".csv")]

#     file_path = f"{path}{currentDate}.csv"
#     if currentDate not in file_name:
#         with open(file_path, 'w') as file:
#             file.write("Name,Time\n")

#     with open(file_path, "r+") as attendance_file:
#         myDataList = attendance_file.readlines()
#         nameList = [line.split(",")[0] for line in myDataList]

#         if name not in nameList:
#             attendance_file.write(f"{name},{currentTime}\n")
#             print(f"{name} at {currentTime} Added to {currentDate}.csv")
#         else:
#             print(f"{name} {currentTime} already recorded for today.")


# q_lty, mup = render_distance("close")

# capture_video = cv2.VideoCapture(0)
# while True:
#     if keyboard.is_pressed('q'):
#         print("\n\n\n\tProgram closed.")
#         break
#     success, img_orginal = capture_video.read()

#     # img_resized = cv2.resize(img_orginal, (0, 0), None, 0.25, 0.25)
#     img_resized = cv2.resize(img_orginal, (0, 0), None, q_lty, q_lty)
#     img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

#     face_location_in_current_frame = face_recognition.face_locations(img_resized)
#     encode_face_for_current_frame = face_recognition.face_encodings(img_resized, face_location_in_current_frame)

#     for face_location, encode_face in zip(face_location_in_current_frame, encode_face_for_current_frame):
#         matched = face_recognition.compare_faces(encoded_list, encode_face)
#         distance = face_recognition.face_distance(encoded_list, encode_face)
#         match_index = np.argmin(distance)

#         threshold = 0.5

#         if matched[match_index] and distance[match_index] <= threshold:
#             name = className[match_index]
#             attendance_entry(name)
#         else:
#             name = "Unknown Person"

#         now = datetime.now()
#         currentTime = now.strftime("%H:%M:%S")
#         print(f"{name} {currentTime}" )
#         y1, x2, y2, x1 = face_location
#         y1, x2, y2, x1 = y1*mup, x2*mup, y2*mup, x1*mup

#         cv2.rectangle(img_orginal, (x1, y1), (x2, y2), (0, 255, 0), 1)
#         cv2.rectangle(img_orginal, (x1, y2 + 35), (x2, y2), (0, 255, 0), cv2.FILLED)
#         cv2.putText(img_orginal, name, (x1 + 6, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

#         if name != "Unknown Person":
#             cv2.putText(img_orginal, f"{round(distance[match_index], 10)}", (x1 + 6, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

#     cv2.imshow("Webcam", img_orginal)
#     cv2.waitKey(1)

import cv2
import numpy as np
import face_recognition
import os
import time
import keyboard
import pickle  # For saving/loading encodings
from datetime import datetime
import threading  # For parallel processing

# Paths
dataset_path = "Data_set/Face_recognition"
attendance_path = "Attendance/"
os.makedirs(attendance_path, exist_ok=True)  # Ensure attendance folder exists

# Load or create encodings
encoded_file = "encoded_faces.pkl"
if os.path.exists(encoded_file):
    with open(encoded_file, 'rb') as f:
        encoded_list, className = pickle.load(f)
    print("Loaded encodings from file.")
else:
    # Load images and names if encoding file doesn't exist
    images, className = [], []
    myList = os.listdir(dataset_path)
    for element in myList:
        if element.endswith(("jpg", "jpeg")):
            img = cv2.imread(f"{dataset_path}/{element}")
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            className.append(element.split("_")[0])
    # Create encoding list and save it
    encoded_list = [face_recognition.face_encodings(img)[0] for img in images]
    with open(encoded_file, 'wb') as f:
        pickle.dump((encoded_list, className), f)
    print("Encodings created and saved.")


# Dynamic image quality adjustment
def render_distance(value):
    if value == "close":
        return 0.25, 4
    elif value == "mid":
        return 0.5, 2
    return 1, 1


# Attendance entry
def attendance_entry(name):
    currentDate = datetime.now().strftime("%d-%m-%Y")
    file_path = f"{attendance_path}{currentDate}.csv"
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("Name,Time\n")

    with open(file_path, "r+") as attendance_file:
        nameList = [line.split(",")[0] for line in attendance_file.readlines()]
        if name not in nameList:
            attendance_file.write(f"{name},{datetime.now().strftime('%H:%M:%S')}\n")
            print(f"{name} recorded in attendance.")

# Initialize video capture
capture_video = cv2.VideoCapture(0)
frame_skip = 3  # Process every 3rd frame to reduce load
q_lty, mup = render_distance("close")


# Threaded attendance writer
def threaded_attendance(name):
    if name != "Unknown Person":
        threading.Thread(target=attendance_entry, args=(name,)).start()


frame_count = 0
while True:
    if keyboard.is_pressed('q'):
        print("Program closed.")
        break

    success, img_orginal = capture_video.read()
    if not success:
        break

    # Skip frames for performance
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize and convert image
    img_resized = cv2.resize(img_orginal, (0, 0), None, q_lty, q_lty)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Face detection
    face_locations = face_recognition.face_locations(img_resized)
    encode_faces = face_recognition.face_encodings(img_resized, face_locations)

    for face_location, encode_face in zip(face_locations, encode_faces):
        # Quick match threshold
        matches = face_recognition.compare_faces(encoded_list, encode_face, tolerance=0.5)
        face_distances = face_recognition.face_distance(encoded_list, encode_face)
        match_index = np.argmin(face_distances)

        if matches[match_index] and face_distances[match_index] < 0.5:
            name = className[match_index]
            threaded_attendance(name)
        else:
            name = "Unknown Person"

        # Draw face box and label
        y1, x2, y2, x1 = [coord * mup for coord in face_location]
        cv2.rectangle(img_orginal, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.rectangle(img_orginal, (x1, y2 + 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img_orginal, name, (x1 + 6, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if name != "Unknown Person":
            cv2.putText(img_orginal, f"{round(face_distances[match_index], 10)}", (x1 + 6, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # y1, x2, y2, x1 = [coord * mup for coord in face_location]
        # cv2.rectangle(img_orginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img_orginal, name, (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Webcam", img_orginal)
    if cv2.waitKey(1) == ord('q'):
        break

capture_video.release()
cv2.destroyAllWindows()
