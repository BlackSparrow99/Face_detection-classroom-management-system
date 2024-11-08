# import cv2
# import numpy as np
# import face_recognition
# import os
# import time

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
# # print(len(encode_list))
# print("Encoding is completed.")

# capture_video = cv2.VideoCapture(0)
# while True:
#     success, img_orginal = capture_video.read()
#     img_resized = cv2.resize(img_orginal, (0, 0), None, 0.25, 0.25)
#     img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

#     face_location_in_current_frame = face_recognition.face_locations(img_resized)
#     encode_face_for_current_frame = face_recognition.face_encodings(img_resized, face_location_in_current_frame)

#     for face_location, encode_face in zip(face_location_in_current_frame, encode_face_for_current_frame):
#         matched = face_recognition.compare_faces(encoded_list, encode_face)
#         distance = face_recognition.face_distance(encoded_list, encode_face)
#         # print(matched[0], distance)
#         match_index = np.argmin(distance)

#         if matched[match_index]:
#             name = className[match_index].upper()
#             print(name, time.time())
#             y1, x2, y2, x1 = face_location
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

#             cv2.rectangle(img_orginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img_orginal, (x1, y2+35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img_orginal, f"{name}", (x1+6, y2+14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
#             cv2.putText(img_orginal, f"{round(distance[match_index], 10)}", (x1+6, y2+28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

#     cv2.imshow("Webcam", img_orginal)
#     cv2.waitKey(1)


import cv2
import numpy as np
import face_recognition
import os
import time
import keyboard


path = "Data_set/Face_recognition"

images = []
className = []

myList = os.listdir(path)
print(myList)

for element in myList:
    extension_type = element.split(".")[-1]
    if extension_type == "jpg" or extension_type == "jpeg":
        # print(element)
        curImg = cv2.imread(f"{path}/{element}")
        images.append(curImg)
        name = element.split("_")[0]
        # print(name)
        className.append(name)


def create_encode_list(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


encoded_list = create_encode_list(images)
print("Encoding is completed.")


def render_distance(value):
    if value == "close":
        quality = 0.25
        multiplier = 4
    elif value == "mid":
        quality = 0.5
        multiplier = 2
    else:
        quality = 1
        multiplier = 1
    return quality, multiplier


q_lty, mup = render_distance("far")

capture_video = cv2.VideoCapture(0)
while True:
    if keyboard.is_pressed('q'):
        print("\n\n\n\tProgram closed.")
        break
    success, img_orginal = capture_video.read()

    # img_resized = cv2.resize(img_orginal, (0, 0), None, 0.25, 0.25)
    img_resized = cv2.resize(img_orginal, (0, 0), None, q_lty, q_lty)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    face_location_in_current_frame = face_recognition.face_locations(img_resized)
    encode_face_for_current_frame = face_recognition.face_encodings(img_resized, face_location_in_current_frame)

    for face_location, encode_face in zip(face_location_in_current_frame, encode_face_for_current_frame):
        matched = face_recognition.compare_faces(encoded_list, encode_face)
        distance = face_recognition.face_distance(encoded_list, encode_face)
        match_index = np.argmin(distance)

        threshold = 0.5

        if matched[match_index] and distance[match_index] <= threshold:
            name = className[match_index].upper()
        else:
            name = "Unknown Person"

        print(name, time.time())
        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = y1*mup, x2*mup, y2*mup, x1*mup

        cv2.rectangle(img_orginal, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img_orginal, (x1, y2 + 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img_orginal, name, (x1 + 6, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if name != "Unknown Person":
            cv2.putText(img_orginal, f"{round(distance[match_index], 10)}", (x1 + 6, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Webcam", img_orginal)
    cv2.waitKey(1)
