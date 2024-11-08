# import cv2
# import numpy as np
# import face_recognition


# def resize_to_fit(image, max_width=800, max_height=600):
#     h, w = image.shape[:2]
#     if w > max_width or h > max_height:
#         aspect_ratio = w / h
#         if w > h:
#             new_width = max_width
#             new_height = int(new_width / aspect_ratio)
#         else:
#             new_height = max_height
#             new_width = int(new_height * aspect_ratio)
#         return cv2.resize(image, (new_width, new_height))
#     else:
#         return image


# img_mahadi = face_recognition.load_image_file("Data_set/PXL_20220921_083543743.PORTRAIT_copy.jpg")
# img_mahadi = cv2.cvtColor(resize_to_fit(img_mahadi), cv2.COLOR_BGR2RGB)

# img_mahadi_test = face_recognition.load_image_file("Data_set/Me.jpg")
# # The line `# img_mahadi_test =
# # face_recognition.load_image_file("Data_set/PXL_20230114_132838982_copy.jpg")` is a commented-out
# # line in the code. This means that it is not currently being executed as part of the program.


# img_mahadi_test = face_recognition.load_image_file("Data_set/PXL_20220921_083543743.PORTRAIT_copy.jpg")
# img_mahadi_test = face_recognition.load_image_file("Data_set/Me.jpg")
# # img_mahadi_test2 = face_recognition.load_image_file("Data_set/PXL_20230114_132838982_copy.jpg")
# # img_mahadi_test3 = cv2.cvtColor(resize_to_fit(img_mahadi_test), cv2.COLOR_BGR2RGB)

# face_loc = face_recognition.face_locations(img_mahadi)[0]
# encode_mahadi = face_recognition.face_encodings(img_mahadi)[0]
# cv2.rectangle(img_mahadi, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

# face_loc = face_recognition.face_locations(img_mahadi_test)[0]
# encode_mahadi_test = face_recognition.face_encodings(img_mahadi_test)[0]
# cv2.rectangle(img_mahadi_test, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

# result = face_recognition.compare_faces([encode_mahadi], encode_mahadi_test)
# faceDis = face_recognition.face_distance([encode_mahadi], encode_mahadi_test)

# # rounded_faceDis = [round(d, 2) for d in faceDis]
# # print([bool(r) for r in result], faceDis)
# # result_text = f"Match: {bool(result[0])}, Distance: {rounded_faceDis[0]}"
# # cv2.putText(img_mahadi_test, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

# cv2.putText(img_mahadi_test, f"{result[0]}:{round((1.0-faceDis[0]), 2)*100}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# # Show resized images
# cv2.imshow("Mahadi", img_mahadi)
# cv2.imshow("Mahadi Test", img_mahadi_test)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
import face_recognition


def resize_to_fit(image, max_width=800, max_height=600):
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        aspect_ratio = w / h
        if w > h:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        return cv2.resize(image, (new_width, new_height))
    else:
        return image


img_mahadi = face_recognition.load_image_file("Data_set/PXL_20220921_083543743.PORTRAIT_copy.jpg")
img_mahadi = cv2.cvtColor(resize_to_fit(img_mahadi), cv2.COLOR_BGR2RGB)

img_mahadi_test = face_recognition.load_image_file("Data_set/Me.jpg")
# The line `# img_mahadi_test =
# face_recognition.load_image_file("Data_set/PXL_20230114_132838982_copy.jpg")` is a commented-out
# line in the code. This means that it is not currently being executed as part of the program.


img_mahadi_test = face_recognition.load_image_file("Data_set/PXL_20220921_083543743.PORTRAIT_copy.jpg")
img_mahadi_test = face_recognition.load_image_file("Data_set/Me.jpg")
img_mahadi_test2 = face_recognition.load_image_file("Data_set/PXL_20230114_132838982_copy.jpg")
img_mahadi_test3 = cv2.cvtColor(resize_to_fit(img_mahadi_test), cv2.COLOR_BGR2RGB)

face_loc = face_recognition.face_locations(img_mahadi)[0]
encode_mahadi = face_recognition.face_encodings(img_mahadi)[0]
cv2.rectangle(img_mahadi, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

face_loc = face_recognition.face_locations(img_mahadi_test)[0]
encode_mahadi_test = face_recognition.face_encodings(img_mahadi_test)[0]
cv2.rectangle(img_mahadi_test, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 0, 255), 2)

result = face_recognition.compare_faces([encode_mahadi], encode_mahadi_test)
faceDis = face_recognition.face_distance([encode_mahadi], encode_mahadi_test)

# rounded_faceDis = [round(d, 2) for d in faceDis]
# print([bool(r) for r in result], faceDis)
# result_text = f"Match: {bool(result[0])}, Distance: {rounded_faceDis[0]}"
# cv2.putText(img_mahadi_test, result_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

cv2.putText(img_mahadi_test, f"{result[0]}:{round((1.0-faceDis[0]), 2)*100}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Show resized images
cv2.imshow("Mahadi", img_mahadi)
cv2.imshow("Mahadi Test", img_mahadi_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
