# directory folder for dropping image to match into

import cv2
import os
import glob
img_dir = "imagestocompare" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')

files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)


# algoririthms
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(img, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images to compare
# more images in directory will take longer to load and process

all_images_to_compare = []
titles = []
for f in glob.iglob("imagestocheckagainst\*"):  # directory where the files to check against are
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)

for image_to_compare, title in zip(all_images_to_compare, titles):
    # 1) Check if 2 images are equals
    if img.shape == image_to_compare.shape:
        difference = cv2.subtract(img, image_to_compare)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print(title + ": 100%")  # PRINTS THE EXACT MATCHES
    # 2) Check for similarities between the 2 images
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    percentage_similarity = len(good_points) / number_keypoints * 100

    #print result
    print(title + " : " + str(int(percentage_similarity)) + "%")
