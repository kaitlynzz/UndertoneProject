import glob
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from tqdm import tqdm as tqdm

warnings.filterwarnings("ignore")


def resize_img(img, desired_height):
    try:
        height, width = img.shape[:2]
        resize_multiple = desired_height / height  # in order to get what factor to multiply by to reach desired size, desired height/true height
        img_height = int(
            height / resize_multiple)  # the new height of the image found through dividing true height by resize
        img_width = int(height / resize_multiple)  # the new width is the same as the height, found by the same method
        img = cv2.resize(img, None, fx=resize_multiple, fy=resize_multiple,
                         interpolation=cv2.INTER_AREA)  # actually resizes the image
        return img
    except:
        return img


def get_face_coords_MTCNN(img):
    detector = MTCNN()  # loads facial recognition object into variable
    detector.detect_faces(img)  # detects the box around ones face in the img inputted as argument
    results = detector.detect_faces(img)  # puts the box inside variable
    if results != []:  # if it detects a face
        b = results[0][
            'box']  # because results is a list of one item, it takes the element at index 0 (the only element)
        # and uses the key 'box' to access  a list of coordinates
        x1 = int(b[0])  # takes the first value of the list and stores it
        x1 = (
            x1 if x1 > 0 else 0)  # checks if this stored value is greater than 0 (still on the screen) if not, then set to 0
        # repeat
        y1 = int(b[1])
        x1 = (y1 if y1 > 0 else 0)

        x2 = int(b[0]) + int(b[2])
        x2 = (x2 if x2 < img.shape[1] else img.shape[1])

        y2 = int(int(b[1])) + int(b[3])
        y2 = (y2 if y2 < img.shape[0] else img.shape[0])
        #
        # print(results[0])  # prints the first element of results list, removing brackets
        # print(results)  # prints results list
        return x1, y1, x2, y2  # returns box coordinates
    else:  # if no face detected:
        return None, None, None, None


def get_HSV_centroid(data_ori):  # creates the clusters and centroids for our data
    clt = KMeans(n_clusters=3,
                 max_iter=5)  # 3 clusters therefore 3 centroids, max repeats while trying to stabilize are 5
    clt.fit(data_ori)  # fits our data to the model which has 3 clusters
    centroids = clt.cluster_centers_  # stores location of centroids

    hist = centroid_histogram(clt)  # make a graph of centroids + clusters
    top1 = np.argsort(hist)[-1]  # returns the index  of the axis in a way that woulkd be sorted
    # top2 = np.argsort(hist)[-2]
    # top3 = np.argsort(hist)[-3]

    col1_hsv = np.uint8([[centroids[top1]]])  # returns unsigned integer of 8 bits which organizes centroid list

    col1_rgb = cv2.cvtColor(col1_hsv, cv2.COLOR_HSV2RGB)

    return col1_hsv, col1_rgb


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def get_GRAYscale_centroid(data_ori):
    clt = KMeans(n_clusters=3, max_iter=20, n_init=3)
    clt.fit(data_ori)
    centroids = clt.cluster_centers_

    hist = centroid_histogram(clt)
    top1 = np.argsort(hist)[-1]

    col1_gray = np.uint8(centroids[top1])
    return col1_gray


def get_colors(img_path, show_rgb=False):
    img = cv2.imread(img_path)  # stores file path to img
    img = resize_img(img, 720)  # resizes image to 720 as desired height
    face_bb = get_face_coords_MTCNN(img)  # receieves a list of the face coordinates
    x1, y1, x2, y2 = face_bb  # reinstates those coordinates into the same order of the original variables
    if x1 == None:  # if no face

        return np.nan, np.nan, np.nan, np.nan  # null

    face = img[y1:y2, x1:x2]  # face's height is this range, faces width is this range
    face = resize_img(face, 50)  # resize the image to 50 desired height again
    face_hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    data = face_hsv.reshape((face_hsv.shape[0] * face_hsv.shape[1], 3))

    col1_hsv, col1_rgb = get_HSV_centroid(data)

    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    data = face_gray.reshape((face_gray.shape[0] * face_gray.shape[1], 1))
    col1_gray = get_GRAYscale_centroid(data)

    if show_rgb == True:
        lo_square1 = np.full((10, 10, 3), col1_rgb, dtype=np.uint8)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

        print(col1_rgb)

    return col1_hsv[0][0], col1_rgb[0][0], col1_gray, face_bb


def init_data():
    df = pd.read_pickle('UTKFace_2.0.pkl')
    df = df.dropna().reset_index(drop=True)

    df['H'] = None
    df['S'] = None
    df['V'] = None

    for i in tqdm(df.index[:]):  # check previous df.index
        H, S, V = df.at[i, 'hsv_color']
        df.at[i, 'H'] = H
        df.at[i, 'S'] = S
        df.at[i, 'V'] = V
    # print(df.head())

    df['H'] = df['H'].astype(np.uint8)
    df['S'] = df['S'].astype(np.uint8)
    df['V'] = df['V'].astype(np.uint8)

    # print(df.describe())

    # outliers = df[np.abs(df.H - df.H.mean()) >= (2 * df.H.std())]
    df = df[np.abs(df.H - df.H.mean()) <= (2 * df.H.std())]
    df = df.reset_index(drop=True)
    # print(outliers.shape, df.shape)

    data = df[['H', 'S', 'V']].values
    return data


def get_tone(img_path):
    data = init_data()
    kmeans = KMeans(n_clusters=12, n_init='auto')
    kmeans.fit(data)
    col1_hsv, col1_rgb, col1_gray, face_bb = get_colors(img_path)
    num = kmeans.predict([col1_hsv])
    color = np.uint8([kmeans.cluster_centers_[num]])
    return color


def get_tone_undertone_data():
    data = []
    labels = []
    for img_path in glob.glob('tone/*'):
        # print(img_path)
        image = cv2.imread(img_path)
        color = image[10, 10]
        # print(color)
        data.append(color.tolist())
        label = Path(img_path).stem
        labels.append(label)
    return data, labels


def predict_tone_undertone(face_tone):
    data, labels = get_tone_undertone_data()
    X = np.array(data)
    y = np.array(labels)
    SVCClf = SVC(kernel='linear', gamma='scale', shrinking=False, )
    SVCClf.fit(X, y)
    prediction = SVCClf.predict([face_tone])
    # print(prediction)
    return prediction.tolist()[0]


def analyze_face_tone(img_path):
    color = get_tone(img_path)
    color_rgb = cv2.cvtColor(color, cv2.COLOR_HSV2RGB).tolist()[0][0]
    print(color_rgb)
    prediction = predict_tone_undertone(color_rgb)
    return color_rgb, prediction

#
# img_path = 'sample/face3.png'
# result = analyze_face_tone(img_path)
# print(result)
