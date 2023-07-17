from sklearn.cluster import KMeans
from tqdm import tqdm as tqdm
from glob import glob

from mtcnn.mtcnn import MTCNN
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances_argmin_min
from itertools import zip_longest

df = pd.DataFrame()  # initializes the panda dataframe into a variable
for i, path in enumerate(
        glob('UTKFace/*jpg')):  # glob returns True or False, when True it returns the files within the directory of the identified file path
    df.loc[i, 'path'] = path  # it stores file path into its located  row and  column (i, and path)

for i in tqdm(df.index):  # iterate through the rows in the dataframe
    path = df.loc[i, 'path']  # stores the coordinate (row, column) into variable path
    gender = path.split('\\')[1].split('_')[
        1]  # splits variable at the \ to create a list between UTKFace and 9_1_4..etc then it splits again at every _ at the FIRST index which then takes the first index once again
    age = path.split('\\')[1].split('_')[0]  # same thing except it takes the element at index 0 for age
    ethnicity = path.split('\\')[1].split('_')[2]  # takes the element at index 2
    df.loc[
        i, 'gender'] = gender  # stores numerical value of gender into its proper ROW (picture #) under the gender column
    df.loc[i, 'age'] = age  # stores numerical value of age into its proper ROW (picture #) under the age column
    df.loc[
        i, 'ethnicity'] = ethnicity  # stores numerical value of ethnicity into its proper ROW (picture #) under the ethnicity column

df = df[
    df.ethnicity.apply(lambda x: x.isnumeric())]  # checks if the charracters within ethnicity dataframe are numerical

df['age'] = df['age'].values.astype('int8')  # converts datatype of entire column into another one
df['gender'] = df['gender'].values.astype('int8')  # converts datatype of entire column into another one
df['ethnicity'] = df['ethnicity'].values.astype('int8')  # converts datatype of entire column into another one
print(df.shape)  # prints out the rows and columns
print(df.dtypes)  # prints out the datatypes of each column
print(df)  # prints out the entire data frame with titles and indexes, then the data values for each respective column
df.describe
print(df.loc[16731, "age"])  # prints the age of image with index 16731
print(path)  # prints the file path of file with index 16731


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

        print(results[0])  # prints the first element of results list, removing brackets
        print(results)  # prints results list
        return x1, y1, x2, y2  # returns box coordinates
    else:  # if no face detected:
        return None, None, None, None


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


def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def get_HSV_centroid(data_ori):  # creates the clusters and centroids for our data
    clt = KMeans(n_clusters=3,
                 max_iter=5)  # 3 clusters therefore 3 centroids, max repeats while trying to stabilize are 5
    clt.fit(data_ori)  # fits our data to the model which has 3 clusters
    centroids = clt.cluster_centers_  # stores location of centroids

    hist = centroid_histogram(clt)  # make a graph of centroids + clusters
    top1 = np.argsort(hist)[-1]  # returns the index  of the axis in a way that woulkd be sorted
    top2 = np.argsort(hist)[-2]
    top3 = np.argsort(hist)[-3]

    col1_hsv = np.uint8([[centroids[top1]]])  # returns unsigned integer of 8 bits which organizes centroid list

    col1_rgb = cv2.cvtColor(col1_hsv, cv2.COLOR_HSV2RGB)

    return col1_hsv, col1_rgb


def get_GRAYscale_centroid(data_ori):
    clt = KMeans(n_clusters=3, max_iter=20, n_init=3)
    clt.fit(data_ori)
    centroids = clt.cluster_centers_

    hist = centroid_histogram(clt)
    top1 = np.argsort(hist)[-1]

    col1_gray = np.uint8(centroids[top1])
    return col1_gray


def get_colors(img_path, show_gray=False, show_rgb=False):
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
        # ax1.imshow(face[:,:,[2,1,0]])
        # ax2.imshow(lo_square1)
        # plt.show()

    if show_gray == True:
        lo_square1 = np.full((10, 10), col1_gray, dtype=np.uint8)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

        print(col1_gray)
        # ax1.imshow(face_gray, cmap='gray', vmin= 0, vmax = 255)
        # ax2.imshow(lo_square1, cmap='gray', vmin= 0, vmax = 255)
        # plt.show()

    return col1_hsv[0][0], col1_rgb[0][0], col1_gray, face_bb


for i in tqdm(df.index[:5]):

    img_path = df.at[i, 'path']
    if img_path:
        col1_hsv, col1_rgb, col1_gray, face_bb = get_colors(img_path, show_gray=True)  # takes unsigned8 bit integer
        # rgb val gray val and face coords from get_colors and stores into variables

for i in tqdm(df.index[:5]):
    img_path = df.at[i, 'path']
    print(img_path)
    col1_hsv, col1_rgb, col1_gray, face_bb = get_colors(img_path, show_rgb=True)

df['hsv_color'] = None
df['hsv_color'] = df['hsv_color'].astype(object)
df['rgb_color'] = None
df['rgb_color'] = df['rgb_color'].astype(object)
df['gray_color'] = None
df['gray_color'] = df['gray_color'].astype(object)
df['face_bb'] = None
df['face_bb'] = df['face_bb'].astype(object)

num_samples = 700  # how many samples we look at
for i in tqdm(df.index[:num_samples]):
    try:
        img_path = df.at[i, 'path']  # path of image
        col1_hsv, col1_rgb, col1_gray, face_bb = get_colors(img_path)  # column values from get colors

        df.at[i, 'hsv_color'] = col1_hsv  # storing variables into the dataframe
        df.at[i, 'rgb_color'] = col1_rgb
        df.at[i, 'gray_color'] = col1_gray
        df.at[i, 'face_bb'] = face_bb
    except Exception as ex:
        print(ex)
df.to_pickle('UTKFace_2.0.pkl')

# files.download('UTKFace_2.0.pkl')

path = 'UTKFace/'
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
print(df.shape)
df.head()

data = df[["H", 'S', 'V']].values
print(data.shape)


n = 50
ppk = np.zeros(n)
for i in tqdm(range(n)):
    kmeans = KMeans(n_clusters=i + 1, n_init=10, )
    kmeans.fit(data)
    ppk[i] = kmeans.inertia_

plt.figure(figsize=(12, 10))
plt.scatter(np.arange(n), ppk)
# plt.show()

# we use lots of pictures to see what colors are good for them, so if we input our own picture, it can return an okay
# color for us as well

df['H'] = df['H'].astype(np.uint8)
df['S'] = df['S'].astype(np.uint8)
df['V'] = df['V'].astype(np.uint8)

df.describe()

outliers = df[np.abs(df.H - df.H.mean()) >= (2 * df.H.std())]
df = df[np.abs(df.H - df.H.mean()) <= (2 * df.H.std())]
df = df.reset_index(drop=True)
outliers.shape, df.shape

data = df[['H', 'S', 'V']].values

kmeans = KMeans(n_clusters=12, n_init=10)
kmeans.fit(data)
df['y_pred'] = kmeans.labels_

clt0 = df[df.y_pred == 0]
clt1 = df[df.y_pred == 1]
clt2 = df[df.y_pred == 2]
clt3 = df[df.y_pred == 3]
clt4 = df[df.y_pred == 4]
clt5 = df[df.y_pred == 5]
clt6 = df[df.y_pred == 6]
clt7 = df[df.y_pred == 7]
clt8 = df[df.y_pred == 8]
clt9 = df[df.y_pred == 9]
clt10 = df[df.y_pred == 10]
clt11 = df[df.y_pred == 11]

print('Count occurences for each cluster')
df.y_pred.value_counts()

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
labels = kmeans.labels_

ax.scatter(data[:, 0], data[:, 1], data[:, 2],
           c=labels.astype(float), edgecolor='k', cmap=plt.cm.get_cmap('Dark2_r'))
ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Value')
ax.set_title('Skin color clustering after removing outliers')
plt.show()

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
closest
for num, i in enumerate(closest):
    img = cv2.imread(df.at[i, 'path'])

    color_rgb = cv2.cvtColor(np.uint8([[kmeans.cluster_centers_[num]]]), cv2.COLOR_HSV2RGB)

    lo_square = np.full((10, 10, 3), color_rgb, dtype=np.uint8)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

    ax1.imshow(img[:, :, [2, 1, 0]])
    ax2.imshow(lo_square)
    plt.show()


def show_batch(i1, i2, i3, i4, i5, path, df, cluster_num):
    color_rgb = cv2.cvtColor(np.uint8([[kmeans.cluster_centers_[cluster_num]]]), cv2.COLOR_HSV2RGB)
    lo_square = np.full((10, 10, 3), color_rgb, dtype=np.uint8)

    img1 = cv2.imread(df.at[i1, 'path'])
    print("1")
    img2 = cv2.imread(df.at[i2, 'path'])
    img3 = cv2.imread(df.at[i3, 'path'])
    img4 = cv2.imread(df.at[i4, 'path'])
    img5 = cv2.imread(df.at[i5, 'path'])

    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(15, 5))
    print("2")
    ax1.imshow(img1[:, :, [2, 1, 0]])
    ax2.imshow(img2[:, :, [2, 1, 0]])
    ax3.imshow(img3[:, :, [2, 1, 0]])
    ax4.imshow(img4[:, :, [2, 1, 0]])
    ax5.imshow(img5[:, :, [2, 1, 0]])
    print("3")
    ax6.imshow(lo_square)
    print("4")
    plt.show()


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


for i1, i2, i3, i4, i5 in grouper(5, clt0.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt1.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt2.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt3.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt4.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt5.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt6.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt7.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt8.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt9.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt10.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)

for i1, i2, i3, i4, i5 in grouper(5, clt11.index[:5]):
    show_batch(i1, i2, i3, i4, i5, path, df, 0)


def get_tone(img_path):
    col1_hsv, col1_rgb, col1_gray, face_bb = get_colors(img_path)
    num = kmeans.predict([col1_hsv])
    color = np.uint8([kmeans.cluster_centers_[num]])
    return color


img_path = 'sample/face.png'
color = get_tone(img_path)

img = cv2.imread(img_path)
color_rgb = cv2.cvtColor(color, cv2.COLOR_HSV2RGB)
print(color_rgb)
lo_square = np.full((10, 10, 3), color_rgb, dtype=np.uint8)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 5))

ax1.imshow(img[:, :, [2, 1, 0]])
ax2.imshow(lo_square)
plt.show()
