import glob
from pathlib import Path

import cv2
import numpy as np
from sklearn.svm import SVC


# reference :https://heygoldie.com/blog/how-to-know-your-undertone
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


def predict_tone_undertone(imgs):
    data, labels = get_tone_undertone_data()

    X = np.array(data)
    y = np.array(labels)

    SVCClf = SVC(kernel='linear', gamma='scale', shrinking=False, )
    SVCClf.fit(X, y)
    prediction = SVCClf.predict(imgs)
    # print(prediction)
    return prediction


pred = predict_tone_undertone(imgs=[[155, 123, 111], [220, 169, 144], [222, 172, 146]])
print(pred)
