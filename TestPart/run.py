from train import emotion_analysis
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from TestPart.model import build_model
from TestPart.gender import gender
import json
if __name__ == '__main__':
    # path = '/home/jing/PycharmProjects/facial/dataset/fer2013/fer2013.csv'
    num_classes = 7

    # x_train, y_train, x_test, y_test = reshape_dataset(path, num_classes)

    model = build_model(num_classes)
    model.load_weights('facial_expression_model_weights.h5')

    # monitor_testset_results = False
    #
    # if monitor_testset_results == True:
    #     # make predictions for test set
    #     predictions = model.predict(x_test)
    #
    #     index = 0
    #     for i in predictions:
    #         if index < 30 and index >= 20:
    #             # print(i) #predicted scores
    #             # print(y_test[index]) #actual scores
    #
    #             testing_img = np.array(x_test[index], 'float32')
    #             testing_img = testing_img.reshape([48, 48])
    #
    #             plt.gray()
    #             plt.imshow(testing_img)
    #             plt.show()
    #
    #             print(i)
    #
    #             emotion_analysis(i)
    #             print("----------------------------------------------")
    #         index = index + 1

    # ------------------------------
    # make prediction for custom image out of test set

    # img = image.load_img("/home/jing/PycharmProjects/facial/dataset/pablo.png", grayscale=True, target_size=(48, 48))
    # img = image.load_img("dataset/monalisa.png", grayscale=True, target_size=(48, 48))
    img_path = "dataset/face_20.png"
    img = image.load_img(img_path, grayscale=True, target_size=(48, 48))
    # img = image.load_img("dataset/jackman.png", grayscale=True, target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255

    file_name = img_path.split("/")[-1].split(".")[0] + '.txt'
    custom = model.predict(x)
    list_outcome = list(custom[0])
    emotion_index = list_outcome.index(max(list_outcome))
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    if emotion_index == 0:
        emotion_kind = objects[0]
    elif emotion_index == 1:
        emotion_kind = objects[1]
    elif emotion_index == 2:
        emotion_kind = objects[2]
    elif emotion_index == 3:
        emotion_kind = objects[3]
    elif emotion_index == 4:
        emotion_kind = objects[4]
    elif emotion_index == 5:
        emotion_kind = objects[5]
    else:
        emotion_kind = objects[6]
    sex = gender(img_path)
    dic = {
           'name': img_path.split("/")[-1].split(".")[0],
           'stu_emotion': emotion_kind,
           'stu_sex': sex}
    js = json.dumps(dic, sort_keys=True)
    file = open(file_name, 'w')
    file.write(js)
    file.close()
    t1 = emotion_analysis(custom[0])
    x = np.array(x, 'float32')
    x = x.reshape([48, 48])
    plt.gray()
    plt.imshow(x)
    plt.show()
    # ------------------------------
