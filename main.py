import numpy as np
from aae import AAE

def load_face_data():
    from glob import glob
    import cv2
    file_list = glob('./lfw/*/*.jpg')
    num_training_examples = 10000
    num_testing_examples = 1000
    train_images = [cv2.cvtColor(cv2.resize(cv2.imread(path), (64, 64)), cv2.COLOR_BGR2RGB) for path in file_list[:num_training_examples]]
    test_images = [cv2.cvtColor(cv2.resize(cv2.imread(path), (64, 64)), cv2.COLOR_BGR2RGB) for path in file_list[-num_testing_examples:]]
    print(np.array(train_images).shape)
    print(np.array(test_images).shape)
    return np.array(train_images), np.array(test_images)

def load_fashion_data():
    import cv2
    import numpy as np
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.array([cv2.resize(img, (64, 64)) for img in x_train])
    x_train = x_train[:, :, :, np.newaxis]
    x_train = np.concatenate([x_train] * 3, axis=3)
    print(x_train.shape)

    x_test = np.array([cv2.resize(img, (64, 64)) for img in x_test])
    x_test = x_test[:, :, :, np.newaxis]
    x_test = np.concatenate([x_test] * 3, axis=3)
    print(x_test.shape)
    return x_train, x_test

def load_anime_data():
    from glob import glob
    import cv2
    file_list = glob('./anime/*.jpg')
    num_training_examples = 50000
    num_testing_examples = 1000
    train_images = [cv2.cvtColor(cv2.resize(cv2.imread(path), (64,64)), cv2.COLOR_BGR2RGB) for path in file_list[:num_training_examples]]
    test_images = [cv2.cvtColor(cv2.resize(cv2.imread(path), (64,64)), cv2.COLOR_BGR2RGB) for path in file_list[-num_testing_examples:]]
    print(np.array(train_images).shape)
    print(np.array(test_images).shape)
    return np.array(train_images),np.array(test_images)

if __name__ == '__main__':

    model = AAE('anime_aae', load_anime_data, 64, 64, 3)
    model.train(epochs=10000, batch_size=32, sample_interval=100)

    model = AAE('fashion_aae', load_fashion_data, 64, 64, 3)
    model.train(epochs=10000, batch_size=32, sample_interval=100)

    model = AAE('humanface_aae', load_face_data, 64, 64, 3)
    model.train(epochs=10000, batch_size=32, sample_interval=100)
    model.hide_message('anime_aae')
    model.hide_message('fashion_aae')
