import tensorflow as tf
import tensorflow.contrib.slim as slim  # TensorFlow-Slim
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import math
import time
import pickle

# Constants
IMG_SIZE = 32 # kích thước ảnh.
NUM_CHANNELS = 3 # Số kênh màu
NUM_CLASSES = 43 # Số lượng lớp

# Model parameters - Các thông số mô hình
LR = 5e-3  # learning rate - Tỷ lệ học
KEEP_PROB = 0.5  # dropout keep probability - Giải phóng neural
OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use - Tối ưu

# Training process
NUM_EPOCH = 5
BATCH_SIZE = 128  # batch size for training (relatively small) Số lượng đem vào 1 lần học
BATCH_SIZE_INF = 2048  # batch size for running inference, e.g. calculating accuracy
VALIDATION_SIZE = 0.2  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './model.ckpt'  # where to save trained model


def preprocess_data(X, y):
    X = X.astype('float32')
    X = (X - 128.) / 128.

    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))

    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.

    y = y_onehot

    return X, y

def next_batch(X, y, batch_size, augment_data):
    start_idx = 0
    while start_idx < X.shape[0]:
        images = X[start_idx: start_idx + batch_size]
        labels = y[start_idx: start_idx + batch_size]

        yield (np.array(images), np.array(labels))

        start_idx += batch_size

def calculate_accuracy(data_gen, data_size, batch_size, accuracy, x, y, keep_prob, sess):
    num_batches = math.ceil(data_size / batch_size)
    last_batch_size = data_size % batch_size
    accs = []  # accuracy for each batch
    for _ in range(num_batches):
        images, labels = next(data_gen)
        acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})
        accs.append(acc)
    acc_full = np.mean(accs[:-1])
    acc = (acc_full * (data_size - last_batch_size) + accs[-1] * last_batch_size) / data_size

    return acc

def neural_network():
    with tf.variable_scope('neural_network'):

        x = tf.placeholder('float', [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
        y = tf.placeholder('float', [None, NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)

        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
            net = slim.conv2d(x, 16, [3, 3], scope='conv0')  # output shape: (32, 32, 16)
            net = slim.max_pool2d(net, [3, 3], 1, padding='SAME', scope='pool0')  # output shape: (32, 32, 16)
            net = slim.conv2d(net, 64, [5, 5], 3, padding='VALID', scope='conv1')  # output shape: (10, 10, 64)
            net = slim.max_pool2d(net, [3, 3], 1, scope='pool1')  # output shape: (8, 8, 64)
            net = slim.conv2d(net, 128, [3, 3], scope='conv2')  # output shape: (8, 8, 128)
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')  # output shape: (8, 8, 64)
            net = slim.max_pool2d(net, [3, 3], 1, scope='pool3')  # output shape: (6, 6, 64)

            net = tf.contrib.layers.flatten(net)
            net = slim.fully_connected(net, 1024, scope='fc4')
            net = tf.nn.dropout(net, keep_prob)
            net = slim.fully_connected(net, 1024, scope='fc5')
            net = tf.nn.dropout(net, keep_prob)
            net = slim.fully_connected(net, NUM_CLASSES, scope='fc6')

        logits = net

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        optimizer = OPT.minimize(loss)

        predictions = tf.argmax(logits, 1)

        correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return x, y, keep_prob, logits, optimizer, predictions, accuracy

def run_training():
    training_file = '/home/hunglv/PycharmProjects/Classification of traffic signs/Classifi_image/data/train.pkl'
    testing_file = '/home/hunglv/PycharmProjects/Classification of traffic signs/Classifi_image/data/test.pkl'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['X'], train['Y']
    X_test, y_test = test['X'], test['Y']

    # n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    # image_shape = X_train.shape[1:3]
    # n_classes = np.unique(y_train).shape[0]

    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)

    with tf.Graph().as_default(), tf.Session() as sess:

        x, y, keep_prob, logits, optimizer, predictions, accuracy = neural_network()

        print('Training model from scratch - Bắt đầu training model')
        train_start_time = time.time()

        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess.run(init)

        for epoch in range(NUM_EPOCH):
            train_gen = next_batch(X_train, y_train, BATCH_SIZE, True)
            num_batches_train = math.ceil(X_train.shape[0] / BATCH_SIZE)
            for _ in range(num_batches_train):
                images, labels = next(train_gen)
                sess.run(optimizer, feed_dict={x: images, y: labels, keep_prob: KEEP_PROB})
            train_gen = next_batch(X_train, y_train, BATCH_SIZE_INF, True)
            train_size = X_train.shape[0]
            train_acc = calculate_accuracy(train_gen, train_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
            valid_gen = next_batch(X_valid, y_valid, BATCH_SIZE_INF, True)
            valid_size = X_valid.shape[0]
            valid_acc = calculate_accuracy(valid_gen, valid_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

            print('Epoch %d -- Train acc.: %.4f, Validation acc.: %.4f' % (epoch + 1, train_acc, valid_acc))

        total_time = time.time() - train_start_time
        print('Total elapsed time: %.2f sec (%.2f min)' % (total_time, total_time / 60))
        print('Calculating test accuracy...')
        test_gen = next_batch(X_test, y_test, BATCH_SIZE_INF, False)
        test_size = X_test.shape[0]
        test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
        print('Test acc.: %.4f' % (test_acc,))

        save_path = saver.save(sess, MODEL_SAVE_PATH)
        print('Trained model saved at: %s' % save_path)
        return test_acc


def run_inference(image_files):
    images = []
    for image_file in image_files:
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        image = np.array(list(image.getdata()), dtype='uint8')
        image = np.reshape(image, (32, 32, 3))

        images.append(image)
    images = np.array(images, dtype='uint8')

    images, _ = preprocess_data(images, np.array([0 for _ in range(images.shape[0])]))

    with tf.Graph().as_default(), tf.Session() as sess:
        x, y, keep_prob, logits, optimizer, predictions, accuracy = neural_network()
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_SAVE_PATH)
        preds = sess.run(predictions, feed_dict={x: images, keep_prob: 1.})
    label_map = {}
    with open('signnames.csv', 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            label_int, label_string = line.split(',')
            label_int = int(label_int)
            label_map[label_int] = label_string
            print()
    final_preds = [label_map[pred] for pred in preds]
    return final_preds


if __name__ == '__main__':
    test_acc, accuracy_history = run_training()
    # print(run_inference(['/home/hunglv/PycharmProjects/Classification of traffic signs/hunglv/2.jpg']))

