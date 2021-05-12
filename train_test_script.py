# train script
"""
@author: mengxue.zhang
"""

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
import os
import time
from tools import *
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras import Model

import tensorflow as tf
import sklearn.metrics as metric


model_file_prefix = 'mdl_simple_k0_wght'
model_file_suffix = '.hdf5'
model_file = model_file_prefix + model_file_suffix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def run_model(model_func=None, mode='DEBUG', base_dir='./soe/MLHDN', train_prefix='train', test_prefix='test',
                run_times=10, shapes=[56, 56], epochs=10, batch_size=32, lr=1e-3):
    assert model_func
    # gpu_setting(init_ratio=0.4)
    verbose = 1 if mode == 'DEBUG' else 0

    t = time.time()
    results = np.zeros((10 + 2, run_times))
    run_time = np.zeros((2, run_times))

    for rt in range(run_times):

        train_mat = './input/' + train_prefix + '_r' + str(rt) + '.mat'
        test_mat = './input/' + test_prefix + '_r' + str(rt) + '.mat'

        task_dir = base_dir + '/' + 'run' + str(rt) + '/'
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        train_gen_batch = get_batch('train', train_mat, batch_size, shapes)
        train_step = get_step(train_mat, bs=batch_size)

        test_gen_batch = get_batch('test', test_mat, 128, shapes)
        test_step = get_step(test_mat, bs=128)
        model = model_func.__next__()

        callbacks, MODEL_FILE = get_model_callbacks(save_file=task_dir)
        if verbose:
            model.summary()
        train_time = time.time()

        model.fit_generator(generator=train_gen_batch,
                            steps_per_epoch=train_step,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=callbacks,
                            shuffle=False)

        train_time = time.time() - train_time
        # print(MODEL_FILE)

        model.load_weights(MODEL_FILE)

        test_gen_batch = get_batch('test', test_mat, 128, shapes)
        test_gt = get_data('test_gt', test_mat)

        test_time = time.time()
        prediction_results = model.predict_generator(generator=test_gen_batch, steps=test_step, verbose=verbose)
        prediction = prediction_results.argmax(axis=-1)
        test_time = time.time() - test_time

        assert (prediction.shape[0] == test_gt.shape[0])

        # sio.savemat(task_dir + 'result.mat', {'gt': test_gt, 'pred': prediction})
        OA, Kappa, ProducerA = CalAccuracy(prediction, test_gt)
        results[0:10, rt] = ProducerA
        results[-2, rt] = OA
        results[-1, rt] = Kappa
        run_time[0, rt] = train_time
        run_time[1, rt] = test_time

        print('rand', rt + 1, ' accuracy:', OA * 100)
        K.clear_session()
        tf.reset_default_graph()
        model = None

    mean_oa = np.mean(results[-2] * 100)
    std_oa = np.std(results[-2] * 100)

    aa = np.mean(results[0:10], axis=0)
    mean_aa = np.mean(aa) * 100
    std_aa = np.std(aa) * 100

    mean_kappa = np.mean(results[-1])
    std_kappa = np.std(results[-1])

    for i in range(10):
        print('Class ', str(i + 1), ' mean:', str(np.mean(results[i])), 'std:', str(np.std(results[i])))

    print('OA mean:', str(mean_oa), 'std:', str(std_oa))
    print('AA mean:', str(mean_aa), 'std:', str(std_aa))
    print('Kappa mean:', str(mean_kappa), 'std:', str(std_kappa))
    print('train_time:', str(np.mean(run_time[0])), 'std:', str(np.std(run_time[0])))
    print('test_time:', str(np.mean(run_time[1])), 'std:', str(np.std(run_time[1])))
    print('\n')


def load_model(model, model_path, i, model_file=model_file):
    trained_cnn_dir = model_path + '/' + 'run' + str(i) + '/' + model_file
    model.load_weights(trained_cnn_dir)
    return model


def gpu_setting(init_ratio=0.4):
    # 40%　GPU
    config = tf.ConfigProto(device_count={'gpu':0})
    config.gpu_options.per_process_gpu_memory_fraction = init_ratio
    session = tf.Session(config=config)
    # SET Session
    KTF.set_session(session)

def CalAccuracy(predict, label):
    label = label.astype(np.uint8)
    n = label.shape[0]
    OA = np.sum(predict == label) * 1.0 / n
    correct_sum = np.zeros((max(label) + 1))
    reali = np.zeros((max(label) + 1))
    predicti = np.zeros((max(label) + 1))
    producerA = np.zeros((max(label) + 1))

    for i in range(0, max(label) + 1):

        correct_sum[i] = np.sum(label[np.where(predict == i)] == i)
        reali[i] = np.sum(label == i)
        predicti[i] = np.sum(predict == i)
        if reali[i] == 0:
            print('Warnning!',str(i),'class have no training samples')
            producerA[i] = 1.0
        else:
            producerA[i] = correct_sum[i] / reali[i]

    Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))
    return OA, Kappa, producerA


def test_images(model_func, mode='DEBUG', model_path='./soe/mv', base_dir='./soe_cp_convnet', test_prefix='test',
                run_times=10, shapes=[56, 56], suffix=None):
    assert model_func
    # gpu_setting(init_ratio=0.4)
    verbose = 1 if mode == 'DEBUG' else 0
    # n_results = []

    results = np.zeros((10 + 2, run_times))

    for rt in range(run_times):
        if suffix:
            test_mat = './input/' + test_prefix + '_r' + str(rt) + '_' + str(suffix) + '.mat'
        else:
            test_mat = './input/' + test_prefix + '_r' + str(rt) + '.mat'

        task_dir = base_dir + '/'
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        model = model_func.__next__()
        if verbose:
            model.summary()
        model = load_model(model, model_path, i=rt)

        test_gen_batch = get_batch('test', test_mat, 128, shapes)
        test_step = get_step(test_mat, bs=128)
        test_gt = get_data('test_gt', test_mat)

        prediction_results = model.predict_generator(generator=test_gen_batch, steps=test_step, verbose=verbose)
        prediction = prediction_results.argmax(axis=-1)

        assert (prediction.shape[0] == test_gt.shape[0])

        # sio.savemat(task_dir+'result.mat', {'gt':test_gt, 'pred': prediction})
        OA, Kappa, ProducerA = CalAccuracy(prediction, test_gt)

        results[0:10, rt] = ProducerA
        results[-2, rt] = OA
        results[-1, rt] = Kappa

        print('rand', rt + 1, ' accuracy:', OA * 100)
        K.clear_session()
        tf.reset_default_graph()
        model = None

    mean_oa = np.mean(results[-2] * 100)
    std_oa = np.std(results[-2] * 100)

    aa = np.mean(results[0:10], axis=0)
    mean_aa = np.mean(aa) * 100
    std_aa = np.std(aa) * 100

    mean_kappa = np.mean(results[-1])
    std_kappa = np.std(results[-1])

    for i in range(10):
        print('Class ', str(i + 1), ' mean:', str(np.mean(results[i])), 'std:', str(np.std(results[i])))

    print('OA mean:', str(mean_oa), 'std:', str(std_oa))
    print('AA mean:', str(mean_aa), 'std:', str(std_aa))
    print('Kappa mean:', str(mean_kappa), 'std:', str(std_kappa))
    print('\n')


def get_model_callbacks(save_file):
    call_backs = []
    MODEL_FILE = save_file + model_file
    call_backs.append(ModelCheckpoint(MODEL_FILE, save_best_only=False, save_weights_only=True))
    return call_backs, MODEL_FILE

