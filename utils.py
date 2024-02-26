import os
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import cupy as cp
from copy import deepcopy
from tqdm import tqdm


def get_dataset(dataset):
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) / 255.0
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], 32, 32, 3) / 255.0
        X_test = X_test.reshape(X_test.shape[0], 32, 32, 3) / 255.0

    y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
    n_classes = len(np.unique(y_train))

    return (X_train, y_train), (X_test, y_test), n_classes


def set_index(nInit, nValid, y_train_raw, random_seed=0):
    n_classes = len(np.unique(y_train_raw))
    np.random.seed(random_seed)
    idx_shuffled = np.random.permutation(len(y_train_raw))
    idx = [idx_shuffled[np.where(y_train_raw[idx_shuffled] == (i % n_classes))[0][i // n_classes]]
           for i in range(nInit+nValid)]
    idx_labeled, idx_valid = np.array(idx[:nInit]), np.array(idx[nInit:])
    idx_unlabeled = np.setdiff1d(idx_shuffled, np.concatenate((idx_labeled, idx_valid)))
    np.random.shuffle(idx_labeled)
    np.random.shuffle(idx_unlabeled)
    np.random.shuffle(idx_valid)

    return idx_labeled, idx_unlabeled, idx_valid


def get_model(network, input_shape, n_classes):
    if network.upper() in ['SCNN', 'S-CNN']:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_initializer='he_normal'),
            keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_classes, kernel_initializer='he_normal'),
            keras.layers.Activation('softmax')
        ])
        opt = keras.optimizers.adam(lr=0.001)
    elif network.upper() in ['KCNN', 'K-CNN']:
        model = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape,
                                kernel_initializer='he_normal'),
            keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_classes, kernel_initializer='he_normal'),
            keras.layers.Activation('softmax')
        ])
        opt = keras.optimizers.RMSprop(lr=0.0001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train_and_test_model(step, path, args, X_labeled, y_labeled, valid_set, X_test, y_test, n_classes):
    weights_best_file = f'{path}/weights_best.hdf5'
    if os.path.exists(weights_best_file): os.remove(weights_best_file)
    model = get_model(args.network, X_labeled.shape[1:], n_classes)
    callback = [tf.keras.callbacks.ModelCheckpoint(filepath=weights_best_file, monitor='val_accuracy',
                                                   save_best_only=True, verbose=1)]
    print(f'Training.. step {step + 1:03d}/{args.nStep:03d}') if step < args.nStep else print(f'Training.. final step')
    model.fit(X_labeled, y_labeled, batch_size=args.nBatch, epochs=args.nEpoch, validation_data=valid_set,
              callbacks=callback)
    if os.path.exists(weights_best_file): model.load_weights(weights_best_file)
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f'Step {step + 1:03d}/{args.nStep:03d} - test acc: {acc:.5f}\n') if step < args.nStep \
        else print(f'Final step - test acc: {acc:.5f}\n')
    if os.path.exists(weights_best_file): os.remove(weights_best_file)

    return model, acc


def get_feature(model, X, n_part=1):
    idx_layer = [l for l in range(len(model.layers)) if len(model.layers[l].get_weights()) > 0][-1]
    model_output = K.function([model.layers[0].input], model.layers[idx_layer - 1].output)
    m = X.shape[0]
    n = m // n_part
    feature = model_output([X[:n]])
    for j in range(1, int(np.ceil(m / n))):
        feature = np.concatenate((feature, model_output([X[j*n:(j+1)*n]])), axis=0)

    return feature


def get_ldm_gpu(model, X_pool, X_S, stop_cond=10, n_part=1):
    m = X_pool.shape[0]
    idx_layer = [l for l in range(len(model.layers)) if len(model.layers[l].get_weights()) > 0][-1]
    sigmas = 10 ** cp.arange(-5, 0.1, 0.1)
    weights_ori = model.layers[idx_layer].get_weights()
    weights_ori = [cp.array(weight, dtype=cp.float32) for weight in weights_ori]
    feature_S = cp.array(get_feature(model, X_S, n_part), dtype=cp.float32)
    feature_P = cp.array(get_feature(model, X_pool, n_part), dtype=cp.float32)
    y0_S = cp.argmax(feature_S @ weights_ori[0] + weights_ori[1].reshape(1, -1), axis=1).astype(cp.int32)
    y0_P = cp.argmax(feature_P @ weights_ori[0] + weights_ori[1].reshape(1, -1), axis=1).astype(cp.int32)
    pbar = tqdm(total=len(sigmas))
    ldm = cp.ones(m, dtype=cp.float32)
    ldm_before = deepcopy(ldm)
    for sigma in sigmas:
        count = 0
        while count < stop_cond:
            count += 1
            rhos_P = cp.ones(m, dtype=cp.float32)
            weights_ = [cp.random.normal(weight, sigma, dtype=cp.float32) for weight in weights_ori]
            y_S = cp.argmax(feature_S @ weights_[0] + weights_[1].reshape(1, -1), axis=1).astype(cp.int32)
            y_P = cp.argmax(feature_P @ weights_[0] + weights_[1].reshape(1, -1), axis=1).astype(cp.int32)
            rho_S = cp.mean(y0_S != y_S).astype(cp.float32)
            rhos_P[y0_P != y_P] = rho_S
            ldm = cp.minimum(ldm, rhos_P).astype(cp.float32)
            if cp.sum(ldm < ldm_before) > 0: count = 0
            ldm_before = deepcopy(ldm)
        pbar.update(1)
    pbar.close()

    return ldm.get()
