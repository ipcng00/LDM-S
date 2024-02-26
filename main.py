from __future__ import print_function
from utils import *
from strategies import query_by_LDMS


def main(args, idx_rep):
    # set session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.99
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # initialize experiment
    path = f'./results/{args.dataset}_{args.network}'     # path for saving results
    os.makedirs(path, exist_ok=True)
    print(f'{args.dataset}/{args.network}: rep-{idx_rep:02d}')
    (X_train_raw, y_train_raw), (X_test, y_test), n_classes = get_dataset(args.dataset)
    idx_labeled, idx_unlabeled, idx_valid = set_index(args.nInit, args.nValid, y_train_raw, idx_rep)
    valid_set = (X_train_raw[idx_valid], y_train_raw[idx_valid])
    X_S = X_train_raw[np.random.permutation(len(y_train_raw))[:args.nSample]]

    # acquisition step
    test_accs = []
    for step in range(args.nStep+1):
        # make, train, and test model
        X_labeled, y_labeled = X_train_raw[idx_labeled], y_train_raw[idx_labeled]
        model, acc = train_and_test_model(step, path, args, X_labeled, y_labeled, valid_set, X_test, y_test, n_classes)
        test_accs.append(acc)
        np.savetxt(f'{path}/test_accs_{idx_rep+1:03d}.txt', test_accs, fmt='%.5f')

        # query samples
        if step < args.nStep:
            np.random.shuffle(idx_unlabeled)
            idx_pool = idx_unlabeled[:args.nPool]
            X_pool = X_train_raw[idx_pool]
            idx_query = query_by_LDMS(model, X_pool, X_S, args.nQuery)
            idx_labeled = np.concatenate((idx_labeled, idx_pool[idx_query]))
            idx_unlabeled = np.setdiff1d(idx_unlabeled, idx_labeled)

        del model
        tf.compat.v1.keras.backend.clear_session()
