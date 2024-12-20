## -> represents changes made from devnet.py to use Fuzzy Similarity Relation

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
sess = tf.compat.v1.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.sparse import vstack, csc_matrix
from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split

import time

MAX_INT = np.iinfo(np.int32).max
data_format = 0

def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)

def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)

def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''    
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)

ref = tf.Variable(np.random.normal(loc=0., scale=1.0, size=5000), dtype=tf.float32)
## modified loss function
def deviation_loss_using_fuzzy_similarity_relation(y_true, y_pred, ref):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 5.     

    # dev = (y_pred - tf.reduce_mean(ref)) / tf.math.reduce_std(ref)
 ## New Fuzzy Similarity Relation
    mean_ref = tf.reduce_mean(ref)
    variance_ref = tf.math.reduce_variance(ref)

    # Split predictions into normal data and anomalies based on y_true labels
    y_pred_normal = tf.boolean_mask(y_pred, y_true == 0)  # Normal data predictions
    y_pred_anomalies = tf.boolean_mask(y_pred, y_true == 1)  # Anomalies predictions

    # Step 1: Compute Dki, Mki, and λ_k for Normal Data (inliers)
    Dki_normal = (y_pred_normal - mean_ref) ** 2 / variance_ref
    Mki_normal = 1 / (1 + tf.exp(Dki_normal * 2.5))  # Assuming w1 is 2.5
    numerator_normal = tf.reduce_sum((y_pred_normal - mean_ref) ** 2 * tf.exp(Mki_normal * 2.5))
    denominator_normal = tf.reduce_sum(tf.exp(Mki_normal * 2.5))
    lambda_k_normal = numerator_normal / denominator_normal

    # Step 2: Compute Dki, Mki, and λ_k for Anomalies (outliers)
    Dki_anomalies = (y_pred_anomalies - mean_ref) ** 2 / variance_ref
    Mki_anomalies = 1 / (1 + tf.exp(Dki_anomalies * 2.5))  # Assuming w1 is 2.5
    numerator_anomalies = tf.reduce_sum((y_pred_anomalies - mean_ref) ** 2 * tf.exp(Mki_anomalies * 2.5))
    denominator_anomalies = tf.reduce_sum(tf.exp(Mki_anomalies * 2.5))
    lambda_k_anomalies = numerator_anomalies / denominator_anomalies

    # Step 3: Calculate deviation (dev) for both classes separately
    dev_normal = lambda_k_normal * ((y_pred_normal - mean_ref) / tf.math.reduce_std(ref))
    dev_anomalies = lambda_k_anomalies * ((y_pred_anomalies - mean_ref) / tf.math.reduce_std(ref))

    # Step 4: Calculate inlier loss and outlier loss
    inlier_loss_normal = tf.abs(dev_normal)
    outlier_loss_anomalies = tf.abs(tf.maximum(confidence_margin - dev_anomalies, 0.0))

    # Combine the losses back into a single loss value
    normal_loss = (1 - tf.boolean_mask(y_true, y_true == 0)) * inlier_loss_normal
    anomaly_loss = tf.boolean_mask(y_true, y_true == 1) * outlier_loss_anomalies

    total_loss = tf.reduce_mean(normal_loss) + tf.reduce_mean(anomaly_loss)
    return total_loss


def deviation_network(input_shape, network_depth):
    '''
    construct the deviation network-based detection model
    '''
    if network_depth == 4:
        model = dev_network_d(input_shape)
    elif network_depth == 2:
        model = dev_network_s(input_shape)
    elif network_depth == 1:
        model = dev_network_linear(input_shape)
    else:
        sys.exit("The network depth is not set properly")
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=lambda y_true, y_pred: deviation_loss_using_fuzzy_similarity_relation(y_true, y_pred, ref), optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0
 
def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

 
def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    '''      
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


def load_model_weight_predict(model_name, input_shape, network_depth, x_test):
    '''
    load the saved weights to make predictions
    '''
    model = deviation_network(input_shape, network_depth)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)    
    
    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()

def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''  
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise

def run_devnet(args):
    names = args.data_set.split(',')
    # names = ['UNSW_NB15_traintest_backdoor']
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)  
        filename = nm.strip()
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()    
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]  
        n_outliers_org = outliers.shape[0]   
        
        train_time = 0
        test_time = 0
        for i in np.arange(runs):  
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            n_noise  = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)                
            
            rng = np.random.RandomState(random_seed)  
            if data_format == 0:                
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)
                
                noises = inject_noise(outliers, n_noise, random_seed)
                x_train = np.append(x_train, noises, axis = 0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)        
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]                               
                
                noises = inject_noise_sparse(outliers, n_noise, random_seed)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            # Output shape updates
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            print(f"Updated training data size: {x_train.shape[0]}, No. outliers: {len(outlier_indices)}, No. inliers: {len(inlier_indices)}, No. noise injected: {n_noise}")
            
            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size
            nb_batch = args.nb_batch
            n_samples_trn = x_train.shape[0]
            
            # Initialize and summarize the model
            model = deviation_network(input_shape, network_depth)
            print(model.summary())
            model_name = f"./model/devnet_{filename}_{args.cont_rate}cr_{args.batch_size}bs_{args.known_outliers}ko_{network_depth}d.weights.h5"
            
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True)
            
            # Train the model and track time
            start_time = time.time()
            model.fit(batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                      steps_per_epoch=nb_batch, epochs=epochs, callbacks=[checkpointer])
            
            # Get predictions
            y_pred_train = model.predict(x_train)

            # Separate normal and anomalous data
            normal_indices = np.where(y_train == 0)[0]
            anomalous_indices = np.where(y_train == 1)[0]
            normal_preds = y_pred_train[normal_indices]
            anomalous_preds = y_pred_train[anomalous_indices]

            # Calculate metrics for normal data
            mean_ref = tf.reduce_mean(ref)
            variance_ref = tf.math.reduce_variance(ref)

            # Normal Data Metrics
            Dki_normal = (normal_preds - mean_ref) ** 2 / variance_ref
            Mki_normal = 1 / (1 + tf.exp(Dki_normal * 2.5))
            numerator_normal = tf.reduce_sum((normal_preds - mean_ref) ** 2 * tf.exp(Mki_normal * 2.5))
            denominator_normal = tf.reduce_sum(tf.exp(Mki_normal * 2.5))
            lambda_k_normal = numerator_normal / denominator_normal

            # Anomalous Data Metrics
            Dki_anomalous = (anomalous_preds - mean_ref) ** 2 / variance_ref
            Mki_anomalous = 1 / (1 + tf.exp(Dki_anomalous * 2.5))
            numerator_anomalous = tf.reduce_sum((anomalous_preds - mean_ref) ** 2 * tf.exp(Mki_anomalous * 2.5))
            denominator_anomalous = tf.reduce_sum(tf.exp(Mki_anomalous * 2.5))
            lambda_k_anomalous = numerator_anomalous / denominator_anomalous

            # Print separate values for normal and anomalous data
            print(f"\nNormal Data (y_train == 0) Metrics for round {i}:")
            print(f"Distance (Dki): {Dki_normal.numpy()}")
            print(f"Dki size: {Dki_normal.shape[0]}")
            print(f"Membership (Mki): {Mki_normal.numpy()}")
            print(f"Mki size: {Mki_normal.shape[0]}")
            print(f"Influence (lambda_k): {lambda_k_normal.numpy()}")

            print(f"\nAnomalous Data (y_train == 1) Metrics for round {i}:")
            print(f"Distance (Dki): {Dki_anomalous.numpy()}")
            print(f"Dki size: {Dki_anomalous.shape[0]}")
            print(f"Membership (Mki): {Mki_anomalous.numpy()}")
            print(f"Mki size: {Mki_anomalous.shape[0]}")
            print(f"Influence (lambda_k): {lambda_k_anomalous.numpy()}")
            
            train_time += time.time() - start_time
            
            # Test time tracking and performance evaluation
            start_time = time.time()
            scores = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
            test_time += time.time() - start_time
            
            # Evaluate AUC performance
            rauc[i], ap[i] = aucPerformance(scores, y_test)
        
        # Final summary per dataset
        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time /= runs
        test_time /= runs
        
        print(f"Dataset: {filename}, Depth: {network_depth}")
        print(f"Average AUC-ROC: {mean_auc:.4f}, Average AUC-PR: {mean_aucpr:.4f}")
        print(f"Average runtime: {train_time + test_time:.4f} seconds")
        
        # Save results
        writeResults(f"{filename}_{network_depth}", x.shape[0], x.shape[1], n_samples_trn, n_outliers_org, n_outliers,
                     network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)


      
parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1','2', '4'], default='2', help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=30, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--data_set", type=str, default='annthyroid_21feat_normalised', help="a list of data set names")
parser.add_argument("--data_format", choices=['0','1'], default='0',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str, default='./results/devnet_auc_performance_30outliers_0.02contrate_2depth_10runs.csv', help="the output file path")
parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
args = parser.parse_args()
run_devnet(args)
