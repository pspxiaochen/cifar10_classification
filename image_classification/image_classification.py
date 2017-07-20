"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# from urllib.request import urlretrieve
# from os.path import isfile, isdir
# from tqdm import tqdm
# import problem_unittests as tests
# import tarfile
# import numpy as np
# cifar10_dataset_folder_path = 'cifar-10-batches-py'
#
# # Use Floyd's cifar-10 dataset if present
# floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
# if isfile(floyd_cifar10_location):
#     tar_gz_path = floyd_cifar10_location
# else:
#     tar_gz_path = 'cifar-10-python.tar.gz'
#
# class DLProgress(tqdm):
#     last_block = 0
#
#     def hook(self, block_num=1, block_size=1, total_size=None):
#         self.total = total_size
#         self.update((block_num - self.last_block) * block_size)
#         self.last_block = block_num
#
# if not isfile(tar_gz_path):
#     with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
#         urlretrieve(
#             'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
#             tar_gz_path,
#             pbar.hook)
#
# if not isdir(cifar10_dataset_folder_path):
#     with tarfile.open(tar_gz_path) as tar:
#         tar.extractall()
#         tar.close()
#
#
#
# #%matplotlib inline
# #%config InlineBackend.figure_format = 'retina'
#
# import helper
# # import numpy as np
# #
# # # Explore the dataset
# # batch_id = 1
# # sample_id = 5
# # helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
# from sklearn import preprocessing
# def normalize(x):
#     max = np.max(x)
#     min = np.min(x)
#     return (x - min) / (max - min)
#
# def one_hot_encode(x):
#     x = np.array(x)
#     num_labels = x.shape[0]
#     index_offset = np.arange(num_labels) * 10
#     x_one_hot = np.zeros((num_labels,10))
#     x_one_hot.flat[index_offset + x.ravel()] = 1
#     return x_one_hot
#
# helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

import pickle
import problem_unittests as tests
import helper

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

import tensorflow as tf
import numpy as np
def neural_net_image_input(image_shape):
    image_input = tf.placeholder(dtype=tf.float32,shape=[None,image_shape[0],image_shape[1],image_shape[2]],name='x')
    return image_input

def neural_net_label_input(n_classes):
    label_input = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='y')
    return label_input

def neural_net_keep_prob_input():
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    return keep_prob

def conv2d_maxpool(x_tensor,conv_num_outputs,conv_ksize,conv_strides,pool_ksize,pool_strides):
    w = tf.Variable(tf.truncated_normal([conv_ksize[0],conv_ksize[1],x_tensor.get_shape().as_list()[3],conv_num_outputs],stddev=0.01))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs],stddev=0.01))
    x = tf.nn.conv2d(x_tensor,w,[1,conv_strides[0],conv_strides[1],1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    x=tf.nn.relu(x)
    x=tf.nn.max_pool(x,[1,pool_ksize[0],pool_ksize[1],1],[1,pool_strides[0],pool_strides[1],1],padding="SAME")
    return x


def flatten(x_tensor):
    x_shape = x_tensor.get_shape().as_list()
    x_tensor = tf.reshape(x_tensor,shape=[-1,x_shape[1]*x_shape[2]*x_shape[3]])
    #x_tensor=tf.contrib.layers.flatten(x_tensor)
    return x_tensor

def fully_conn(x_tensor,num_outputs):
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.01))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.01))
    fc1 = tf.matmul(x_tensor,w)
    fc1 = tf.add(fc1,b)
    fc1 = tf.nn.relu(fc1)
    #x_tensor=tf.contrib.layers.fully_connected(x_tensor,num_outputs)
    return fc1

def output(x_tensor,num_outputs):
    #x_tensor=tf.contrib.layers.fully_connected(x_tensor,num_outputs,activation_fn=None)
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.01))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.01))
    fc1 = tf.matmul(x_tensor,w)
    fc1 = tf.add(fc1,b)
    return fc1

def conv_net(x,keep_prob):
    net = conv2d_maxpool(x,64,(3,3),(2,2),(2,2),(2,2))
    net = conv2d_maxpool(net,128,(3,3),(1,1),(2,2),(2,2))
    net = conv2d_maxpool(net,256,(3,3),(1,1),(2,2),(2,2))
    net = conv2d_maxpool(net, 512, (3, 3), (1, 1), (2, 2), (2, 2))
    net = conv2d_maxpool(net, 512, (3, 3), (1, 1), (2, 2), (2, 2))

    net = flatten(net)
    net = tf.nn.dropout(net,keep_prob)
    net = fully_conn(net,4096)
    net = tf.nn.dropout(net, keep_prob)
    net = fully_conn(net,4096)
    net = tf.nn.dropout(net, keep_prob)
    out = output(net,10)
    return out

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
#optimizer = tf.train.AdamOptimizer().minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')



def train_neural_network(session,optimizer,keep_probability,feature_batch,label_batch):
    # logits=conv_net(feature_batch,keep_probability)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=label_batch))
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    session.run(optimizer,feed_dict={x:feature_batch,y:label_batch,keep_prob:keep_probability})

def print_stats(session,feature_batch,label_batch,cost,accuracy):
    feed_dict={x:feature_batch,y:label_batch,keep_prob:1.0}
    print(session.run(cost,feed_dict=feed_dict))
    print('loss: ' + str((session.run(accuracy, feed_dict = feed_dict) * 100.0)) + '%')
    print('Valid accuracy: ' + str((session.run(accuracy, feed_dict = {x: valid_features, y: valid_labels, keep_prob:1.0}) * 100.0)) + '%')


epochs = 50
batch_size = 1024
keep_probability = 0.5

save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

# import tensorflow as tf
# import pickle
# import helper
# import random
#
# # Set batch size if not already set
# try:
#     if batch_size:
#         pass
# except NameError:
#     batch_size = 1024
#
# save_model_path = './image_classification'
# n_samples = 4
# top_n_predictions = 3
#
#
# def test_model():
#     """
#     Test the saved model against the test dataset
#     """
#
#     test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
#     loaded_graph = tf.Graph()
#
#     with tf.Session(graph=loaded_graph) as sess:
#         # Load model
#         loader = tf.train.import_meta_graph(save_model_path + '.meta')
#         loader.restore(sess, save_model_path)
#
#         # Get Tensors from loaded model
#         loaded_x = loaded_graph.get_tensor_by_name('x:0')
#         loaded_y = loaded_graph.get_tensor_by_name('y:0')
#         loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
#         loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
#         loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
#
#         # Get accuracy in batches for memory limitations
#         test_batch_acc_total = 0
#         test_batch_count = 0
#
#         for test_feature_batch, test_label_batch in helper.batch_features_labels(test_features, test_labels,
#                                                                                  batch_size):
#             test_batch_acc_total += sess.run(
#                 loaded_acc,
#                 feed_dict={loaded_x: test_feature_batch, loaded_y: test_label_batch, loaded_keep_prob: 1.0})
#             test_batch_count += 1
#
#         print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))
#
#         # Print Random Samples
#         random_test_features, random_test_labels = tuple(
#             zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
#         random_test_predictions = sess.run(
#             tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
#             feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
#         helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)
#
#
# test_model()