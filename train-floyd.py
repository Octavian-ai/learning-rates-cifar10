import numpy as np
import tensorflow as tf
from time import time
import math
import pickle
import numpy as np
import os
import time
import csv
from urllib.request import urlretrieve
from include.model import model

def get_data_set(name="train"):
    x = None
    y = None
    folder_name = "cifar-10-batches-py"
    f = open('/data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()
    if name is "train":
        for i in range(5):
            f = open('/data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()
            _X = datadict["data"]
            _Y = datadict['labels']
            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1, 32*32*3)
            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)
    elif name is "test":
        f = open('/data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()
        x = datadict["data"]
        y = np.array(datadict['labels'])
        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32*32*3)
    return x, dense_to_one_hot(y)

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")
x, y, output, y_pred_cls, global_step, learning_rate = model()
global_accuracy = 0

# PARAMS
_BATCH_SIZE = 128
_EPOCH = 20000
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"
# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)

# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
    
def train(epoch, learningrate):
    #learning_rate = tf.convert_to_tensor(learningrate, dtype=tf.float32)
    batch_size = int(math.ceil(len(train_x) / _BATCH_SIZE))
    i_global = 0
    for s in range(batch_size):
        batch_xs = train_x[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        batch_ys = train_y[s*_BATCH_SIZE: (s+1)*_BATCH_SIZE]
        start_time = time.time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: learningrate})
        duration = time.time() - start_time
        if s % 10 == 0:
            percentage = int(round((s/batch_size)*100))
            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)
            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _BATCH_SIZE / duration))
    test_and_save(i_global, epoch, learningrate)
    
def test_and_save(_global_step, epoch, learningrate):
    global global_accuracy
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: learningrate}
        )
        i = j
    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{})"
    print(mes.format((epoch+1), acc, correct_numbers, len(test_x)))
    if global_accuracy != 0 and global_accuracy < acc:
        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)
        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)
        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc
    elif global_accuracy == 0:
        global_accuracy = acc
    print("###########################################################################################################")
          
def LRRange(mul=5):
	
	for i in range(mul*6, 0, -1):
		lr = pow(0.1, i/mul)
		yield lr
	for i in range(1, 2*mul+1):
		lr = pow(10, i/mul)
		yield lr
    
    
    
      
        
def main():
    
    for learningrate in LRRange(mul=5):
        global sess, train_writer, global_accuracy
        epoch_sum = 0
        no_improvement_count = 0
        old_acc = 0
        status = 0
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
        global_accuracy = 0
        sess.run(tf.global_variables_initializer())
        with sess.as_default(): 
            start = time.time()
            for i in range(_EPOCH):
                epoch_start = time.time()
                print("\nEpoch: {0}/{1}\n".format((i+1), _EPOCH))
                train(i, learningrate)
                epoch_end = time.time()
                epoch_sum = epoch_sum + (epoch_end - epoch_start)
                i = 0
                predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
                while i < len(test_x):
                    j = min(i + _BATCH_SIZE, len(test_x))
                    batch_xs = test_x[i:j, :]
                    batch_ys = test_y[i:j, :]
                    predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
                    i = j
                correct = (np.argmax(test_y, axis=1) == predicted_class)
                acc = correct.mean() * 100
                correct_numbers = correct.sum()
                print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
                if acc > old_acc:
                    no_improvement_count = 0
                    old_acc = acc
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= 15:
                        status = 2#No improvement
                        break
                if(epoch_sum >= 14400):
                    status = 0 #0 for timeout
                    break
                elif (acc >= 70):
                    status = 1 #1 for successful
                    break
                
                    
            end = time.time()
            duration = (end - start)
            #Writing to CSV  
            i = 0
            predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
            while i < len(test_x):
                j = min(i + _BATCH_SIZE, len(test_x))
                batch_xs = test_x[i:j, :]
                batch_ys = test_y[i:j, :]
                predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
                i = j
            correct = (np.argmax(test_y, axis=1) == predicted_class)
            acc = correct.mean() * 100
            correct_numbers = correct.sum()
            print()
            print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
            fields=[learningrate, acc, duration, status]
            with open(r'/output/output_data.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        sess.close()
            
        
        
    
if __name__ == "__main__":
    main()
    

