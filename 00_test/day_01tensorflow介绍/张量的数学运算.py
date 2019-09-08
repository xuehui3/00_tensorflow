import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

a = tf.fill([2, 2], 2.)
sess=tf.Session()
with sess.as_default():
    print(a.eval())
    print(type(a))

