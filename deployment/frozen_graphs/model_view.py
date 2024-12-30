import tensorflow as tf
from tensorflow.python.platform import gfile
LOGDIR='content/result.jpe'
train_writer = tf.summary.FileWriter(LOGDIR)
with tf.Session() as sess:
    model_filename ='classificator_full_model.pb'
    with tf.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        train_writer.flush()
        train_writer.close()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
train_writer.add_graph(sess.graph)
