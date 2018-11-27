import numpy as np
import tensorflow as tf

class nn_model:
    def __init__(self, dim, a_list):
        self.dim = dim
        self.a_list = a_list
        self.w = {}
        self.w_t = {}
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        self._build()

    def __del__(self):
        self._sess.close()

    def _build(self):
        with self._graph.as_default():
            global_step = tf.Variable(0, trainable=False, name='global')
            s = tf.placeholder(dtype=tf.float32, shape=[None, self.dim], name='states')
            a = tf.placeholder(dtype=tf.uint8, shape=[None], name='actions')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='targets')
            lrate = tf.placeholder(dtype=tf.float32, shape=[], name='lrate')
            onehot = tf.one_hot(a, depth=len(self.a_list))
            n_features = self.dim + len(self.a_list)
            x = tf.concat([s, onehot],1)
            with tf.variable_scope('train'):
                self.w['l1w'] = l1w = tf.Variable(
                    tf.truncated_normal(shape=[n_features,10]))
                self.w['l1b'] = l1b = tf.Variable(
                    tf.truncated_normal(shape=[10]))
                L1 = tf.nn.relu(tf.matmul(x, l1w) + l1b)
                self.w['l2w'] = l2w = tf.Variable(
                    tf.truncated_normal(shape=[10,10]))
                self.w['l2b'] = l2b = tf.Variable(
                    tf.truncated_normal(shape=[10]))
                L2 = tf.nn.relu(tf.matmul(L1, l2w) + l2b)
                self.w['l3w'] = l3w = tf.Variable(
                    tf.truncated_normal(shape=[10,10]))
                self.w['l3b'] = l3b = tf.Variable(
                    tf.truncated_normal(shape=[10]))
                L3 = tf.nn.relu(tf.matmul(L2, l3w) + l3b)
                self.w['l4w'] = l4w = tf.Variable(
                    tf.truncated_normal(shape=[10,1]))
                self.pred = tf.matmul(L3,l4w)
                self.loss = tf.reduce_mean((self.pred - tf.reshape(y,[-1,1]))**2)
            with tf.variable_scope('target'):
                self.w_t['l1w'] = l1w = tf.Variable(
                    tf.constant(0., shape=[n_features,10]))
                self.w_t['l1b'] = l1b = tf.Variable(
                    tf.constant(0., shape=[10]))
                L1 = tf.nn.relu(tf.matmul(x, l1w) + l1b)
                self.w_t['l2w'] = l2w = tf.Variable(
                    tf.constant(0., shape=[10,10]))
                self.w_t['l2b'] = l2b = tf.Variable(
                    tf.constant(0., shape=[10]))
                L2 = tf.nn.relu(tf.matmul(L1, l2w) + l2b)
                self.w_t['l3w'] = l3w = tf.Variable(
                    tf.constant(0., shape=[10,10]))
                self.w_t['l3b'] = l3b = tf.Variable(
                    tf.constant(0., shape=[10]))
                L3 = tf.nn.relu(tf.matmul(L2, l3w) + l3b)
                self.w_t['l4w'] = l4w = tf.Variable(
                    tf.constant(0., shape=[10,1]))
                self.pred_t = tf.matmul(L3, l4w)
            with tf.variable_scope('optimize'):
                optimizer = tf.train.AdamOptimizer(learning_rate=
                    lrate)
                self.train_op = optimizer.minimize(loss=self.loss,
                    global_step=global_step)
                init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def fit(self, states, actions, targets, lrate):
        with self._graph.as_default():
            feed_dict = {
                'states:0':states,
                'actions:0':actions,
                'targets:0':targets,
                'lrate:0':lrate
            }
            self._sess.run(self.train_op, feed_dict)

    def update(self):
        for key in self.w.keys():
            self.w_t[key].assign(self.w[key])

    # evaluate the trained network
    def Q(self, state, action):
        feed_dict = {
            'states:0': np.array(state).reshape((1,-1)),
            'actions:0': [action]
        }

        return self._sess.run(self.pred, feed_dict)

    # evaluate the target network
    def Qhat(self, state, action):
        feed_dict = {
            'states:0': np.array(state).reshape((1,-1)),
            'actions:0': [action]
        }
        return self._sess.run(self.pred, feed_dict)
