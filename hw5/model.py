import numpy as np
import tensorflow as tf

def clipped_error(x):
    # Huber loss
    try:
      return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
      return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class nn_model:
    def __init__(self, dim, a_list, name, lambda_, lrate, load=False):
        self.lrate = lrate
        self.lambda_ = lambda_
        self._name = name
        self._path = 'trained_agents/'
        self.dim = dim
        self.a_list = a_list
        self.w = {}
        self.w_t = {}
        self.assign_op = {}
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        try:
            self._build()
        except Exception as e:
            print('Model Initialization Fails! Because {}'.format(e))
        else:
            print('Model Generated!')
        if load == True:
            try:
                self.saver.restore(self._sess, self._path + name)
            except:
                print('Restoring Variables Fails!')
            else:
                print('Model Restored!')
        else:
            self._sess.run(self.init_op)
            print('Model Initialized!')

    def get_loss(self, states, actions, targets):
        if hasattr(self, 'loss'):
            feed_dict = {
                'states:0':states,
                'actions:0':actions,
                'targets:0':targets,
            }
            return self._sess.run(self.loss, feed_dict=feed_dict)
        else:
            print('loss has not been calculated.')

    def __del__(self):
        self._sess.close()

    def _build(self):
        with self._graph.as_default():
            global_step = tf.Variable(0, trainable=False, name='global')
            s = tf.placeholder(dtype=tf.float32, shape=[None, self.dim], name='states')
            a = tf.placeholder(dtype=tf.uint8, shape=[None], name='actions')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='targets')
            onehot = tf.one_hot(a, depth=len(self.a_list))
            output_dim = len(self.a_list)
            with tf.variable_scope('train'):
                self.w['l1w'] = l1w = tf.Variable(
                    tf.glorot_uniform_initializer()((self.dim,64)))
                self.w['l1b'] = l1b = tf.Variable(
                    tf.constant(0., shape=[64]))
                L1 = tf.nn.relu(tf.matmul(s, l1w) + l1b)
                # self.w['l2w'] = l2w = tf.Variable(
                #     tf.glorot_uniform_initializer()((10,10)))
                # self.w['l2b'] = l2b = tf.Variable(
                #     tf.constant(0., shape=[10]))
                # L2 = tf.nn.relu(tf.matmul(L1, l2w) + l2b)
                # self.w['l3w'] = l3w = tf.Variable(
                #     tf.glorot_uniform_initializer()((10,10)))
                # self.w['l3b'] = l3b = tf.Variable(
                #     tf.constant(0., shape=[10]))
                # L3 = tf.nn.relu(tf.matmul(L2, l3w) + l3b)
                self.w['l4w'] = l4w = tf.Variable(
                    tf.glorot_uniform_initializer()((64,output_dim)))
                self.pred = tf.matmul(L1,l4w)
                q_act = tf.reduce_mean(self.pred * onehot, reduction_indices=1)
                self.loss = tf.reduce_mean(
                    (clipped_error(q_act  - tf.reshape(y,[-1,1])))**2) \
                    # + self.lambda_ * (tf.norm(l1w) + tf.norm(l4w))
            with tf.variable_scope('target'):
                self.w_t['l1w'] = l1w = tf.Variable(
                    tf.constant(0., shape=[self.dim,64]))
                self.w_t['l1b'] = l1b = tf.Variable(
                    tf.constant(0., shape=[64]))
                L1 = tf.nn.relu(tf.matmul(s, l1w) + l1b)
                # self.w_t['l2w'] = l2w = tf.Variable(
                #     tf.constant(0., shape=[10,10]))
                # self.w_t['l2b'] = l2b = tf.Variable(
                #     tf.constant(0., shape=[10]))
                # L2 = tf.nn.relu(tf.matmul(L1, l2w) + l2b)
                # self.w_t['l3w'] = l3w = tf.Variable(
                #     tf.constant(0., shape=[10,10]))
                # self.w_t['l3b'] = l3b = tf.Variable(
                #     tf.constant(0., shape=[10]))
                # L3 = tf.nn.relu(tf.matmul(L2, l3w) + l3b)
                self.w_t['l4w'] = l4w = tf.Variable(
                    tf.constant(0., shape=[64,output_dim]))
                self.pred_t = tf.matmul(L1, l4w)
            with tf.variable_scope('optimize'):
                optimizer = tf.train.AdamOptimizer(learning_rate=
                    self.lrate)
                gvs = optimizer.compute_gradients(self.loss, self.w)
                capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                self.train_op = optimizer.apply_gradients(capped_gvs)
                # optimizer = tf.train.GradientDescentOptimizer(
                #     learning_rate=self.lrate
                # )
                # optimizer = tf.train.RMSPropOptimizer(learning_rate = self.lrate)
                # self.train_op = optimizer.minimize(loss=self.loss,
                #     global_step=global_step)
                self.init_op = tf.global_variables_initializer()
            with tf.variable_scope('assign'):
                for key in self.w.keys():
                    self.assign_op[key] = tf.assign(self.w_t[key], self.w[key])
            self.saver = tf.train.Saver()
        self._graph.finalize()

    def fit(self, states, actions, targets):
        with self._graph.as_default():
            feed_dict = {
                'states:0':states,
                'actions:0':actions,
                'targets:0':targets,
            }
            self._sess.run(self.train_op, feed_dict)

    def update(self):
        for key in self.w.keys():
            self._sess.run(self.assign_op[key])

    # evaluate the trained network
    def Q(self, state):
        feed_dict = {
            'states:0': np.array(state).reshape((1,-1)),
        }
        return self._sess.run(self.pred, feed_dict)[0]

    # evaluate the target network
    def Qhat(self, state):
        feed_dict = {
            'states:0': np.array(state).reshape((1,-1)),
        }
        return self._sess.run(self.pred_t, feed_dict)[0]

    def save(self):
        self.saver.save(self._sess, self._path + self._name)
