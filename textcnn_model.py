import tensorflow as tf
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from tqdm import tqdm
import json
import numpy as np
from sklearn import metrics
from copy import deepcopy

CONTINUE_TRAIN = False
MAX_SEQLEN = 7
NUM_WORDS = 10000
MAX_TEXTLEN = 64
EMBEDDING_SIZE = 200
FILTERS = 100
BATCH_SIZE = 256
EPOCHS = 20
INIT_LEARNING_RATE = 5e-3
MIN_LEARNING_RATE = 1e-5
DECAY_STEPS = 100
DECAY_RATE = 0.9
ATT_SIZE = 200
MODEL_PATH = 'model/'
DROPOUT_RATE = 0.0
DISPLAY_ITER = 1

relationships = {'夫妻': 0, '恋情': 1, '前任': 2, '好友': 3, '搭档': 4, '血缘关系': 5, '其他': 6}
reversed_relationships = {0: '夫妻', 1: '恋情', 2: '前任', 3: '好友', 4: '搭档', 5: '血缘关系', 6: '其他'}
class GenerateData():
    def __init__(self, rpath, tokenizer=None):
        self.rpath = rpath
        self.tokenizer = tokenizer

        self.read_data()
        self.preprocessing()
        self.shuffle()
        # initialize batch id
        self.batch_id = 0

    def read_data(self):
        self.text_seq = []
        self.people = []
        self.labels = []
        self.mask = []
        with open(self.rpath, 'r') as rf:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                content = json.loads(line.strip())
                if content['label'] in ['绯闻', '情侣']:
                    content['label'] = '恋情'
                if content['label'] in ['母子', '母女', '父女', '父子', '兄弟', '姐妹', '兄妹', '姐弟']:
                    content['label'] = '血缘关系'
                #if content['label'] in ['好友', '搭档']:
                #    content['label'] = '好友'
                #if content['label'] in ['兄弟', '姐妹', '兄妹', '姐弟']:
                #    content['label'] = '兄弟姐妹'
                elif content['label'] not in relationships:
                    #continue
                    content['label'] = '其他'
                text_seq = content["sentences"][:MAX_SEQLEN]
                self.mask.append([True] * len(text_seq) + [False] * (MAX_SEQLEN - len(text_seq)))
                if len(text_seq) < MAX_SEQLEN:
                    text_seq.extend([['NaN'] * MAX_TEXTLEN for _ in range(MAX_SEQLEN - len(text_seq))])
                self.text_seq.append(text_seq)
                self.people.append(content['people'])
                self.labels.append(relationships[content['label']])

    def preprocessing(self):
        text_reshape = np.reshape(self.text_seq, (-1,))
        # initialize tokenizer
        if not self.tokenizer:
            self.tokenizer = Tokenizer(num_words=NUM_WORDS)
            self.tokenizer.fit_on_texts(text_reshape)
        text_seq = self.tokenizer.texts_to_sequences(text_reshape)
        # padding text
        text_seq_padding = sequence.pad_sequences(text_seq, maxlen=MAX_TEXTLEN)
        self.text_seq = text_seq_padding.reshape((-1, MAX_SEQLEN, MAX_TEXTLEN))

        # get one-hot label
        self.labels = np.eye(len(relationships))[self.labels]


    #def preprocessing(self):
    #    text_seq_reshape = np.reshape(self.text_seq, (-1,))
    #    print(np.shape(text_seq_reshape))
    #    # padding text
    #    text_seq_padding = sequence.pad_sequences(text_seq_reshape, maxlen=MAX_TEXTLEN)
    #    self.text_seq = text_seq_padding.reshape((-1, MAX_SEQLEN, MAX_TEXTLEN))

        # get one-hot label
    #    self.labels = np.eye(len(relationships))[self.labels]


    def next(self):
        if BATCH_SIZE <= len(self.text_seq) - self.batch_id:
            batch_text_seq = deepcopy(self.text_seq[self.batch_id:(self.batch_id + BATCH_SIZE)])
            batch_people = deepcopy(self.people[self.batch_id:(self.batch_id + BATCH_SIZE)])
            batch_mask = deepcopy(self.mask[self.batch_id:(self.batch_id + BATCH_SIZE)])
            batch_labels = deepcopy(self.labels[self.batch_id:(self.batch_id + BATCH_SIZE)])
            self.batch_id = self.batch_id + BATCH_SIZE
        else:
            batch_text_seq = deepcopy(self.text_seq[self.batch_id:])
            batch_people = deepcopy(self.people[self.batch_id:])
            batch_mask = deepcopy(self.mask[self.batch_id:])
            batch_labels = deepcopy(self.labels[self.batch_id:])

            # shuffle
            self.shuffle()
            # reset batch id
            self.batch_id = 0

        #for i, p in enumerate(batch_people):
        #    print(p, reversed_relationships[np.argmax(batch_labels[i])])

        return batch_text_seq, batch_people, batch_mask, batch_labels

    def shuffle(self):
        np.random.seed(1117)
        np.random.shuffle(self.text_seq)
        np.random.seed(1117)
        np.random.shuffle(self.people)
        np.random.seed(1117)
        np.random.shuffle(self.mask)
        np.random.seed(1117)
        np.random.shuffle(self.labels)

class TextCNN():
    def __init__(self):
        self.build_inputs()
        self.build_model()
        self.build_loss()
        self.build_optimizer()

    def build_inputs(self):
        self.text_seq = tf.placeholder(tf.int32, [None, MAX_SEQLEN, MAX_TEXTLEN])
        self.mask = tf.placeholder(tf.int32, [None, MAX_SEQLEN])
        self.targets = tf.placeholder(tf.float32, (None, len(relationships)))
        self.dropout_rate = tf.placeholder(tf.float32)

    def attention(self, inputs, mask, return_alphas=False):
        # hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
        # Trainable parameters
        w_omega = tf.Variable(tf.variance_scaling_initializer(scale=1., mode='fan_in')((int(inputs.shape[-1]), ATT_SIZE)))
        b_omega = tf.Variable(tf.zeros([ATT_SIZE]))
        u_omega = tf.Variable(tf.variance_scaling_initializer(scale=1., mode='fan_in')((ATT_SIZE,)))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1)  # (B,T) shape
        # padding部分减去一个很大的整数，使其softmax后接近于0
        vu = tf.where(tf.equal(mask, True), vu, vu - float('inf'))

        alphas = tf.nn.softmax(vu)  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def build_model(self):
        text_seq_reshape = tf.reshape(self.text_seq, (-1, MAX_TEXTLEN))
        emb_text = tf.keras.layers.Embedding(NUM_WORDS, EMBEDDING_SIZE)(text_seq_reshape)
        #emb_text = tf.keras.layers.Embedding(
        #    NUM_WORDS+1, EMBEDDING_SIZE,
        #    embeddings_initializer=tf.keras.initializers.constant(embeddings_matrix),
        #    trainable=True
        #)(text_seq_reshape)
        # conv layers
        convs = []
        filter_sizes = [2, 3, 4, 5]
        for fsz in filter_sizes:
            l_conv = tf.keras.layers.Conv1D(filters=FILTERS, kernel_size=fsz, activation='relu')(emb_text)
            l_pool = tf.keras.layers.MaxPooling1D(MAX_TEXTLEN - fsz + 1)(l_conv)
            l_pool_flatten = tf.reshape(l_pool, (-1, FILTERS))
            convs.append(l_pool_flatten)
        merge = tf.keras.layers.concatenate(convs, axis=1)
        merge_seq = tf.reshape(merge, (-1, MAX_SEQLEN, len(filter_sizes) * FILTERS))
        # attention
        hidden, alpha = self.attention(merge_seq, self.mask, return_alphas=True)
        hidden = tf.keras.layers.Dense(units=32, activation='relu')(hidden)
        # dropout
        drop_hidden = tf.nn.dropout(hidden, rate=self.dropout_rate)
        #drop_hidden = tf.concat([drop_hidden, self.targets], axis=1)
        self.logits = tf.keras.layers.Dense(units=len(relationships), activation=None)(drop_hidden)
        self.probs = tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.probs, 1)

    def build_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.targets)
        self.loss = tf.reduce_mean(cross_entropy)

    def build_optimizer(self):
        self.current_epoch = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(
            INIT_LEARNING_RATE,
            self.current_epoch,
            decay_steps=DECAY_STEPS,
            decay_rate=DECAY_RATE
        )
        self.learning_rate = tf.maximum(self.learning_rate, MIN_LEARNING_RATE)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.current_epoch)

    def initialize_variables(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        K.set_session(sess)

    def train(self, trainset, testset):
        with tf.Session() as sess:
            if CONTINUE_TRAIN:
                try:
                    saver = tf.train.Saver()
                    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
                    print('Successfully load model!')
                except:
                    print('Fail to load model!')
                    self.initialize_variables(sess)
            else:
                self.initialize_variables(sess)
            step = 1

            saver = tf.train.Saver()
            while step <= EPOCHS:
                self.run_one_epoch(sess, step, trainset, is_training=True)
                saver.save(sess, MODEL_PATH + 'model')
                print('----- Validation -----')
                self.run_one_epoch(sess, step, testset, is_training=False)
                print('---------------------')
                step += 1

            print("Optimization Finished!")

    def run_one_epoch(self, sess, step, dataset, is_training=False):
        iter = 0
        predictions = []
        ground_truth = []
        people = []
        while iter * BATCH_SIZE < len(dataset.text_seq):
            iter += 1
            self.current_epoch = step
            batch_text_seq, batch_people, batch_mask, batch_labels = dataset.next()
            feed = {
                self.text_seq: batch_text_seq,
                self.mask: batch_mask,
                self.targets: batch_labels
            }

            if is_training:
                feed[self.dropout_rate] = DROPOUT_RATE
                sess.run(self.optimizer, feed_dict=feed)
            else:
                feed[self.dropout_rate] = 0.0

            y_pred, lr, loss = sess.run([self.predictions, self.learning_rate, self.loss], feed_dict=feed)
            y_true = np.argmax(batch_labels, 1)
            if iter % DISPLAY_ITER == 0:
                print(
                    "Epoch " + str(step) + ", Iter " + str(iter) +
                    ", Learning Rate = " + "{:.5f}".format(lr) +
                    ", Minibatch Loss = " + "{:.5f}".format(loss) +
                    ", Accuracy = " + "{:.5f}".format(metrics.accuracy_score(y_true, y_pred)) +
                    ", Precision = " + "{:.5f}".format(metrics.precision_score(y_true, y_pred, average='weighted')) +
                    ", Recall = " + "{:.5f}".format(metrics.recall_score(y_true, y_pred, average='weighted')) +
                    ", f1 score = " + "{:.5f}".format(metrics.f1_score(y_true, y_pred, average='weighted'))
                )

            people.extend(batch_people)
            predictions.extend(list(y_pred))
            ground_truth.extend(list(y_true))

        with open(str(is_training)+'result.txt', 'w') as wf:
             for i, pred in enumerate(predictions):
                 wf.write(json.dumps({
                    'pred': reversed_relationships[pred],
                    'truth': reversed_relationships[ground_truth[i]],
                    'people': people[i]
                 }) + '\n')
        #    if pred != ground_truth[i]:
        #        print('pred:', reversed_relationships[pred], 'truth:', reversed_relationships[ground_truth[i]], 'people:', people[i])

        print(metrics.classification_report(ground_truth, predictions, digits=4))

    def test(self, dataset):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
            self.run_one_epoch(sess, 0, dataset, is_training=False)

if __name__ == '__main__':
    trainset = GenerateData('train_data.json')
    testset = GenerateData('test_data.json', trainset.tokenizer)
    #embeddings_matrix = get_word2vec_dictionaries(trainset.tokenizer)
    model = TextCNN()
    model.train(trainset, testset)
