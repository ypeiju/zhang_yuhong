############################################
###  Filename: LSTM_PennTreeBank.py 
###  Function: TensorFlow实现 LSTM_PennTreeBank.py
### Reference：https://www.tensorflow.org/tutorials/recurrent
###     Paper：https://arxiv.org/abs/1409.2329
############################################
import numpy as np
import tensorflow as tf
import reader

DATA_PATH = 'simple-examples/data/'
VOCAB_SIZE = 10000

HIDDEN_SIZE = 200
NUM_LAYERS = 2
LEARNING_RATE = 1.0
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):

        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE,state_is_tuple = True)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS, state_is_tuple = True)

        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)
        #定义输出层
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        
        #定义softmax层
        softmax_weight = tf.get_variable("softmax_w", [HIDDEN_SIZE, VOCAB_SIZE])
        softmax_bias = tf.get_variable("softmax_b", [VOCAB_SIZE])
        logits = tf.matmul(output, softmax_weight) + softmax_bias
        #定义损失函数
        #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training: return
        trainable_variables = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for step in range(epoch_size):
        x, y = session.run(data)
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)
	
def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("language_model", reuse = None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse = True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = session, coord = coord)

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)
		
if __name__ == "__main__":
    main()
