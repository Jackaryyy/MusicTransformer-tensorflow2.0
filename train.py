from model import MusicTransformer
from custom.layers import *
from custom import callback
import params as par
from tensorflow.python.keras.optimizer_v2.adam import Adam
from data import Data
import utils
import argparse
import sys

tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=0.0001, help='learning rate')
parser.add_argument('--batch_size', default=2, help='batch size')
parser.add_argument('--pickle_dir', default='music', help='path to dataset')
parser.add_argument('--max_seq', default=2048, help='max sequence length')
parser.add_argument('--epochs', default=100, help='training epochs')
parser.add_argument('--load_path', default=None, help='model load path', type=str)
parser.add_argument('--save_path', default="result", help='model save path')
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=False)

args = parser.parse_args()


# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu


# load data
dataset = Data('dataset/processed')
print(dataset)


# load model
learning_rate = callback.CustomSchedule(par.embedding_dim)
opt = Adam(l_r, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# define model
mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=2,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)


# Train Start
for e in range(epochs):
    mt.reset_metrics()
    for b in range(len(dataset.files) // batch_size):
        try:
                batch_x, batch_y = dataset.seq2seq_batch(batch_size, max_seq)
        except:
                continue
        result_metrics = mt.train_on_batch(batch_x, batch_y)
        if b % 100 == 0:
              eval_x, eval_y = dataset.seq2seq_batch(batch_size, max_seq, 'eval')
              eval_result_metrics = mt.evaluate(eval_x, eval_y)
              mt.save(save_path)
              print('\n====================================================')
              print('Epoch/Batch: {}/{}'.format(e, b))
              print('Train >>>> Loss: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1]))
              print('Eval >>>> Loss: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1]))

