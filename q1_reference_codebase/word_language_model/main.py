# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import pandas as pd

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='FNN',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer or FNN)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--ngram_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', default=False,
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='models/model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')

args = parser.parse_args()


"""Initialise dataframe to store validation results and plotting in jupyter notebook"""
result_df = pd.DataFrame()

"""Set saved_filename for model file based on model and whether tied option is enabled"""
if args.tied:
    saved_filename = args.save[:-3] + "-" + args.model + "-tied" + args.save[-3:]
else:
    saved_filename = args.save[:-3] + "-" + args.model + "-not_tied" + args.save[-3:]

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.


"""Adapted batchify function

    Should the model be FNN, the sliding window mechanism is implemented. For 8 ngram,
    the sliding window (bsz) is set to 8, and subsequently will be split into 7 word inputs 
    and 1 target word in the get_batch function below. If the model is Transformer or RNN
    architecture, the original batchify function will be used for preprocessing the data.    
"""
def batchify(data, bsz):
    if args.model == "FNN":
        # Implement sliding window to generate data in sizes of bsz
        data = [np.array(data[i:i+bsz]) for i in range(data.shape[0] - bsz + 1)]
        data = torch.Tensor(data).to(torch.int64)
        return data.to(device)
    else:
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)     

eval_ngram_size = args.ngram_size
train_data = batchify(corpus.train, args.ngram_size)
val_data = batchify(corpus.valid, eval_ngram_size)
test_data = batchify(corpus.test, eval_ngram_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
elif args.model == "FNN":
    model = model.FNNModel(ntokens, args.emsize, args.nhid, args.ngram_size, args.dropout, args.tied).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.batch_size.
# If source is equal to the example output of the batchify function, with
# a batch_size-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

"""Adapted get_batch function

    For FNN model, the data is selected to all the words except the words in the
    last column. The target will be the last column of words. For transformer or
    RNN based models, the original get_batch function will be used for the data
    preprocessing
"""
def get_batch(source, i):
    seq_len = min(args.batch_size, len(source) - 1 - i)
    if args.model == "FNN":
        # Generate batches based on sliding window
        data = source[i:i+seq_len]
        target = data[:, -1]
        data = data[:, :-1]
        return data, target
    else:
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target  


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer' and args.model != 'FNN':
        hidden = model.init_hidden(eval_ngram_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.batch_size):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            elif args.model == "FNN":
                output = model(data)
                output = output.view(-1, ntokens)                
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # initialise Adam optimizer with configured learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer' and args.model != 'FNN':
        hidden = model.init_hidden(args.ngram_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.batch_size)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        elif args.model == "FNN":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        optimizer.zero_grad()        
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, ngram_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * ngram_size).zero_().view(-1, ngram_size).to(device)
    hidden = model.init_hidden(ngram_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                print("==== SAVING BEST MODEL",saved_filename,"====")
                torch.save(model, saved_filename)
            best_val_loss = val_loss

        # Append results to pandas DataFrame for analysis of validation results 
        result_df = result_df.append({
            "epoch": epoch, 
            "perpexity": math.exp(val_loss),
            "val_loss": val_loss, 
        }, ignore_index=True)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Save results from results_df into csv for further analysis
csv_filename = "logs" + saved_filename[6:-3] + '.csv'
result_df.to_csv(csv_filename)

# Load the best saved model.
with open(saved_filename, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, ngram_size=1, seq_len=args.batch_size)
