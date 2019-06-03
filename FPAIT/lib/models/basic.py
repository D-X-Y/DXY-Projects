from .omniglotnet import omniglotnet
from .cnn_lstm import CNN_RNN
from .cnn_lstm import CNN_SIMPLE
from .cnn_fait import CNN_FAIT
from .cnn_tcn import CNN_TCN

def obtain_model(option):
  if option.arch == 'omniglotnet':
    net = omniglotnet(option.num_classes)
  elif option.arch == 'cnn_lstm':
    net = CNN_RNN   (option.vocab_size+1, option.answer_class, 'LSTM')
  elif option.arch == 'cnn_gru':
    net = CNN_RNN   (option.vocab_size+1, option.answer_class, 'GRU')
  elif option.arch == 'cnn_vanilla':
    net = CNN_RNN   (option.vocab_size+1, option.answer_class, 'Vanilla')
  elif option.arch == 'cnn_simple':
    net = CNN_SIMPLE(option.vocab_size+1, option.answer_class)
  elif option.arch == 'cnn_tcn':
    net = CNN_TCN(option.vocab_size+1, option.answer_class)
  elif option.arch == 'cnn_fait':
    net = CNN_FAIT(option.vocab_size+1, option.answer_class)
  else:
    raise TypeError ('Does not find {:}'.format(option.arch))

  if option.use_cuda:
    net = net.cuda()
  return net

def obtain_fait_model(option):
  if option.arch == 'cnn_lstm':
    net = CNN_RNN (option.vocab_size+1, option.num_classes, 'LSTM')
  elif option.arch == 'cnn_gru':
    net = CNN_RNN (option.vocab_size+1, option.num_classes, 'GRU')
  elif option.arch == 'cnn_vanilla':
    net = CNN_RNN (option.vocab_size+1, option.num_classes, 'Vanilla')
  elif option.arch == 'cnn_tcn':
    net = CNN_TCN (option.vocab_size+1, option.num_classes)
  elif option.arch == 'cnn_fait':
    net = CNN_FAIT(option.vocab_size+1, option.num_classes)
  else:
    raise TypeError ('Does not find {:}'.format(option.arch))

  if option.use_cuda:
    net = net.cuda()
  return net
