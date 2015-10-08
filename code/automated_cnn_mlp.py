import os
import sys
import timeit
import pylab
from PIL import Image

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from cnn_tools import HiddenLayer, _dropout_from_layer, DropoutHiddenLayer, LeNetConvPoolLayer, DropoutLeNetConvPoolLayer

class cnn_mlp(object):
  """A Convolutional Neural Network followed by a MultiLayer Perceptron,
  with all trappings required to do dropout.
  """

  def __init__(self,
               rng,
               input,
               filter_shapes,
               image_shape,
               poolsize,
               layer_sizes,
               dropout_rates,
               activations):

    """
    Allocate a cnn_mlp (ConvNet followed by MLP) with shared variable internal parameters.

    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type input: theano.tensor.dtensor4
    :param input: symbolic image tensor, of shape image_shape

    :type filter_shapes:  list of (list of length 4)
    :param filter_shapes: list of the filters whith their respective properties ((number of kernels, num input feature maps,
                                                                                  filter height, filter width), ...)
                          len(filter_shapes) = number of LeNetConvPoolLayer layers

    :type image_shape: tuple or list of length 4
    :param image_shape: (batch size, num input feature maps,
                             image height, image width)

    :type poolsize: tuple or list of length 2
    :param poolsize: the downsampling (pooling) factor (#rows, #cols)

    :type layer_sizes: list of int
    :param layer_sizes: sizes (number of units) of each HiddenLayer (
                        len(layer_sizes) = number of HiddenLayer layers)

    :type dropout_rates: list of float
    :param dropout_rates: dropout rate used for each layer (including dropout on the input)

    :type activations: list of theano.function
    :param activations: list of the activation functions to use at each layer

    """


    #######################################
    # Set up all the convolutional layers #
    #######################################

    self.layers = []
    self.dropout_layers = []

    next_layer_input = input.reshape(image_shape)
    next_dropout_layer_input = _dropout_from_layer(rng, next_layer_input, p=dropout_rates[0])

    layer_counter = 0

    for i in range(len(filter_shapes)):

      filter_shape = filter_shapes[i]

      next_dropout_layer = DropoutLeNetConvPoolLayer(
        rng=rng,
        input=next_dropout_layer_input,
        image_shape=image_shape,
        filter_shape=filter_shape,
        poolsize=poolsize,
        dropout_rate=dropout_rates[layer_counter + 1],
        activation=activations[layer_counter]
        )

      self.dropout_layers.append(next_dropout_layer)
      next_dropout_layer_input = next_dropout_layer.output

      # Reuse parameters from the dropout layer here
      next_layer = LeNetConvPoolLayer(
        rng=rng,
        input=next_layer_input,
        image_shape=image_shape,
        filter_shape=filter_shape,
        W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
        b=next_dropout_layer.b,
        poolsize=poolsize,
        activation=activations[layer_counter]
        )

      self.layers.append(next_layer)
      next_layer_input = next_layer.output

      image_shape = (image_shape[0],
                     filter_shape[0],
                     (image_shape[2] - filter_shape[2] + 1) / poolsize[0],
                     (image_shape[3] - filter_shape[3] + 1) / poolsize[1])

      layer_counter += 1

    ################################
    # Set up all the hidden layers #
    ################################

    weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])

    next_layer_input = next_layer_input.flatten(2)
    next_dropout_layer_input = next_dropout_layer_input.flatten(2)

    assert (layer_sizes[0] == numpy.prod(image_shape[1:])), "The dimension of the first hidden layer does not match last convolutional layer size."

    for n_in, n_out in weight_matrix_sizes[:-1]:

      next_dropout_layer = DropoutHiddenLayer(
        rng=rng,
        input=next_dropout_layer_input,
        activation=activations[layer_counter],
        n_in=n_in,
        n_out=n_out,
        dropout_rate=dropout_rates[layer_counter + 1])

      self.dropout_layers.append(next_dropout_layer)
      next_dropout_layer_input = next_dropout_layer.output

      # Reuse the paramters from the dropout layer here
      next_layer = HiddenLayer(
        rng=rng,
        input=next_layer_input,
        activation=activations[layer_counter],
        # scale the weight matrix W with (1-p)
        W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
        b=next_dropout_layer.b,
        n_in=n_in,
        n_out=n_out)

      self.layers.append(next_layer)
      next_layer_input = next_layer.output

      layer_counter += 1

    ###########################
    # Set up the output layer #
    ###########################

    n_in, n_out = weight_matrix_sizes[-1]

    dropout_output_layer = LogisticRegression(
      input=next_dropout_layer_input,
      n_in=n_in,
      n_out=n_out)
    self.dropout_layers.append(dropout_output_layer)

    # Again, reuse paramters in the dropout output.
    output_layer = LogisticRegression(
      input=next_layer_input,
      # scale the weight matrix W with (1-p)
      W=dropout_output_layer.W * (1 - dropout_rates[-1]),
      b=dropout_output_layer.b,
      n_in=n_in,
      n_out=n_out)
    self.layers.append(output_layer)

    # Use the negative log likelihood of the logistic regression layer as
    # the objective.
    self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
    self.dropout_errors = self.dropout_layers[-1].errors

    self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
    self.errors = self.layers[-1].errors

    # Grab all the parameters together.
    self.params = [ param for layer in self.dropout_layers for param in layer.params ]


def test_cnn_mlp(
  initial_learning_rate=0.1,
  learning_rate_decay = 1,
  dropout_rates = [0.2, 0.2, 0.2, 0.5],
  n_epochs=200,
  dataset='mnist.pkl.gz',
  filter_shapes=((20,1,5,5),(50,20,5,5)),
  layer_sizes=[800, 500, 10],
  activations=[T.tanh, T.tanh, T.tanh],
  batch_size=500,
  random_seed=23455):

  # Need to add assertions here to assert:
    # 1. len(activations) = len(filter_shapes) + len(layer_sizes)

  ################
  # LOADING DATA #
  ################

  datasets = load_data(dataset)

  train_set_x, train_set_y = datasets[0]
  valid_set_x, valid_set_y = datasets[1]
  test_set_x, test_set_y = datasets[2]

  # compute number of minibatches for training, validation and testing
  n_train_batches = train_set_x.get_value(borrow=True).shape[0]
  n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
  n_test_batches = test_set_x.get_value(borrow=True).shape[0]
  n_train_batches /= batch_size
  n_valid_batches /= batch_size
  n_test_batches /= batch_size

  ######################
  # BUILD ACTUAL MODEL #
  ######################
  print '... building the model'

  # allocate symbolic variables for the data
  index = T.lscalar()  # index to a [mini]batch
  epoch = T.scalar()

  x = T.matrix('x')   # the data is presented as rasterized images
  y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

  rng = numpy.random.RandomState(random_seed)

  learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
                                              dtype=theano.config.floatX))

  # construct the cnn_mlp class
  classifier = cnn_mlp(
    rng=rng,
    input=x,
    filter_shapes=filter_shapes,
    image_shape=(batch_size, 1, 28, 28),
    poolsize=(2,2),
    layer_sizes= layer_sizes,
    dropout_rates=dropout_rates,
    activations=activations)

  # Build the expresson for the cost function.
  cost = classifier.negative_log_likelihood(y)
  dropout_cost = classifier.dropout_negative_log_likelihood(y)

  # create a function to compute the mistakes that are made by the model
  test_model = theano.function(
    [index],
    classifier.errors(y),
    givens={
      x: test_set_x[index * batch_size: (index + 1) * batch_size],
      y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
  )

  validate_model = theano.function(
    [index],
    classifier.errors(y),
    givens={
      x: valid_set_x[index * batch_size: (index + 1) * batch_size],
      y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
  )

  # create a list of all model parameters to be fit by gradient descent
  params = classifier.params

  # create a list of gradients for all model parameters
  grads = T.grad(dropout_cost, params)

  # train_model is a function that updates the model parameters by SGD
  updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
  ]

  train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
      x: train_set_x[index * batch_size: (index + 1) * batch_size],
      y: train_set_y[index * batch_size: (index + 1) * batch_size]
    }
  )

  # Theano function to decay the learning rate
  decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
                                        updates={learning_rate:
                                                 learning_rate * learning_rate_decay})

  ###############
  # TRAIN MODEL #
  ###############
  print '... training'
  # early-stopping parameters
  patience = 10000  # look as this many examples regardless
  patience_increase = 2  # wait this much longer when a new best is
                           # found
  improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
  validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

  best_validation_loss = numpy.inf
  best_iter = 0
  test_score = 0.
  start_time = timeit.default_timer()

  epoch = 0
  done_looping = False

  while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in xrange(n_train_batches):

      iter = (epoch - 1) * n_train_batches + minibatch_index

      if iter % 100 == 0:
        print 'training @ iter = ', iter
        cost_ij = train_model(minibatch_index)

      if (iter + 1) % validation_frequency == 0:

        # compute zero-one loss on validation set
        validation_losses = [validate_model(i) for i
                              in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                (epoch, minibatch_index + 1, n_train_batches,
                 this_validation_loss * 100.))

        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:

          #improve patience if loss improvement is good enough
          if this_validation_loss < best_validation_loss *  \
             improvement_threshold:
            patience = max(patience, iter * patience_increase)

          # save best validation score and iteration number
          best_validation_loss = this_validation_loss
          best_iter = iter

          # test it on the test set
          test_losses = [
            test_model(i)
            for i in xrange(n_test_batches)
          ]
          test_score = numpy.mean(test_losses)
          print(('     epoch %i, minibatch %i/%i, test error of '
                 'best model %f %%') %
                 (epoch, minibatch_index + 1, n_train_batches,
                  test_score * 100.))

      if patience <= iter:
        done_looping = True
        break

      new_learning_rate = decay_learning_rate()

  end_time = timeit.default_timer()
  print('Optimization complete.')
  print('Best validation score of %f %% obtained at iteration %i, '
        'with test performance %f %%' %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
  print >> sys.stderr, ('The code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_cnn_mlp()