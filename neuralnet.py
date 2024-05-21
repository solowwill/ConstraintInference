# Neural net class for Proactive Feedback agent

import numpy as np

class NeuralNet:

  def __init__(self, input_dim, output_dim=1, hidden_dim=100, num_layers=4):
 
    if num_layers == 1:
      self.layers = [LinearLayer(input_dim, output_dim), ReLU()]
    else:

      self.layers = [LinearLayer(input_dim,hidden_dim), ReLU()]
      for _ in range (num_layers-2):
        self.layers.append(LinearLayer(hidden_dim, hidden_dim))
        self.layers.append(ReLU())
        
      self.layers.append(LinearLayer(hidden_dim,output_dim))
      self.layers.append(ReLU())


  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X


  def backward(self, grad):
    for layer in reversed(self.layers):
      grad = layer.backward(grad)


  def step(self, step_size, momentum, weight_decay):
    for layer in self.layers:
      layer.step(step_size, momentum, weight_decay)

class LinearLayer:

  def __init__(self, input_dim, output_dim):
    self.W = np.random.randn(input_dim,output_dim) * np.sqrt(2 / input_dim)

    self.b = np.zeros((1,output_dim)) +.01

    self.prev_w_update = np.zeros((input_dim,output_dim))
    self.prev_b_update = np.zeros((1,output_dim))


  def forward(self, input):

    self.input = input

    return input @ self.W + self.b

  def backward(self, grad):
    
    self.grad_weights = self.input.T @ grad
  
    self.grad_bias = np.sum(grad, axis=0, keepdims=True)

    return grad @ self.W.T


  def step(self, step_size, momentum = 0.8, weight_decay = 0.0):

    self.prev_w_update = momentum * self.prev_w_update + (1-momentum) * self.grad_weights
    self.prev_b_update = momentum * self.prev_b_update + (1-momentum) * self.grad_bias


    self.W -= step_size * self.prev_w_update - weight_decay * self.W
    self.b -= step_size * self.prev_b_update - weight_decay * self.b

class SigmoidBinaryEntropy:
  
  def forward(self, logits, labels):

    self.logits = logits
    self.labels = labels

    sig_pred = 1 / (1 + np.exp(-logits))

    loss = (1/logits.shape[0]) * np.sum(-(labels * np.log(sig_pred) + (1-labels) * np.log(1-sig_pred)))

    pred_labels = (sig_pred > .5) 

    tn = np.sum(self.labels == pred_labels)
    fn = np.sum(self.labels != pred_labels)
    acc = tn / (tn+fn)

    return loss, acc

  def backward(self):

    sig_pred = 1 / (1 + np.exp(-self.logits))

    grad = (sig_pred - self.labels) * self.logits

    return grad

  def predict(self, logits):

    sig_pred = 1 / (1 + np.exp(-logits))
    pred_labels = (sig_pred > .5) 

    return pred_labels
  
class NegSigmoidSoftMax:

  def forward(self, logits, labels):
    self.logits = logits
    self.labels = labels

    sig_pred = 1 / (1 + np.exp(-logits))

    loss = (1/logits.shape[0]) * np.sum(-(labels * np.log(sig_pred) + (1-labels) * np.log(1-sig_pred)))

    pred_labels = (sig_pred > .5) 

    tn = np.sum(self.labels == pred_labels)
    fn = np.sum(self.labels != pred_labels)
    acc = tn / (tn+fn)

    return loss, acc

class ReLU:

  def forward(self, input):

    self.input = input 

    return np.maximum(0, input)

  def backward(self, grad):
  
    return (self.input > 0) * grad
  

  def step(self,step_size, momentum = 0, weight_decay = 0):
    return


# TODO: Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, loss_func, X_val, Y_val):
  input = X_val

  # Compute the logits 
  for layer in model.layers:
    input = layer.forward(input)

  cross_ent, acc = loss_func.forward(input, Y_val)

  return cross_ent, acc

class TrainNet():

  def __init__(self, net, loss_func, max_epochs=5, batch_size=64, step_size=.0001, momentum=.8, weight_decay=0):
    self.net = net
    self.loss_func = loss_func
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.step_size = step_size
    self.momentum = momentum
    self.weight_decay = weight_decay

  def train(self, X_train, Y_train):

    for ep in range(self.max_epochs):
      # Scramble order of examples
      shuffle_inds = np.arange(X_train.shape[0])
      np.random.shuffle(shuffle_inds)

      X_train = X_train[shuffle_inds]
      Y_train = Y_train[shuffle_inds]

      batch_loss = []
      batch_acc = []

      # for each batch in data:
      for i in range(int(X_train.shape[0] / self.batch_size)):

        # Gather batch
        batch = X_train[i*self.batch_size:(i+1)*self.batch_size]
        batch_labels = Y_train[i*self.batch_size:(i+1)*self.batch_size]

        # Compute forward pass
        output = self.net.forward(batch)

        # Compute loss
        loss, acc = self.loss_func.forward(output, batch_labels)

        # Store for epoch output
        batch_loss.append(loss)
        batch_acc.append(acc)
        
        # Backward loss and networks
        error = self.loss_func.backward()
        self.net.backward(error)

        # Take optimizer step
        self.net.step(self.step_size, self.momentum, self.weight_decay)



  

