from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LSTM, Dropout, concatenate, Flatten, Dense, Input, Lambda

import tensorflow

class Cascade:
    def __init__(self, window_size,conv1_filters,conv2_filters,conv3_filters,
                 conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                 padding1,padding2,padding3,conv1_activation,conv2_activation,
                 conv3_activation,dense_nodes,dense_activation,dense_dropout,
                 lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,depth,
                 final_dropout):
        
        self.number_classes = 4
        self.mesh_rows = 20
        self.mesh_columns = 21
        
        self.window_size = window_size        
        
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.conv3_filters = conv3_filters
        
        self.conv1_kernel_shape = conv1_kernel_shape
        self.conv2_kernel_shape = conv2_kernel_shape
        self.conv3_kernel_shape = conv3_kernel_shape
        
        self.padding1 = padding1
        self.padding2 = padding2
        self.padding3 = padding3
        
        self.conv1_activation = conv1_activation
        self.conv2_activation = conv2_activation
        self.conv3_activation = conv3_activation
        
        self.dense_nodes = dense_nodes
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        
        self.lstm1_cells = lstm1_cells
        self.lstm2_cells = lstm2_cells
        
        self.dense3_nodes = dense3_nodes
        self.dense3_activation = dense3_activation
        self.final_dropout = final_dropout
        
        self.depth = depth
        
        self.model = self.get_model()

    def get_model(self):
      inputs = []
      convs = []
      for i in range(self.window_size):
          input_layer = Input(shape=(self.mesh_rows, self.mesh_columns, self.depth), name = "input"+str(i+1))
          inputs.append(input_layer)

      for i in range(self.window_size):
          conv1 = Conv2D(self.conv1_filters, self.conv1_kernel_shape, padding = self.padding1, activation=self.conv1_activation, name = str(i+1)+"conv"+str(1))(inputs[i])
          
          conv2 = Conv2D(self.conv2_filters, self.conv2_kernel_shape, padding = self.padding2, activation=self.conv1_activation,name = str(i+1)+"conv"+str(2))(conv1)
          
          conv3 = Conv2D(self.conv3_filters, self.conv3_kernel_shape, padding = self.padding3, activation=self.conv1_activation,name = str(i+1)+"conv"+str(3))(conv2)

          flat = Flatten(name = str(i+1)+"flatten")(conv3)
          dense = Dense(self.dense_nodes, activation = self.dense_activation,name = str(i+1)+"dense")(flat)
          dense2 = Dropout(self.dense_dropout,name = str(i+1)+"dropout")(dense)

          dense2 = Lambda(lambda X: tensorflow.expand_dims(X, axis=1))(dense2)
          convs.append(dense2)
      
      merge = concatenate(convs,axis=1,name = "merge")  
      lstm1 = LSTM(self.lstm1_cells, return_sequences=True,name = "lstm"+str(1))(merge)
      lstm2 = LSTM(self.lstm2_cells, return_sequences=False,name = "lstm"+str(2))(lstm1)
      dense3 = Dense(self.dense3_nodes, activation=self.dense3_activation,name = "dense"+str(2))(lstm2)
      final = Dropout(self.final_dropout, name = "dropout"+str(1))(dense3)
      output = Dense(self.number_classes, activation="softmax",name = "dense"+str(3))(final)
      
      model = Model(inputs=inputs, outputs=output)
      return model
  
def test_cascade_model():
    window_size = 10
    
    conv1_filters = 1
    conv2_filters = 2
    conv3_filters = 4
    
    conv1_kernel_shape = (7,7)
    conv2_kernel_shape = conv1_kernel_shape
    conv3_kernel_shape = conv1_kernel_shape
    
    padding1 = "same"
    padding2 = padding1
    padding3 = padding1
    
    conv1_activation = "relu"
    conv2_activation = conv1_activation
    conv3_activation = conv1_activation
    
    dense_nodes = 125
    dense_activation = "relu"
    dense_dropout = 0.5
    
    lstm1_cells = 10
    lstm2_cells = lstm1_cells
    
    dense3_nodes = dense_nodes
    dense3_activation = "relu"
    final_dropout = 0.5
    
    
    cascade_object = Cascade(window_size,conv1_filters,conv2_filters,conv3_filters,
                 conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                 padding1,padding2,padding3,conv1_activation,conv2_activation,
                 conv3_activation,dense_nodes,dense_activation,dense_dropout,
                 lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation,
                 final_dropout)
    cascade_model = cascade_object.model
    print(cascade_model.summary())
    
# test_cascade_model()