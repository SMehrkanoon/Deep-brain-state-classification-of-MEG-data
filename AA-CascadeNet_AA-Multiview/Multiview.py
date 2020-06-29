from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, LSTM, concatenate, Permute, Add

class Multiview:
    def __init__(self, window_size,conv1_filters,conv2_filters,conv3_filters,
                 conv1_kernel_shape,conv2_kernel_shape,conv3_kernel_shape,
                 padding1,padding2,padding3,conv1_activation,conv2_activation,
                 conv3_activation,dense_nodes,dense_activation,depth,
                 lstm1_cells,lstm2_cells,dense3_nodes,dense3_activation):
        
        self.number_classes = 4
        self.number_channels = 248
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
        
        self.lstm1_cells = lstm1_cells
        self.lstm2_cells = lstm2_cells
        
        self.dense3_nodes = dense3_nodes
        self.dense3_activation = dense3_activation
        
        self.depth = depth
        
        self.model = self.get_model()
    
    def get_model(self):

        inputs_cnn = []
        inputs_lstm = []
        
        outputs_cnn = []
        lstm = []
        
        for i in range(self.window_size):
            input_cnn = Input(shape=(self.mesh_rows,self.mesh_columns, self.depth), name = "input"+str(i+1))  
            input_lstm = Input(shape=(self.number_channels,self.depth), name = "input"+str(i+1+self.window_size))
            inputs_cnn.append(input_cnn)
            inputs_lstm.append(input_lstm)
            
            conv1 = Conv2D(self.conv1_filters, self.conv1_kernel_shape, padding = self.padding1,activation=self.conv1_activation,input_shape=(self.mesh_rows,self.mesh_columns,self.depth),name = str(i+1)+"conv"+str(1))(input_cnn)
              
            conv2 = Conv2D(self.conv2_filters, self.conv2_kernel_shape, padding = self.padding2, activation=self.conv2_activation)(conv1)
            
            conv3 = Conv2D(self.conv3_filters, self.conv3_kernel_shape, padding = self.padding3, activation=self.conv3_activation)(conv2)
        
            flat = Flatten()(conv3)
            dense = Dense(self.dense_nodes, activation=self.dense_activation)(flat)
            outputs_cnn.append(dense)
        
            permut = Permute((2,1), input_shape=(self.number_channels,1))(input_lstm)
            dense = Dense(self.dense_nodes, activation=self.dense_activation, input_shape=(1,self.number_channels))(permut)
            lstm.append(dense)
        
        merge = concatenate(lstm,axis=1)
        lstm1 = LSTM(self.lstm1_cells, return_sequences=True)(merge)
        lstm2 = LSTM(self.lstm2_cells, return_sequences=False)(lstm1)
        dense3 = Dense(self.dense3_nodes, activation=self.dense3_activation)(lstm2)
        
        added = Add()([i for i in outputs_cnn])
        final = concatenate([dense3,added], axis=-1)
        output = Dense(4, activation="softmax")(final)
        
        model = Model(inputs=inputs_cnn+inputs_lstm, outputs=output)
        return model