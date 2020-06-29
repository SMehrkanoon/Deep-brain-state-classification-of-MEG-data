import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D, LSTM
from tensorflow.keras.layers import Lambda,Dropout,dot,Activation
from tensorflow.keras.layers import BatchNormalization

class CascadeSelfGlobalAttention:
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
        
        self.depth_k = 4
        self.depth_v = 2
        self.num_heads = 2  
        self.relative = False
        
        self.depth = depth
        self.model = self.get_model()

    def shape_list(self,x):
      """Return list of dims, statically where possible."""
      static = x.get_shape().as_list()
      shape = tf.shape(x)
      ret = []
      for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
      return ret
    
    
    def split_heads_2d(self,inputs, Nh):
      """Split channels into multiple heads."""
      B, H, W, d = self.shape_list(inputs)
      ret_shape = [B, H, W, Nh, d // Nh]
      split = tf.reshape(inputs, ret_shape)
      return tf.transpose(split, [0, 3, 1, 2, 4])
    
    
    def combine_heads_2d(self,inputs):
      """Combine heads (inverse of split heads 2d)."""
      transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
      Nh, channels = self.shape_list(transposed)[-2:]
      ret_shape = self.shape_list(transposed)[:-2] + [Nh * channels]
      return tf.reshape(transposed, ret_shape)
    
    
    def rel_to_abs(self,x):
      """Converts tensor from relative to aboslute indexing."""
      # [B, Nh, L, 2L1]
      B, Nh, L, _ = self.shape_list(x)
      # Pad to shift from relative to absolute indexing.
      col_pad = tf.zeros((B, Nh, L, 1))
      x = tf.concat([x, col_pad], axis=3)
      flat_x = tf.reshape(x, [B, Nh, L * 2 * L])
      flat_pad = tf.zeros((B, Nh, L-1))
      flat_x_padded = tf.concat([flat_x, flat_pad], axis=2)
      # Reshape and slice out the padded elements.
      final_x = tf.reshape(flat_x_padded, [B, Nh, L+1, 2*L-1])
      final_x = final_x[:, :, :L, L-1:]
      return final_x
    
    
    def relative_logits_1d(self,q, rel_k, H, W, Nh, transpose_mask):
      """Compute relative logits along one dimenion."""
      rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
      # Collapse height and heads
      rel_logits = tf.reshape(rel_logits, [-1, Nh * H, W, 2 * W-1])
      rel_logits = self.rel_to_abs(rel_logits)
      # Shape it and tile height times
      rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
      rel_logits = tf.expand_dims(rel_logits, axis=3)
      rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
      # Reshape for adding to the logits.
      rel_logits = tf.transpose(rel_logits, transpose_mask)
      rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
      return rel_logits
    
    
    def relative_logits(self,q, H, W, Nh, dkh):
      """Compute relative logits."""
      # Relative logits in width dimension first.
      rel_embeddings_w = tf.get_variable('r_width', shape=(2*W - 1, dkh),initializer=tf.random_normal_initializer(dkh**-0.5)) # [B, Nh, HW, HW]
      rel_logits_w = self.relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
      # Relative logits in height dimension next.
      # For ease, we 1) transpose height and width,
      # 2) repeat the above steps and
      # 3) transpose to eventually put the logits
      # in their right positions.
      rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh),initializer=tf.random_normal_initializer(dkh**-0.5))
      # [B, Nh, HW, HW]
      rel_logits_h = self.relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
      return rel_logits_h, rel_logits_w
    
    
    def self_attention_2d(self,inputs, dk, dv, Nh, relative=True):
      """2d relative selfattention."""
      _, H, W, _ = self.shape_list(inputs)
      dkh = dk // Nh
      dvh = dv // Nh
      flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])
      # Compute q, k, v
      kqv = Conv2D(2 * dk + dv, 1)(inputs)
      k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
      q *= dkh ** -0.5 # scaled dotproduct
      # After splitting, shape is [B, Nh, H, W, dkh or dvh]
      q = self.split_heads_2d(q, Nh)
      k = self.split_heads_2d(k, Nh)
      v = self.split_heads_2d(v, Nh)
      # [B, Nh, HW, HW]
      logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh),transpose_b=True)
      if relative:
        rel_logits_h, rel_logits_w = self.relative_logits(q, H, W, Nh,dkh)
        logits += rel_logits_h
        logits += rel_logits_w
      weights = tf.nn.softmax(logits)
      attn_out = tf.matmul(weights, flatten_hw(v, dvh))
      attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
      attn_out = self.combine_heads_2d(attn_out)
      # Project heads
      attn_out = Conv2D(dv, 1)(attn_out)
      return attn_out
    
    
    def tfaugmented_conv2d(self,X, Fout, k, dk, dv, Nh, relative):
      if Fout-dv < 0:
          filters = 1
      else:
          filters = Fout - dv
      conv_out = Conv2D(filters=filters,kernel_size=k, padding='same')(X)
      attn_out = self.self_attention_2d(X, dk, dv, Nh, relative=relative)
      return tf.concat([conv_out, attn_out], axis=3)

    def attention_block(self,hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector
  
    def get_model(self):
      inputs = []
      convs = []
      for i in range(self.window_size):
          input_layer = Input(shape=(self.mesh_rows, self.mesh_columns, self.depth), name = "input"+str(i+1))
          inputs.append(input_layer)

      for i in range(self.window_size):
          conv1 = self.tfaugmented_conv2d(inputs[i], self.conv1_filters, self.conv1_kernel_shape, dk=self.depth_k, dv=self.depth_v, Nh=self.num_heads, relative=self.relative)
          norm1 = BatchNormalization(axis = -1)(conv1)
          
          conv2 = Conv2D(self.conv2_filters, self.conv2_kernel_shape, padding = self.padding2, activation=self.conv1_activation,name = str(i+1)+"conv"+str(2))(norm1)
          
          conv3 = Conv2D(self.conv3_filters, self.conv3_kernel_shape, padding = self.padding3, activation=self.conv1_activation,name = str(i+1)+"conv"+str(3))(conv2)
          
          flat = Flatten(name = str(i+1)+"flatten")(conv3)
          dense = Dense(self.dense_nodes, activation = self.dense_activation,name = str(i+1)+"dense")(flat)
          dense2 = Dropout(self.dense_dropout,name = str(i+1)+"dropout")(dense)

          dense2 = Lambda(lambda X: tf.expand_dims(X, axis=1))(dense2)
          
          convs.append(dense2)
          
      merge = concatenate(convs,axis=1,name = "merge")  
      lstm1 = LSTM(self.lstm1_cells, return_sequences=True,name = "lstm"+str(1))(merge)
      lstm2 = LSTM(self.lstm2_cells, return_sequences=True,name = "lstm"+str(2))(lstm1)

      attention_output = self.attention_block(lstm2)
      
      dense3 = Dense(self.dense3_nodes, activation=self.dense3_activation,name = "dense"+str(2))(attention_output)
      final = Dropout(self.final_dropout, name = "dropout"+str(1))(dense3)
      output = Dense(self.number_classes, activation="softmax",name = "dense"+str(3))(final)
      
      model = Model(inputs=inputs, outputs=output)
      return model





