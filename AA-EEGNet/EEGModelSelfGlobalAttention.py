from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten, Lambda, dot, concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow as tf

########### SELF ATTENTION LAYER  ###########


def shape_list(x):
  """Return list of dims, statically where possible."""
  static = x.get_shape().as_list()
  shape = tf.shape(x)
  ret = []
  for i, static_dim in enumerate(static):
    dim = static_dim or shape[i]
    ret.append(dim)
  return ret


def split_heads_2d(inputs, Nh):
  """Split channels into multiple heads."""
  B, H, W, d = shape_list(inputs)
  ret_shape = [B, H, W, Nh, d // Nh]
  split = tf.reshape(inputs, ret_shape)
  return tf.transpose(split, [0, 3, 1, 2, 4])


def combine_heads_2d(inputs):
  """Combine heads (inverse of split heads 2d)."""
  transposed = tf.transpose(inputs, [0, 2, 3, 1, 4])
  Nh, channels = shape_list(transposed)[-2:]
  ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
  return tf.reshape(transposed, ret_shape)


def rel_to_abs(x):
  """Converts tensor from relative to aboslute indexing."""
  # [B, Nh, L, 2L−1]
  B, Nh, L, _ = shape_list(x)
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


def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
  """Compute relative logits along one dimenion."""
  rel_logits = tf.einsum('bhxyd,md->bhxym', q, rel_k)
  # Collapse height and heads
  rel_logits = tf.reshape(rel_logits, [-1, Nh * H, W, 2 * W-1])
  rel_logits = rel_to_abs(rel_logits)
  # Shape it and tile height times
  rel_logits = tf.reshape(rel_logits, [-1, Nh, H, W, W])
  rel_logits = tf.expand_dims(rel_logits, axis=3)
  rel_logits = tf.tile(rel_logits, [1, 1, 1, H, 1, 1])
  # Reshape for adding to the logits.
  rel_logits = tf.transpose(rel_logits, transpose_mask)
  rel_logits = tf.reshape(rel_logits, [-1, Nh, H*W, H*W])
  return rel_logits


def relative_logits(q, H, W, Nh, dkh):
  """Compute relative logits."""
  # Relative logits in width dimension first.
  rel_embeddings_w = tf.get_variable('r_width', shape=(2*W - 1, dkh),initializer=tf.random_normal_initializer(dkh**-0.5)) # [B, Nh, HW, HW]
  rel_logits_w = relative_logits_1d(q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5])
  # Relative logits in height dimension next.
  # For ease, we 1) transpose height and width,
  # 2) repeat the above steps and
  # 3) transpose to eventually put the logits
  # in their right positions.
  rel_embeddings_h = tf.get_variable('r_height', shape=(2 * H - 1, dkh),initializer=tf.random_normal_initializer(dkh**-0.5))
  # [B, Nh, HW, HW]
  rel_logits_h = relative_logits_1d(tf.transpose(q, [0, 1, 3, 2, 4]),rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3])
  return rel_logits_h, rel_logits_w


def self_attention_2d(inputs, dk, dv, Nh, relative=True):
  """2d relative self−attention."""
  _, H, W, _ = shape_list(inputs)
  dkh = dk // Nh
  dvh = dv // Nh
  flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])
  # Compute q, k, v
  kqv = tf.keras.layers.Conv2D(2 * dk + dv, 1)(inputs)
  k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
  q *= dkh ** -0.5 # scaled dot−product
  # After splitting, shape is [B, Nh, H, W, dkh or dvh]
  q = split_heads_2d(q, Nh)
  k = split_heads_2d(k, Nh)
  v = split_heads_2d(v, Nh)
  # [B, Nh, HW, HW]
  logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh),transpose_b=True)
  if relative:
    rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh,dkh)
    logits += rel_logits_h
    logits += rel_logits_w
  weights = tf.nn.softmax(logits)
  attn_out = tf.matmul(weights, flatten_hw(v, dvh))
  attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
  attn_out = combine_heads_2d(attn_out)
  # Project heads
  attn_out = tf.keras.layers.Conv2D(dv, 1)(attn_out)
  return attn_out


def tfaugmented_conv2d(X, Fout, k, dk, dv, Nh, relative):
  conv_out = tf.keras.layers.Conv2D(filters=Fout - dv,kernel_size=k, padding='same')(X)
  attn_out = self_attention_2d(X, dk, dv, Nh, relative=relative)
  return tf.concat([conv_out, attn_out], axis=3)


########### GLOBAL ATTENTION LAYER  ###########

def attention_block(hidden_states):
    print(hidden_states.shape)
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


#################  EEGNET  #################


def EEGNetAttention(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout', depth_k=6, depth_v=4, num_heads=2, relative=False):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (1, Chans, Samples))

    block1       = tfaugmented_conv2d(input1, F1, (1, kernLength), dk=depth_k, dv=depth_v, Nh=num_heads, relative=relative)

    block1       = BatchNormalization(axis = 1)(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.), data_format='channels_first')(block1)
    
    block1       = BatchNormalization(axis = 1)(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4), data_format='channels_first')(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization(axis = 1)(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8), data_format='channels_first')(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    #Expanding dimension
    flatten = tf.expand_dims(flatten, axis=1)
    #Global attention
    attention_output = attention_block(flatten)
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(attention_output)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)



