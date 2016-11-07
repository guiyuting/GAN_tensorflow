import io
import tensorflow as tf
from dataprepare import *
import lmdb
from PIL import Image
from tensorflow.contrib.layers import batch_norm

# directory to which you unpack the funny 55 GB zip.
DB_PATH = '/Users/YutingGui/Desktop/0725_NLI/lsun/church_outdoor_train_lmdb/'
START_IDX = 0
def load_image(val):
    """LSUN images are stored as bytes of JPEG representation.
        This function converts those bytes into a a 3D tensor
        of shape (64,64,3) in range [0.0, 1.0].
        """
    
    img = Image.open(io.BytesIO(val))
    rp = 64.0 / min(img.size)
    img = img.resize(np.rint(rp * np.array(img.size)).astype(np.int32), Image.BICUBIC)
    img = img.crop((0,0,64,64))
    img = np.array(img, dtype=np.float32) / 255.
    return img

def iterate_images(start_idx=None):
    """Iterates over the images returns pairs of
        (index, image_tensor). It is never the case that all the images
        are loaded into memory at the same time (hopefully, lmdb, please?).
        
        give it a start_idx, not to start from the beginning"""
    with lmdb.open(DB_PATH, map_size=1099511627776,
                   max_readers=100, readonly=True) as env:
        with env.begin(write=False) as txn:
            
            with txn.cursor() as cursor:
                for i, (key, val) in enumerate(cursor):
                    if start_idx is None or start_idx <= i:
                        yield i, load_image(val)

def show_image(img, resize=True):
    """Given a tensor displays it in IPython notebook cell.
        must be the last line. Set resize=False to keep original size."""
    from PIL import Image
    i = Image.fromarray((img * 255.).astype(np.uint8))
    return i.resize((512,512),Image.BICUBIC) if resize else i

def batched_images(start_idx=None):
    """Yields pairs (start_idx_of_next_batch, batch).
        Every batch is of shape (DISCRIMIN_BATCH, 64, 64, 3)"""
    batch, next_idx = None, None
    for idx, image in iterate_images(start_idx):
        if batch is None:
            batch = np.empty((128, 64, 64, 3))
            next_idx = 0
        batch[next_idx] = image
        next_idx += 1
        if next_idx == 128:
            yield idx + 1, batch
            batch = None
###################




def Linear(x, out_shape, name=None):#
    with tf.variable_scope(name,reuse=None):
        w = tf.get_variable(name="w", shape=[x.get_shape()[-1], out_shape[0]],initializer=tf.truncated_normal_initializer(stddev = 0.02))
        b = tf.get_variable(name="b", shape=out_shape,initializer=tf.constant_initializer(0.01))
    return tf.matmul(x, w) + b, w, b

'''https://github.com/tensorflow/tensorflow/issues/4079'''
def Lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def Deconv(x, out_shape, name=None):
    '''
    input: input layer x, output shape
    output: deconv * w + b
    initialize/retrieve the existing weight and bias variable 
    '''
    with tf.variable_scope(name,reuse=None):

        w = tf.get_variable(name="w", shape=[5,5,out_shape[-1],x.get_shape()[-1]],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name="b", shape=[out_shape[-1]], dtype=tf.float32,initializer=tf.constant_initializer(0.))
        deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, out_shape, strides = [1,2,2,1]), b)
    return deconv


def Conv(x, out_shape, name=None, re=None):
    '''
    input: input layer x, outout shape
    output: conv * w + b
    '''
    with tf.variable_scope(name,reuse=re):
        w = tf.get_variable(name="w", shape=[5,5,x.get_shape()[-1],out_shape[-1]],\
                            dtype=tf.float32,\
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name="b", shape=[out_shape[-1]], dtype=tf.float32,\
                            initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(tf.nn.conv2d(x, w, strides = [1,2,2,1], padding="SAME"), b)
    return conv

def Batchnorm(x, name="Batch_norm"):
    return batch_norm(x, scope=name,is_training=True, updates_collections=None, scale=True, decay=0.99)



###########################################
class gan():
    def __init__(self):
        ''' hyparameters '''
        self.epoch = 10
        self.batch_size = 128
        self.learning_rate = 0.0002
        self.beta_adam = 0.5
        self.model() # get the loss for discriminator and generator
        self.init = tf.initialize_all_variables()

        self.sess = tf.Session()
    def generator(self, z):

        ''' take uniform sample z, generate G(z), an image '''
        
        ''' trained parameters'''
        
        Relu = tf.nn.relu
        z_, g_W0, g_b0 = Linear(z, [4*4*1024],name="zinit")
        
        g_h0 = tf.reshape(z_, [-1,4,4,1024])
        g_h1 = Relu(Batchnorm(Deconv(g_h0, [self.batch_size,8,8,512], name="g_h1"),\
               name="bg_h1"))
        g_h2 = Relu(Batchnorm(Deconv(g_h1, [self.batch_size,16,16,256], name="g_h2"),\
               name="bg_h2"))
        g_h3 = Relu(Batchnorm(Deconv(g_h2, [self.batch_size,32,32,128], name="g_h3"),\
               name="bg_h3"))
        g_out = Relu(Batchnorm(Deconv(g_h3, [self.batch_size,64,64,3], name="g_out"),\
               name="bg_out"))
        # g_out is a batch of fake pictures
        return g_out

    def discriminator(self, x, reuse):
        ''' take a fake/true image as input, generate lable:fake/true'''
        ''' image size: 64*64*3, output logit '''
        # why the first and second dimension changed?
        # because of strides = 2
        # x:64,64,3
        if reuse == True:
            tf.get_variable_scope().reuse_variables()
        print("11")
        d_h1 = Lrelu(Batchnorm(Conv(x,[64*2], name="d_h1"), name="bd_h1")) #32*32*128
        d_h2 = Lrelu(Batchnorm(Conv(d_h1,[64*4], name="d_h2"), name="bd_h2")) #16,16,256
        d_h3 = Lrelu(Batchnorm(Conv(d_h2,[64*8], name="d_h3"), name="bd_h3")) #8,8,512
        d_out,_,_= Linear(tf.reshape(d_h3, [self.batch_size, -1]), [1], name="d_out") #outputshape=[1]
        return d_out # batch_size * number in [0,1]

    def model(self):
        '''optimization'''
        self.image = tf.placeholder(tf.float32, [None, 64,64,3])
        self.z = tf.placeholder(tf.float32, [None, 100]) 


        ''' loss of discriminator '''
        self.generated_image = self.generator(self.z) # a batch of fake images
        self.d_fake_logit = self.discriminator(self.generated_image, reuse=False)
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                           self.d_fake_logit, tf.zeros_like(self.d_fake_logit)))

        self.d_true_logit = self.discriminator(self.image, reuse=True)
        self.d_loss_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                           self.d_fake_logit, tf.ones_like(self.d_fake_logit)))
        self.d_loss = self.d_loss_fake + self.d_loss_true

        ''' loss of generator '''
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                      self.d_fake_logit, tf.ones_like(self.d_fake_logit)))
        
        self.d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        self.g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta_adam) \
                                          .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta_adam) \
                                          .minimize(self.g_loss, var_list=self.g_vars)

    def train(self):
        print("Training")
        # XXX dont know why initialize_all_variables() doesn't work
        self.sess.run(tf.initialize_variables(tf.all_variables()))
        #self.sess.run(self.init)

        total_batch = int(trainset.images.shape[0] / self.batch_size)
        START_IDX = 0
        for e in xrange(self.epoch):
            for next_idx, batch in batched_images(START_IDX):
                START_IDX = next_idx
                batch_z = np.random.uniform(-1., 1., size=[self.batch_size, 100])
                feed={self.image:batch, self.z:batch_z }

                _, g_loss, _, d_loss = self.sess.run([self.g_optim, self.g_loss,\
                                       self.d_optim, self.d_loss], feed_dict=feed)
                print("generator_loss: {:f}, discriminator_loss: {:f}".format(g_loss, d_loss))

        



def main():
    #trainset, validset, testset = read_mnist()
    mnist_gan = gan()
    mnist_gan.train()


if __name__ == "__main__":
    main()
