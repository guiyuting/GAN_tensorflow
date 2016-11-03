import tensorflow as tf
from dataprepare import *
from tf.contrib.layers import batch_norm

def Linear(x, shape, name=None, reusestate=None):#
    with tf.variable_scope(name,reuse=reusestate):
        w = tf.get_variable(name="w", shape=shape,\
            initializer=tf.truncated_normal_initializer(stddev = 0.02))
        b = tf.get_variable(name="b", shape=shape,\
            initializer=tf.constant_initializer(0.01))
    return tf.matmul(x, w) + b, w, b

'''https://github.com/tensorflow/tensorflow/issues/4079'''
def Lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def Deconv(x, out_shape, name=None, reusestate=None):
    '''
    input: input layer x, output shape
    output: deconv * w + b
    initialize/retrieve the existing weight and bias variable 
    '''
    with tf.variable_scope(name,reuse=reusestate):

        w = tf.get_variable(name="w", shape=[5,5,out_shape[-1],x.get_shape()[-1]],\
                            dtype=tf.float32,\
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name="b", shape=[out_shape[-1]], dtype=tf.float32,\
                            initializer=tf.constant_initializer(0.))
        deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, out_shape, \
                                strides = [1,2,2,1]), b)
    return deconv


def Conv(x, out_shape, name=None, reusestate=None):
    '''
    input: input layer x, outout shape
    output: conv * w + b
    '''
    with tf.variable_scope(name,reuse=reusestate):
        w = tf.get_variable(name="w", shape=[5,5,x.get_shape()[-1],out_shape[-1]],\
                            dtype=tf.float32,\
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable(name="b", shape=[out_shape[-1]], dtype=tf.float32,\
                            initializer=tf.constant_initializer(0.))
        conv = tf.nn.bias_add(tf.nn.conv2d(x, w, strides = [1,2,2,1]), b)
    return conv


class gan():
    def __init__(self):
        ''' hyparameters '''
        self.epoch = 10
        self.batch_size = 32
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()

    def generator(self, z, reusestate):

        ''' take uniform sample z, generate G(z), an image '''
        
        ''' trained parameters'''
#TODO: add batch norm 
        Relu = tf.nn.relu
        z_, g_W0, g_b0 = Linear(z, [1,4*4*1024],name="zinit")
        
        g_h0 = tf.reshape(self.z_, [-1,4,4,1024])
        g_h1 = Relu(Deconv(g_h0, [self.batch_size,8,8,512], name="g_h1",reuse=reusestate))
        g_h2 = Relu(Deconv(g_h1, [self.batch_size,16,16,256], name="g_h2",reusestate))
        g_h3 = Relu(Deconv(g_h2, [self.batch_size,32,32,128], name="g_h3",reusestate))
        g_out = Relu(Deconv(g_h3, [self.batch_size,64,64,3], name="g_out",reusestate))
        # g_out is a batch of fake pictures
        return g_out

    def discriminator(self, x, reusestate):
        ''' take a fake/true image as input, generate lable:fake/true'''
        ''' image size: 64*64*3, output logit '''
        d_h1 = Lrelu(Conv(x,[64*2], name="d_h1",reusestate)) #64*64*128
        d_h2 = Lrelu(Conv(d_h1,[64*4], name="d_h2",reusestate))
        d_h3 = Lrelu(Conv(d_h2,[64*8], name="d_h3",reusestate))
        d_h4 = Lrelu(Conv(d_h3,[64*16], name="d_h4",reusestate)) # 64*64*1024
        return d_h4


    def train(self):

        self.sess.run(self.init)
        zz = tf.random_normal([100,1],stddev=0.001)
        self.sess.run(self.generator(zz,None))

        tf.get_variable_scope().reuse_variables()

        




def main():
    #train, valid, test = read_mnist()
    mnist_gan = gan()
    mnist_gan.train()


if __name__ == "__main__":
    main()
