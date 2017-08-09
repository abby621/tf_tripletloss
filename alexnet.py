from network import Network
import tensorflow as tf

class CaffeNetPlaces365(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 2e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(365, relu=False, name='fc8')
             .softmax(name='prob'))
        
def test():
    #input_node = tf.placeholder(tf.float32, shape=(None, 227,227,3))
    input_node = tf.random_uniform(shape=(1,227,227,3))*255
    net = CaffeNetPlaces365({'data': input_node})
    
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0"
    
    with tf.Session(config=c) as sess:
        print('Loading the model')
        net.load('../../models/alexnet.npy', sess)
        temp = sess.run(net.layers['prob'])
        print (temp)
        print(temp.shape)
        print('Done!')
        
if __name__ == '__main__':
    test()