# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 10 19:13:56 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
	This python code creates a Stacked Hourglass Model
	(Credits : A.Newell et al.)
	(Paper : https://arxiv.org/abs/1603.06937)
	
	Code translated from 'anewell' github
	Torch7(LUA) --> TensorFlow(PYTHON)
	(Code : https://github.com/anewell/pose-hg-train)
	
	Modification are made and explained in the report
	Goal : Achieve Real Time detection (Webcam)
	----- Modifications made to obtain faster results (trade off speed/accuracy)
	
	This work is free of use, please cite the author if you use it!
"""
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
from PIL import Image

class HourglassModel():
    """ HourglassModel class: (to be renamed)
    Generate TensorFlow model to train and predict Human Pose from images (soon videos)
    Please check README.txt for further information on model management.
    """
    def __init__(self, nFeat = 128, nStack = 4, nLow = 4, inputDim = 3,outputDim = 3, batch_size = 20, drop_rate = 0.2, lear_rate = 2.5e-4, decay = 0.96, decay_step = 2000, training = True, w_summary = True, logdir_train = None, logdir_test = None,tiny = True,modif = False,w_loss = False, name = 'tiny_hourglass'):
        """ Initializer
        Args:
            nStack				: number of stacks (stage/Hourglass modules)
            nFeat				: number of feature channels on conv layers
            nLow				: number of downsampling (pooling) per module
            outputDim			: number of output Dimension (16 for MPII)
            batch_size			: size of training/testing Batch
            dro_rate			: Rate of neurons disabling for Dropout Layers
            lear_rate			: Learning Rate starting value
            decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
            decay_step			: Step to apply decay
            dataset			: Dataset (class DataGenerator)
            training			: (bool) True for training / False for prediction
            w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
            tiny				: (bool) Activate Tiny Hourglass
            modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
            name				: name of the model
        """
        self.nStack = nStack
        self.nFeat = nFeat
        self.inputDim = inputDim
        self.outDim = outputDim
        self.batchSize = batch_size
        self.training = training
        self.w_summary = w_summary
        self.tiny = tiny
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.name = name
        self.decay_step = decay_step
        self.nLow = nLow
        self.modif = modif
        #self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.w_loss = False

    # ACCESSOR

    def get_input(self):
        """ Returns Input (Placeholder) Tensor
        Image Input :
            Shape: (None,256,256,3)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.img
    def get_output(self):
        """ Returns Output Tensor
        Output Tensor :
            Shape: (None, nbStacks, 64, 64, outputDim)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.output
    def get_label(self):
        """ Returns Label (Placeholder) Tensor
        Image Input :
            Shape: (None, nbStacks, 64, 64, outputDim)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.gtMaps
    def get_loss(self):
        """ Returns Loss Tensor
        Image Input :
            Shape: (1,)
            Type : tf.float32
        Warning:
            Be sure to build the model first
        """
        return self.loss
    def get_saver(self):
        """ Returns Saver
        /!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
        Warning:
            Be sure to build the model first
        """
        return self.saver

    def generate_model(self):
        """ Create the complete graph
        """
        startTime = time.time()
        print('CREATE MODEL:')
    # Set up input placeholder
        with tf.device(self.gpu):
            with tf.name_scope('inputs'):
                # Shape Input Image - batchSize: None, height: 128, width: 128, channel: inputDim (color,mask,edge)
                self.img = tf.placeholder(dtype= tf.float32, shape= (None, 128, 128, self.inputDim), name = 'input_img')
                if self.w_loss:
                    self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
                # Shape Ground Truth Map: batchSize x nStack x 128 x 128 x outDim
                self.gtMaps = tf.placeholder(dtype = tf.float32, shape = (None, 128, 128, 3))#self.nStack, 128, 128, 3))
                self.mask = tf.placeholder(dtype= tf.float32, shape= (None, 128, 128))
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')
            
    # Build the graph TODO: check the structure of HG
            self.output = self._graph_hourglass(self.img)
            graphTime = time.time()
            print('---Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')

    # Calculate loss. TODO: modify the loss
            with tf.name_scope('loss'):
                if self.w_loss:
                    self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
                else:
                    # ground truth:[b,4,128,128,3]; output:same size
                    # TODO: change to sum format, change size of gtMAP
                    self.loss,self.loss_out = self.MAE_loss()
                    # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
            lossTime = time.time()	
            print('---Loss : Done (' + str(int(abs(graphTime-lossTime))) + ' sec.)')

    # Calculate accuracy ?
        with tf.device(self.cpu):
        #     with tf.name_scope('accuracy'):
        #         self._accuracy_computation()
            accurTime = time.time()
            print('---Acc : Done (' + str(int(abs(accurTime-lossTime))) + ' sec.)')

    # Set up training parameter, using exp decay
            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
            with tf.name_scope('lr'):
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
            lrTime = time.time()
            print('---LR : Done (' + str(int(abs(accurTime-lrTime))) + ' sec.)')

    # Define optimizer TODO: change to Adam
        with tf.device(self.gpu):
            with tf.name_scope('adam'):
                self.rmsprop = tf.train.AdamOptimizer(learning_rate= self.lr)
            optimTime = time.time()
            print('---Optim : Done (' + str(int(abs(optimTime-lrTime))) + ' sec.)')
            with tf.name_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
            minimTime = time.time()
            print('---Minimizer : Done (' + str(int(abs(optimTime-minimTime))) + ' sec.)')

    # Init all the variables
        self.init = tf.global_variables_initializer()
        initTime = time.time()
        print('---Init : Done (' + str(int(abs(initTime-minimTime))) + ' sec.)')

    # Keep summary
        with tf.device(self.cpu):
            with tf.name_scope('training'):
                tf.summary.scalar('loss', self.loss, collections = ['train'])
                tf.summary.scalar('learning_rate', self.lr, collections = ['train'])

        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')
        endTime = time.time()
        print('Model created (' + str(int(abs(endTime-startTime))) + ' sec.)')
        del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime

    def restore(self, load = None):
        """ Restore a pretrained model
        Args:
            load	: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary(summary = False)
                if load is not None:
                    print('Loading Trained Model')
                    t = time.time()
                    self.saver.restore(self.Session, load)
                    print('Model Loaded (', time.time() - t,' sec.)')
                else:
                    print('Please give a Model in args (see README for further information)')

    
    def test_init(self, data_gen, load = None, save = None):
        """ Initialize the training
        Args:
            nEpochs		: Number of Epochs to train
            epochSize		: Size of one Epoch
            saveStep		: Step to save 'train' summary (has to be lower than epochSize)
            dataset		: Data Generator (see generator.py)
            load			: Model to load (None if training from scratch) (see README for further information)
        """
        order = data_gen.file_order
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()
                if load is not None:
                    self.saver.restore(self.Session, load)

                for i in range(2000):
                    img_test, mask_test = data_gen(1)
                    img_out = self.Session.run([self.output], feed_dict = {self.img : img_test, self.mask: mask_test})
                    image = np.asarray(img_out[0][0][3])
                    mask = np.asarray(mask_test)
                    # image = image[0,0,...]
                    image = (image/255.0-0.5)*2
                    norm_factor = np.square(image).sum(axis=2)
                    image= np.divide(image,np.expand_dims(np.sqrt(norm_factor),-1))
                    image = (image/2+0.5)*255

                    image = np.multiply(image,np.expand_dims(mask/255,-1))[0]

                    f = Image.fromarray(image.astype(np.uint8))
                    f.save(os.path.join(save, order[i]))
        return 0

    def _train(self, data_gen, nEpochs = 10, epochSize = 1000, batchSize=20, saveStep = 500, validIter = 10):
        """
        """
        with tf.name_scope('Train'):
            # self.generator = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'train')
            # self.valid_gen = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'valid')
            startTime = time.time()
            self.resume = {}
            self.resume['accur'] = []
            self.resume['loss'] = []
            self.resume['err'] = []
            c_out = 0.
            for epoch in range(nEpochs):
                epochstartTime = time.time()
                avg_cost = 0.
                cost = 0.
                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                # Training Set
                for i in range(epochSize):
                    # DISPLAY PROGRESS BAR
                    # TODO : Customize Progress Bar
                    percent = ((i+1)/epochSize) * 100
                    num = np.int(20*percent/100)
                    tToEpoch = int((time.time() - epochstartTime) * (100 - percent)/(percent))
                    sys.stdout.write('\r Train: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.'+'-loss_batch: '+str(c_out))
                    sys.stdout.flush()
                    img_train, gt_train,mask_train = data_gen(batchSize)  #weight_train = next(self.generator) 
                    weight_train = 0
                    if i % saveStep == 0:
                        if self.w_loss:
                            _, c,c_out, summary = self.Session.run([self.train_rmsprop, self.loss, self.loss_out,self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
                        else:
                            _, c,c_out, summary = self.Session.run([self.train_rmsprop, self.loss, self.loss_out,self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.mask: mask_train})
                        # Save summary (Loss + Accuracy)
                        self.train_summary.add_summary(summary, epoch*epochSize + i)
                        self.train_summary.flush()
                    else:
                        if self.w_loss:
                            _, c,c_out = self.Session.run([self.train_rmsprop, self.loss,self.loss_out], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
                        else:
                            _, c,c_out = self.Session.run([self.train_rmsprop, self.loss,self.loss_out], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.mask: mask_train})
                    cost += c
                    avg_cost += c/epochSize
                epochfinishTime = time.time()
                #Save Weight (axis = epoch)
                # if self.w_loss:
                #     weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
                # else :
                #     weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.mask: mask_train})
                # self.train_summary.add_summary(weight_summary, epoch)
                # self.train_summary.flush()
                #self.weight_summary.add_summary(weight_summary, epoch)
                #self.weight_summary.flush()
                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime-epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime-epochstartTime)/epochSize))[:4] + ' sec.')
                with tf.name_scope('save'):
                    self.saver.save(self.Session, os.path.join(os.getcwd(),str(self.name + '_' + str(epoch + 1))))
                self.resume['loss'].append(cost)

                # # Validation Set
                # accuracy_array = np.array([0.0]*len(self.joint_accur))
                # for i in range(validIter):
                #     img_valid, gt_valid, mask_valid = data_gen(batchSize) #, w_valid = next(self.generator)
                #     accuracy_pred = self.Session.run(self.joint_accur, feed_dict = {self.img : img_valid, self.gtMaps: gt_valid})
                #     accuracy_array += np.array(accuracy_pred, dtype = np.float32) / validIter
                # print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%' )
                # self.resume['accur'].append(accuracy_pred)
                # self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))
                # valid_summary = self.Session.run(self.test_op, feed_dict={self.img : img_valid, self.gtMaps: gt_valid})
                # self.test_summary.add_summary(valid_summary, epoch)
                # self.test_summary.flush()
            print('Training Done')
            print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * batchSize) )
            print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
            print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
            print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - startTime)))

    def record_training(self, record):
        """ Record Training Data and Export them in CSV file
        Args:
            record		: record dictionnary
        """
        out_file = open(self.name + '_train_record.csv', 'w')
        for line in range(len(record['accur'])):
            out_string = ''
            labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
            for label in labels:
                out_string += str(label) + ', '
            out_string += '\n'
            out_file.write(out_string)
        out_file.close()
        print('Training Record Saved')


    def training_init(self, data_gen, nEpochs = 10, epochSize = 1000, batchSize=20, saveStep = 500, load = None):
        """ Initialize the training
        Args:
            nEpochs		: Number of Epochs to train
            epochSize		: Size of one Epoch
            saveStep		: Step to save 'train' summary (has to be lower than epochSize)
            dataset		: Data Generator (see generator.py)
            load			: Model to load (None if training from scratch) (see README for further information)
        """
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_weight()
                self._define_saver_summary()
                if load is not None:
                    self.saver.restore(self.Session, load)
                    #try:
                        #	self.saver.restore(self.Session, load)
                    #except Exception:
                        #	print('Loading Failed! (Check README file for further information)')
                self._train(data_gen, nEpochs, epochSize, batchSize, saveStep, validIter=10)

    def weighted_bce_loss(self):
        """ Create Weighted Loss Function
        WORK IN PROGRESS
        """
        self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
        e1 = tf.expand_dims(self.weights,axis = 1, name = 'expdim01')
        e2 = tf.expand_dims(e1,axis = 1, name = 'expdim02')
        e3 = tf.expand_dims(e2,axis = 1, name = 'expdim03')
        return tf.multiply(e3,self.bceloss, name = 'lossW')

    def MAE_loss(self):
        """
        self.img [b,128,128,3]
        self.output [b,stack,128,128,3]
        self.gtMAPs [b,128,128,3]
        """
        x = self.gtMaps
        y_stack = self.output
        mask = self.mask
        mask = tf.not_equal(mask,tf.zeros_like(mask))

        a = tf.reduce_sum(tf.square(x),3)
        a_m = tf.boolean_mask(a,mask)
        loss = 0

        tmp = 0
        count = 0.
        y = x
        z_mask = mask
        for i in range(self.nStack):
            output = self.output[:,i,:,:,:]
            
            output = (output/255.0-0.5)*2
            output_mask = tf.abs(output) < 1e-5
            output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(output), output)
            norm_factor = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)), -1)
            norm_output = tf.divide(output_no0,norm_factor)

            a11=tf.boolean_mask(tf.reduce_sum(tf.square(norm_output),3),z_mask)
            a22=tf.boolean_mask(tf.reduce_sum(tf.square(y),3),z_mask)
            a12=tf.boolean_mask(tf.reduce_sum(tf.multiply(y,norm_output),3),z_mask)
            cos_dist = tf.clip_by_value(tf.where(tf.is_nan(a12 / tf.sqrt(tf.multiply(a11,a22))), -1*tf.ones_like(a12 / tf.sqrt(tf.multiply(a11,a22))), a12 / tf.sqrt(tf.multiply(a11,a22))), -1, 1)
            loss_i = tf.reduce_mean(3.1415926/2-(cos_dist+tf.pow(cos_dist,3)/6+tf.pow(cos_dist,5)*3/40+tf.pow(cos_dist,7)*15/336+tf.pow(cos_dist,9)*105/3456))
            loss += loss_i
            count = loss_i
        # for i in range(self.nStack):
        #     # y = y_stack[:,i,:,:,:]

        #     output = y_stack[:,i,:,:,:]
        #     output = (output/255.0-0.5)*2
        #     output_mask = tf.abs(output) < 1e-5
        #     output_no0 = tf.where(output_mask, 1e-5*tf.ones_like(output), output)
        #     norm_factor = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(output_no0),3)), -1)
        #     y = tf.divide(output_no0,norm_factor)


        #     b = tf.reduce_sum(tf.square(y),3)
        #     ab = tf.reduce_sum(tf.multiply(x,y),3)
        #     b_m = tf.boolean_mask(b,mask)
        #     ab_m = tf.boolean_mask(ab,mask)
            
        #     cos_dist = ab_m/tf.sqrt(tf.multiply(a_m,b_m))

        #     # assign 1 if it is NAN
        #     cos_tmp = tf.where(tf.is_nan(cos_dist),tf.ones_like(cos_dist),tf.zeros_like(cos_dist))
        #     count = tf.reduce_mean(cos_tmp)

        #     cos_dist = tf.where(tf.is_nan(cos_dist),-1*tf.ones_like(cos_dist),cos_dist)
        #     cos_dist = tf.clip_by_value(cos_dist,-1,1)
        #     tmp = tf.reduce_mean(tf.acos(cos_dist))
        #     loss += tmp
        return loss,count


    # def _accuracy_computation(self):
    #     """ Computes accuracy tensor
    #     """
    #     self.joint_accur = []
    #     for i in range(len(self.joints)):
    #         self.joint_accur.append(self._accur(self.output[:, self.nStack - 1, :, :,i], self.gtMaps[:, self.nStack - 1, :, :, i], self.batchSize))
        
    def _define_saver_summary(self, summary = True):
        """ Create Summary and Saver
        Args:
            logdir_train		: Path to train summary directory
            logdir_test		: Path to test summary directory
        """
        if (self.logdir_train == None) or (self.logdir_test == None):
            raise ValueError('Train/Test directory not assigned')
        else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver()
            if summary:
                with tf.device(self.gpu):
                    self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                    self.test_summary = tf.summary.FileWriter(self.logdir_test)
                    #self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

    def _init_weight(self):
        """ Initialize weights
        """
        print('Session initialization')
        self.Session = tf.Session()
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _init_session(self):
        """ Initialize Session
        """
        print('Session initialization')
        t_start = time.time()
        config = tf.ConfigProto(allow_soft_placement = True)
        self.Session = tf.Session(config = config)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')
        
    def _graph_hourglass(self, inputs):
        """Create the Network
        Args:
            inputs : TF Tensor (placeholder) of shape (None, 128, 128, inputDim=3) 
        """
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 128 x 128 x inputDim
                pad1 = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]], name='pad_1')
                # Dim pad1 : nbImages x 132 x 132 x inputDim
                # W-F+2P / S + 1
                conv1 = self._conv_bn_relu(pad1, filters= 64, kernel_size = 7, strides = 1, name = 'conv_256_to_128')
                # Dim conv1 : nbImages x 64 x 64 x 64
                r1 = self._residual(conv1, numOut = 128, name = 'r1')
                # Dim pad1 : nbImages x 64 x 64 x 128


                # pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
                # Dim pool1 : nbImages x 32 x 32 x 128
                if self.tiny:
                    # r3 = self._residual(pool1, numOut=self.nFeat, name='r3')
                    r3 = self._residual(r1, numOut=self.nFeat, name='r3')
                else:
                    r2 = self._residual(pool1, numOut= int(self.nFeat/2), name = 'r2')
                    r3 = self._residual(r2, numOut= self.nFeat, name = 'r3')
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack
            if self.tiny:
                with tf.name_scope('stacks'):
                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                        drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, name = 'll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], ll[0], r3], name = 'merge')
                    for i in range(1, self.nStack - 1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name= 'll')
                            if self.modif:
                                # TEST OF BATCH RELU
                                out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], ll[i], sum_[i-1]], name= 'merge')
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack-1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack-1], self.nFeat,1,1, 'VALID', 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
                print(out[0].shape)
                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis= 1 , name= 'stack_output'),name = 'final_output')
                else:
                    return tf.stack(out, axis= 1 , name = 'final_output')	
            else:
                with tf.name_scope('stacks'):
                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                        drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1,1, 'VALID', name = 'conv')
                        ll_[0] =  self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                    for i in range(1, self.nStack -1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name= 'conv')
                            ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                            if self.modif:
                                out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], sum_[i-1], ll_[0]], name= 'merge')
                    with tf.name_scope('stage_' + str(self.nStack -1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack-1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack-1], self.nFeat, 1, 1, 'VALID', 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis= 1 , name= 'stack_output'),name = 'final_output')
                else:
                    return tf.stack(out, axis= 1 , name = 'final_output')
                        
                
    def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
            if self.w_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
            return conv
            
    def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		: Number of filters (channels)
            kernel_size	: Size of kernel
            strides		: Stride
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
        Returns:
            norm			: Output Tensor
        """
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
            return norm

    def _conv_block(self, inputs, numOut, name = 'conv_block'):
        """ Convolutional Block
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        if self.tiny:
            with tf.name_scope(name):
                norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                pad = tf.pad(norm, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
                return conv
        else:
            with tf.name_scope(name):
                with tf.name_scope('norm_1'):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                    conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
                with tf.name_scope('norm_2'):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                    pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
                    conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
                with tf.name_scope('norm_3'):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
                    conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
                return conv_3

    def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
        """ Skip Layer
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the bloc
        Returns:
            Tensor of shape (None, inputs.height, inputs.width, numOut)
        """
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
                return conv				

    def _residual(self, inputs, numOut, name = 'residual_block'):
        """ Residual Unit
        Args:
            inputs	: Input Tensor
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            if self.modif:
                return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
            else:
                return tf.add_n([convb, skipl], name = 'res_block')

    def _hourglass(self, inputs, n, numOut, name = 'hourglass'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name = 'up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
            low_1= self._residual(low_, numOut, name = 'low_1')
            
            if n > 0:
                low_2 = self._hourglass(low_1, n-1, numOut, name = 'low_2')
            else:
                low_2 = self._residual(low_1, numOut, name = 'low_2')
                
            low_3 = self._residual(low_2, numOut, name = 'low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name = 'upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2,up_1]), name='out_hg')
            else:
                return tf.add_n([up_2,up_1], name='out_hg')

    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of max position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.arg_max(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
        Args:
            u		: 2D - Tensor (Height x Width : 64x64 )
            v		: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            (float) : Distance (in [0,1])
        """
        u_x,u_y = self._argmax(u)
        v_x,v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(91))

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
        returns one minus the mean distance.
        Args:
            pred		: Prediction Batch (shape = num_image x 64 x 64)
            gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
            num_image 	: (int) Number of images in batch
        Returns:
            (float)
        """
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err/num_image)

