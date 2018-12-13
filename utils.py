from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

#get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

#def merge_images(images, size):
#  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def inverse_transform(images):
  return (images+1.)/2.


def f(t):
    ret = 7.787*t + 4.0/29.0
    ret = np.where(t>0.008856,np.power(t,1.0/3.0),ret)
    return ret

def segment(samples):
  X_fire_trans = (samples.transpose(0,1,3,2)+1.0)/2.0*255.0
  xyz_mat = np.array([[0.412453,0.357580,0.180423],[0.212671,0.715160,0.072169],[0.019334,0.119193,0.950227]])
  X_xyz = np.matmul(xyz_mat,X_fire_trans).transpose(0,1,3,2)
  temp = np.matmul(xyz_mat,np.array([255,255,255]))
  Xn = temp[0]; Yn = temp[1]; Zn = temp[2]
  X = X_xyz[:,:,:,0]; Y = X_xyz[:,:,:,1]; Z = X_xyz[:,:,:,2]
  L = 903.3*Y/Yn
  L = np.where(Y/Yn>0.008856,116.0*np.power(Y/Yn,1.0/3.0)-16,L)
  L = L.reshape(L.shape+(1,))
  f_X = f(X/Xn); f_Y = f(Y/Yn); f_Z = f(Z/Zn)
  a = 500*(f_X-f_Y).reshape(f_Y.shape+(1,))
  b = 200*(f_Y-f_Z).reshape(f_Y.shape+(1,))
  X_lab = np.concatenate((L,a,b),axis=3)
  rng = [100.0, 108.1750890269474, 156.04417152036933]
  mn = [0.0, -36.667932404020966, -61.816033812662056]
  Lm=67.4671137460132; am=2.5818119101690304; bm=6.201981527281288;
  samples_seg = np.where((X_lab[:,:,:,0]>Lm) & (X_lab[:,:,:,1] + 1.0*X_lab[:,:,:,2]>am+1.0*bm+5) & (X_lab[:,:,:,1]>am-15),1.0,-1.0)
  samples_seg = samples_seg.reshape(samples_seg.shape+(1,))
  samples_seg = np.concatenate((samples_seg,samples_seg,samples_seg),axis=3)
  return samples_seg
    
# Generates visual images to test wildfire generated images
def visualize(sess, dcgan, config, option):
    
  # Basic arrangement of random generated samples  
  if option==0:
    batch_size = 8*4
    jdx = 0
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))[:batch_size,:]
      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_arange_%s.png' % (idx))
        
  # Produces images where each hidden dimension is varied from -1 to 1
  # Inspection of these images reveals which dimensions correlate with larger firer, brighter images, etc
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    batch_size = 8*4
    z_sample_orig = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    jdx = 0
    for idx in xrange(10):
      print(" [*] %d" % idx)
      z_sample = z_sample_orig.copy()
      idx2 = 0
      while idx2 < batch_size:
          z_sample[idx2:idx2+8,jdx] = np.linspace(-1,1,8)
          idx2 += 8
          jdx += 1

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/varyDimension_%s.png' % (idx))
  
  # Save images that slowly get larger fires. Useful for creating videos
  elif option == 2:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    for fireSize in range(21):
        np.random.seed(571)
        for idx in xrange(batch_size):
          z_sample[idx,:] = np.random.uniform(-1,1,dcgan.z_dim)
          z_sample[idx,4] = -fireSize*0.1-1 
          z_sample[idx,26] = fireSize*0.1-1 
          z_sample[idx,31] = -fireSize*0.1+1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/fireSize%02d.png'%(fireSize))
   
  # After inspection of images from Option 1, Option 2 changes 
  # key dimensions to vary fire size from small to large
  elif option == 3:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    np.random.seed(573)
    idx = 0
    while idx < batch_size:
        fireImage = np.random.uniform(-1,1,dcgan.z_dim)
        for fireSize in range(8):
          # FIRE SIZE
          z_sample[idx,:] = fireImage.copy()
          z_sample[idx,4] = -fireSize*2.0/7.0+1
          z_sample[idx,26] = fireSize*2.0/7.0-1
          z_sample[idx,31] = -fireSize*2.0/7.0+1
        
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/fireSize_arange.png')
  
  # After inspection of images from Option 1, Option 2 changes 
  # key dimensions to vary image brightness from dark to light  
  elif option == 4:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    np.random.seed(573)
    idx = 0
    while idx < batch_size:
        fireImage = np.random.uniform(-1,1,dcgan.z_dim)
        for fireSize in range(8):
          # TIME OF DAY
          z_sample[idx,:] = fireImage.copy()
          z_sample[idx,36] = -fireSize*2.0/7.0+1
          z_sample[idx,8] = fireSize*2.0/7.0-1
          z_sample[idx,19] = -fireSize*2.0/7.0+1
          z_sample[idx,15] = fireSize*2.0/7.0-1
          idx+=1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/timeOfDay_arange.png')
    
  # Interpolate image between five random wildfires
  elif option == 5:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    np.random.seed(583)
    fireImage2 = np.random.uniform(-1,1,dcgan.z_dim)
    idx = 0
    while idx < batch_size:
        fireImage1 = fireImage2.copy()
        fireImage2 = np.random.uniform(-1,1,dcgan.z_dim)
        fireImage2 = np.random.uniform(-1,1,dcgan.z_dim)
        
        for interp in range(8):
          z_sample[idx,:] = fireImage1.copy()*(7.0-interp)/7.0 + fireImage2.copy()*interp/7.0
          idx+=1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/interpFires_arange.png')
  
  # Segment wildfire images
  elif option==6:
    batch_size = 8*4
    jdx = 0
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})[:batch_size//2,:,:,:]
      samples_seg = segment(samples)
      samples = np.concatenate((samples,samples_seg),axis=0)
      save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/segment_%s.png' % (idx))
        
def image_manifold_size(num_images):
  return num_images//8, 8