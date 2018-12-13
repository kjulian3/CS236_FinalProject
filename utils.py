"""
Some codes from https://github.com/Newmu/dcgan_code
"""
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

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

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

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

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
    
def visualize(sess, dcgan, config, option):
    
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option==-1:
    batch_size = 8*4
    #z_sample_orig = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    jdx = 0
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      #z_sample = z_sample_orig.copy()
      #idx2 = 0
      #while idx2 < batch_size:
      #    z_sample[idx2:idx2+8,jdx] = np.linspace(-1,1,8)
      #    idx2 += 8
      #    jdx += 1

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})[:batch_size//2,:,:,:]
      print("MIN:",np.min(samples))
      samples_seg = segment(samples)
      samples = np.concatenate((samples,samples_seg),axis=0)
      save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_seg_%s.png' % (idx))
        
  elif option==0:
    batch_size = 8*4
    #z_sample_orig = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    jdx = 0
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      #z_sample = z_sample_orig.copy()
      #idx2 = 0
      #while idx2 < batch_size:
      #    z_sample[idx2:idx2+8,jdx] = np.linspace(-1,1,8)
      #    idx2 += 8
      #    jdx += 1

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      print(samples.shape)
      save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_arange_%s.png' % (idx))
        
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    batch_size = 8*4
    z_sample_orig = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    jdx = 0
    for idx in xrange(10):#dcgan.z_dim):
      print(" [*] %d" % idx)
      #z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      z_sample = z_sample_orig.copy()
      idx2 = 0
      while idx2 < batch_size:
          z_sample[idx2:idx2+8,jdx] = np.linspace(-1,1,8)
          idx2 += 8
          jdx += 1
      #print("Before: ",z_sample)
      #for kdx, z in enumerate(z_sample):
      #  z[idx] = values[kdx]

      #print("After: ",z_sample)
      if config.dataset == "wildfire":
        #y = np.random.choice(8, config.batch_size)
        y = np.array([i for i in range(8) for j in range(8)])
        y_one_hot = np.zeros((config.batch_size, 8))
        y_one_hot[np.arange(config.batch_size), y] = 1
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        save_images(samples, [image_frame_dim, image_frame_dim], config.sample_dir + '/test_arange_%s.png' % (idx))
    
      elif config.dataset == "wildfire_withForest":
        #y = np.random.choice(8, config.batch_size)
        y = np.array([i for i in range(2) for j in range(16)])
        y_one_hot = np.zeros((32, 2))
        y_one_hot[np.arange(32), y] = 1
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        save_images(samples, image_manifold_size(32), config.sample_dir + '/test_arange_%s.png' % (idx))
        
      elif config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y = np.array([i for i in range(8) for j in range(8)])
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
        save_images(samples, [image_frame_dim, image_frame_dim], config.sample_dir + '/test2_arange_%s.png' % (idx))
        
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_arange_%s.png' % (idx))
        
      #save_images(samples, [image_frame_dim, image_frame_dim], config.sample_dir + '/test_arange_%s.png' % (idx))
  elif option == 2:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    for fireSize in range(21):
        np.random.seed(571)
        for idx in xrange(batch_size):#dcgan.z_dim):
          z_sample[idx,:] = np.random.uniform(-1,1,dcgan.z_dim)
          z_sample[idx,25] = fireSize*0.1-1 #np.random.uniform(-1,-0.75)
          z_sample[idx,38] = fireSize*0.1-1 #np.random.uniform(0.75,1)
          z_sample[idx,35] = -fireSize*0.1+1 #np.random.uniform(-1,-0.5)
          #z_sample[idx,16] = np.random.uniform(0.5,1)
          #z_sample[idx,12] = np.random.uniform(-1,-0.5)
          #z_sample[idx,8] = np.random.uniform(0.5,1)
          #z_sample[idx,7] = np.random.uniform(0.5,1)
          #z_sample[idx,1] = np.random.uniform(-1,-0.5)
          #z_sample[idx,0] = np.random.uniform(-1,-0.5)

        #print(z_sample)
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_arange_fireSize%02d.png'%(fireSize))
   
  elif option == 3:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    #for fireSize in range(8):
    np.random.seed(573)
    idx = 0
    while idx < batch_size:#dcgan.z_dim):
        fireImage = np.random.uniform(-1,1,dcgan.z_dim)
        for fireSize in range(8):
          # FIRE SIZE
          #z_sample[idx,:] = fireImage.copy()
          #z_sample[idx,4] = -fireSize*2.0/7.0+1 #np.random.uniform(-1,-0.75)
          #z_sample[idx,26] = fireSize*2.0/7.0-1 #np.random.uniform(0.75,1)
          #z_sample[idx,31] = -fireSize*2.0/7.0+1 #np.random.uniform(-1,-0.5)
          
          # TIME OF DAY
          z_sample[idx,:] = fireImage.copy()
          z_sample[idx,36] = -fireSize*2.0/7.0+1 #np.random.uniform(-1,-0.75)
          z_sample[idx,8] = fireSize*2.0/7.0-1 #np.random.uniform(0.75,1)
          z_sample[idx,19] = -fireSize*2.0/7.0+1 #np.random.uniform(-1,-0.5)
          z_sample[idx,15] = fireSize*2.0/7.0-1 #np.random.uniform(0.75,1)
            
          idx+=1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_arange_timeOfDay.png')
  elif option == 4:
    batch_size = 8*4
    z_sample = np.tile(np.random.uniform(-1, 1, size=(1, dcgan.z_dim)), (batch_size, 1))
    np.random.seed(583)
    fireImage2 = np.random.uniform(-1,1,dcgan.z_dim)
    idx = 0
    while idx < batch_size:#dcgan.z_dim):
        fireImage1 = fireImage2.copy()
        fireImage2 = np.random.uniform(-1,1,dcgan.z_dim)
        fireImage2 = np.random.uniform(-1,1,dcgan.z_dim)
        print(np.sum(np.abs(fireImage1-fireImage2)))
        
        for interp in range(8):
          z_sample[idx,:] = fireImage1.copy()*(7.0-interp)/7.0 + fireImage2.copy()*interp/7.0
          idx+=1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, image_manifold_size(batch_size), config.sample_dir + '/test_arange_interpFires4.png')

def image_manifold_size(num_images):
  if num_images==32:
    return 4,8
  elif num_images==16:
    return 2,8
  else:
    return num_images//8, 8

 # manifold_h = int(np.floor(np.sqrt(num_images)))
  #manifold_w = int(np.ceil(np.sqrt(num_images)))
  #assert manifold_h * manifold_w == num_images
  #return manifold_h, manifold_w
