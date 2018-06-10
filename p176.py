import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/home/shenxj/tf-work/000000.png", 'r').read()

with tf.Session() as sess:
	img_data = tf.image.decode_png(image_raw_data)
	
	img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
	
	croped = tf.image.resize_image_with_crop_or_pad(img_data,300,300)
	print croped.get_shape()
	plt.imshow(croped.eval())
	plt.show()

	padded = tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)
	print padded.get_shape()
	plt.imshow(padded.eval())
	plt.show()

	central_corpped = tf.image.central_crop(img_data, 0.5)
	print central_corpped.get_shape()
	plt.imshow(central_corpped.eval())
	plt.show()
