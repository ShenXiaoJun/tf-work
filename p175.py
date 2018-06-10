import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/home/shenxj/tf-work/000000.png", 'r').read()

with tf.Session() as sess:
	img_data = tf.image.decode_png(image_raw_data)
	#print img_data.eval()	
	#plt.imshow(img_data.eval())
	#plt.show()
	
	img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
	#img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint16)
	#encoded_image = tf.image.encode_png(img_data)
	#with tf.gfile.GFile("/home/shenxj/tf-work/output/000000.png", "wb") as f:
		#f.write(encoded_image.eval())
	
	resized = tf.image.resize_images(img_data,[300,300],method=0)
	print resized.get_shape()
	plt.imshow(resized.eval())
	plt.show()
