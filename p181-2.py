import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile("/home/shenxj/tf-work/000000.png", 'r').read()

with tf.Session() as sess:
	img_data = tf.image.decode_png(image_raw_data)
	#print img_data.eval()
	#plt.imshow(img_data.eval())
	#plt.show()
	
	img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)	

	img_data = tf.image.resize_images(img_data,[180,267],method=1)
	batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
	boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	result = tf.image.draw_bounding_boxes(batched,boxes)
	
	result = tf.squeeze(result,0)
	print result.eval()	
	plt.imshow(result.eval())
	plt.show()
	
	
