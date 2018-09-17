# tf-work
## 3
p41.py-----------在不同计算图上定义和使用变量<br>
p41-2.py---------将加法计算泡在CPU和GPU上<br>
p43.py-----------tf.constan是一个计算，这个计算的结果为一个张量，保存在变量a中<br>
p55.py-----------通过变量实现神经网络的参数，并实现前向传播的过程<br>
p60.py-----------通过placeholder实现前向传播算法<br>
p60-2.py---------因为x在定义时指定了n为3,所以在运行前向传播过程时需要提供三个样例数据<br>
p62.py-----------完整的程序来训练神经网络解决二分类问题<br>
## 4
p76.py-----------矩阵乘法(tf.matmul)与元素间直接相乘<br>
p79.py-----------tf.select函数和tf.greater函数<br>
p79-2.py---------实现一个拥有两个输入节点，一个输出节点，没有隐藏层的神经网络<br>
p88.py-----------通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法<br>
p90.py-----------滑动平均，解释ExponentialMovingAcerage是如何被使用<br>
p92.py-----------用圆做样例并训练<br>
## 5
p95.py-----------下载解压MNIST和batch操作<br>
p96.py-----------在MNIST数据集上实现 使用带指数衰减的学习率设置，使用正则化来避免过度你和，以及使用滑动平均模型来使得最终模型更加健壮 的完整TensorFlow程序<br>
p101.py----------计算滑动平均模型在测试数据和验证数据上的正确率<br>
p109.py----------当tf.variable_scope函数嵌套时，reuse参数的取值是如何确定的<br>
p109-2.py--------通过tf.variable_scope来管理变量的名称<br>
p110.py----------通过tf.variable_scope和tf.get_variable函数定义计算前向传播函数<br>
p111.py----------保存TensorFlow计算图的方法<br>
p112.py----------加载已经保存的TensorFlow模型<br>
p112-2.py--------直接加载已经持久化的图<br>
p113.py----------保存滑动平均模型的样例<br>
p114.py----------计算滑动平均模型前向传播的结果<br>
p114-2.py--------variable_to_restore函数的使用样例<br>
p115.py----------整个TensorFlow计算图可以统一存放在一个文件中<br>
p116.py----------通过该程序可以直接计算定义的加法运算的结果。当只需要得到计算图中某个节点的取值时，这提供了一个更加方便的方法<br>
p117.py----------以json格式导出MetaGraphDefProtocol Buffer<br>
p124.py----------如何使用tf.train.NewCheckpointReader类<br>
p126_mnist_inference.py--MNIST数字识别，定义前向传播的过程以及神经网络中的参数<br>
p127_mnist_train.py--MNIST数字识别，定义了神经网络的训练过程<br>
p129_mnist_eval.py--MNIST数字识别，定义了测试过程<br>
## 6
p146.py----------TensorFlow实现卷积层<br>
p150_mnist_eval.py--定义卷积训练的测试过程<br>
p150_mnist_train.py--实现类似与LeNet-5模型结构的前向传播<br>
p151_mnist_inference.py--因为卷积神经网络的输入层为一个三维矩阵，所以需要调整一个输入数据的格式<br>
p158.py----------Inception-v3模型的一个Inception模块<br>
p161-data_process.py--<br>
p161-fine_tuning.py--<br>
p161.py----------迁移学习<br>
## 7
p170.py----------将MNIST输入数据转化为TFRecord的格式<br>
p172.py----------读取TFRecord文件中的数据<br>
p173.py----------TensorFlow中对jpeg格式图像的编码/解码函数<br>
p173-2.py--------TensorFlow中对png格式图像的编码/解码函数<br>
p175.py----------修改图片尺寸<br>
p175-2.py--------使用不同算法的修改图片尺寸<br>
p176.py----------图片裁剪或填充<br>
p176-2.py--------图片裁剪或填充<br>
p177.py----------图片翻转<br>
p178.py----------图片色彩调整<br>
p178-2.py--------图片色彩调整<br>
p179.py----------图片调整色相<br>
p180.py----------图片饱和度调整<br>
p181.py----------图片加入标注框<br>
p181-2.py--------随机截取图像<br>
p181-3.py--------随机截取图像<br>
p182.py----------从图像片段截取，到图像大小调整在到图像翻转及色彩调整的整个图像与处理过程<br>
p184.py----------操作队列<br>
p187.py----------多线程控制tf.Coordinator<br>
p188.py----------tf.Coordinator 和 tf.QueueRunner管理多线程队列操作<br>
p190.py----------生成样例数据<br>
p191.py----------tf.train_filenames_once函数和tf.train.string_input_producer 函数的使用<br>
p192.py----------组合训练数据(batching, tf.train.batch & tf.train.shuffle_batch)<br>
p195.py----------完整的tf输入数据处理框架<br>
p195-2.py--------完整的tf输入数据处理框架,github源码<br>
## 8 
<br>
p204.py----------简单的循环神经网络前向传播过程<br>
p207.py----------tensorflow中实现使用LSTM结构的循环神经网络的前向传播过程<br>
p209.py----------tensorflow使用MultiRNNCell类来实现深层循环神经网络的前向传播过程<br>
p211.py----------tensorflow实现带dropout的循环神经网络<br>
p215.py----------tensorflow ptb_raw_data函数读取PTB的原始数据，并将原始数据中的单词转化为单词ID<br>
p223.py----------不能运行，通过TFLearn快速解决iris分类问题<br>
