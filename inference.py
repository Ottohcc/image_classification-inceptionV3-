import glob
import os.path
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import time


# 模型目录
INCEPTION_MODEL_FILE = 'inceptionV3/tensorflow_inception_graph.pb'

# inception-v3模型参数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  
# inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  
# 图像输入张量对应的名称
background_dict={0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

def inference(photo):
    image_data = tf.gfile.GFile(photo, 'rb').read()
    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:

            # 读取训练好的inception-v3模型
            with tf.gfile.GFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

            # 使用inception-v3处理图片获取特征向量
            image_values = sess.run(bottleneck_tensor,{jpeg_data_tensor: image_data})
            # 将四维数组压缩成一维数组，由于全连接层输入时有batch的维度，所以用列表作为输入
            image_values = [np.squeeze(image_values)]

            image = tf.image.decode_jpeg(image_data)
            if image.dtype != tf.float32:
                print('.', end='')
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.compat.v1.image.resize_images(image, [299, 299])
                image_values = sess.run(image)

            # 加载图和变量
            saver = tf.train.import_meta_graph('Model/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(
                'Model/'))
            # 通过名字从图中获取输入占位符
            input_x = graph.get_operation_by_name(
                'BottleneckInputPlaceholder').outputs[0]

            # 我们想要评估的tensors
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

            # 收集预测值
            all_predictions = sess.run(predictions, {input_x: image_values})


            # 打印出预测结果
            index=str(all_predictions)[1]
            print(all_predictions)
            index=int(index)
            print('预测为：'+background_dict[index])

    return background_dict[index]

# 处理好之后的数据文件。
INPUT_DATA = 'test_images_processed_data.npy'

def batch_inference():
    processed_data = np.load(INPUT_DATA, allow_pickle=True)
    test_images = processed_data[4]

    with tf.Graph().as_default() as graph:
        with tf.Session().as_default() as sess:
            # 读取训练好的inception-v3模型
            with tf.gfile.GFile(INCEPTION_MODEL_FILE, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # 加载图和变量
            saver = tf.train.import_meta_graph('Model/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(
                'Model/'))
            # 通过名字从图中获取输入占位符
            input_x = graph.get_operation_by_name(
                'BottleneckInputPlaceholder').outputs[0]

            # 我们想要评估的tensors
            predictions = graph.get_operation_by_name('evaluation/ArgMax').outputs[0]

            # 收集预测值
            all_predictions = sess.run(predictions, {input_x: test_images})

            # 打印出预测结果
            #print('预测为：'+background_dict[index])
    return all_predictions





f = open('answer1.csv', 'w')
start_time = time.time()
for i in range(3407):
    answer = inference('Image_Classification/test/'+str(i)+'.jpg')
    f.write(str(i)+','+answer+'\n')
end_time = time.time()

print(end_time-start_time)

#answers = batch_inference()
#for i in range(len(answers)):
#    f.write(str(i)+','+background_dict[answers[i]]+'\n')

f.close()