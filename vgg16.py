from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import os
from keras import optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.applications import VGG16
import time
import tensorflow as tf


#關閉GPU加速功能
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#計算爬蟲程式執行時間(起點)
start = time.time()

#TensorFlow 默認情況下會映射幾乎所有GPU內存，所以需在運行時分配內存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	# Currently, memory growth needs to be the same across GPUs
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)

# 器材照片預處理(分訓練集和測試集)
train_dir = 'vgg16/train'
val_dir = 'vgg16/test'

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 不使用數據增強
# train_datagen = ImageDataGenerator(rescale=1. / 255)
# val_datagen = ImageDataGenerator(rescale=1. / 255)

# 使用數據增強
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40., width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)


# 使用迭代器生成圖片張量
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=20)
val_gen = train_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=20)

# 微调模型，凍結VGG16部分層
conv_base.trainable = False
flag = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        flag = True
    if flag:
        layer.trainable = True

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))

# 使用dropout
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

# 編譯模型
model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy', metrics=['acc'])
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# 訓練模型
history = model.fit_generator(train_gen, epochs=30, validation_data=val_gen)

# 儲存訓練後模型
model.save('model_CnnModelTrainWorkout_v2.h5')
# model.save('E:/KaggleDatas/idenprof-jpg/idenprof/identprof.h5')

#計算爬蟲程式執行時間(終點)
end = time.time()
spend = end - start
hour = spend // 3600
minu = (spend - 3600 * hour) // 60
sec = spend - 3600 * hour - 60 * minu
print(f'一共花費了{hour}小時{minu}分鐘{sec}秒')

# 绘制训练精度损失曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc,'r',color ='green', label='training acc')
plt.plot(epochs, val_acc, '--r', label='val acc')
plt.title('training & val accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r',color = 'green', label='training loss')
plt.plot(epochs, val_loss, '--r', label='val loss')
plt.title('training & val loss')
plt.legend()

plt.show()