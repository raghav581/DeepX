import tensorflow as tf

print("Loading Inception Chest Model...")
model = tf.keras.models.load_model(r"G:\DeepX\models\inceptionv3_chest.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Converting Inception Chest Model...")
tflite_quant_model = converter.convert()
open("inceptionv3_chest.tflite", "wb").write(tflite_quant_model)
print("Done!")

print("Loading Inception CT Model...")
model = tf.keras.models.load_model(r"G:\DeepX\models\inception_ct.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Converting Inception CT Model...")
tflite_quant_model = converter.convert()
open("inception_ct.tflite", "wb").write(tflite_quant_model)
print("Done!")