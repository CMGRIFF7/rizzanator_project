# convert_model.py

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

def main():
    # Load the trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../training/trained_model')
    model = TFAutoModelForSeq2SeqLM.from_pretrained('../training/trained_model')

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optional: Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open('rizzanator_model.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    main()
