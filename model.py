import numpy as np
import tensorflow as tf

class Model:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def test(self, image_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], image_data.unsqueeze(0))
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(output_data[0])