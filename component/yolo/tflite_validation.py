import tensorflow as tf

# Load the TFLite model
model_path = "../../train/PTQ_384_640/best_saved_model/best_integer_quant.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input and output information
print("Input Details:")
for detail in input_details:
    print(f"Name: {detail['name']}, Dtype: {detail['dtype']}, Shape: {detail['shape']}")

print("\nOutput Details:")
for detail in output_details:
    print(f"Name: {detail['name']}, Dtype: {detail['dtype']}, Shape: {detail['shape']}")



