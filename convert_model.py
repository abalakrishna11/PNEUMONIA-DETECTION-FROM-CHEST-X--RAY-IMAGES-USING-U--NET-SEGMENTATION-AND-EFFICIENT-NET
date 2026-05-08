import tensorflow as tf

model_path = r"C:\Users\Balakrishna\Downloads\pneumoai_final (1)\pneumoai_final\models\best_final_v2.keras"

print("Loading model...")
model = tf.keras.models.load_model(model_path, compile=False)

print("Saving as H5...")
model.save("pneumo_model.h5")

print("Conversion complete!")