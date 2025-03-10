from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import time


model_path = 'animal_classifier_model.h5'
model = load_model(model_path)
print("\033[92mModel loaded successfully.\033[0m")


json_path = 'class_names.json'
if not os.path.exists(json_path):
    raise FileNotFoundError(f"Error: JSON file '{json_path}' not found. Please provide the class names file.")
with open(json_path, 'r') as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())
names = ''
for name in class_names:
    names += f'{name}\n'

print(f"Only predicts within these {len(class_names)} class names:\n \033[96m{names}\033[0m")
print("\033[93minput of another animal images will gives unexpected results.!\033[0m")


def predict_animal(image_path):
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 

    predictions = model.predict({'input_layer_2': img_array})
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    return predicted_class_name, confidence


if __name__ == "__main__":
    while True:
        image_path = input('''
copy the path of image you want to get predictions.(e.g.,'downloads/images/test_image.jpg)
        
\033[1;95m enter the path of image:\033[0;0m ''')

        if not os.path.exists(image_path):
            print(f"\033[91mError: Image file '{image_path}' not found.\033[0m")
        else:
            animal_name, confidence = predict_animal(image_path)
            print(f"Predicted Animal:\033[92m {animal_name}\033[0m")
            print(f"Confidence:\033[92m {confidence:.2f}%\033[0m")


            print("\n\n\n____Advaced answer____")

            preds = model.predict({
                'input_layer_2': preprocess_input(
                    np.expand_dims(
                        image.img_to_array(
                            image.load_img(image_path, target_size=(224, 224))
                        ),
                        axis=0
                    )
                )
            })
            top_3_idx = np.argsort(preds[0])[-3:][::-1]
            print("\nTop 3 Predictions:")
            for idx in top_3_idx:
                print(f"{class_names[idx]}:\033[92m {preds[0][idx]*100:.2f}%\033[0m")


            print("Enter 0 to Exit")
            print("Enter any other number to Restart")

            choice = input("Enter your choice: ").strip()

            if choice == '0':
                print("Exiting.")
                time.sleep(1)
                print("Exiting..")
                time.sleep(1)
                print("Exiting...")
                time.sleep(1)
                break
            else:
                os.system('cls' if os.name == 'nt' else 'clear')

