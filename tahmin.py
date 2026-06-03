import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("pnomoni_modeli.h5")

img_path = "chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("Tahmin: PNEUMONIA")
    print("Güven oranı:", prediction)
else:
    print("Tahmin: NORMAL")
    print("Güven oranı:", 1 - prediction)