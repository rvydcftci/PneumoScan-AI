import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

base_dir = "chest_xray/chest_xray"
test_dir = base_dir + "/test"

img_size = 224
batch_size = 32

model = load_model("pnomoni_modeli.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype("int32").ravel()

true_classes = test_data.classes
class_names = list(test_data.class_indices.keys())

print("Sınıf isimleri:", test_data.class_indices)
print("Accuracy:", accuracy_score(true_classes, predicted_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))