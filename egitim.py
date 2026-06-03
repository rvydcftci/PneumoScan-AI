import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# =========================
# 1. VERİ SETİ AYARLARI
# =========================

base_dir = "chest_xray/chest_xray"
train_dir = base_dir + "/train"

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Sınıf isimleri:", train_data.class_indices)
print("Eğitim ve doğrulama verisi hazır.")

# =========================
# 2. CNN MODELİ
# =========================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# 3. CALLBACKS
# =========================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "pnomoni_modeli.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]

# NORMAL sınıfı az olduğu için ağırlık verildi
class_weights = {
    0: 2.0,  # NORMAL
    1: 1.0   # PNEUMONIA
}

# =========================
# 4. MODEL EĞİTİMİ
# =========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# Modeli kaydet
model.save("pnomoni_modeli.h5")
print("Model eğitildi ve kaydedildi.")

# =========================
# 5. ACCURACY GRAFİĞİ
# =========================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Eğitim Doğruluğu")
plt.plot(history.history["val_accuracy"], label="Doğrulama Doğruluğu")
plt.title("Eğitim ve Doğrulama Doğruluk Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_graph.png")
plt.show()

# =========================
# 6. LOSS GRAFİĞİ
# =========================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Eğitim Kaybı")
plt.plot(history.history["val_loss"], label="Doğrulama Kaybı")
plt.title("Eğitim ve Doğrulama Kayıp Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
plt.show()

print("Accuracy ve Loss grafikleri kaydedildi.")