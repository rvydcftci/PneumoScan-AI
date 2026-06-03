import os

base_dir = "chest_xray/chest_xray"

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")

print("Train klasörü var mı?", os.path.exists(train_dir))
print("Test klasörü var mı?", os.path.exists(test_dir))
print("Val klasörü var mı?", os.path.exists(val_dir))

print("Train NORMAL:", len(os.listdir(os.path.join(train_dir, "NORMAL"))))
print("Train PNEUMONIA:", len(os.listdir(os.path.join(train_dir, "PNEUMONIA"))))

print("Test NORMAL:", len(os.listdir(os.path.join(test_dir, "NORMAL"))))
print("Test PNEUMONIA:", len(os.listdir(os.path.join(test_dir, "PNEUMONIA"))))

print("Val NORMAL:", len(os.listdir(os.path.join(val_dir, "NORMAL"))))
print("Val PNEUMONIA:", len(os.listdir(os.path.join(val_dir, "PNEUMONIA"))))