import os

base = "data/fer2013/train"

print("FER-2013 dataset check:\n")

for emotion in sorted(os.listdir(base)):
    path = os.path.join(base, emotion)
    if os.path.isdir(path):
        print(emotion, "->", len(os.listdir(path)))
