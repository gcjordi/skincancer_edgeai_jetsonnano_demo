# skincancer_edgeai_jetsonnano_demo
Skin Cancer Detector - Edge AI classifier - Jetson Nano (DEMO)

The purpose of this repo is to be a demo for the "Certification Jetson AI Specialist"

It's a demo (no medical uses at the moment)

It's a demo for Edge AI Computing based on NVIDIA JETSON NANO 2GB DEVELOPER KIT

Based on the structure: https://github.com/dusty-nv/jetson-inference

Dataset: HAM10000 (only part for demo)

Functionality: classify skin lesions (malignant cancer or benign tumor) using Edge AI Computing, with a simple image taken by the user.

P.S: It can also work in WebApp: https://github.com/gcjordi/skinlesionanalyzer_webapp_jgc

RUN (in terminal) (ONLY TEST MODE, NOT REAL MEDICAL TEST):

cd jetson-inference/

docker/run.sh

cd python/training/classification

python3 train.py --model-dir=models/tools --batch-size=4 --workers=1 --epochs=1 data/skin/

python3 onnx_export.py --model-dir=models/skin (previously renamed name folder: tools to skin)

imagenet --model=models/skin/resnet18.onnx --labels=data/skin/labels.txt --input_blob=input_0 --output_blob=output_0 /dev/video0
