echo "The GCLOUD training starts now!"

echo " "

# Upscale x4

python3 train_SRCNN.py --inputDir '../dataset/300W-3D-crap-56/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 4

python3 train_SUB_CNN.py --inputDir '../dataset/300W-3D-crap-56/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 4

python3 train_VDSR.py --inputDir '../dataset/300W-3D-crap-56/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 4

python3 train_SRGAN.py --inputDir '../dataset/300W-3D-crap-56/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 4

# Upscale x2

python3 train_SRCNN.py --inputDir '../dataset/300W-3D-crap-112/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 2

python3 train_SUB_CNN.py --inputDir '../dataset/300W-3D-crap-112/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 2

python3 train_VDSR.py --inputDir '../dataset/300W-3D-crap-112/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 2

python3 train_SRGAN.py --inputDir '../dataset/300W-3D-crap-112/train' --targetDir '../dataset/300W-3D-low-res-224/train' --upscale_factor 2

