echo "The GCLOUD training starts now!"

echo " "

# Upscale x4

echo "Upscale x4"

python3 train_SRCNN.py --inputDir '../dataset/LR_56/train' --targetDir '../dataset/HR/train' --upscale_factor 4

python3 train_SUB_CNN.py --inputDir '../dataset/LR_56/train' --targetDir '../dataset/HR/train' --upscale_factor 4

python3 train_SRCNN_coord.py --inputDir '../dataset/LR_56/train' --targetDir '../dataset/HR/train' --upscale_factor 4

python3 train_SUB_CNN_coord.py --inputDir '../dataset/LR_56/train' --targetDir '../dataset/HR/train' --upscale_factor 4


# Upscale x2

echo "Upscale x2"

python3 train_SRCNN.py --inputDir '../dataset/LR_112/train' --targetDir '../dataset/HR/train' --upscale_factor 2

python3 train_SUB_CNN.py --inputDir '../dataset/LR_112/train' --targetDir '../dataset/HR/train' --upscale_factor 2

python3 train_SRCNN_coord.py --inputDir '../dataset/LR_112/train' --targetDir '../dataset/HR/train' --upscale_factor 2

python3 train_SUB_CNN_coord.py --inputDir '../dataset/LR_112/train' --targetDir '../dataset/HR/train' --upscale_factor 2


# Now let's train the long ones

echo "Time for the really deep models."

python3 train_VDSR.py --inputDir '../dataset/LR_56/train' --targetDir '../dataset/HR/train' --upscale_factor 4

python3 train_VDSR.py --inputDir '../dataset/LR_112/train' --targetDir '../dataset/HR/train' --upscale_factor 2

echo "End of training!!!"

sudo shutdown -h now