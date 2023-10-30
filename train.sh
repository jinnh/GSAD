## train uncertainty model
python train.py -uncertainty --config config/llie_train_u.json --dataset ./config/lolv1.yml 

## train global structure-aware diffusion
python train.py --config config/lolv1_train.json --dataset ./config/lolv1.yml