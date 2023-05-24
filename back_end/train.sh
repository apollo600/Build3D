# python prepare_phototourism.py --root_dir data --img_downscale 8

python train.py \
  --root_dir data/brandenburg_gate --dataset_name phototourism \
  --img_downscale 2 --use_cache --N_importance 64 --N_samples 64 \
  --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
  --num_epochs 40 --batch_size 3600 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name brandenburg_gate_2_v1 \
  --num_gpus 1 \
  # --ckpt_path ckpts/brandenburg_scale2_nerfw/epoch=19.ckpt \