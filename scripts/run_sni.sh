# Baseline
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task super_ni --batch_size 16 --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task super_ni --batch_size 16 --is_decoder --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH


# Instructive Decoding
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task super_ni --batch_size 16 --noisy --eps -0.3 --neg_type opposite --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task super_ni --batch_size 16 --noisy --eps -0.3 --neg_type opposite --is_decoder --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH