# Baseline
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name WeOpenML/Alpaca-7B-v1 --task mmlu --batch_size 4 --is_decoder --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH

# Instructive Decoding
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --noisy --eps -0.3 --neg_type opposite --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name WeOpenML/Alpaca-7B-v1 --task mmlu --batch_size 4 --is_decoder --noisy --neg_type opposite --eps -0.3 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH