# Baseline
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task mmlu --batch_size 4 --is_decoder --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH # You should make your own opensni-7b weights, since it is not loaded from huggingface hub.

# Instructive Decoding

# Null
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --noisy --eps -0.3 --neg_type none --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --noisy --eps -0.3 --neg_type none_options --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
# Opposite
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --noisy --eps -0.3 --neg_type opposite --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --noisy --eps -0.3 --neg_type opposite_options --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name allenai/tk-instruct-3b-def --task mmlu --batch_size 16 --noisy --eps -0.3 --neg_type opposite_full --candidate --max_target_length 1 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH

# Null
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task mmlu --batch_size 4 --is_decoder --noisy --neg_type none --eps -0.3 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task mmlu --batch_size 4 --is_decoder --noisy --neg_type none_options --eps -0.3 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
# Opposite
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task mmlu --batch_size 4 --is_decoder --noisy --neg_type opposite --eps -0.3 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task mmlu --batch_size 4 --is_decoder --noisy --neg_type opposite_options --eps -0.3 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH
CUDA_VISIBLE_DEVICES=0 python src/run_eval.py --model_name weights/opensni-7b --task mmlu --batch_size 4 --is_decoder --noisy --neg_type opposite_full --eps -0.3 --task_files_path data/mmlu --wandb_group YOUR_WANDB_GROUP --wandb_name YOUR_WANDB_NAME --output_path YOUR_OUTPUT_PATH

