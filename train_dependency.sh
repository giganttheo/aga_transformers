
conda activate train-jax

export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080
cd ~/graph-transformer/aga_transformers
export PATH=/usr/local/cuda-11.2/bin:$PATH.
export PATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$PATH.
export TOKENIZERS_PARALLELISM=false

# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# python ./train_lora.py \
# 	--output_dir "./lora-t5-blockgraph-base-8k" \
# 	--model_name_or_path "google/flan-t5-base" \
# 	--tokenizer_name "google/flan-t5-base" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--num_train_epochs 3 \
# 	--learning_rate 1e-1 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 14 \
# 	--per_device_eval_batch_size 14 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 8192 \
# 	--val_max_target_length 512 \
# 	--max_train_samples 64 \
# 	--max_eval_samples 64 \
# 	--gradient_checkpointing

python ./train_lora_graph.py \
	--output_dir "./8k-global-dependency" \
	--model_name_or_path "google/flan-t5-base" \
	--tokenizer_name "google/flan-t5-base" \
	--dataset_name="gigant/tib_dependency" \
	--source_prefix "summarize: " \
	--do_train \
	--do_eval \
	--num_train_epochs 4 \
	--learning_rate 1e-2 \
	--warmup_steps 100 \
	--per_device_train_batch_size 6 \
	--per_device_eval_batch_size 6 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 8192 \
	--val_max_target_length 512 \
	--gradient_checkpointing \
	# --resume_from_checkpoint \
	# --run_id "fv3mirpt"

# python ./train_lora.py \
# 	--output_dir "./lora-t5-graph-small-8k" \
# 	--model_name_or_path "google/flan-t5-small" \
# 	--tokenizer_name "google/flan-t5-small" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--num_train_epochs 1 \
# 	--learning_rate 1e-2 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 20 \
# 	--per_device_eval_batch_size 20 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 8192 \
# 	--val_max_target_length 512 \
# 	--max_eval_samples 60 \
# 	--gradient_checkpointing \
# 	--seed 45 \
# 	--resume_from_checkpoint \
# 	--run_id "qoftnzim"

# python ./train_lora.py \
# 	--output_dir "./_tmp" \
# 	--model_name_or_path "google/flan-t5-base" \
# 	--tokenizer_name "google/flan-t5-base" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--num_train_epochs 4 \
# 	--learning_rate 1e-2 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 6 \
# 	--per_device_eval_batch_size 6 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 8192 \
# 	--val_max_target_length 512 \
# 	--seed 43 \
# 	--gradient_checkpointing