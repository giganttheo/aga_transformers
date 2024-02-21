
# conda activate train-lora
conda activate train-jax #py38

export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080
cd ~/graph-transformer/aga_transformers
export PATH=/usr/local/cuda-11.2/bin:$PATH.
export PATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$PATH.
export TOKENIZERS_PARALLELISM=false
wandb login

# export XLA_PYTHON_CLIENT_PREALLOCATE=false

python ./train_lora_dependency.py \
	--output_dir "./8k-global-dependency-bias" \
	--model_name_or_path "google/long-t5-local-base" \
	--tokenizer_name "google/long-t5-local-base" \
	--dataset_name="gigant/tib_dependency" \
	--source_prefix "summarize: " \
	--do_train \
	--do_eval \
	--num_train_epochs 10 \
	--learning_rate 1e-2 \
	--warmup_steps 100 \
	--per_device_train_batch_size 14 \
	--per_device_eval_batch_size 14 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 8192 \
	--val_max_target_length 512 \
	--gradient_checkpointing \
	--max_train_samples 64 \
	--max_eval_samples 64 \
	# --resume_from_checkpoint \#8192 \
	# --run_id "fv3mirpt"
	# --max_train_samples 64 \
	# --max_eval_samples 64 \