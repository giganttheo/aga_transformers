
conda activate train-jax

export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080
cd ~/graph-transformer/aga_transformers
export PATH=/usr/local/cuda-11.2/bin:$PATH.
export PATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$PATH.
export TOKENIZERS_PARALLELISM=false
wandb login

python ./train_FIRE_txt.py \
	--output_dir "./8k-firet5" \
	--model_name_or_path "gigant/flan-t5fire-small" \
	--tokenizer_name "gigant/flan-t5fire-small" \
	--dataset_name="gigant/tib" \
	--source_prefix "summarize: " \
	--do_train \
	--do_eval \
	--num_train_epochs 10 \
	--learning_rate 2e-2 \
	--warmup_steps 100 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 8192 \
	--val_max_target_length 512 \
	--gradient_checkpointing \
	# --max_train_samples 50 \
	# --max_eval_samples 50 \
	# --seed 43 \
