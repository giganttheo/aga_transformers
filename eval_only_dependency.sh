
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

python ./train_dependency_only_bias.py \
	--output_dir "./8k-global-dependency-bias-only" \
	--model_name_or_path "gigant/longt5-global-3epoch" \
	--tokenizer_name "gigant/longt5-global-3epoch" \
	--dataset_name="gigant/tib_dependency" \
	--source_prefix "summarize: " \
	--do_eval \
    --do_train \
    --max_train_samples 1 \
	--num_train_epochs 1 \
	--learning_rate 0 \
	--warmup_steps 100 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 14 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 8192 \
	--val_max_target_length 512 \
	--gradient_checkpointing \