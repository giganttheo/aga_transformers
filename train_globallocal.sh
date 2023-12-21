export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080
python ./train_lora.py \
	--output_dir "./lora-t5-graph-base-16k" \
	--model_name_or_path "google/flan-t5-base" \
	--tokenizer_name "google/flan-t5-base" \
	--dataset_name="gigant/tib" \
	--source_prefix "summarize: " \
	--do_train \
	--do_eval \
	--do_predict \
	--predict_with_generate \
	--num_train_epochs 3 \
	--learning_rate 1e-6 \
	--warmup_steps 10 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 16384 \
	--val_max_target_length 512