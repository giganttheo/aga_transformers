export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080

conda activate gatr-train
cd ~/graph-transformer/aga_transformers

conda install cuda=11.2.0 cudnn cudatoolkit-dev -c nvidia
pip install  --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda11_pip]"==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

export PATH=/usr/local/cuda-11.2/bin:$PATH.

nvidia-smi

python ./train_lora.py \
	--output_dir "./lora-t5-graph-small-4k" \
	--model_name_or_path "google/flan-t5-small" \
	--tokenizer_name "google/flan-t5-small" \
	--dataset_name="gigant/tib" \
	--source_prefix "summarize: " \
	--do_train \
	--do_eval \
	--do_predict \
	--predict_with_generate \
	--num_train_epochs 3 \
	--learning_rate 1e-6 \
	--warmup_steps 100 \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 2 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 4096 \
	--val_max_target_length 512