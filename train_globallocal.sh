
# conda activate gatr-train

# conda activate train-jax

# export http_proxy=http://webproxy.lab-ia.fr:8080
# export https_proxy=http://webproxy.lab-ia.fr:8080
# export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
# export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080
# cd ~/graph-transformer/aga_transformers
# export PATH=/usr/local/cuda-11.2/bin:$PATH.
# export PATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$PATH.


# export PATH=/usr/local/cuda-10.2/bin:$PATH.
# export PATH=/usr/local/cuda-10.2/targets/x86_64-linux/include:$PATH.
# export PATH=/usr/local/cuda/bin:$PATH.


# export PATH=~/miniconda3/pkgs/cudnn-8.9.2.26-cuda11_0/include:$PATH.

# # export PATH=~/miniconda3/envs/train-jax/bin:$PATH.

# CONDA_OVERRIDE_CUDA="10.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia


# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda102]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# CONDA_OVERRIDE_CUDA="10.2" conda install cudnn cudatoolkit cuda-nvcc=10.2 -c nvidia -c conda-forge

# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda112]"==0.4.15 jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# # pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade git+https://github.com/giganttheo/qax.git@compat-py38


# pip install --proxy=http://webproxy.lab-ia.fr:8080 https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.19+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl

# pip install --proxy=http://webproxy.lab-ia.fr:8080 en_core_web_trf spacy

# # CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* jax=0.4.15 cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia
# # pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade numpy wheel build
# # conda install cudnn=8.9 cudatoolkit=11.2.142 -c nvidia
# # pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda11_pip]"==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# #  cuda=11.2.142

#CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* jax=0.4.16 cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia

# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda11_pip]"==0.4.15 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda11_pip]"==0.4.15 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda11_local]"==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# conda install cudnn cudatoolkit cuda-nvcc -c nvidia -c conda-forge

# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl jax==0.4.13
# # CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia
# CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia
# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade jaxlib jax==0.4.13

# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade jaxlib jax==0.4.15
# pip install --proxy=http://webproxy.lab-ia.fr:8080 flash-attention-jax

# CONDA_OVERRIDE_CUDA="11.2" conda install cuda=11.2 jaxlib=*=*cuda* jax cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia


### works?

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

# CONDA_OVERRIDE_CUDA="10.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia

# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export JAX_NUMPY_RANK_PROMOTION=warn
# export XLA_PYTHON_CLIENT_PREALLOCATE=false


# CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia


#XLA performance flags recommended by https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#xla-performance-flags

# export XLA_FLAGS='--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true --xla_gpu_enable_async_collectives=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_highest_priority_async_stream=true'

# export XLA_FLAGS='--xla_gpu_triton_gemm_any=true'

# export XLA_FLAGS=''

# ###
# nvidia-smi

# python ./train_lora.py \
# 	--output_dir "./lora-t5-graph-base-16k" \
# 	--model_name_or_path "google/flan-t5-base" \
# 	--tokenizer_name "google/flan-t5-base" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--num_train_epochs 3 \
# 	--learning_rate 1e-3 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 16384 \
# 	--val_max_target_length 512 \
# 	--max_train_samples 64 \
# 	--max_eval_samples 64 \
# 	--gradient_checkpointing #\
# 	--resume_from_checkpoint \
# 	--run_id "294lkdvh"

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

# #--predict_with_generate \

# python ./train_lora.py \
# 	--output_dir "./lora-t5-graph-base-8k" \
# 	--model_name_or_path "google/flan-t5-base" \
# 	--tokenizer_name "google/flan-t5-base" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--do_predict \
# 	--predict_with_generate \
# 	--num_train_epochs 3 \
# 	--learning_rate 1e-6 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 1 \
# 	--per_device_eval_batch_size 1 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 8192 \
# 	--val_max_target_length 512


# python ./train_lora.py \
# 	--output_dir "./lora-t5-graph-small-4k" \
# 	--model_name_or_path "google/flan-t5-small" \
# 	--tokenizer_name "google/flan-t5-small" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--do_predict \
# 	--predict_with_generate \
# 	--num_train_epochs 3 \
# 	--learning_rate 1e-6 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 2 \
# 	--per_device_eval_batch_size 2 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 4096 \
# 	--val_max_target_length 512 \
# 	--max_train_samples 100 \
# 	--max_eval_samples 100 \
# 	--max_predict_samples 100


# python ./train_lora.py \
# 	--output_dir "./lora-t5-graph-small-8k" \
# 	--model_name_or_path "google/flan-t5-small" \
# 	--tokenizer_name "google/flan-t5-small" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--num_train_epochs 3 \
# 	--learning_rate 1e-3 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 16 \
# 	--per_device_eval_batch_size 16 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 8192 \
# 	--val_max_target_length 512 \
# 	--max_eval_samples 32 \
# 	--gradient_checkpointing

# python ./train_lora.py \
# 	--output_dir "./lora-t5-graph-base-8k" \
# 	--model_name_or_path "google/flan-t5-base" \
# 	--tokenizer_name "google/flan-t5-base" \
# 	--dataset_name="gigant/tib" \
# 	--source_prefix "summarize: " \
# 	--do_train \
# 	--do_eval \
# 	--num_train_epochs 2 \
# 	--learning_rate 1e-2 \
# 	--warmup_steps 100 \
# 	--per_device_train_batch_size 10 \
# 	--per_device_eval_batch_size 10 \
# 	--overwrite_output_dir \
# 	--dtype "bfloat16" \
# 	--max_target_length 512 \
# 	--max_source_length 8192 \
# 	--val_max_target_length 512 \
# 	--max_eval_samples 20 \
# 	--gradient_checkpointing #\
	# --resume_from_checkpoint \
	# --run_id "294lkdvh"

python ./train_lora.py \
	--output_dir "./8k-global-local" \
	--model_name_or_path "google/long-t5-local-base" \
	--tokenizer_name "google/long-t5-local-base" \
	--dataset_name="gigant/tib" \
	--source_prefix "summarize: " \
	--do_train \
	--do_eval \
	--num_train_epochs 2 \
	--learning_rate 1e-2 \
	--warmup_steps 100 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--overwrite_output_dir \
	--dtype "bfloat16" \
	--max_target_length 512 \
	--max_source_length 8192 \
	--val_max_target_length 512 \
	--gradient_checkpointing \
	--resume_from_checkpoint \
	--run_id "2sgjrcax"
	# --max_train_samples 50 \
	# --max_eval_samples 50 \
	# --seed 43 \


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