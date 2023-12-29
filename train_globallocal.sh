
conda activate gatr-train

export http_proxy=http://webproxy.lab-ia.fr:8080
export https_proxy=http://webproxy.lab-ia.fr:8080
export HTTP_PROXY=http://webproxy.lab-ia.fr:8080
export HTTPS_PROXY=http://webproxy.lab-ia.fr:8080
cd ~/graph-transformer/aga_transformers
export PATH=/usr/local/cuda-11.2/bin:$PATH.


pip install --proxy=http://webproxy.lab-ia.fr:8080 -f https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.19+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl

# CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* jax=0.4.15 cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia
# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade numpy wheel build
# conda install cudnn=8.9 cudatoolkit=11.2.142 -c nvidia
# pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade "jax[cuda11_pip]"==0.4.19 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#  cuda=11.2.142

pip install --proxy=http://webproxy.lab-ia.fr:8080 --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.18+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl jax==0.4.18
# CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia
CONDA_OVERRIDE_CUDA="11.2" conda install jaxlib=*=*cuda* cuda-nvcc cudnn cudatoolkit -c conda-forge -c nvidia



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