### installation
conda create -n adlcv-ex-32 python=3.10
conda activate adlcv-ex-32
module load cuda/11.8

```
h100sh 
module load python3/3.10.12

python3 -m venv venv
source venv/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers scipy
pip install 'numpy<2'
pip install matplotlib wandb tqdm lovely-tensors einops
pip install accelerate
pip install filelock
```


### References
[https://huggingface.co/blog/stable_diffusion](stable diffusion tutorial)