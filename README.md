

## Installation

1. Basic install
```bash
conda create -n vizwiz-culture python=3.10
conda activate vizwiz-culture
pip install -r requirements.txt
```

2. (Optional) Install flash-attention

```bash
pip install flash-attn --no-build-isolation

# Verify import; if output is empty installation was successful
python -c "import torch; import flash_attn_2_cuda"
```

## OpenAI Setup

1. Login, setup billing, and create API key on [https://platform.openai.com/](https://platform.openai.com/)

2. Run `export OPENAI_API_KEY=<your_key>`

## Gemini Setup

1. Login, setup billing, and create project on [https://console.cloud.google.com/](https://console.cloud.google.com/)

2. Go to [https://cloud.google.com/sdk/docs/install#linux](https://cloud.google.com/sdk/docs/install#linux) and follow the instructions to install `gcloud`

2. Run `gcloud init` and follow the instructions

3. Run `gcloud auth application-default login` and follow the instructions

4. Run `export GOOGLE_CLOUD_PROJECT=<your_project>`

## Examples

### GPT-4V
```bash
python run.py \
  model_name="gpt-4" \
  dataset_path=data/xm3600_images \
  parse_json=true \
  template_name=default_json
```

### Gemini

> [!NOTE]
> Pass `model_name="gemini"` for Gemini 1.0 Pro Vision and `model_name="gemini-1.5"` for Gemini 1.5 Pro

```bash
python run.py \
  model_name="gemini-1.5" \
  dataset_path=data/xm3600_images \
  parse_json=true \
  template_name=default_json
```

### HuggingFace Models

#### InstructBLIP (w/ default JSON template)

```bash
python run.py \
  model_name="instructblip" \
  dataset_path=data/xm3600_images \
  parse_json=true \
  template_name=default_json \
  save_path=outputs/instructblip_outputs
```

#### LLaVa-1.6 (w/ culture template and wandb logging)

```bash
export WANDB_API_KEY=<your_key>
python run.py \
  model_name="llava-v1.6" \
  dataset_path=data/xm3600_images \
  template_name=llava7b_culture \
  save_strategy=WANDB
```

### Running on SLURM

Pass `--multirun run=slurm` to run on SLURM.

> [!IMPORTANT]
> You might need to adjust the Slurm parameters (see defaults in [configs/run/slurm.yaml](configs/run/slurm.yaml)).
> To do so, either change them directly in the `slurm.yaml`, create a new `yaml` file, or pass them as hydra overrides, e.g. via `hydra.launcher.partition=gpu` or `hydra.launcher.gpus_per_node=0`.

You can launch different configurations in parallel using comma-separated arguments, e.g. `model_name=gemini,gpt-4`.

Example: 

```bash
python run.py --multirun run=slurm \
  model_name=gemini,gpt-4 \
  dataset_path=data/xm3600_images \
  parse_json=true \
  template_name=default_json \
  save_strategy=WANDB \
  hydra.sweep.dir=./paid_models_sweep \
  hydra.launcher.gpus_per_node=0 \
  hydra.launcher.cpus_per_task=4 \
  hydra.launcher.mem_gb=4
```

## Data

### XM3600

Download the images to a folder named `xm3600_images` like this:
```bash
mkdir -p xm3600_images
wget -O - https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz | tar -xvzf - -C xm3600_images
```
