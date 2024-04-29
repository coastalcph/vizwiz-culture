

## Installation

1. Basic install
```bash
conda create -n vizwiz-culture python=3.10
conda activate vizwiz-culture
pip install -e .
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

### General Usage

### Registering models

You can register new models, datasets, and callbacks by adding them under [src/vlm_inference/configuration/registry.py](src/vlm_inference/configuration/registry.py). We currently support the Google Gemini API, the OpenAI API and HuggingFace.

#### Callbacks

We currently use callbacks for logging, local saving of outputs, and uploading to Wandb.

You can get rid of default callbacks via `'~_callback_dict.<callback_name>'`, e.g. remove the Wandb callback via `'~_callback_dict.wandb'` (mind the quotation marks).

You can also easily override values of the callbacks, e.g. `_callback_dict.wandb.project=new-project`.

### GPT-4V

> [!NOTE]
> Pass `model="gpt-4"` for GPT-4V (gpt-4-1106-vision-preview) and `model="gpt-4-turbo"` for GPT-4 Turbo with Vision (gpt-4-turbo-2024-04-09)

```bash
python run.py \
  model="gpt-4-turbo" \
  model.json_mode=true \
  dataset=cultural_captioning \
  dataset.path=data/xm3600_images \
  dataset.template_name=culture_json
```

### Gemini

> [!NOTE]
> Pass `model="gemini-1.0"` for Gemini 1.0 Pro Vision and `model="gemini-1.5"` for Gemini 1.5 Pro

```bash
python run.py \
  model="gemini-1.5" \
  model.json_mode=true \
  dataset=cultural_captioning \
  dataset.path=data/xm3600_images \
  dataset.template_name=culture_json
```


### HuggingFace Models

#### InstructBLIP (w/ non-JSON continuation template and regular captioning)

```bash
python run.py \
  model="instructblip" \
  model.json_mode=false \
  dataset.path=data/xm3600_images \
  dataset.template_name=continuation
```

#### LLaVa-1.6 (w/ JSON culture template and cultural captioning)

```bash
python run.py \
  model="llava" \
  model.json_mode=true \
  dataset=cultural_captioning \
  dataset.path=data/xm3600_images \
  dataset.template_name=llava7b_culture_json
```

### Running on SLURM

Pass `--multirun run=slurm` to run on SLURM.

> [!IMPORTANT]
> You might need to adjust the Slurm parameters (see defaults in [configs/run/slurm.yaml](configs/run/slurm.yaml)).
> To do so, either change them directly in the `slurm.yaml`, create a new `yaml` file, or pass them as hydra overrides, e.g. via `hydra.launcher.partition=gpu` or `hydra.launcher.gpus_per_node=0`.

You can launch different configurations in parallel using comma-separated arguments, e.g. `model=gemini-1.0,gpt-4`.

Example: 

```bash
python run.py --multirun run=slurm \
  model=gemini-1.0,gpt-4 \
  model.json_mode=true \
  dataset=cultural_captioning \
  dataset.path=data/xm3600_images \
  dataset.template_name=culture_json \
  hydra.sweep.dir=./closed_models_sweep \
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
