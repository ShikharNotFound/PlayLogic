# Training PlayLogic on Hugging Face

Use the Space for the running web demo. Use Hugging Face Jobs or Colab for model training.

## 1. Push this repo first

The training job clones the repo, so push these files before running training:

- `training/train_grpo.py`
- `training/requirements-training.txt`
- `training/playlogic_grpo_colab.ipynb`

## 2. Run a small Hugging Face Job

Replace `YOUR_USERNAME/playlogic-football-grpo` with your model repo name.

```bash
hf auth login

hf jobs run \
  --flavor a10g-small \
  --timeout 4h \
  --secrets HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc "git clone https://github.com/arjune4dev/PlayLogic.git && cd PlayLogic && pip install -q --upgrade pip && pip install -q -r requirements.txt -r training/requirements-training.txt -e . && python training/train_grpo.py --max-steps 50 --push-to-hub --hub-model-id YOUR_USERNAME/playlogic-football-grpo"
```

Start with `--max-steps 5` if you only want to test that the job works.

## 3. Colab submission

Open `training/playlogic_grpo_colab.ipynb` in Colab, use a GPU runtime, and run all cells.

## 4. Bigger training

After the smoke run succeeds, increase:

```bash
--num-resets 16
--max-prompts 256
--max-steps 500
```

For larger models or longer runs, use stronger hardware such as `a10g-large` or `a100-large`.
