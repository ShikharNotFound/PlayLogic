import argparse
import json
import os
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from playlogic.models import Action
from playlogic.server.environment import MultiAgentFootball


MOVE_RE = re.compile(
    r"\bMOVE\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\b",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PlayLogic action model with GRPO.")
    parser.add_argument("--model-name", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--output-dir", default="./grpo_football")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--num-resets", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-completion-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(local_obs, team_name):
    return (
        f"You are a {team_name} football player.\n"
        f"Observation: {local_obs}\n"
        "Return exactly one action in this format: MOVE dx dy.\n"
        "Use dx and dy between -8 and 8.\n"
        "Action:"
    )


def collect_prompts(num_resets, max_prompts):
    prompts = []
    env = MultiAgentFootball()
    for _ in range(num_resets):
        obs = env.reset()
        obs_dict = json.loads(obs.text)
        for pid, local_obs in obs_dict.items():
            team_name = "Team A" if int(pid) < 11 else "Team B"
            prompts.append(build_prompt(local_obs, team_name))

    random.shuffle(prompts)
    return prompts[:max_prompts]


def football_action_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion.strip()
        match = MOVE_RE.search(text)

        if not match:
            rewards.append(-1.0)
            continue

        dx = float(match.group(1))
        dy = float(match.group(2))
        score = 1.0

        score += 0.5 if -8.0 <= dx <= 8.0 else -0.75
        score += 0.5 if -8.0 <= dy <= 8.0 else -0.75
        score += 0.25 if abs(dx) + abs(dy) > 0.25 else -0.25

        extra_text = MOVE_RE.sub("", text).strip()
        if extra_text:
            score -= 0.25

        rewards.append(score)

    return rewards


def smoke_test_environment():
    env = MultiAgentFootball()
    env.reset()
    actions = {i: "MOVE 0 0" for i in range(22)}
    _, rewards, done, _ = env.step(Action(text=json.dumps(actions)))
    print("Environment smoke test OK")
    print("Rewards:", rewards)
    print("Done:", done)


def save_reward_plot(trainer, output_dir):
    logs = trainer.state.log_history
    reward_values = [log["reward"] for log in logs if "reward" in log]
    if not reward_values:
        print("No reward values were logged yet.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / "reward_plot.png"

    plt.figure()
    plt.plot(reward_values, marker="o")
    plt.title("GRPO Reward")
    plt.xlabel("Log step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward plot to {plot_path}")


def maybe_login_to_hub():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return

    from huggingface_hub import login

    login(token=token)


def main():
    args = parse_args()
    set_seed(args.seed)
    smoke_test_environment()

    prompts = collect_prompts(args.num_resets, args.max_prompts)
    dataset = Dataset.from_dict({"prompt": prompts})
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=football_action_reward,
        processing_class=tokenizer,
    )

    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    save_reward_plot(trainer, args.output_dir)
    print(f"Saved final model to {final_dir}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set.")
        maybe_login_to_hub()
        trainer.model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)
        print(f"Pushed model to https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()