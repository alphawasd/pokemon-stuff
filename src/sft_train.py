"""
SFT Training: Qwen2.5-1.5B -> Gen 9 OU Showdown agent
=====================================================

Fine-tunes a small instruct model on the (state -> action) pairs produced by
sft_data_prep.py, using Unsloth + QLoRA so it fits a free T4.

KEY DESIGN DECISIONS (these matter for a paper)
-----------------------------------------------
1. COMPLETION-ONLY LOSS. We mask the prompt tokens and compute loss ONLY on
   the action JSON. Reason: we want the model to learn "given this state,
   produce this action" — NOT to memorise/reproduce the state description.
   Training on the prompt too would waste capacity and bias the model toward
   parroting state text. This is done via DataCollatorForCompletionOnlyLM.

2. PROMPT FORMAT IS FROZEN. The text built here must byte-match what
   sft_data_prep.py wrote AND what the RL environment / inference will send.
   Any drift = train/inference skew = silent performance loss. The prompts
   already live in the JSONL, so we use them verbatim and only append the
   completion + EOS.

3. SMALL MODEL ON PURPOSE. 1.5B is the right size for a T4 and for the RL
   phase later (rollout throughput). A clean SFT->RL improvement on a small
   model is a better paper than a 7B you can barely train once.

USAGE (Colab T4)
----------------
  !pip install unsloth -q
  !python sft_train.py --data sft_train_clean.jsonl --epochs 2

Then it saves a LoRA adapter to ./sft_qwen_showdown/ and runs a quick
sanity generation so you can see the model produce an action.
"""

import argparse
import json
from pathlib import Path


def load_pairs(path: str):
    """Load JSONL produced by sft_data_prep.py into a HF Dataset.

    Uses the prompt-completion dataset format. With completion_only_loss=True
    in SFTConfig, TRL masks the prompt tokens automatically and computes loss
    ONLY on the completion — no manual response-template collator needed.
    """
    from datasets import Dataset
    rows = [json.loads(l) for l in open(path)]
    records = [{"prompt": r["prompt"], "completion": r["completion"]}
               for r in rows]
    return Dataset.from_list(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sft_train_clean.jsonl")
    ap.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    ap.add_argument("--out", default="sft_qwen_showdown")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer, SFTConfig

    # ---- model + LoRA ----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        dtype=None,            # auto (fp16 on T4)
        load_in_4bit=True,     # QLoRA
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    # ---- data ----
    dataset = load_pairs(args.data)
    print(f"[sft] {len(dataset)} training examples")
    print(f"[sft] example prompt:\n{dataset[0]['prompt']}")
    print(f"[sft] example completion: {dataset[0]['completion']}\n")

    # ---- train ----
    # completion_only_loss=True masks the prompt and trains only on the action.
    cfg = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=args.seed,
        max_length=args.max_seq,
        completion_only_loss=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=cfg,
    )

    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[sft] saved LoRA adapter -> {args.out}/")

    # ---- quick sanity generation ----
    # IMPORTANT: use the SAME chat template the trainer applied, or the model
    # sees an out-of-distribution prompt and produces garbage (chatty prose).
    # No trailing "Decision: " — the assistant turn is the sole answer boundary.
    FastLanguageModel.for_inference(model)
    test_prompt = (
        "You are an expert Gen 9 OU Pokemon battler. Given the state, respond "
        'with ONE line of JSON: {"action":"move"|"switch","value":"<name>"}.\n'
        "Turn: 4\n"
        "Your active: Great Tusk (HP 100/100)\n"
        "Your revealed team: Great Tusk, Kingambit, Dragapult, Slowking-Galar, "
        "Gholdengo, Corviknight\n"
        "Opponent active: Gholdengo (HP 100/100)\n"
        "Opponent revealed: Gholdengo, Landorus-Therian"
    )
    messages = [{"role": "user", "content": test_prompt}]
    templated = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(templated, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    print("\n[sanity] model output for a test state:")
    print(text)
    print("\nDoes it produce valid JSON with a plausible action? If yes, SFT "
          "worked. Next: eval move-accuracy on the held-out split.")


if __name__ == "__main__":
    main()
