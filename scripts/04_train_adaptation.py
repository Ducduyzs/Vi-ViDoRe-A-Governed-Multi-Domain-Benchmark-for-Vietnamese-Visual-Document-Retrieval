import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from PIL import Image

from src.config import PathConfig, ModelConfig, TrainConfig
from src.data.schema import PageMetadata, QueryItem, QrelItem, BenchmarkSplit
from src.training.dataset import ViViDoReDataset
from src.training.hard_negative_miner import HardNegativeMiner
from src.training.loss import LateInteractionInfoNCELoss
from src.models.maxsim import maxsim_pytorch


def collate_fn(batch):
    """Collate function for DataLoader to handle variable-length sequences."""
    query_ids = [item["query_id"] for item in batch]
    query_texts = [item["query_text"] for item in batch]
    pos_page_ids = [item["pos_page_id"] for item in batch]
    pos_images = [item["pos_image"] for item in batch]
    neg_page_ids = [item["neg_page_ids"] for item in batch]
    neg_images = [item["neg_images"] for item in batch]
    domains = [item["domain"] for item in batch]
    
    return {
        "query_ids": query_ids,
        "query_texts": query_texts,
        "pos_page_ids": pos_page_ids,
        "pos_images": pos_images,
        "neg_page_ids": neg_page_ids,
        "neg_images": neg_images,
        "domains": domains,
    }


def encode_queries_with_model(model, processor, queries, device, batch_size=4, torch_dtype=torch.bfloat16):
    """Encode queries using the model directly (with gradients if model.train())."""
    all_embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        # Create dummy images for PaliGemma processor
        dummy_image = Image.new("RGB", (16, 16), color=(255, 255, 255))
        dummy_images = [dummy_image] * len(batch)
        
        inputs = processor(
            text=batch,
            images=dummy_images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        # Forward pass WITH gradients
        outputs = model(**inputs, output_hidden_states=True)
        # Use last hidden state as multi-vector representation
        embeds = outputs.hidden_states[-1]  # (B, N_q, D)
        all_embeddings.append(embeds)
    
    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    return torch.empty(0, device=device)


def encode_documents_with_model(model, processor, images, device, batch_size=2, torch_dtype=torch.bfloat16):
    """Encode documents using the model directly (with gradients if model.train())."""
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(
            images=batch, return_tensors="pt"
        ).to(device)
        
        # Forward pass WITH gradients
        outputs = model(**inputs, output_hidden_states=True)
        embeds = outputs.hidden_states[-1]  # (B, N_d, D)
        all_embeddings.append(embeds)
    
    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    return torch.empty(0, device=device)


def main():
    parser = argparse.ArgumentParser(description="Step 4: Train Vietnamese Late Interaction Adaptation with LoRA.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--backbone", type=str, default="vidore/colpali-v1.2", help="HuggingFace model ID")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    paths = PathConfig()
    train_dir = paths.benchmark_dir / "train"
    dev_dir = paths.benchmark_dir / "dev"

    if not train_dir.exists():
        print(f"[!] Training split not found at {train_dir}. Please run scripts/02_generate_benchmark.py first.")
        return

    # Load training queries and corpus metadata
    train_queries = []
    with open(train_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_queries.append(QueryItem.from_dict(json.loads(line)))

    meta_path = paths.processed_dir / "all_pages_metadata.jsonl"
    all_pages = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_pages.append(PageMetadata.from_dict(json.loads(line)))

    print(f"[*] Training Adaptation on {len(train_queries)} queries and {len(all_pages)} total corpus pages.")

    # Initialize Negative Miner
    miner = HardNegativeMiner(all_pages)
    dataset = ViViDoReDataset(
        queries=train_queries,
        corpus_pages=all_pages,
        negative_miner=miner,
        num_hard_negatives=3,
    )

    # DataLoader with batching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    # Load model and processor directly (bypass retriever for training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"[*] Loading backbone: {args.backbone} on {device}...")
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    
    processor = AutoProcessor.from_pretrained(args.backbone, trust_remote_code=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.backbone,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = LateInteractionInfoNCELoss(temperature=0.05)

    # Learning rate scheduler
    total_steps = len(dataloader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print("\n[+] Starting LoRA Vietnamese Retrieval Adaptation...")
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            query_texts = batch["query_texts"]
            pos_images = batch["pos_images"]
            neg_images_list = batch["neg_images"]  # List of lists
            
            # Encode queries and positive documents WITH gradients
            q_embeds = encode_queries_with_model(model, processor, query_texts, device, args.batch_size, torch_dtype)
            doc_embeds = encode_documents_with_model(model, processor, pos_images, device, args.batch_size, torch_dtype)
            
            # Encode hard negatives if available
            neg_doc_embeds = None
            if any(len(negs) > 0 for negs in neg_images_list):
                # Flatten all negative images
                all_neg_images = []
                neg_counts = []
                for negs in neg_images_list:
                    neg_counts.append(len(negs))
                    all_neg_images.extend(negs)
                
                if all_neg_images:
                    all_neg_embeds = encode_documents_with_model(
                        model, processor, all_neg_images, device, args.batch_size, torch_dtype
                    )
                    
                    # Reshape to (B, K_neg, N_d, D)
                    B = len(query_texts)
                    max_negs = max(neg_counts) if neg_counts else 0
                    if max_negs > 0:
                        neg_doc_embeds = torch.zeros(
                            B, max_negs, all_neg_embeds.shape[1], all_neg_embeds.shape[2],
                            device=device, dtype=all_neg_embeds.dtype
                        )
                        idx = 0
                        for i, count in enumerate(neg_counts):
                            for j in range(count):
                                neg_doc_embeds[i, j] = all_neg_embeds[idx]
                                idx += 1

            # Compute loss
            loss = loss_fn(q_embeds, doc_embeds, neg_doc_embeds)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"[!] NaN loss detected at epoch {epoch}, batch {batch_idx}. Skipping...")
                optimizer.zero_grad()
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / args.grad_accum
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * args.grad_accum
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_loss = total_loss / max(1, num_batches)
        print(f"[*] Epoch {epoch} Complete - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_dir = paths.checkpoints_dir / f"vi_colpali_epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        print(f"[+] Saved checkpoint to: {ckpt_dir}")

        # Dev evaluation placeholder
        if dev_dir.exists():
            print(f"[*] Running dev evaluation...")
            # TODO: Add dev evaluation

    print("\n[+] Training adaptation pipeline completed successfully!")


if __name__ == "__main__":
    main()