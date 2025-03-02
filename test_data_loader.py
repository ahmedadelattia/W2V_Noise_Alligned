from dataloader import SimClassDataset
from train import data_collator
from torch.utils.data import DataLoader
import time
dataset = SimClassDataset(split="development")

# Create DataLoader with a small batch size and no additional workers (for debugging)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=data_collator
)

print("Starting DataLoader check...")
start_time = time.time()
for i, batch in enumerate(dataloader):
    print(f"Batch {i+1}:")
    if batch["input_values_noisy"] is not None:
        print("  input_values_noisy shape:", batch["input_values_noisy"].shape)
    if batch["input_values_clean"] is not None:
        print("  input_values_clean shape:", batch["input_values_clean"].shape)
    # Print a couple of transcript examples from the batch
    if batch["transcript"] is not None:
        print("  transcripts:", batch["transcript"][:2])
    # Stop after 3 batches
    if i >= 2:
        break
end_time = time.time()
print("DataLoader check finished in {:.2f} seconds.".format(end_time - start_time))

if __name__ == "__main__":
    ()