# This code uses the decoder loss directly.
#
#

# Standard library
import os
import random

import args

# Deep learning
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader

# Parallel
from torch.utils.data.distributed import DistributedSampler
from torch_optimizer.lamb import Lamb
from trainer import TrainerDirectDecoder

# Data
from utils import MoleculeModule, get_optim_groups


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_train_objs(config):
    # load data
    train_loader = MoleculeModule(
        config.max_len,
        config.train_load,
        config.data_root,
        output_data_path=config.output_data_root,
    )
    train_loader.setup()

    loader = DataLoader(
        train_loader.pubchem,
        batch_size=config.n_batch,
        pin_memory=True,
        shuffle=False,
        collate_fn=train_loader.text_encoder.process,
        num_workers=config.n_workers,
    )

    # Get output dataset if in paired mode
    output_dataset = train_loader.get_output_dataset()

    # Verify paired mode status
    if config.output_data_root and config.output_data_root != "":
        if output_dataset is not None:
            print(
                f"✅ Paired mode active: Input dataset ({len(train_loader.pubchem)} samples), "
                f"Output dataset ({len(output_dataset)} samples)"
            )

            # Sample a few SMILES from both datasets to confirm they're different
            if len(train_loader.pubchem) > 0 and len(output_dataset) > 0:
                sample_idx = min(5, len(train_loader.pubchem) - 1)
                print(
                    f"\nInput SMILES example: {train_loader.pubchem[sample_idx]['text']}"
                )
                print(f"Output SMILES example: {output_dataset[sample_idx]['text']}\n")
        else:
            print(
                f"❌ Paired mode requested but output dataset could not be loaded. "
                f"Using regular (non-paired) training mode."
            )

    # load model
    if config.smi_ted_version == "v1":
        from smi_ted_light.load import Smi_ted
    elif config.smi_ted_version == "v2":
        from smi_ted_large.load import Smi_ted

    model = Smi_ted(config, train_loader.get_vocab()).to("cuda")

    # Handle different initialization modes
    if config.init_mode == "weights_only" and config.init_weights_from:
        print(f"Loading only model weights from {config.init_weights_from}")
        loc = "cuda" if torch.cuda.is_available() else "cpu"
        weights = torch.load(config.init_weights_from, map_location=loc)
        model.load_state_dict(weights["MODEL_STATE"])
    elif config.init_mode == "full_checkpoint" and config.load_checkpoint_path:
        print(f"Loading full checkpoint from {config.load_checkpoint_path}")
        # Trainer handles this case
        pass
    else:  # scratch
        print("Initializing model weights from scratch")
        model.apply(model._init_weights)

    # load optimizer
    optim_groups = get_optim_groups(model)
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.lr_decoder, betas=(0.9, 0.99), fused=True
    )

    return loader, output_dataset, model, optimizer


def main(
    config,
    save_every: int,
    total_epochs: int,
    save_checkpoint_path: str,
    load_checkpoint_path: str,
):
    # ddp_setup()

    # training objects
    train_loader = MoleculeModule(
        config.max_len,
        config.train_load,
        config.data_root,
        output_data_path=config.output_data_root,
    )
    train_loader.setup()

    # Get the datasets
    if train_loader.paired_mode:

        def paired_collate(batch):
            # Get corresponding output samples
            output_batch = train_loader.output_dataset.select(range(len(batch)))
            # # DEBUG
            # print(f"\nProcessing batch with {len(batch)} pairs")
            # if random.random() < 0.3:  # 30% chance to print sample
            #     idx = random.randint(0, len(batch) - 1)
            #     print(f"Sample pair - Input: {batch[idx]['text']}")
            #     print(f"          -> Output: {output_batch[idx]['text']}")
            return train_loader.text_encoder.process_with_output(batch, output_batch)

        train_data = DataLoader(
            train_loader.pubchem,
            batch_size=config.n_batch,
            pin_memory=True,
            shuffle=False,
            collate_fn=paired_collate,
            num_workers=0,  # Keep at 0 for debugging
        )
    else:
        # Regular non-paired mode
        train_data = DataLoader(
            train_loader.pubchem,
            batch_size=config.n_batch,
            pin_memory=True,
            shuffle=False,
            collate_fn=train_loader.text_encoder.process,
            num_workers=config.n_workers,
        )

    # Get output dataset if in paired mode
    output_dataset = train_loader.get_output_dataset()

    # Verify paired mode status and output examples
    if config.output_data_root and config.output_data_root != "":
        if output_dataset is not None:
            print(
                f"✅ Paired mode active: Input dataset ({len(train_loader.pubchem)} samples), "
                f"Output dataset ({len(output_dataset)} samples)"
            )
        else:
            print(
                f"❌ Paired mode requested but output dataset could not be loaded. "
                f"Using regular (non-paired) training mode."
            )

    # load model
    if config.smi_ted_version == "v1":
        from smi_ted_light.load import Smi_ted
    elif config.smi_ted_version == "v2":
        from smi_ted_large.load import Smi_ted

    model = Smi_ted(config, train_loader.get_vocab()).to("cuda")

    # Handle different initialization modes
    if config.init_mode == "weights_only" and config.init_weights_from:
        print(f"Loading only model weights from {config.init_weights_from}")
        loc = "cuda" if torch.cuda.is_available() else "cpu"
        weights = torch.load(config.init_weights_from, map_location=loc)
        model.load_state_dict(weights["MODEL_STATE"])
    elif config.init_mode == "full_checkpoint" and config.load_checkpoint_path:
        print(f"Loading full checkpoint from {config.load_checkpoint_path}")
        # Trainer handles this case
        pass
    else:  # scratch
        print("Initializing model weights from scratch")
        model.apply(model._init_weights)

    # load optimizer
    optim_groups = get_optim_groups(model)
    optimizer = torch.optim.AdamW(
        optim_groups, lr=config.lr_decoder, betas=(0.9, 0.99), fused=True
    )

    # init trainer - pass the text_encoder
    trainer = TrainerDirectDecoder(
        model,
        train_data,
        optimizer,
        save_every,
        save_checkpoint_path,
        load_checkpoint_path,
        config,
        text_encoder=train_loader.text_encoder,
    )
    trainer.train(total_epochs)
    # destroy_process_group()


if __name__ == "__main__":
    parser = args.get_parser()
    args = parser.parse_args()
    main(
        args,
        args.checkpoint_every,
        args.max_epochs,
        save_checkpoint_path=args.save_checkpoint_path,
        load_checkpoint_path=args.load_checkpoint_path,
    )
