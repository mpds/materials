# Deep learning
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from fast_transformers.masking import LengthMask
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Standard library
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        # self.local_rank = int(os.environ["LOCAL_RANK"])
        # self.global_rank = int(os.environ["RANK"])
        self.model = model.to("cuda")  # .to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.last_batch_idx = -1
        self.save_checkpoint_path = save_checkpoint_path
        self.config = config

        if os.path.exists(load_checkpoint_path):
            print(f"Loading checkpoint at {load_checkpoint_path}...")
            self._load_checkpoint(load_checkpoint_path)

        # self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_checkpoint(self, checkpoint_path):
        if self.config.init_mode != "full_checkpoint":
            print(
                f"Skipping checkpoint loading because init_mode is {self.config.init_mode}"
            )
            return

        opt_dict = None
        # loc = f"cuda:{self.local_rank}"
        loc = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_dict = torch.load(checkpoint_path, map_location=loc)
        if os.path.exists(
            os.path.join(self.save_checkpoint_path, "OPTIMIZER_STATES.pt")
        ):
            opt_dict = torch.load(
                os.path.join(self.save_checkpoint_path, "OPTIMIZER_STATES.pt"),
                map_location=loc,
            )

        self.model.load_state_dict(ckpt_dict["MODEL_STATE"])
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict["OPTIMIZER_STATE"])
            print("Optimizer states restored!")

        self.last_batch_idx = (
            ckpt_dict["last_batch_idx"] if "last_batch_idx" in ckpt_dict else -1
        )
        self.epochs_run = (
            ckpt_dict["EPOCHS_RUN"] + 1
            if self.last_batch_idx == -1
            else ckpt_dict["EPOCHS_RUN"]
        )

        # load RNG states each time the model and states are loaded from checkpoint
        if "rng" in ckpt_dict:
            rng = ckpt_dict["rng"]
            for key, value in rng.items():
                if key == "torch_state":
                    torch.set_rng_state(value.cpu())
                elif key == "cuda_state":
                    torch.cuda.set_rng_state(value.cpu())
                elif key == "numpy_state":
                    np.random.set_state(value)
                elif key == "python_state":
                    random.setstate(value)
                else:
                    print("unrecognized state")

        print(f"Resuming training from checkpoint at Epoch {self.epochs_run}.")

    def _save_checkpoint(self, epoch, config, last_idx):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict["torch_state"] = torch.get_rng_state()
        out_dict["cuda_state"] = torch.cuda.get_rng_state()
        if np:
            out_dict["numpy_state"] = np.random.get_state()
        if random:
            out_dict["python_state"] = random.getstate()

        # model states
        ckpt_dict = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "hparams": vars(config),
            "last_batch_idx": last_idx,
            "rng": out_dict,
        }

        # optimizer states
        opt_dict = {
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
        }

        if last_idx == -1:
            filename = f"{str(self.model)}_{epoch}.pt"
        else:
            filename = f"{str(self.model)}_{last_idx}_{epoch}.pt"

        torch.save(ckpt_dict, os.path.join(self.save_checkpoint_path, filename))
        torch.save(
            opt_dict, os.path.join(self.save_checkpoint_path, "OPTIMIZER_STATES.pt")
        )

        print(
            f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}."
        )

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            # if self.local_rank == 0:
            self._save_checkpoint(epoch, self.config, last_idx=-1)

    def _run_epoch(self, epoch):
        print(
            f"Epoch {epoch} | Batchsize: {self.config.n_batch} | Steps: {len(self.train_data)} | Last batch: {self.last_batch_idx}"
        )
        # self.train_data.sampler.set_epoch(epoch)
        loss_list = pd.Series()

        for idx, data in enumerate(tqdm(self.train_data)):
            # skip batches
            if idx <= self.last_batch_idx:
                continue

            # run batch
            bucket_idx_masked = data[0]
            bucket_targets = data[1]
            bucket_idx_not_masked = data[2]
            with torch.cuda.device(0):
                loss = self._run_batch(
                    bucket_idx_masked, bucket_targets, bucket_idx_not_masked
                )
            torch.cuda.empty_cache()

            # track loss
            # if self.local_rank == 0:
            loss_list = pd.concat([loss_list, pd.Series([loss])], axis=0)

            # checkpoint
            # if self.local_rank == 0 and idx % self.save_every == 0 and idx != 0:
            if idx % self.save_every == 0 and idx != 0:
                self._save_checkpoint(epoch, self.config, idx)
                # WARN: due to job limit time - save loss for each iter
                loss_list.to_csv(
                    os.path.join(
                        self.config.save_checkpoint_path,
                        f"training_loss_{idx}_epoch{epoch}.csv",
                    ),
                    index=False,
                )
                loss_list = pd.Series()

        self.last_batch_idx = -1

        # if self.local_rank == 0:
        loss_list.to_csv(
            os.path.join(
                self.config.save_checkpoint_path, f"training_loss_epoch{epoch}.csv"
            ),
            index=False,
        )

    def _run_batch(self, bucket_idx_masked, bucket_targets, bucket_idx_not_masked):
        raise NotImplementedError


class TrainerEncoderDecoder(Trainer):

    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
    ) -> None:
        super().__init__(
            model,
            train_data,
            optimizer,
            save_every,
            save_checkpoint_path,
            load_checkpoint_path,
            config,
        )
        self.criterionC = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterionR = nn.MSELoss()

        self.optimE = self.optimizer[0]
        self.optimD = self.optimizer[1]

        self.ngpus_per_node = torch.cuda.device_count()
        self.total_batches = len(self.train_data)
        self.batch_thresh = int(
            self.total_batches - (self.total_batches * 0.05 * self.ngpus_per_node)
        )
        print("batch_thresh:", self.batch_thresh)

    def _load_checkpoint(self, checkpoint_path):
        opt_dict = None
        # loc = f"cuda:{self.local_rank}"
        loc = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_dict = torch.load(checkpoint_path, map_location=loc)
        if os.path.exists(
            os.path.join(self.save_checkpoint_path, "OPTIMIZER_STATES.pt")
        ):
            opt_dict = torch.load(
                os.path.join(self.save_checkpoint_path, "OPTIMIZER_STATES.pt"),
                map_location=loc,
            )

        self.model.load_state_dict(ckpt_dict["MODEL_STATE"])
        if opt_dict is not None:
            self.optimizer[0].load_state_dict(opt_dict["OPTIMIZER_STATE_ENCODER"])
            self.optimizer[1].load_state_dict(opt_dict["OPTIMIZER_STATE_DECODER"])
            print("Optimizer states restored!")

        self.last_batch_idx = (
            ckpt_dict["last_batch_idx"] if "last_batch_idx" in ckpt_dict else -1
        )
        self.epochs_run = (
            ckpt_dict["EPOCHS_RUN"] + 1
            if self.last_batch_idx == -1
            else ckpt_dict["EPOCHS_RUN"]
        )

        # load RNG states each time the model and states are loaded from checkpoint
        if "rng" in ckpt_dict:
            rng = ckpt_dict["rng"]
            for key, value in rng.items():
                if key == "torch_state":
                    torch.set_rng_state(value.cpu())
                elif key == "cuda_state":
                    torch.cuda.set_rng_state(value.cpu())
                elif key == "numpy_state":
                    np.random.set_state(value)
                elif key == "python_state":
                    random.setstate(value)
                else:
                    print("unrecognized state")

        print(f"Resuming training from checkpoint at Epoch {self.epochs_run}.")

    def _save_checkpoint(self, epoch, config, last_idx):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict["torch_state"] = torch.get_rng_state()
        out_dict["cuda_state"] = torch.cuda.get_rng_state()
        if np:
            out_dict["numpy_state"] = np.random.get_state()
        if random:
            out_dict["python_state"] = random.getstate()

        # model states
        ckpt_dict = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "hparams": vars(config),
            "last_batch_idx": last_idx,
            "rng": out_dict,
        }

        # optimizer states
        opt_dict = {
            "OPTIMIZER_STATE_ENCODER": self.optimizer[0].state_dict(),
            "OPTIMIZER_STATE_DECODER": self.optimizer[1].state_dict(),
        }

        if last_idx == -1:
            filename = f"{str(self.model)}_{epoch}.pt"
        else:
            filename = f"{str(self.model)}_{last_idx}_{epoch}.pt"

        torch.save(ckpt_dict, os.path.join(self.save_checkpoint_path, filename))
        torch.save(
            opt_dict, os.path.join(self.save_checkpoint_path, "OPTIMIZER_STATES.pt")
        )

        print(
            f"Epoch {epoch} | Training checkpoint saved at {os.path.join(self.save_checkpoint_path, filename)}."
        )

    def _run_epoch(self, epoch):
        print(
            f"Epoch {epoch} | Batchsize: {self.config.n_batch} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        loss_list = pd.DataFrame()

        for idx, data in enumerate(tqdm(self.train_data)):
            bucket_idx_masked = data[0]
            bucket_targets = data[1]
            bucket_idx_not_masked = data[2]
            lossE, lossD = self._run_batch(
                idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked
            )
            torch.cuda.empty_cache()

            # if self.local_rank == 0:
            df = pd.DataFrame(
                {
                    "lossE": [lossE.cpu().item()],
                    "lossD": [lossD.cpu().item()],
                }
            )
            loss_list = pd.concat([loss_list, df], axis=0)

        # if self.local_rank == 0:
        loss_list.to_csv(
            os.path.join(
                self.config.save_checkpoint_path, f"training_loss_epoch{epoch}.csv"
            ),
            index=False,
        )

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs

        return custom_forward

    def _run_batch(
        self, batch_idx, bucket_idx_masked, bucket_targets, bucket_idx_not_masked
    ):
        self.optimE.zero_grad(set_to_none=True)
        self.optimD.zero_grad(set_to_none=True)

        can_train_encoder = (batch_idx + 1) <= self.batch_thresh
        can_train_decoder = (batch_idx + 1) > self.batch_thresh

        padding_idx = 2
        errorE = torch.zeros(1).to("cuda")  # .to(self.local_rank)
        errorD = torch.zeros(1).to("cuda")  # .to(self.local_rank)
        errorE_tmp = 0.0
        errorD_tmp = 0.0

        for chunk in range(len(bucket_idx_masked)):
            idx_masked = bucket_idx_masked[chunk].to("cuda")  # .to(self.local_rank)
            targets = bucket_targets[chunk].to("cuda")  # .to(self.local_rank)
            idx_not_masked = bucket_idx_not_masked[chunk]
            idx_not_masked = list(
                map(
                    lambda x: F.pad(
                        x, pad=(0, self.config.max_len - x.shape[0]), value=2
                    ).unsqueeze(0),
                    idx_not_masked,
                )
            )
            idx_not_masked = torch.cat(idx_not_masked, dim=0).to(
                "cuda"
            )  # .to(self.local_rank)
            mask = idx_masked != padding_idx

            ###########
            # Encoder #
            ###########
            if can_train_encoder:
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

                # encoder forward
                with torch.cuda.device(0):
                    x = self.model.encoder.tok_emb(idx_masked)
                    x = self.model.encoder.drop(x)
                    x = checkpoint.checkpoint(self.custom(self.model.encoder.blocks), x)
                    logits = self.model.encoder.lang_model(x)

                    # loss function
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)
                    errorE_tmp = self.criterionC(logits, targets) / len(
                        bucket_idx_masked
                    )

                    if chunk < len(bucket_idx_masked) - 1:
                        errorE_tmp.backward()
                        errorE += errorE_tmp.detach()
                    else:
                        errorE += errorE_tmp

            ###########
            # Decoder #
            ###########
            if can_train_decoder:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = True

                self.model.encoder.eval()

                # encoder forward
                with torch.no_grad():
                    true_set, true_cte = self.model.encoder(
                        idx_masked, mask=mask, inference=True
                    )

                # add padding
                input_mask_expanded = mask.unsqueeze(-1).expand(true_cte.size()).float()
                mask_embeddings = true_cte * input_mask_expanded
                true_cte = F.pad(
                    mask_embeddings,
                    pad=(0, 0, 0, self.config.max_len - mask_embeddings.shape[1]),
                    value=0,
                )
                true_cte = true_cte.view(-1, self.config.max_len * self.config.n_embd)

                # decoder forward
                pred_set, pred_ids = self.model.decoder(true_cte)

                # losses
                pred_ids = pred_ids.view(-1, pred_ids.size(-1))
                true_ids = idx_not_masked.view(-1)

                error_ids = self.criterionC(pred_ids, true_ids) / len(bucket_idx_masked)
                error_set = self.criterionR(pred_set, true_set) / len(bucket_idx_masked)
                errorD_tmp = error_ids + error_set

                if chunk < len(bucket_idx_masked) - 1:
                    errorD_tmp.backward()
                    errorD += errorD_tmp.detach()
                else:
                    errorD += errorD_tmp

        if can_train_decoder:
            errorD.backward()
            self.optimD.step()
        elif can_train_encoder:
            errorE.backward()
            self.optimE.step()

        # if self.local_rank == 0:
        print(f"LossE: {errorE.item()} | LossD: {errorD.item()}")
        return errorE, errorD


class TrainerDirectDecoder(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        config,
        text_encoder=None,
    ) -> None:
        super().__init__(
            model,
            train_data,
            optimizer,
            save_every,
            save_checkpoint_path,
            load_checkpoint_path,
            config,
        )
        self.criterionC = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterionR = nn.MSELoss()
        self.text_encoder = text_encoder
        self.paired_mode = (
            hasattr(train_data.dataset, "paired_mode")
            and train_data.dataset.paired_mode
        )

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0], length_mask=inputs[1])
            return inputs

        return custom_forward

    def _run_batch(
        self,
        bucket_idx_masked,
        bucket_targets,
        bucket_idx_not_masked,
        bucket_output_ids=None,
    ):
        padding_idx = 2
        error = torch.zeros(1).to("cuda")
        error_tmp = 0.0
        self.optimizer.zero_grad(set_to_none=True)

        for chunk in range(len(bucket_idx_masked)):
            idx_masked = bucket_idx_masked[chunk].to("cuda")
            targets = bucket_targets[chunk].to("cuda")

            # Use output SMILES if in paired mode, otherwise use input SMILES
            if self.paired_mode and bucket_output_ids is not None:
                idx_not_masked = bucket_output_ids[chunk]
            else:
                idx_not_masked = bucket_idx_not_masked[chunk]

            idx_not_masked = list(
                map(
                    lambda x: F.pad(
                        x, pad=(0, self.config.max_len - x.shape[0]), value=2
                    ).unsqueeze(0),
                    idx_not_masked,
                )
            )
            idx_not_masked = torch.cat(idx_not_masked, dim=0).to("cuda")
            mask = idx_masked != padding_idx

            # encoder forward
            x = self.model.encoder.tok_emb(idx_masked)
            x = self.model.encoder.drop(x)
            x = checkpoint.checkpoint(
                self.custom(self.model.encoder.blocks),
                x,
                LengthMask(
                    mask.sum(-1),
                    max_len=idx_masked.shape[1],
                ),
            )

            # mean pooling
            input_masked_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            sum_embeddings = torch.sum(x * input_masked_expanded, 1)
            sum_mask = torch.clamp(input_masked_expanded.sum(1), min=1e-9)
            true_set = sum_embeddings / sum_mask
            true_cte = x
            del x
            torch.cuda.empty_cache()

            # add padding
            input_mask_expanded = mask.unsqueeze(-1).expand(true_cte.size()).float()
            mask_embeddings = true_cte * input_mask_expanded
            true_cte = F.pad(
                mask_embeddings,
                pad=(0, 0, 0, self.config.max_len - mask_embeddings.shape[1]),
                value=0,
            )
            true_cte = true_cte.view(-1, self.config.max_len * self.config.n_embd)

            # decoder forward
            pred_set, pred_ids = self.model.decoder(true_cte)

            # losses
            pred_ids = pred_ids.view(-1, pred_ids.size(-1))
            true_ids = idx_not_masked.view(-1)

            # # Before computing error_ids
            # if random.random() < 0.1:  # 10% chance to print
            #     print("\n=== Target Verification ===")
            #     print("Input Tokens:", idx_masked[0].tolist()[:10])
            #     print("Target Tokens:", true_ids[:10].cpu().tolist())
            #     print(
            #         "Decoded Input:", self.text_encoder.decode(idx_masked[0].tolist())
            #     )
            #     print(
            #         "Decoded Target:",
            #         self.text_encoder.decode(true_ids[:50].cpu().tolist()),
            #     )

            error_ids = self.criterionC(pred_ids, true_ids) / len(bucket_idx_masked)
            error_set = self.criterionR(pred_set, true_set) / len(bucket_idx_masked)
            error_tmp = error_ids + error_set

            if chunk < len(bucket_idx_masked) - 1:
                error_tmp.backward()
                error += error_tmp.detach()
            else:
                error += error_tmp

            torch.cuda.empty_cache()

        error.backward()
        self.optimizer.step()

        print(f"Loss: {error.item()}")
        return error.item()

    def _run_epoch(self, epoch):
        print(
            f"Epoch {epoch} | Batchsize: {self.config.n_batch} | Steps: {len(self.train_data)} | Last batch: {self.last_batch_idx}"
        )
        loss_list = pd.Series()

        for idx, data in enumerate(tqdm(self.train_data)):
            if idx <= self.last_batch_idx:
                continue

            # Handle both paired and non-paired data
            if self.paired_mode:
                bucket_idx_masked = data[0]
                bucket_targets = data[1]
                bucket_idx_not_masked = data[2]
                bucket_output_ids = data[4]  # Output SMILES IDs
                with torch.cuda.device(0):
                    loss = self._run_batch(
                        bucket_idx_masked,
                        bucket_targets,
                        bucket_idx_not_masked,
                        bucket_output_ids,
                    )
            else:
                bucket_idx_masked = data[0]
                bucket_targets = data[1]
                bucket_idx_not_masked = data[2]
                with torch.cuda.device(0):
                    loss = self._run_batch(
                        bucket_idx_masked, bucket_targets, bucket_idx_not_masked
                    )

            torch.cuda.empty_cache()
            loss_list = pd.concat([loss_list, pd.Series([loss])], axis=0)

            if idx % self.save_every == 0 and idx != 0:
                self._save_checkpoint(epoch, self.config, idx)
                loss_list.to_csv(
                    os.path.join(
                        self.config.save_checkpoint_path,
                        f"training_loss_{idx}_epoch{epoch}.csv",
                    ),
                    index=False,
                )
                loss_list = pd.Series()

        self.last_batch_idx = -1
        loss_list.to_csv(
            os.path.join(
                self.config.save_checkpoint_path, f"training_loss_epoch{epoch}.csv"
            ),
            index=False,
        )
