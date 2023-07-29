from typing import List, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch_geometric.utils import to_dense_batch
from torchmetrics import MeanMetric, MetricCollection, MinMetric
from torchmetrics.classification import AUROC
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG

from manner.data.components.mind_batch import MINDRecBatch
from manner.models.components.click_predictors import DotProduct
from manner.models.components.losses import SupConLoss
from manner.models.components.news_encoder import MannerNewsEncoder
from manner.models.components.user_encoder import NAMLUserEncoder as UserEncoder


class CRModule(LightningModule):
    def __init__(
        self,
        supcon_loss: bool,
        late_fusion: bool,
        temperature: float,
        plm_model: str,
        frozen_layers: List[int],
        dropout_probability: float,
        use_entities: bool,
        pretrained_entity_embeddings_path: str,
        entity_embedding_dim: int,
        num_attention_heads: int,
        query_vector_dim: int,
        text_embedding_dim: int,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if self.hparams.use_entities:
            # load pretrained entity embeddings
            pretrained_entity_embeddings = torch.from_numpy(
                np.load(pretrained_entity_embeddings_path)
            ).float()
        else:
            pretrained_entity_embeddings = torch.empty(self.hparams.entity_embedding_dim)

        # model components
        self.news_encoder = MannerNewsEncoder(
            plm_model=self.hparams.plm_model,
            frozen_layers=self.hparams.frozen_layers,
            dropout_probability=self.hparams.dropout_probability,
            use_entities=self.hparams.use_entities,
            entity_embeddings=pretrained_entity_embeddings,
            entity_embedding_dim=self.hparams.entity_embedding_dim,
            num_attention_heads=self.hparams.num_attention_heads,
            query_vector_dim=self.hparams.query_vector_dim,
            text_embedding_dim=self.hparams.text_embedding_dim,
        )

        if not self.hparams.late_fusion:
            # user encoder used only for early fusion
            self.user_encoder = UserEncoder(
                news_embedding_dim=self.hparams.text_embedding_dim,
                query_vector_dim=self.hparams.query_vector_dim,
            )

        self.click_predictor = DotProduct()

        # loss function
        if self.hparams.supcon_loss:
            self.criterion = SupConLoss(temperature=self.hparams.temperature)
        else:
            self.criterion = CrossEntropyLoss()

        # metric objects for calculating and averaging performance across batches
        metrics = MetricCollection(
            {
                "auc": AUROC(task="binary", num_classes=2),
                "mrr": RetrievalMRR(),
                "ndcg@5": RetrievalNormalizedDCG(k=5),
                "ndcg@10": RetrievalNormalizedDCG(k=10),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()
        
        # collect outputs of `*_step`
        self.keys = ['preds', 'targets', 'cand_news_size']
        self.training_step_outputs = {key: [] for key in self.keys}
        self.validation_step_outputs = {key: [] for key in self.keys}
        self.test_step_outputs = {key: [] for key in self.keys}

    def forward(self, batch: MINDRecBatch) -> torch.Tensor:
        # encode clicked news
        clicked_news_vector = self.news_encoder(batch["x_hist"])
        clicked_news_vector_agg, mask_hist = to_dense_batch(
            clicked_news_vector, batch["batch_hist"]
        )

        # encode candidate news
        candidate_news_vector = self.news_encoder(batch["x_cand"])
        candidate_news_vector_agg, _ = to_dense_batch(candidate_news_vector, batch["batch_cand"])

        if self.hparams.late_fusion:
            hist_size = torch.tensor(
                [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
                device=self.device,
            )
            user_vector = torch.div(
                clicked_news_vector_agg.sum(dim=1), hist_size.unsqueeze(dim=-1)
            )
        else:
            user_vector = self.user_encoder(clicked_news_vector_agg)

        scores = self.click_predictor(
            user_vector.unsqueeze(dim=1), candidate_news_vector_agg.permute(0, 2, 1)
        )

        return scores

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_metrics.reset()

    def model_step(self, batch: MINDRecBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.forward(batch)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])

        if self.hparams.supcon_loss:
            # indices of positive pairs for loss calculation
            pos_idx = [torch.where(y_true[i])[0] for i in range(mask_cand.shape[0])]
            pos_repeats = torch.tensor([len(pos_idx[i]) for i in range(len(pos_idx))])
            q_p = torch.repeat_interleave(torch.arange(mask_cand.shape[0]), pos_repeats)
            p = torch.cat(pos_idx)

            # indices of negative pairs for loss calculation
            neg_idx = [
                torch.where(~y_true[i].bool())[0][
                    : len(torch.where(mask_cand[i])[0]) - pos_repeats[i]
                ]
                for i in range(mask_cand.shape[0])
            ]
            neg_repeats = torch.tensor([len(t) for t in neg_idx])
            q_n = torch.repeat_interleave(torch.arange(mask_cand.shape[0]), neg_repeats)
            n = torch.cat(neg_idx)

            indices_tuple = (q_p, p, q_n, n)
            loss = self.criterion(
                embeddings=scores,
                labels=None,
                indices_tuple=indices_tuple,
                ref_emb=None,
                ref_labels=None,
            )
        else:
            loss = self.criterion(scores, y_true)

        # predictions, targets, indexes for metric computation
        preds = torch.cat(
            [scores[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).detach()
        targets = torch.cat(
            [y_true[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()
        cand_news_size = torch.tensor(
            [torch.where(mask_cand[n])[0].shape[0] for n in range(mask_cand.shape[0])]
        )

        return loss, preds, targets, cand_news_size

    def training_step(self, batch: MINDRecBatch, batch_idx: int):
        loss, preds, targets, cand_news_size = self.model_step(batch)

        # update and log loss
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.training_step_outputs['preds'].append(preds) 
        self.training_step_outputs['targets'].append(targets) 
        self.training_step_outputs['cand_news_size'].append(cand_news_size) 

        return loss

    def on_train_epoch_end(self):
        preds = torch.cat([output for output in self.training_step_outputs["preds"]])
        targets = torch.cat([output for output in self.training_step_outputs["targets"]])
        cand_news_size = torch.cat([output for output in self.training_step_outputs["cand_news_size"]])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        self.train_metrics(preds, targets, **{"indexes": indexes})
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
        # clean memory for the next epoch
        for key in self.keys:
            self.training_step_outputs[key].clear()

    def validation_step(self, batch: MINDRecBatch, batch_idx: int):
        loss, preds, targets, cand_news_size = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log(
            "val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.validation_step_outputs['preds'].append(preds) 
        self.validation_step_outputs['targets'].append(targets) 
        self.validation_step_outputs['cand_news_size'].append(cand_news_size) 

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val loss
        self.val_loss_best(loss)  # update best so far val loss

        # log `val_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        preds = torch.cat([output for output in self.validation_step_outputs["preds"]])
        targets = torch.cat([output for output in self.validation_step_outputs["targets"]])
        cand_news_size = torch.cat([output for output in self.validation_step_outputs["cand_news_size"]])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        self.val_metrics(preds, targets, **{"indexes": indexes})
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # clean memory for the next epoch
        for key in self.keys:
            self.validation_step_outputs[key].clear()

    def test_step(self, batch: MINDRecBatch, batch_idx: int):
        loss, preds, targets, cand_news_size = self.model_step(batch)

        # update and log loss
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.test_step_outputs['preds'].append(preds) 
        self.test_step_outputs['targets'].append(targets) 
        self.test_step_outputs['cand_news_size'].append(cand_news_size) 

    def on_test_epoch_end(self):
        preds = torch.cat([output for output in self.test_step_outputs["preds"]])
        targets = torch.cat([output for output in self.test_step_outputs["targets"]])
        
        cand_news_size = torch.cat([output for output in self.test_step_outputs["cand_news_size"]])
        indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)

        self.test_metrics(preds, targets, **{"indexes": indexes})
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # clean memory for the next epoch
        for key in self.keys:
            self.test_step_outputs[key].clear()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.parameters())

        return {"optimizer": optimizer}
