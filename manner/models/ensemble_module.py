from typing import Optional, Tuple, Union

import torch
from lightning import LightningModule
from torch_geometric.utils import to_dense_batch
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalNormalizedDCG

from manner.data.components.mind_batch import MINDRecBatch
from manner.metrics.diversity import Diversity
from manner.metrics.personalization import Personalization
from manner.models.a_module import AModule
from manner.models.components.click_predictors import DotProduct
from manner.models.cr_module import CRModule


class EnsembleModule(LightningModule):
    def __init__(
        self,
        cr_module_module_ckpt: str,
        a_module_categ_ckpt: Optional[str],
        a_module_sent_ckpt: Optional[str],
        categ_weight: Optional[float],
        sent_weight: Optional[float],
        num_categ_classes: int,
        num_sent_classes: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # load ensemble components
        self.cr_module = CRModule.load_from_checkpoint(
            checkpoint_path=self.hparams.cr_module_module_ckpt
        )

        if self.hparams.categ_weight != 0:
            assert isinstance(self.hparams.a_module_categ_ckpt, str)
            self.a_module_categ = AModule.load_from_checkpoint(
                checkpoint_path=self.hparams.a_module_categ_ckpt
            )
        if self.hparams.sent_weight != 0:
            assert isinstance(self.hparams.a_module_sent_ckpt, str)
            self.a_module_sent = AModule.load_from_checkpoint(
                checkpoint_path=self.hparams.a_module_sent_ckpt
            )

        self.click_predictor = DotProduct()

        recommendation_metrics = MetricCollection(
            {
                "ndcg@5": RetrievalNormalizedDCG(k=5), 
                "ndcg@10": RetrievalNormalizedDCG(k=10)
            }
        )
        categ_div_metrics = MetricCollection(
            {
                "categ_div@5": Diversity(num_classes=self.hparams.num_categ_classes, k=5),
                "categ_div@10": Diversity(num_classes=self.hparams.num_categ_classes, k=10),
            }
        )
        sent_div_metrics = MetricCollection(
            {
                "sent_div@5": Diversity(num_classes=self.hparams.num_sent_classes, k=5),
                "sent_div@10": Diversity(num_classes=self.hparams.num_sent_classes, k=10),
            }
        )
        categ_pers_metrics = MetricCollection(
            {
                "categ_pers@5": Personalization(num_classes=self.hparams.num_categ_classes, k=5),
                "categ_pers@10": Personalization(num_classes=self.hparams.num_categ_classes, k=10),
            }
        )
        sent_pers_metrics = MetricCollection(
            {
                "sent_pers@5": Personalization(num_classes=self.hparams.num_sent_classes, k=5),
                "sent_pers@10": Personalization(num_classes=self.hparams.num_sent_classes, k=10),
            }
        )
        self.test_recommendation_metrics = recommendation_metrics.clone(prefix="test/")
        self.test_categ_div_metrics = categ_div_metrics.clone(prefix="test/")
        self.test_sent_div_metrics = sent_div_metrics.clone(prefix="test/")
        self.test_categ_pers_metrics = categ_pers_metrics.clone(prefix="test/")
        self.test_sent_pers_metrics = sent_pers_metrics.clone(prefix="test/")
        
        # collect outputs of `*_step`
        self.keys = [
                'preds', 'targets', 
                'cand_news_size', 'hist_news_size',
                'target_categories', 'target_sentiments', 
                'hist_categories', 'hist_sentiments'
                ]
        self.test_step_outputs = {key: [] for key in self.keys}

    def forward(self, batch: MINDRecBatch) -> torch.Tensor:
        # recommendation (ideal) scores
        scores = self._submodel_forward(batch, model=self.cr_module)

        # category-based scores
        if self.hparams.categ_weight != 0:
            categ_scores = self._submodel_forward(batch, model=self.a_module_categ)
            scores += self.hparams.categ_weight * categ_scores

        # sentiment-based scores
        if self.hparams.sent_weight != 0:
            sent_scores = self._submodel_forward(batch, model=self.a_module_sent)
            scores += self.hparams.sent_weight * sent_scores

        return scores

    def _submodel_forward(
        self, batch: MINDRecBatch, model: Union[CRModule, AModule]
    ) -> torch.Tensor:
        # encode clicked news
        clicked_news_vector = model.news_encoder(batch["x_hist"])
        clicked_news_vector_agg, mask_hist = to_dense_batch(
            clicked_news_vector, batch["batch_hist"]
        )

        # encode candidate news
        candidate_news_vector = model.news_encoder(batch["x_cand"])
        candidate_news_vector_agg, mask_cand = to_dense_batch(
            candidate_news_vector, batch["batch_cand"]
        )

        # aggregated history
        hist_size = torch.tensor(
            [torch.where(mask_hist[i])[0].shape[0] for i in range(mask_hist.shape[0])],
            device=self.device,
        )
        user_vector = torch.div(clicked_news_vector_agg.sum(dim=1), hist_size.unsqueeze(dim=-1))

        scores = self.click_predictor(
            user_vector.unsqueeze(dim=1), candidate_news_vector_agg.permute(0, 2, 1)
        )

        # z-score normalization
        cand_size = torch.tensor(
            [torch.where(mask_cand[i])[0].shape[0] for i in range(mask_cand.shape[0])],
            device=self.device,
        )
        std_devs = torch.stack(
            [torch.std(scores[i][mask_cand[i]]) for i in range(mask_cand.shape[0])]
        ).unsqueeze(-1)
        scores = torch.div(
            scores
            - torch.div(torch.sum(scores, dim=1), cand_size).unsqueeze(-1).expand_as(scores),
            std_devs,
        )

        return scores

    def model_step(self, batch: MINDRecBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.forward(batch)
        y_true, mask_cand = to_dense_batch(batch["labels"], batch["batch_cand"])

        candidate_categories, _ = to_dense_batch(batch["x_cand"]["category"], batch["batch_cand"])
        candidate_sentiments, _ = to_dense_batch(batch["x_cand"]["sentiment"], batch["batch_cand"])

        clicked_categories, mask_hist = to_dense_batch(
            batch["x_hist"]["category"], batch["batch_hist"]
        )
        clicked_sentiments, _ = to_dense_batch(batch["x_hist"]["sentiment"], batch["batch_hist"])

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
        hist_news_size = torch.tensor(
            [torch.where(mask_hist[n])[0].shape[0] for n in range(mask_hist.shape[0])]
        )

        target_categories = torch.cat(
            [candidate_categories[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()
        target_sentiments = torch.cat(
            [candidate_sentiments[n][mask_cand[n]] for n in range(mask_cand.shape[0])], dim=0
        ).long()

        hist_categories = torch.cat(
            [clicked_categories[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0
        ).long()
        hist_sentiments = torch.cat(
            [clicked_sentiments[n][mask_hist[n]] for n in range(mask_hist.shape[0])], dim=0
        ).long()

        return preds, targets, cand_news_size, hist_news_size, target_categories, target_sentiments, hist_categories, hist_sentiments

    def training_step(self, batch: MINDRecBatch, batch_idx: int):
        pass

    def validation_step(self, batch: MINDRecBatch, batch_idx: int):
        pass

    def test_step(self, batch: MINDRecBatch, batch_idx: int):
        preds, targets, cand_news_size, hist_news_size, target_categories, target_sentiments, hist_categories, hist_sentiments = self.model_step(batch)

        self.test_step_outputs['preds'].append(preds) 
        self.test_step_outputs['targets'].append(targets) 
        self.test_step_outputs['cand_news_size'].append(cand_news_size) 
        self.test_step_outputs['hist_news_size'].append(hist_news_size) 
        self.test_step_outputs['target_categories'].append(target_categories) 
        self.test_step_outputs['target_sentiments'].append(target_sentiments) 
        self.test_step_outputs['hist_categories'].append(hist_categories) 
        self.test_step_outputs['hist_sentiments'].append(hist_sentiments) 

    def on_test_epoch_end(self):
        preds = torch.cat([output for output in self.test_step_outputs["preds"]])
        targets = torch.cat([output for output in self.test_step_outputs["targets"]])
        
        target_categories = torch.cat([output for output in self.test_step_outputs["target_categories"]])
        target_sentiments = torch.cat([output for output in self.test_step_outputs["target_sentiments"]])

        hist_categories = torch.cat([output for output in self.test_step_outputs["hist_categories"]])
        hist_sentiments = torch.cat([output for output in self.test_step_outputs["hist_sentiments"]])
        
        cand_news_size = torch.cat([output for output in self.test_step_outputs["cand_news_size"]])
        hist_news_size = torch.cat([output for output in self.test_step_outputs["hist_news_size"]])
        
        cand_indexes = torch.arange(cand_news_size.shape[0]).repeat_interleave(cand_news_size)
        hist_indexes = torch.arange(hist_news_size.shape[0]).repeat_interleave(hist_news_size)
        
        self.test_recommendation_metrics(preds, targets, **{"indexes": cand_indexes})
        self.test_categ_div_metrics(preds, target_categories, cand_indexes)
        self.test_sent_div_metrics(preds, target_sentiments, cand_indexes)
        self.test_categ_pers_metrics(
            preds, target_categories, hist_categories, cand_indexes, hist_indexes
        )
        self.test_sent_pers_metrics(
            preds, target_sentiments, hist_sentiments, cand_indexes, hist_indexes
        )

        self.log_dict(self.test_recommendation_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(
            self.test_categ_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_div_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_categ_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log_dict(
            self.test_sent_pers_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        # clean memory for the next epoch
        for key in self.keys:
            self.test_step_outputs[key].clear()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        pass
