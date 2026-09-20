"""The bi-encoder baseline: SigLIP-2, scored by dot product.

This is the thing to beat, and it is deliberately not a straw man. Three
points of fairness:

  * it uses SigLIP-2, which is the same family as Qwen3-VL's own vision tower,
    so both arms saw the pixels through the same kind of encoder;
  * it gets a trained calibration head on frozen features, exactly as the
    cross-encoder arm does, rather than being left at raw zero-shot softmax
    which is known to be badly calibrated;
  * it gets a learned abstention rule. "none of these" has no sensible text
    embedding, so instead its logit is a learned function of how well the best
    real option matched. That is the standard remedy and it uses the
    information actually available to a bi-encoder.

If the cross-encoder cannot beat this, it is not worth running a language
model per query.
"""
import torch
import torch.nn as nn

from .data import NONE_OPTION

PROMPT = "a photo of a {}, a type of pet"


class SiglipFeatures:
    """Frozen image and label embeddings. Labels are embedded once."""

    def __init__(self, cfg):
        from transformers import AutoModel, AutoProcessor
        self.cfg = cfg.resolved()
        self.processor = AutoProcessor.from_pretrained(cfg.siglip)
        self.model = (AutoModel.from_pretrained(cfg.siglip,
                                                dtype=cfg.torch_dtype())
                      .to(cfg.device).eval().requires_grad_(False))

    @torch.no_grad()
    def image_embeds(self, images):
        px = self.processor(images=list(images), return_tensors="pt")
        px = {k: v.to(self.cfg.device, self.cfg.torch_dtype())
              for k, v in px.items()}
        e = self.model.get_image_features(**px).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu()

    @torch.no_grad()
    def label_embeds(self, labels):
        text = [PROMPT.format(l) for l in labels]
        tk = self.processor(text=text, padding="max_length", truncation=True,
                            return_tensors="pt")
        tk = {k: v.to(self.cfg.device) for k, v in tk.items()}
        e = self.model.get_text_features(**tk).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu()


class BiEncoderHead(nn.Module):
    """Temperature, bias, and a learned abstention rule. Five parameters."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        # logit("none of these") = n0 + n1 * (best real cosine)
        self.n0 = nn.Parameter(torch.tensor(0.0))
        self.n1 = nn.Parameter(torch.tensor(-1.0))

    def forward(self, cos, slot_mask, is_none):
        """cos [B,K] cosine per option, is_none [B,K] marks the escape hatch."""
        real = cos.masked_fill(is_none | ~slot_mask, -1e4)
        best = real.max(dim=-1).values                     # [B]
        logits = self.scale * cos + self.bias
        none_logit = self.n0 + self.n1 * best
        logits = torch.where(is_none, none_logit[:, None].expand_as(logits),
                             logits)
        return logits.masked_fill(~slot_mask, -1e4)


class DiagHead(nn.Module):
    """A richer baseline head: a learned diagonal reweighting of the joint
    image-label features, so the comparison is not 5 parameters against 2,049.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.n0 = nn.Parameter(torch.tensor(0.0))
        self.n1 = nn.Parameter(torch.tensor(-1.0))

    def forward(self, img, txt, slot_mask, is_none):
        """img [B,d], txt [B,K,d]."""
        joint = (img[:, None, :] * txt * self.w).sum(-1)    # [B,K]
        real = joint.masked_fill(is_none | ~slot_mask, -1e4)
        best = real.max(dim=-1).values
        logits = self.scale * joint + self.bias
        none_logit = self.n0 + self.n1 * best
        logits = torch.where(is_none, none_logit[:, None].expand_as(logits),
                             logits)
        return logits.masked_fill(~slot_mask, -1e4)


def is_none_matrix(option_lists, K):
    m = torch.zeros(len(option_lists), K, dtype=torch.bool)
    for i, opts in enumerate(option_lists):
        for j, o in enumerate(opts):
            if o == NONE_OPTION:
                m[i, j] = True
    return m
