"""VSLM model architecture and checkpoint/generation helpers."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 256
DEPTH = 12
EMBEDDING_DIM = 384
EMBEDDING_RANK = 36
ATTENCIONER_WIDTH = 256
ATTENCIONER_DEPTH = 6
DEVECTORIZER_WIDTH = 256
DEVECTORIZER_DEPTH = 6
MHC_STREAMS = 3
MAIN_KERNEL_SIZE = 128
DEFAULT_TOKENIZER = "gpt2"  # public tokenizer, vocabulary size 50,257
DEFAULT_DATASET = "HuggingFaceTB/cosmopedia"
DEFAULT_SUBSET = "stories"


def make_projection_mlp(input_width: int, hidden_width: int, depth: int,
                        output_width: int) -> nn.Sequential:
    if hidden_width <= 0 or depth <= 0:
        raise ValueError("A largura e a profundidade do Attencioner/De-vectorizer precisam ser positivas.")
    layers: list[nn.Module] = []
    current_width = input_width
    for _ in range(depth):
        layers.extend((nn.Linear(current_width, hidden_width), nn.GELU()))
        current_width = hidden_width
    layers.append(nn.Linear(current_width, output_width))
    return nn.Sequential(*layers)


class MHCAttnResidual(nn.Module):
    """Small mHC-style multi-stream residual plus attention over its two stages."""

    def __init__(self, width: int, streams: int = MHC_STREAMS):
        super().__init__()
        self.streams = streams
        self.residual_logits = nn.Parameter(torch.eye(streams) * 2.0)
        self.pre_logits = nn.Parameter(torch.zeros(streams))
        self.post_logits = nn.Parameter(torch.zeros(streams))
        self.query = nn.Parameter(torch.ones(width))
        self.key = nn.Parameter(torch.ones(width))

    def doubly_stochastic(self) -> torch.Tensor:
        matrix = self.residual_logits.float().exp()
        for _ in range(8):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        return matrix.to(dtype=self.residual_logits.dtype)

    def forward(self, base: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        # HC residual mixing: A is projected to the doubly stochastic manifold;
        # pre/post maps are simplex weights over the parallel streams.
        streams = base.unsqueeze(-2).expand(*base.shape[:-1], self.streams, base.shape[-1])
        streams = torch.einsum("ij,...jd->...id", self.doubly_stochastic(), streams)
        post = self.post_logits.softmax(dim=0)
        streams = streams + post.view(*([1] * (streams.ndim - 2)), self.streams, 1) * update.unsqueeze(-2)
        pre = self.pre_logits.softmax(dim=0)
        mixed = torch.einsum("i,...id->...d", pre, streams)

        # Attention Residuals: content-dependent softmax over the prior stage
        # and the newly mixed stage.
        values = torch.stack((base, mixed), dim=-2)
        query = (update * self.query).unsqueeze(-2)
        keys = values * self.key
        scores = (query * keys).sum(dim=-1) / math.sqrt(base.shape[-1])
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * values).sum(dim=-2)


class Attencioner(nn.Module):
    """Shared factorized token embedding plus projection to WIDTH-dimensional vectors."""

    def __init__(self, embedding: FactorizedTokenEmbedding, width: int = WIDTH,
                 hidden_width: int = ATTENCIONER_WIDTH, depth: int = ATTENCIONER_DEPTH,
                 embedding_dim: int = EMBEDDING_DIM,
                 mhc_streams: int = MHC_STREAMS):
        super().__init__()
        self.embedding = embedding
        self.projection = make_projection_mlp(
            embedding_dim, hidden_width, depth, width,
        )
        self.norm = nn.LayerNorm(width)
        self.residual = MHCAttnResidual(width, mhc_streams)

    @staticmethod
    def quantize(vector: torch.Tensor) -> torch.Tensor:
        """Fake quantization to signed 4-bit levels; STE passes training gradients."""
        clipped = vector.clamp(-1.0, 1.0)
        q = torch.round(clipped * 7.0).clamp(-8, 7) / 7.0
        return clipped + (q - clipped).detach()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids has shape [batch, sequence]. Embedding/projection operate
        # over both dimensions in one call; no token waits for another here.
        projected = self.projection(self.embedding(token_ids))
        vector = self.residual(projected, self.norm(projected))
        return self.quantize(vector)


class CausalDepthwiseConv(nn.Module):
    """Causal temporal mixing computed for every sequence position at once."""

    def __init__(self, width: int, kernel_size: int = MAIN_KERNEL_SIZE):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(width, width, kernel_size, groups=width, bias=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        channels_first = sequence.transpose(1, 2)
        padded = F.pad(channels_first, (self.kernel_size - 1, 0))
        return self.conv(padded).transpose(1, 2)

    def step(self, vector: torch.Tensor, cache: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            cache = vector.new_zeros(vector.shape[0], self.kernel_size - 1, vector.shape[-1])
        window = torch.cat((cache, vector.unsqueeze(1)), dim=1)
        output = self.conv(window.transpose(1, 2)).squeeze(-1)
        return output, window[:, 1:, :]


class MainModel(nn.Module):
    """Causal-convolution layers with attention over the layer outputs."""

    def __init__(self, *, width: int = WIDTH, depth: int = DEPTH,
                 kernel_size: int = MAIN_KERNEL_SIZE, mhc_streams: int = MHC_STREAMS):
        super().__init__()
        self.width = width
        self.depth = depth
        self.causal_convs = nn.ModuleList(CausalDepthwiseConv(width, kernel_size) for _ in range(depth))
        self.layer_projections = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.residuals = nn.ModuleList(MHCAttnResidual(width, mhc_streams) for _ in range(depth))
        self.depth_query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.depth_query, std=width ** -0.5)

    def _finish(self, layer_outputs: list[torch.Tensor]) -> torch.Tensor:
        layer_stack = torch.stack(layer_outputs, dim=-2)
        scores = torch.einsum("w,...dw->...d", self.depth_query, layer_stack) / math.sqrt(self.width)
        return (scores.softmax(dim=-1).unsqueeze(-1) * layer_stack).sum(dim=-2)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """Training path: all positions in the block are processed in parallel."""
        layer_input = vectors
        layer_outputs = []
        for causal_conv, projection, norm, residual in zip(
            self.causal_convs, self.layer_projections, self.norms, self.residuals,
        ):
            context = causal_conv(layer_input)
            candidate = norm(projection(torch.cat((layer_input, context), dim=-1)))
            layer_input = residual(layer_input, candidate)
            layer_outputs.append(layer_input)
        return self._finish(layer_outputs)

    def step(self, vector: torch.Tensor,
             caches: list[torch.Tensor | None] | None = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Incremental causal path used only during autoregressive generation."""
        if caches is None:
            caches = [None] * self.depth
        next_caches = []
        layer_input = vector
        layer_outputs = []
        for causal_conv, projection, norm, residual, cache in zip(
            self.causal_convs, self.layer_projections, self.norms, self.residuals, caches,
        ):
            context, next_cache = causal_conv.step(layer_input, cache)
            candidate = norm(projection(torch.cat((layer_input, context), dim=-1)))
            layer_input = residual(layer_input, candidate)
            next_caches.append(next_cache)
            layer_outputs.append(layer_input)
        return self._finish(layer_outputs), next_caches

    def scan(self, vectors: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Scan is reserved for prompt setup in generation; training uses forward."""
        caches: list[torch.Tensor | None] | None = None
        outputs = []
        for t in range(vectors.shape[1]):
            output, caches = self.step(vectors[:, t, :], caches)
            outputs.append(output)
        return torch.stack(outputs, dim=1), [cache for cache in caches if cache is not None]


class FactorizedTokenEmbedding(nn.Module):
    """Tied V×rank and rank×dimension factors for input and output embeddings."""

    def __init__(self, vocab_size: int, rank: int = EMBEDDING_RANK,
                 embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.rank = rank
        self.embedding_dim = embedding_dim
        self.token_factors = nn.Parameter(torch.empty(vocab_size, rank))
        self.projection = nn.Parameter(torch.empty(rank, embedding_dim))
        nn.init.normal_(self.token_factors, mean=0.0, std=0.02)
        nn.init.normal_(self.projection, mean=0.0, std=rank ** -0.5)

    @property
    def num_embeddings(self) -> int:
        return self.token_factors.shape[0]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        factors = F.embedding(token_ids, self.token_factors)
        # [.., EMBEDDING_RANK] × [EMBEDDING_DIM, EMBEDDING_RANK]^T
        return F.linear(factors, self.projection.t())

    def output_logits(self, vectors: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # Input vectors are 256-D; project them back to rank 48 before scoring tokens.
        # self.projection is [EMBEDDING_RANK, EMBEDDING_DIM], the weight shape
        # expected by F.linear for a 256-D input.
        factors = F.linear(vectors, self.projection)
        return F.linear(factors, self.token_factors, bias)


class DeVectorizer(nn.Module):
    """Project to the shared 256-dimensional space and score every tokenizer token."""

    def __init__(self, embedding: FactorizedTokenEmbedding, width: int = WIDTH,
                 hidden_width: int = DEVECTORIZER_WIDTH,
                 depth: int = DEVECTORIZER_DEPTH, embedding_dim: int = EMBEDDING_DIM,
                 mhc_streams: int = MHC_STREAMS):
        super().__init__()
        self.embedding = embedding
        self.projection = make_projection_mlp(
            width, hidden_width, depth, embedding_dim,
        )
        self.residual = MHCAttnResidual(width, mhc_streams)
        self.residual_projection = nn.Linear(width, width)
        self.bias = nn.Parameter(torch.zeros(embedding.num_embeddings))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self.residual(state, self.residual_projection(state))
        reduced = self.projection(state)
        return self.embedding.output_logits(reduced, self.bias)


class VSLM(nn.Module):
    def __init__(self, compact_to_original: list[int] | torch.Tensor,
                 original_vocab_size: int,
                 architecture: dict[str, int] | None = None):
        super().__init__()
        arch = {
            "width": WIDTH, "depth": DEPTH, "attencioner_width": ATTENCIONER_WIDTH,
            "attencioner_depth": ATTENCIONER_DEPTH,
            "devectorizer_width": DEVECTORIZER_WIDTH,
            "devectorizer_depth": DEVECTORIZER_DEPTH,
            "embedding_dim": EMBEDDING_DIM, "embedding_rank": EMBEDDING_RANK,
            "main_kernel_size": MAIN_KERNEL_SIZE, "mhc_streams": MHC_STREAMS,
        }
        if architecture:
            arch.update({k: int(v) for k, v in architecture.items() if k in arch})
        self.architecture = arch
        compact_to_original = torch.as_tensor(compact_to_original, dtype=torch.long)
        if len(compact_to_original) != original_vocab_size:
            raise ValueError("O mapa precisa incluir todos os tokens do tokenizer, sem descartar IDs.")
        if torch.unique(compact_to_original).numel() != original_vocab_size:
            raise ValueError("O mapa de tokens precisa conter todos os IDs exatamente uma vez.")
        if int(compact_to_original.min()) < 0 or original_vocab_size <= int(compact_to_original.max()):
            raise ValueError("O vocabulário original não cobre os IDs compactos informados.")
        self.vocab_size = len(compact_to_original)
        self.original_vocab_size = original_vocab_size
        token_to_compact = torch.zeros(original_vocab_size, dtype=torch.long)
        token_to_compact[compact_to_original] = torch.arange(self.vocab_size)
        self.register_buffer("compact_to_original", compact_to_original, persistent=True)
        self.register_buffer("token_to_compact", token_to_compact, persistent=True)
        shared_embedding = FactorizedTokenEmbedding(
            self.vocab_size, rank=arch["embedding_rank"], embedding_dim=arch["embedding_dim"])
        self.attencioner = Attencioner(
            shared_embedding, width=arch["width"], hidden_width=arch["attencioner_width"],
            depth=arch["attencioner_depth"],
            embedding_dim=arch["embedding_dim"], mhc_streams=arch["mhc_streams"])
        self.main = MainModel(
            width=arch["width"], depth=arch["depth"],
            kernel_size=arch["main_kernel_size"], mhc_streams=arch["mhc_streams"])
        self.devectorizer = DeVectorizer(
            shared_embedding, width=arch["width"], hidden_width=arch["devectorizer_width"],
            depth=arch["devectorizer_depth"], embedding_dim=arch["embedding_dim"],
            mhc_streams=arch["mhc_streams"])

    def map_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_to_compact[token_ids]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Token vectors and all causal-convolution positions are batched together.
        vectors = self.attencioner(token_ids)
        states = self.main(vectors)
        return self.devectorizer(states)

    @torch.no_grad()
    def generate(self, prompt: str, tokenizer: Any, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_k: int = 40, device: str = "cpu",
                 only_new_text: bool = False) -> str:
        if temperature <= 0:
            raise ValueError("temperature precisa ser maior que zero")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.eos_token_id]
        ids = self.map_token_ids(torch.tensor(prompt_ids, dtype=torch.long, device=device)).unsqueeze(0)
        vectors = self.attencioner(ids)
        prompt_states, caches = self.main.scan(vectors)
        state = prompt_states[:, -1, :]
        generated = []
        for _ in range(max_new_tokens):
            scores = self.devectorizer(state) / temperature
            if top_k > 0:
                cutoff = torch.topk(scores, min(top_k, self.vocab_size), dim=-1).values[:, -1:]
                scores = scores.masked_fill(scores < cutoff, -torch.inf)
            next_id = torch.multinomial(F.softmax(scores, dim=-1), 1)
            generated.append(int(next_id.item()))
            next_vector = self.attencioner(next_id).squeeze(1)
            state, caches = self.main.step(next_vector, caches)
        if only_new_text:
            new_original_ids = self.compact_to_original[
                torch.tensor(generated, dtype=torch.long, device=device)
            ].tolist()
            return tokenizer.decode(new_original_ids, skip_special_tokens=True)
        compact_ids = torch.cat((ids.squeeze(0), torch.tensor(generated, dtype=torch.long, device=device)))
        original_ids = self.compact_to_original[compact_ids].tolist()
        return tokenizer.decode(original_ids, skip_special_tokens=True)

    def parameter_report(self) -> dict[str, int]:
        shared_ids = {id(self.attencioner.embedding.token_factors),
                      id(self.attencioner.embedding.projection)}
        groups = {
            "Embedding compartilhado fatorado": [self.attencioner.embedding.token_factors,
                                                  self.attencioner.embedding.projection],
            "Attencioner": [p for p in self.attencioner.parameters() if id(p) not in shared_ids],
            "Modelo Principal": list(self.main.parameters()),
            "De-vectorizer": [p for p in self.devectorizer.parameters() if id(p) not in shared_ids],
        }
        report = {name: sum(p.numel() for p in params) for name, params in groups.items()}
        report["Total"] = sum(p.numel() for p in self.parameters())
        return report

    def save(self, path: str | Path, tokenizer_name: str,
             config: dict[str, Any] | None = None, step: int = 0,
             optimizer_state: dict[str, Any] | None = None,
             loss: float | None = None, best_loss: float | None = None) -> None:
        path = Path(path)
        temp_path = path.with_name(path.name + ".tmp")
        torch.save({"arch_version": 14, "state_dict": self.state_dict(), "vocab_size": self.vocab_size,
                    "original_vocab_size": self.original_vocab_size,
                    "compact_to_original": self.compact_to_original.cpu(),
                    **self.architecture, "vector_bits": 4,
                    "tokenizer": tokenizer_name,
                    "step": step, "loss": loss, "best_loss": best_loss,
                    "optimizer_state": optimizer_state,
                    "config": config or {}}, temp_path)
        os.replace(temp_path, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple["VSLM", dict[str, Any], str, dict[str, Any] | None, int, float]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        arch_version = ckpt.get("arch_version")
        if arch_version not in (13, 14):
            raise ValueError(f"Versão de checkpoint {arch_version!r} incompatível; esperava arch_version 13 ou 14.")
        if ckpt.get("vocab_size") != len(ckpt.get("compact_to_original", [])):
            raise ValueError("Checkpoint inválido: tamanho do vocabulário não corresponde ao mapa de tokens.")
        saved_state = ckpt["state_dict"]
        architecture = {
            key: int(ckpt[key]) for key in (
                "width", "depth", "attencioner_width", "attencioner_depth",
                "devectorizer_width", "devectorizer_depth", "embedding_dim",
                "embedding_rank", "main_kernel_size",
            ) if key in ckpt
        }
        # mHC stream count is retained; infer it for checkpoints that predate
        # explicitly storing it as architecture metadata.
        architecture.setdefault("mhc_streams", int(saved_state["main.residuals.0.residual_logits"].shape[0]))
        required = {"width", "depth", "attencioner_width", "attencioner_depth",
                    "devectorizer_width", "devectorizer_depth", "embedding_dim",
                    "embedding_rank", "main_kernel_size", "mhc_streams"}
        if required - architecture.keys():
            raise ValueError("Checkpoint incompleto: faltam parâmetros para reconstruir a arquitetura.")
        model = cls(ckpt["compact_to_original"], ckpt["original_vocab_size"],
                    architecture=architecture).to(device)
        optimizer_state = ckpt.get("optimizer_state")
        try:
            if arch_version == 13:
                # Migrate V13 weights: remove the dense MoE/CTM/TRM tensors and
                # retain the input + causal-context columns of each layer projection.
                migrated_state = {}
                removed_prefixes = ("main.router.", "main.ctm.", "main.trm.")
                for name, tensor in saved_state.items():
                    if name.startswith(removed_prefixes):
                        continue
                    if name.startswith("main.layer_projections.") and name.endswith(".weight"):
                        tensor = tensor[:, :2 * architecture["width"]].contiguous()
                    migrated_state[name] = tensor
                model.load_state_dict(migrated_state)
                # Optimizer slots refer to removed parameters and old projection shapes.
                optimizer_state = None
            else:
                model.load_state_dict(saved_state)
        except RuntimeError as exc:
            raise ValueError("Checkpoint incompatível com a arquitetura atual.") from exc
        saved_best = ckpt.get("best_loss")
        best_loss = math.inf if saved_best is None else float(saved_best)
        return (model, ckpt.get("config", {}), ckpt["tokenizer"],
                optimizer_state, int(ckpt.get("step", 0)), best_loss)
