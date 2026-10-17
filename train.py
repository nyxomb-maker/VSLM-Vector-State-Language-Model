"""Cosmopedia streaming input, optimization, validation, and VSLM training."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model import DEFAULT_DATASET, DEFAULT_SUBSET, DEFAULT_TOKENIZER, VSLM


class LossHistory:
    """Append-only per-step log, retained across checkpoint resumes."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.latest: dict[str, Any] = {}
        self.min_train: float | None = None
        self.first_validation: tuple[int, float] | None = None
        self.best_validation: float | None = None
        self.best_step: int | None = None
        self.improvement_deltas: list[float] = []
        self._load_summary()
        self.file = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)

    def _accept(self, record: dict[str, Any]) -> None:
        self.latest = record
        loss = record.get("train_loss")
        if loss is not None:
            self.min_train = float(loss) if self.min_train is None else min(self.min_train, float(loss))
        validation = record.get("validation_loss")
        if validation is not None:
            validation = float(validation)
            step = int(record["step"])
            if self.first_validation is None:
                self.first_validation = (step, validation)
            if self.best_validation is None or validation < self.best_validation:
                if self.best_validation is not None:
                    self.improvement_deltas.append(self.best_validation - validation)
                self.best_validation = validation
                self.best_step = step

    def _load_summary(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    self._accept(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue

    def append(self, record: dict[str, Any]) -> None:
        with self.lock:
            self.file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.file.flush()
            self._accept(record)

    def flush(self) -> None:
        with self.lock:
            self.file.flush()

    def summary(self) -> dict[str, Any]:
        with self.lock:
            mean_delta = (sum(self.improvement_deltas) / len(self.improvement_deltas)
                          if self.improvement_deltas else None)
            return {
                "latest": self.latest,
                "min_train_loss": self.min_train,
                "first_validation": self.first_validation,
                "best_validation_loss": self.best_validation,
                "best_validation_step": self.best_step,
                "validation_improvements": len(self.improvement_deltas),
                "mean_improvement": mean_delta,
            }

    def read_chunk(self, offset: int, max_bytes: int = 1_048_576) -> tuple[str, int]:
        with self.lock:
            self.file.flush()
            with self.path.open("rb") as source:
                source.seek(max(0, offset))
                lines: list[bytes] = []
                count = 0
                while count < max_bytes:
                    line = source.readline()
                    if not line:
                        break
                    lines.append(line)
                    count += len(line)
                return b"".join(lines).decode("utf-8"), source.tell()

    def close(self) -> None:
        self.flush()
        self.file.close()

    def truncate_after(self, step_limit: int) -> None:
        """Keep the history consistent with the checkpoint being resumed."""
        with self.lock:
            self.file.flush()
            records: list[dict[str, Any]] = []
            with self.path.open("r", encoding="utf-8") as source:
                for line in source:
                    try:
                        record = json.loads(line)
                        if int(record["step"]) <= step_limit:
                            records.append(record)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
            self.file.close()
            with self.path.open("w", encoding="utf-8") as target:
                for record in records:
                    target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.latest = {}
            self.min_train = None
            self.first_validation = None
            self.best_validation = None
            self.best_step = None
            self.improvement_deltas = []
            for record in records:
                self._accept(record)
            self.file = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)


def seed_legacy_history(history: LossHistory, html_path: Path, step_limit: int) -> None:
    """Import the old chart's embedded points once when resuming without a log."""
    if history.latest or not html_path.exists() or step_limit <= 0:
        return
    html = html_path.read_text(encoding="utf-8")
    train_match = re.search(r"const legacyTrain\s*=\s*(\[.*?\]);", html, re.S)
    valid_match = re.search(r"const legacyValid\s*=\s*(\[.*?\]);", html, re.S)
    if not train_match or not valid_match:
        return
    try:
        train_points = json.loads(train_match.group(1))
        valid_points = json.loads(valid_match.group(1))
    except json.JSONDecodeError:
        return
    by_step: dict[int, dict[str, Any]] = {}
    for step, loss in train_points:
        if int(step) <= step_limit:
            by_step.setdefault(int(step), {"step": int(step), "epoch": 1,
                                           "train_loss": None, "validation_loss": None})["train_loss"] = float(loss)
    for step, loss in valid_points:
        if int(step) <= step_limit:
            by_step.setdefault(int(step), {"step": int(step), "epoch": 1,
                                           "train_loss": None, "validation_loss": None})["validation_loss"] = float(loss)
    for step in sorted(by_step):
        row = by_step[step]
        history._accept(row)
        row["best_validation_loss"] = history.best_validation
        row["best_validation_step"] = history.best_step
        history.file.write(json.dumps(row, separators=(",", ":")) + "\n")
    history.flush()


class LiveDashboard:
    def __init__(self, html_path: Path, history: LossHistory, initial_state: dict[str, Any]):
        self.html_path = html_path
        self.history = history
        self.state = initial_state
        self.lock = threading.RLock()
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    try:
                        page = dashboard.html_path.read_bytes()
                    except OSError:
                        self.send_error(404, "code.html não encontrado")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(page)
                elif parsed.path == "/api/state":
                    with dashboard.lock:
                        payload = dict(dashboard.state)
                    payload.update(dashboard.history.summary())
                    self.send_json(payload)
                elif parsed.path == "/api/history":
                    try:
                        offset = int(parse_qs(parsed.query).get("offset", ["0"])[0])
                    except ValueError:
                        offset = 0
                    chunk, next_offset = dashboard.history.read_chunk(offset)
                    self.send_json({"text": chunk, "offset": next_offset})
                else:
                    self.send_error(404)

            def send_json(self, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()
        opened = False
        try:
            opened = webbrowser.get("firefox").open(self.url, new=2)
        except (webbrowser.Error, OSError):
            pass
        print(f"Painel de treino: {self.url}")
        if opened:
            print("code.html aberto no Firefox.")
        else:
            print("Não encontrei o Firefox neste ambiente; abra o endereço acima no Firefox.")

    def update(self, **values: Any) -> None:
        with self.lock:
            self.state.update(values)


def device_report(device: str, model: nn.Module | None = None) -> str:
    selected = torch.device(device)
    if selected.type != "cuda":
        if model is not None and next(model.parameters()).device.type != selected.type:
            raise RuntimeError(f"O modelo está em {next(model.parameters()).device}, mas foi pedido {selected}.")
        return f"Dispositivo ativo: {selected}"
    if not torch.cuda.is_available():
        raise SystemExit("Foi pedido CUDA, mas torch.cuda.is_available() retornou False. Confira PyTorch CUDA e drivers.")
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index < 0 or index >= torch.cuda.device_count():
        raise SystemExit(f"GPU cuda:{index} não existe; GPUs disponíveis: {torch.cuda.device_count()}.")
    if model is not None:
        model_device = next(model.parameters()).device
        if model_device.type != "cuda" or model_device.index != index:
            raise RuntimeError(f"Modelo em {model_device}; esperado cuda:{index}.")
        torch.cuda.synchronize(index)
    allocated = torch.cuda.memory_allocated(index) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(index) / (1024 ** 3)
    total = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    name = torch.cuda.get_device_name(index)
    return (f"GPU ativa: cuda:{index} ({name}) | memória CUDA alocada {allocated:.2f} GiB, "
            f"reservada {reserved:.2f}/{total:.2f} GiB")


def token_batches(
    args: argparse.Namespace,
    tokenizer: Any,
    device: str,
    token_map: torch.Tensor,
    *,
    validation: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    stream = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    if args.validation_documents:
        stream = stream.take(args.validation_documents) if validation else stream.skip(args.validation_documents)
    if not validation:
        stream = stream.shuffle(seed=args.seed if seed is None else seed, buffer_size=args.shuffle_buffer)
    token_buffer: list[int] = []
    needed = args.batch_size * (args.block_size + 1)
    for row in stream:
        text = row.get(args.text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        token_buffer.append(tokenizer.eos_token_id)
        while len(token_buffer) >= needed:
            values = torch.tensor(token_buffer[:needed], dtype=torch.long, device=device)
            values = token_map[values]
            token_buffer = token_buffer[needed:]
            batch = values.view(args.batch_size, args.block_size + 1)
            yield batch[:, :-1], batch[:, 1:]


def steps_per_epoch(args: argparse.Namespace) -> int:
    tokens_per_step = args.batch_size * args.block_size
    return math.ceil(args.tokens_per_epoch / tokens_per_step)


def training_batches(args: argparse.Namespace, tokenizer: Any, device: str,
                     token_map: torch.Tensor) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """An epoch is a fixed token budget, independent of stream exhaustion."""
    epoch = 0
    while args.epochs is None or epoch < args.epochs:
        epoch += 1
        cycle = 0
        source = token_batches(args, tokenizer, device, token_map, seed=args.seed + epoch + cycle)
        for _ in range(steps_per_epoch(args)):
            try:
                x, y = next(source)
            except StopIteration:
                cycle += 1
                source = token_batches(args, tokenizer, device, token_map, seed=args.seed + epoch + cycle)
                try:
                    x, y = next(source)
                except StopIteration as exc:
                    raise RuntimeError("O stream não produziu nenhum bloco de treino válido.") from exc
            yield epoch, x, y


def collect_validation_batches(args: argparse.Namespace, tokenizer: Any, device: str,
                               token_map: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if args.validation_documents <= 0 or args.validation_batches <= 0:
        raise ValueError("Validação requer --validation-documents e --validation-batches maiores que zero.")
    source = token_batches(args, tokenizer, device, token_map, validation=True)
    batches = []
    for _ in range(args.validation_batches):
        try:
            batches.append(next(source))
        except StopIteration:
            break
    if not batches:
        raise RuntimeError("Nenhum bloco de validação foi gerado; aumente --validation-documents.")
    return batches


def tokenizer_token_ids(tokenizer: Any) -> list[int]:
    """Use every ID from the tokenizer; no frequency-based vocabulary pruning."""
    token_ids = list(range(len(tokenizer)))
    print(f"Vocabulário completo do tokenizer: {len(token_ids):,} tokens; nenhum ID foi descartado.")
    return token_ids


@torch.no_grad()
def evaluate(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    was_training = model.training
    model.eval()
    losses = []
    try:
        for x, y in batches:
            logits = model(x)
            losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)))
    finally:
        model.train(was_training)
    return torch.stack(losses).mean().item()


def build_optimizers(model: VSLM, args: argparse.Namespace) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Este treino usa torch.optim.Muon; atualize para PyTorch 2.9 ou superior.")

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        use_muon = (
            name.startswith("attencioner.projection.") and name.endswith("weight")
            or name.startswith("devectorizer.projection.") and name.endswith("weight")
            or name.startswith("main.layer_projections.") and name.endswith("weight")
        )
        (muon_params if use_muon else adamw_params).append(param)

    muon = torch.optim.Muon(
        muon_params, lr=args.muon_lr, weight_decay=args.weight_decay,
        momentum=0.95, nesterov=True,
    )
    adamw = torch.optim.AdamW(
        adamw_params, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    return muon, adamw


def train(args: argparse.Namespace) -> None:
    if args.steps is None and args.epochs is None:
        args.steps = 10000
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps precisa ser maior que zero.")
    if args.epochs is not None and args.epochs <= 0:
        raise SystemExit("--epochs precisa ser maior que zero.")
    if args.tokens_per_epoch <= 0:
        raise SystemExit("--tokens-per-epoch precisa ser maior que zero.")
    if args.batch_size <= 0 or args.block_size <= 0:
        raise SystemExit("--batch-size e --block-size precisam ser maiores que zero.")
    if args.save_every <= 0 or args.log_every <= 0 or args.validation_every <= 0:
        raise SystemExit("--save-every, --log-every e --validation-every precisam ser maiores que zero.")
    epoch_steps = steps_per_epoch(args)
    planned_steps = args.steps if args.steps is not None else args.epochs * epoch_steps
    loss_base = Path(args.out)
    loss_path = Path(args.loss_log) if args.loss_log else loss_base.with_suffix(".losses.jsonl")
    args.loss_log = str(loss_path)
    history = LossHistory(loss_path)
    dashboard = LiveDashboard(
        Path(__file__).with_name("code.html"), history,
        {
            "status": "Inicializando treino",
            "command": shlex.join(sys.argv),
            "current_step": 0,
            "start_step": 0,
            "final_step": planned_steps,
            "steps_per_epoch": epoch_steps,
            "requested_epochs": args.epochs,
            "loss_log": str(loss_path),
            "history_revision": 0,
        },
    )
    dashboard.start()
    device = args.device
    print(device_report(device))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.add_tokens(["<|unk|>"])
    if tokenizer.eos_token_id is None:
        raise SystemExit("O tokenizer precisa definir eos_token_id para separar documentos.")
    optimizer_state = None
    start_step = 0
    best_loss = math.inf
    last_loss: float | None = None
    out_path = Path(args.out)
    best_out = Path(args.best_out) if args.best_out else out_path.with_name(f"{out_path.stem}.best{out_path.suffix}")
    if out_path.resolve() == best_out.resolve():
        raise SystemExit("--best-out precisa ser diferente de --out.")
    dashboard.update(best_out=str(best_out))
    if args.resume:
        model, _, saved_tokenizer, optimizer_state, start_step, best_loss = VSLM.load(args.resume, device)
        if saved_tokenizer != args.tokenizer or model.original_vocab_size != len(tokenizer):
            raise SystemExit("O checkpoint usa um tokenizer/vocabulário diferente do configurado.")
        if model.ctm_synchrony != args.ctm_synchrony:
            raise SystemExit("A ablação --ctm-synchrony precisa corresponder à configuração do checkpoint.")
        print(f"Retomando de {args.resume}, passo salvo {start_step}.")
        history.truncate_after(start_step)
    else:
        compact_to_original = tokenizer_token_ids(tokenizer)
        model = VSLM(compact_to_original, len(tokenizer), ctm_synchrony=args.ctm_synchrony).to(device)
    if args.resume and not history.latest:
        seed_legacy_history(history, dashboard.html_path, start_step)
    if args.resume:
        dashboard.update(history_revision=1)
    if history.best_validation is not None:
        best_loss = min(best_loss, history.best_validation)
    best_step = history.best_step
    if best_step is None and math.isfinite(best_loss):
        best_step = start_step
    dashboard.update(
        status="Preparando dados",
        current_step=start_step,
        run_step=0,
        current_epoch=1,
        start_step=start_step,
        final_step=start_step + planned_steps,
        best_loss=best_loss if math.isfinite(best_loss) else None,
        best_step=best_step,
    )
    print(device_report(device, model))
    report = model.parameter_report()
    print("Parâmetros treináveis:")
    for name, count in report.items():
        print(f"  {name}: {count:,}")
    if report["Total"] > 256_000_000:
        raise RuntimeError(f"O modelo ultrapassa o alvo de 256 milhões: {report['Total']:,}")
    muon, adamw = build_optimizers(model, args)
    if optimizer_state and "muon" in optimizer_state and "adamw" in optimizer_state:
        muon.load_state_dict(optimizer_state["muon"])
        adamw.load_state_dict(optimizer_state["adamw"])
        print("Estados do Muon e do AdamW restaurados do checkpoint.")
    elif args.resume:
        print("Checkpoint sem estados compatíveis dos otimizadores: pesos restaurados, Muon e AdamW reiniciados.")
    validation_batches = collect_validation_batches(args, tokenizer, device, model.token_to_compact)
    batches = training_batches(args, tokenizer, device, model.token_to_compact)
    forward_model: nn.Module = model
    if args.compile:
        if not hasattr(torch, "compile"):
            raise SystemExit("Este PyTorch não oferece torch.compile.")
        forward_model = torch.compile(model)
        print("torch.compile habilitado; a compilação inicial pode demorar.")
    model.train()
    max_steps = planned_steps
    final_step = start_step + max_steps
    completed_step = start_step
    last_epoch = 0
    try:
        for epoch, x, y in batches:
            if max_steps is not None and completed_step - start_step >= max_steps:
                break
            if epoch != last_epoch:
                if last_epoch:
                    suffix = f"/{args.epochs}" if args.epochs is not None else ""
                    print(f"Época {last_epoch}{suffix} concluída ({args.tokens_per_epoch:,} tokens).")
                last_epoch = epoch
            step = completed_step + 1
            logits = forward_model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            muon.step()
            adamw.step()
            completed_step = step
            last_loss = loss.item()
            validation_loss: float | None = None
            if step == 1 or step % args.validation_every == 0:
                validation_loss = evaluate(forward_model, validation_batches)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_step = step
                    states = {"muon": muon.state_dict(), "adamw": adamw.state_dict()}
                    model.save(best_out, args.tokenizer, vars(args), step, states,
                               loss=last_loss, best_loss=best_loss)
                    print(f"Nova melhor validação no passo {step}: {best_loss:.4f} | {best_out}")
            run_step = step - start_step
            history.append({
                "step": step,
                "epoch": epoch,
                "run_step": run_step,
                "train_loss": last_loss,
                "validation_loss": validation_loss,
                "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
                "best_validation_step": best_step,
            })
            dashboard.update(
                status="Treinando",
                current_step=step,
                run_step=run_step,
                current_epoch=epoch,
                best_loss=best_loss if math.isfinite(best_loss) else None,
                best_step=best_step,
                final_step=final_step,
                steps_per_epoch=epoch_steps,
            )
            if step == 1 or step % args.log_every == 0:
                history.flush()
                message = f"passo {step}/{final_step} | época {epoch} | treino {loss.item():.4f}"
                if validation_loss is not None:
                    message += f" | validação {validation_loss:.4f}"
                if torch.device(device).type == "cuda":
                    message += " | " + device_report(device, model)
                print(message)
            if step % args.save_every == 0:
                history.flush()
                states = {"muon": muon.state_dict(), "adamw": adamw.state_dict()}
                model.save(args.out, args.tokenizer, vars(args), step, states,
                           loss=last_loss, best_loss=best_loss)
                print(f"Checkpoint automático no passo {step}: {args.out}")
        if args.epochs is not None and last_epoch:
            print(f"Época {last_epoch}/{args.epochs} concluída ({args.tokens_per_epoch:,} tokens).")
    except KeyboardInterrupt:
        print("Interrupção solicitada; salvando o último passo concluído...")
        history.flush()
        states = {"muon": muon.state_dict(), "adamw": adamw.state_dict()}
        model.save(args.out, args.tokenizer, vars(args), completed_step, states,
                   loss=last_loss, best_loss=best_loss)
        print(f"Checkpoint salvo no passo {completed_step}: {args.out}")
        dashboard.update(status="Interrompido", current_step=completed_step)
        history.close()
        return
    except Exception as exc:
        history.flush()
        dashboard.update(status=f"Erro: {exc}", current_step=completed_step)
        history.close()
        raise
    history.flush()
    states = {"muon": muon.state_dict(), "adamw": adamw.state_dict()}
    model.save(args.out, args.tokenizer, vars(args), completed_step, states,
               loss=last_loss, best_loss=best_loss)
    tokenizer.save_pretrained(str(Path(args.out).with_suffix(".tokenizer")))
    print(f"Checkpoint final salvo em: {args.out}")
    dashboard.update(status="Concluído", current_step=completed_step)
    history.close()
def main() -> None:
    if sys.argv[1:2] == ["generate"]:
        parser = argparse.ArgumentParser(description="Gera texto com um checkpoint VSLM")
        parser.add_argument("--model", required=True)
        parser.add_argument("--prompt", default="")
        parser.add_argument("--new-tokens", type=int, default=200)
        parser.add_argument("--temperature", type=float, default=0.8)
        parser.add_argument("--top-k", type=int, default=40)
        parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        args = parser.parse_args(sys.argv[2:])
        print(device_report(args.device))
        model, _, tokenizer_name, _, _, _ = VSLM.load(args.model, args.device)
        print(device_report(args.device, model))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        tokenizer.add_tokens(["<|unk|>"])
        model.eval()
        print(model.generate(args.prompt, tokenizer, args.new_tokens, args.temperature, args.top_k, args.device))
        return

    parser = argparse.ArgumentParser(description="Treina o VSLM em Cosmopedia")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", default="vslm-256m.pt")
    parser.add_argument("--best-out", default=None)
    parser.add_argument("--loss-log", default=None,
                        help="Arquivo JSONL persistente do histórico de loss; padrão: ao lado do checkpoint.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int, default=None)
    duration.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tokens-per-epoch", type=int, default=1_000_000)
    parser.add_argument("--validation-documents", type=int, default=128)
    parser.add_argument("--vocab-sample-tokens", type=int, default=1_000_000,
                        help="Opção legada, mantida por compatibilidade; o vocabulário completo é usado.")
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--shuffle-buffer", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--ctm-synchrony", action=argparse.BooleanOptionalAction, default=True,
                        help="Ativa/desativa os recursos de sincronização amostrada do CTM.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
