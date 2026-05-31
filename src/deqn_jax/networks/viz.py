"""Graphviz diagrams for Equinox policy networks.

torchview-style architecture visualization. Walks an Equinox network's
declared structure (not a forward trace) and emits a DOT diagram with
Linear / activation / LayerNorm / LSTMCell / attention boxes plus shape
annotations on the edges.

Usage:

    from deqn_jax.networks.viz import to_dot, render_to_file

    dot_src = to_dot(model)             # DOT source as a string
    render_to_file(model, "mlp.png")    # writes PNG (needs `dot` binary)

Currently supports MLP, ResMLP, MultiHeadMLP, LSTMPolicy,
TransformerPolicy, and LinearPlusMLP. Anything else raises
``UnsupportedModelError``.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

import equinox as eqx


class UnsupportedModelError(TypeError):
    """Raised when ``to_dot`` is called on a model class it doesn't know how to render."""


@dataclass
class _DotBuilder:
    """Tiny DOT builder. No external deps; produces deterministic source."""

    name: str = "network"
    nodes: List[Tuple[str, str, str, str]] = field(default_factory=list)
    edges: List[Tuple[str, str, str, str]] = field(default_factory=list)
    clusters: List[Tuple[str, str, List[str]]] = field(default_factory=list)

    def node(
        self,
        node_id: str,
        label: str,
        shape: str = "box",
        fillcolor: str = "#E3F2FD",
    ) -> str:
        self.nodes.append((node_id, label, shape, fillcolor))
        return node_id

    def edge(self, src: str, dst: str, label: str = "", style: str = "solid") -> None:
        self.edges.append((src, dst, label, style))

    def cluster(self, cluster_id: str, label: str, node_ids: List[str]) -> None:
        self.clusters.append((cluster_id, label, list(node_ids)))

    def build(self) -> str:
        lines: List[str] = [f'digraph "{_escape(self.name)}" {{']
        lines.append("  rankdir=TB;")
        lines.append('  graph [fontname="Helvetica" fontsize=11];')
        lines.append(
            '  node [fontname="Helvetica" fontsize=10 style="rounded,filled"];'
        )
        lines.append('  edge [fontname="Helvetica" fontsize=9];')

        clustered: set[str] = set()
        for cid, clabel, members in self.clusters:
            clustered.update(members)
            lines.append(f'  subgraph "cluster_{cid}" {{')
            lines.append(f'    label="{_escape(clabel)}";')
            lines.append('    style="rounded,dashed";')
            lines.append('    color="#9E9E9E";')
            lines.append("    fontsize=10;")
            for nid, nlabel, nshape, nfill in self.nodes:
                if nid in members:
                    lines.append(
                        f'    "{nid}" [label="{_escape(nlabel)}" '
                        f'shape={nshape} fillcolor="{nfill}"];'
                    )
            lines.append("  }")

        for nid, nlabel, nshape, nfill in self.nodes:
            if nid in clustered:
                continue
            lines.append(
                f'  "{nid}" [label="{_escape(nlabel)}" '
                f'shape={nshape} fillcolor="{nfill}"];'
            )

        for src, dst, label, style in self.edges:
            attrs = []
            if label:
                attrs.append(f'label="{_escape(label)}"')
            if style != "solid":
                attrs.append(f'style="{style}"')
            attr_str = " [" + " ".join(attrs) + "]" if attrs else ""
            lines.append(f'  "{src}" -> "{dst}"{attr_str};')

        lines.append("}")
        return "\n".join(lines)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Color palette
_C_IO = "#E8F5E9"  # input/output (green)
_C_LINEAR = "#E3F2FD"  # Linear (blue)
_C_ACT = "#F3E5F5"  # activation (purple)
_C_NORM = "#FFF3E0"  # normalize / LayerNorm (orange)
_C_BOUNDS = "#FFEBEE"  # bounds (red)
_C_RECURRENT = "#E0F7FA"  # LSTM cell (cyan)
_C_ATTN = "#FCE4EC"  # attention (pink)
_C_RESIDUAL = "#F1F8E9"  # residual / pos_embed (light green)


def _act_name(fn: Callable) -> str:
    name = getattr(fn, "__name__", None)
    if name:
        return name
    return repr(fn).split(" at ")[0].lstrip("<")


def _maybe_normalize(model, b: _DotBuilder, prev: str, in_features: int) -> str:
    if model.input_shift is None:
        return prev
    nid = b.node("normalize", "normalize\n(shift, scale)", fillcolor=_C_NORM)
    b.edge(prev, nid, label=f"[{in_features}]")
    return nid


def _maybe_bounds(model, b: _DotBuilder, prev: str, out_features: int) -> str:
    if model.output_lower is None and model.output_upper is None:
        return prev
    nid = b.node("bounds", "_apply_bounds\n(softplus / sigmoid)", fillcolor=_C_BOUNDS)
    b.edge(prev, nid, label=f"[{out_features}]")
    return nid


def _render_mlp(model, b: _DotBuilder) -> None:
    in_features = model.layers[0].in_features
    out_features = model.layers[-1].out_features
    n_layers = len(model.layers)

    inp = b.node("input", f"input\n[{in_features}]", shape="ellipse", fillcolor=_C_IO)
    prev = _maybe_normalize(model, b, inp, in_features)

    for i, layer in enumerate(model.layers):
        is_last = i == n_layers - 1
        lid = b.node(
            f"linear_{i}",
            f"Linear\n{layer.in_features} → {layer.out_features}",
            fillcolor=_C_LINEAR,
        )
        b.edge(prev, lid, label=f"[{layer.in_features}]")
        prev = lid

        if not is_last:
            aid = b.node(
                f"act_{i}",
                _act_name(model.activations[i]),
                shape="diamond",
                fillcolor=_C_ACT,
            )
            b.edge(prev, aid, label=f"[{layer.out_features}]")
            prev = aid

    prev = _maybe_bounds(model, b, prev, out_features)
    out = b.node(
        "output", f"output\n[{out_features}]", shape="ellipse", fillcolor=_C_IO
    )
    b.edge(prev, out, label=f"[{out_features}]")


def _render_resmlp(model, b: _DotBuilder) -> None:
    in_features = model.layers[0].in_features
    out_features = model.layers[-1].out_features
    n_layers = len(model.layers)

    inp = b.node("input", f"input\n[{in_features}]", shape="ellipse", fillcolor=_C_IO)
    prev = _maybe_normalize(model, b, inp, in_features)

    for i, layer in enumerate(model.layers):
        is_last = i == n_layers - 1
        residual_src = prev
        lid = b.node(
            f"linear_{i}",
            f"Linear\n{layer.in_features} → {layer.out_features}",
            fillcolor=_C_LINEAR,
        )
        b.edge(prev, lid, label=f"[{layer.in_features}]")

        if is_last:
            prev = lid
            continue

        aid = b.node(
            f"act_{i}",
            _act_name(model.activations[i]),
            shape="diamond",
            fillcolor=_C_ACT,
        )
        b.edge(lid, aid, label=f"[{layer.out_features}]")

        add_id = b.node(f"add_{i}", "+", shape="circle", fillcolor=_C_RESIDUAL)
        b.edge(aid, add_id, label=f"[{layer.out_features}]")
        proj = model.skip_projs[i]
        if proj is None:
            b.edge(residual_src, add_id, label="identity", style="dashed")
        else:
            pid = b.node(
                f"skip_proj_{i}",
                f"Linear (skip)\n{proj.in_features} → {proj.out_features}",
                fillcolor=_C_LINEAR,
            )
            b.edge(residual_src, pid, label=f"[{proj.in_features}]", style="dashed")
            b.edge(pid, add_id, label=f"[{proj.out_features}]", style="dashed")
        prev = add_id

    prev = _maybe_bounds(model, b, prev, out_features)
    out = b.node(
        "output", f"output\n[{out_features}]", shape="ellipse", fillcolor=_C_IO
    )
    b.edge(prev, out, label=f"[{out_features}]")


def _render_multihead_mlp(model, b: _DotBuilder) -> None:
    in_features = model.trunk_layers[0].in_features
    n_heads = len(model.heads)

    inp = b.node("input", f"input\n[{in_features}]", shape="ellipse", fillcolor=_C_IO)
    prev = _maybe_normalize(model, b, inp, in_features)

    for i, layer in enumerate(model.trunk_layers):
        lid = b.node(
            f"trunk_{i}",
            f"Linear\n{layer.in_features} → {layer.out_features}",
            fillcolor=_C_LINEAR,
        )
        b.edge(prev, lid, label=f"[{layer.in_features}]")
        aid = b.node(
            f"act_{i}",
            _act_name(model.activations[i]),
            shape="diamond",
            fillcolor=_C_ACT,
        )
        b.edge(lid, aid, label=f"[{layer.out_features}]")
        prev = aid

    head_in = model.heads[0].in_features
    head_ids = []
    for j, head in enumerate(model.heads):
        hid = b.node(
            f"head_{j}",
            f"Head {j}\nLinear {head.in_features} → {head.out_features}",
            fillcolor=_C_LINEAR,
        )
        b.edge(prev, hid, label=f"[{head_in}]")
        head_ids.append(hid)
    b.cluster("heads", f"per-policy heads ({n_heads})", head_ids)

    concat = b.node("concat", "concat", shape="trapezium", fillcolor=_C_RESIDUAL)
    for hid in head_ids:
        b.edge(hid, concat, label="[1]")

    prev = _maybe_bounds(model, b, concat, n_heads)
    out = b.node("output", f"output\n[{n_heads}]", shape="ellipse", fillcolor=_C_IO)
    b.edge(prev, out, label=f"[{n_heads}]")


def _render_lstm(model, b: _DotBuilder) -> None:
    in_features = model.input_proj.in_features
    hidden = model.input_proj.out_features
    out_features = model.output_proj.out_features
    H = model.history_len

    inp = b.node(
        "input", f"input\n[H={H}, {in_features}]", shape="ellipse", fillcolor=_C_IO
    )
    prev = inp
    if model.input_shift is not None:
        nid = b.node("normalize", "vmap normalize\n(shift, scale)", fillcolor=_C_NORM)
        b.edge(prev, nid, label=f"[H, {in_features}]")
        prev = nid

    proj = b.node(
        "input_proj",
        f"vmap Linear\n{in_features} → {hidden}\n+ tanh",
        fillcolor=_C_LINEAR,
    )
    b.edge(prev, proj, label=f"[H, {in_features}]")
    prev = proj

    cell_ids = []
    for i, cell in enumerate(model.cells):
        cid = b.node(
            f"lstm_{i}",
            f"LSTMCell layer {i}\nin={cell.input_size}, hidden={cell.hidden_size}\nscan over H",
            fillcolor=_C_RECURRENT,
        )
        b.edge(prev, cid, label=f"[H, {hidden}]")
        cell_ids.append(cid)
        prev = cid
    if cell_ids:
        b.cluster("lstm_stack", f"stacked LSTM ({len(cell_ids)} layer(s))", cell_ids)

    extract = b.node(
        "extract_last", "take last h\n(scan output[-1])", fillcolor=_C_RESIDUAL
    )
    b.edge(prev, extract, label=f"[H, {hidden}]")

    out_proj = b.node(
        "output_proj",
        f"Linear\n{hidden} → {out_features}",
        fillcolor=_C_LINEAR,
    )
    b.edge(extract, out_proj, label=f"[{hidden}]")

    prev = _maybe_bounds(model, b, out_proj, out_features)
    out = b.node(
        "output", f"output\n[{out_features}]", shape="ellipse", fillcolor=_C_IO
    )
    b.edge(prev, out, label=f"[{out_features}]")


def _render_transformer_block(
    b: _DotBuilder, block, idx: int, hidden_dim: int
) -> Tuple[str, str, List[str]]:
    """Render one Pre-LN block. Returns (entry_node, exit_node, all_node_ids)."""
    pre = f"block_{idx}"
    members: List[str] = []

    in_node = b.node(
        f"{pre}_in", f"in\n[H, {hidden_dim}]", shape="point", fillcolor="#000000"
    )
    members.append(in_node)

    ln1 = b.node(f"{pre}_ln1", "LayerNorm 1", fillcolor=_C_NORM)
    members.append(ln1)
    b.edge(in_node, ln1, label=f"[H, {hidden_dim}]")

    n_heads = block.attn.num_heads
    head_dim = block.attn.head_dim
    attn = b.node(
        f"{pre}_attn",
        f"MultiHeadAttn\nheads={n_heads}, head_dim={head_dim}\nQ/K/V/O Linear",
        fillcolor=_C_ATTN,
    )
    members.append(attn)
    b.edge(ln1, attn, label=f"[H, {hidden_dim}]")

    add1 = b.node(f"{pre}_add1", "+", shape="circle", fillcolor=_C_RESIDUAL)
    members.append(add1)
    b.edge(attn, add1, label=f"[H, {hidden_dim}]")
    b.edge(in_node, add1, label="residual", style="dashed")

    ln2 = b.node(f"{pre}_ln2", "LayerNorm 2", fillcolor=_C_NORM)
    members.append(ln2)
    b.edge(add1, ln2, label=f"[H, {hidden_dim}]")

    ffn_up_dim = block.ffn_up.out_features
    ffn = b.node(
        f"{pre}_ffn",
        f"FFN\nLinear {hidden_dim} → {ffn_up_dim}\ngelu\nLinear {ffn_up_dim} → {hidden_dim}",
        fillcolor=_C_LINEAR,
    )
    members.append(ffn)
    b.edge(ln2, ffn, label=f"[H, {hidden_dim}]")

    add2 = b.node(f"{pre}_add2", "+", shape="circle", fillcolor=_C_RESIDUAL)
    members.append(add2)
    b.edge(ffn, add2, label=f"[H, {hidden_dim}]")
    b.edge(add1, add2, label="residual", style="dashed")

    return in_node, add2, members


def _render_transformer(model, b: _DotBuilder) -> None:
    in_features = model.input_proj.in_features
    hidden = model.input_proj.out_features
    out_features = model.output_proj.out_features
    H = model.history_len

    inp = b.node(
        "input", f"input\n[H={H}, {in_features}]", shape="ellipse", fillcolor=_C_IO
    )
    prev = inp
    if model.input_shift is not None:
        nid = b.node("normalize", "vmap normalize\n(shift, scale)", fillcolor=_C_NORM)
        b.edge(prev, nid, label=f"[H, {in_features}]")
        prev = nid

    proj = b.node(
        "input_proj",
        f"vmap Linear\n{in_features} → {hidden}",
        fillcolor=_C_LINEAR,
    )
    b.edge(prev, proj, label=f"[H, {in_features}]")

    pos = b.node("pos_embed", f"+ pos_embed\n[H, {hidden}]", fillcolor=_C_RESIDUAL)
    b.edge(proj, pos, label=f"[H, {hidden}]")
    prev = pos

    for i, block in enumerate(model.blocks):
        entry, exit_, members = _render_transformer_block(b, block, i, hidden)
        b.edge(prev, entry, label=f"[H, {hidden}]")
        b.cluster(f"block_{i}", f"TransformerBlock {i}", members)
        prev = exit_

    extract = b.node("extract_last", "x[-1]\n(last timestep)", fillcolor=_C_RESIDUAL)
    b.edge(prev, extract, label=f"[H, {hidden}]")

    final_ln = b.node("final_ln", "LayerNorm", fillcolor=_C_NORM)
    b.edge(extract, final_ln, label=f"[{hidden}]")

    out_proj = b.node(
        "output_proj",
        f"Linear\n{hidden} → {out_features}",
        fillcolor=_C_LINEAR,
    )
    b.edge(final_ln, out_proj, label=f"[{hidden}]")

    prev = _maybe_bounds(model, b, out_proj, out_features)
    out = b.node(
        "output", f"output\n[{out_features}]", shape="ellipse", fillcolor=_C_IO
    )
    b.edge(prev, out, label=f"[{out_features}]")


def _render_linear_plus_mlp(model, b: _DotBuilder) -> None:
    n_states = model.ss_state.shape[0]
    n_policies = model.ss_policy.shape[0]

    inp = b.node("input", f"input\n[{n_states}]", shape="ellipse", fillcolor=_C_IO)

    linear = b.node(
        "linear_branch",
        f"linear branch\nss_policy + P @ (state - ss_state)\n[{n_states}] → [{n_policies}]",
        fillcolor=_C_LINEAR,
    )
    b.edge(inp, linear, label=f"[{n_states}]")

    mlp_label_lines = ["MLP correction (delta)"]
    # use_zlb_feature, r_lag_idx, r_lb only exist on DisasterPolicyNet, not on
    # the generic LinearPlusMLP. getattr keeps this renderer compatible with both.
    if getattr(model, "use_zlb_feature", False):
        mlp_label_lines.append(
            f"input augmented with (R_lag - R_lb)\nat idx {model.r_lag_idx}, R_lb={model.r_lb}"
        )
    inner = model.mlp
    sizes = [inner.layers[0].in_features] + [
        layer.out_features for layer in inner.layers
    ]
    mlp_label_lines.append("sizes: " + " → ".join(str(s) for s in sizes))
    mlp = b.node("mlp_branch", "\n".join(mlp_label_lines), fillcolor=_C_LINEAR)
    b.edge(inp, mlp, label=f"[{n_states}]")

    add = b.node("add", "+", shape="circle", fillcolor=_C_RESIDUAL)
    b.edge(linear, add, label=f"[{n_policies}]")
    b.edge(mlp, add, label=f"[{n_policies}]")

    prev = add
    if model.policy_lower is not None or model.policy_upper is not None:
        clip = b.node(
            "clip", "hard clip\n(policy_lower / policy_upper)", fillcolor=_C_BOUNDS
        )
        b.edge(prev, clip, label=f"[{n_policies}]")
        prev = clip

    out = b.node("output", f"output\n[{n_policies}]", shape="ellipse", fillcolor=_C_IO)
    b.edge(prev, out, label=f"[{n_policies}]")


_RENDERERS: List[Tuple[str, Callable]] = []

# Renderers contributed by model packages (e.g. a model's own policy net),
# populated via register_network_renderer at the model module's import time.
# This inverts the old hard dependency where this generic viz module imported
# a specific model class (audit networks-03).
_EXTERNAL_RENDERERS: List[Tuple[str, Callable]] = []


def register_network_renderer(name, predicate, renderer) -> None:
    """Register a diagram renderer for a model-specific network type.

    ``predicate(model) -> bool`` selects instances; ``renderer(model, builder)``
    draws them. A model's network module calls this so this package never has
    to import model-specific classes. Idempotent per ``name``.
    """
    if any(n == name for n, _ in _EXTERNAL_RENDERERS):
        return
    _EXTERNAL_RENDERERS.append((name, lambda m, b: (predicate(m), renderer)))


def _register_renderers() -> None:
    if _RENDERERS:
        return
    from deqn_jax.networks.linear_plus_mlp import LinearPlusMLP
    from deqn_jax.networks.lstm import LSTMPolicy
    from deqn_jax.networks.mlp import MLP, MultiHeadMLP, ResMLP
    from deqn_jax.networks.transformer import TransformerPolicy

    _RENDERERS.extend(
        [
            (ResMLP.__name__, lambda m, b: (isinstance(m, ResMLP), _render_resmlp)),
            (
                MultiHeadMLP.__name__,
                lambda m, b: (isinstance(m, MultiHeadMLP), _render_multihead_mlp),
            ),
            (
                LinearPlusMLP.__name__,
                lambda m, b: (isinstance(m, LinearPlusMLP), _render_linear_plus_mlp),
            ),
            (MLP.__name__, lambda m, b: (isinstance(m, MLP), _render_mlp)),
            (
                LSTMPolicy.__name__,
                lambda m, b: (isinstance(m, LSTMPolicy), _render_lstm),
            ),
            (
                TransformerPolicy.__name__,
                lambda m, b: (isinstance(m, TransformerPolicy), _render_transformer),
            ),
        ]
    )


def to_dot(model: eqx.Module, name: str = "network") -> str:
    """Return DOT source for a graphviz diagram of ``model``.

    Currently supports MLP, ResMLP, MultiHeadMLP, LSTMPolicy,
    TransformerPolicy, and LinearPlusMLP. Anything else raises
    ``UnsupportedModelError``.
    """
    _register_renderers()
    b = _DotBuilder(name=name)
    for _name, dispatcher in _RENDERERS + _EXTERNAL_RENDERERS:
        matches, renderer = dispatcher(model, b)
        if matches:
            renderer(model, b)
            return b.build()
    raise UnsupportedModelError(
        f"to_dot does not know how to render {type(model).__name__}. "
        "Supported: MLP, ResMLP, MultiHeadMLP, LSTMPolicy, "
        "TransformerPolicy, LinearPlusMLP."
    )


def render_to_file(
    model: eqx.Module,
    output_path: str | Path,
    fmt: str = "png",
    name: str = "network",
) -> Path:
    """Render ``model`` to ``output_path`` via the ``dot`` binary.

    Requires graphviz's ``dot`` to be on PATH (``brew install graphviz``).
    Returns the absolute output path.
    """
    if shutil.which("dot") is None:
        raise RuntimeError(
            "`dot` binary not found on PATH. Install graphviz "
            "(e.g. `brew install graphviz`) or use `to_dot()` to get DOT source."
        )
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dot_src = to_dot(model, name=name)
    subprocess.run(
        ["dot", f"-T{fmt}", "-o", str(output_path)],
        input=dot_src.encode("utf-8"),
        check=True,
    )
    return output_path
