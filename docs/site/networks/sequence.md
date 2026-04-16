# MLP / LSTM / Transformer

DEQN-JAX networks are Equinox modules. The framework supports both
**Markovian** and **history-dependent** policies transparently.

## MLP (Markov)

Standard feedforward, takes a single state vector.

```yaml
network:
  type: mlp
  hidden_sizes: [128, 128]
  activation: tanh
  init: xavier_normal
```

## LSTM (history-dependent)

Sequence policy. Takes a window of recent states `[H, n_states]`.

```yaml
network:
  type: lstm
  hidden_sizes: [64]
  history_len: 8
```

## Transformer (history-dependent)

Multi-head attention over a state-window sequence. Useful when the
policy benefits from longer context than LSTM can carry.

```yaml
network:
  type: transformer
  hidden_sizes: [64]
  history_len: 16
  n_heads: 4
```

## How dispatch works

`compute_residuals` checks the input rank — `[B, D]` (Markov) vs
`[B, H, D]` (sequence) — and routes accordingly. Episode simulation
similarly handles both via `make_constant_history` and
`build_history_windows` helpers.

For the residual-style architecture that combines MLPs with a
linearization prior, see [LinearPlusMLP](linear_plus_mlp.md).
