# Typing Audit (deqn-jax src/)

Generated 2026-04-25 by Ralph Loop typing iteration 1.
Tool versions: `ty 0.0.32`, `pyright` (ad-hoc via `uvx`).

## Summary

- Total: **30** diagnostics on `uvx ty check src/` (was 104 at the start of phase 2; **stop target reached**).
- Bucket counts: REAL_BUG=2, ANNOTATION_LIE=2, EQX_NOISE=11, JAX_NOISE=4, PYDANTIC_DICT=0, OPTIONAL_NARROWING=9, DECISION_NEEDED=0.
- Stop target: **≤ 30 diagnostics** (the 27 OPTIONAL_NARROWING + 2 PYDANTIC_DICT collapse to one source-of-truth fix each, which leaves the residual JAX/EQX framework noise).

## Suppression syntax

ty and Pyright don't share ignore-comment grammars and use different error codes for the same issue:

```python
# Both suppressed (project standard for SUPPRESSED items)
foo()  # pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]
```

`# type: ignore` (legacy mypy form) is recognised by Pyright but **not** by ty 0.0.32, so it's not enough on its own.

## Workplan (ordered by leverage)

### 1. ModelSpec optional fields that are practically required  [ANNOTATION_LIE]  [STATUS: BLOCKED: dropping `Optional` makes the fields positionally required, which breaks 7 toy-model constructions in `tests/`. Test edits beyond import-path updates are out of this loop's scope. Split into 1a (names → default to `()`, no test impact) and 1b (`definitions_fn` → requires test updates, do separately).]

`types.py` declares `definitions_fn`, `state_names`, `policy_names` as `Optional[...]`, but every live model sets them and the framework calls them unconditionally. Producing 27 `OPTIONAL_NARROWING` errors downstream (`model.definitions_fn(...)`, `list(model.state_names)`, etc.).

Sample sites:
- `irf.py:83` — `model.definitions_fn(state, policy_net(state), constants)`
- `trainer.py:1034` — `lambda s, p: model.definitions_fn(s, p, model.constants)`
- `disaster/diagnostics.py:46` — same pattern
- `composite_loss.py:87` — `model.definitions_fn(ss_state, ss_policy, model.constants)`
- `warm_start.py:323` — `list(model.state_names)`
- `warm_start.py:326` — `list(model.policy_names)`
- 21 more of the same shape

**Plan:** In `types.py`, drop the `Optional` from `definitions_fn`, `state_names`, `policy_names`; require all model packages to provide them (a quick `git grep` confirms they already do). Keep `steady_state_fn` and `init_state_fn` Optional — those legitimately don't apply to every model. This kills ~25 errors with one annotation change.
**Cost:** S.

### 1a. ModelSpec name fields default to ``()`` instead of None  [ANNOTATION_LIE]  [STATUS: DONE]

Subset of #1 that doesn't break tests: change `state_names`, `policy_names`, `equation_names` from `Optional[Tuple[str, ...]] = None` to `Tuple[str, ...] = ()`. Tests' toy models construct without these fields, which then default to empty tuple instead of None. Truthy-check call sites (`if model.state_names: ...`) keep working; `list(model.state_names)` returns `[]` instead of raising. Eliminates ~10 narrowing errors at the call sites.

**Plan:** Three lines in `types.py`. Verify tests still pass; verify no code path actually relies on the difference between `None` and `()`.
**Cost:** S.

### 1b. ``definitions_fn`` should be required  [ANNOTATION_LIE]  [STATUS: DONE: took the audit's option (b) -- assert at use sites instead of changing the type. Five sites (composite_loss.py, disaster/diagnostics.py x2, trainer.py, composite_loss.py inner lambda) now bind ``defs_fn = model.definitions_fn`` after asserting non-None so lambdas keep the narrowing.]

The harder half of #1. Every shipped model defines `definitions_fn` and the framework calls it unconditionally on every cycle log path. Dropping the `Optional` requires updating ~7 toy `ModelSpec(...)` constructions in `tests/test_basic.py`, `tests/test_history_persistence.py`, `tests/test_training_contracts.py` to add a `definitions_fn=lambda s, p, c: {}`.

That's a test edit, which the loop's prompt forbids beyond import-path updates. Punting to a human iteration. Saves ~15 errors at use sites.

**Plan:** Either (a) human-driven test updates, or (b) accept Optional + add `assert model.definitions_fn is not None, "..."` at the use sites (mechanical but pollutes hot paths).
**Cost:** M (because it crosses the test boundary).

### 2. ``Optional[steady_state_fn]`` legitimately optional but called everywhere  [OPTIONAL_NARROWING]  [STATUS: DONE]

`steady_state_fn` is genuinely optional (models like `aiyagari` skip it) but warm_start / linearize / composite_loss / cycle assume it exists. Five errors:
- `warm_start.py:302`
- `composite_loss.py:72, 87`
- `linearize.py:47, 205`
- `cycle.py:60`

**Plan:** Each call site is reached only after the caller already required SS (e.g. composite_loss only runs when `loss_type=composite` which requires linearisation which requires SS). Add `assert model.steady_state_fn is not None, "<reason>"` immediately above each call. Five small edits.
**Cost:** S.

### 3. ``Metrics``-style annotation lies in optimizer states  [ANNOTATION_LIE]  [STATUS: DONE]

`gauss_newton.py:119, 239` build `GaussNewtonState(last_loss=new_loss, ...)` where `new_loss` is a JAX `Array` but `last_loss: float`. Identical pattern to the `Metrics` fix in commit `3ae741f`.

**Plan:** Change `GaussNewtonState.last_loss: float` to `Array` in `gauss_newton.py`. Verify nothing in the optimiser actually consumes it as a Python `float`.
**Cost:** S.

### 4. ``compute_loss(shock_scale: float)`` is called with ``Array``  [ANNOTATION_LIE]  [STATUS: DONE]

`training/loss.py:262` declares `shock_scale: float = 1.0`, but every grad-step factory passes `shock_scale: Array` (see `optimizers/standard.py:65`, `optimizers/lbfgs.py:65, 87`, `optimizers/mao.py:185, 202`, `optimizers/pcgrad.py:55, 70`). 7 `invalid-argument-type` errors.

`compute_loss` already broadcasts `shock_scale` against the per-sample shocks, so the runtime accepts both scalar and `Array`.

**Plan:** Loosen the parameter annotation in `compute_loss` and any helper that flows from it: `shock_scale: float | Array = 1.0`. (Same for `barrier_weight`, `huber_delta` if those also flow from JIT-traced state — check.)
**Cost:** S.

### 5. ``benchmark.py`` is broken at runtime  [REAL_BUG]  [STATUS: DONE]

Three real bugs — the script will crash if anyone runs it:
- `benchmark.py:67, 85` — `train_step(...)` called without `lr_scale` (signature requires it).
- `benchmark.py:68, 95` — `state.params` accessed on the 3-tuple `(state, opt, kind)` returned from `create_train_state(...)`; should unpack first.
- `benchmark.py:122` — return type lies: `Dict[str, str | int | float]` returned but signature promises `Dict[str, float]`.

**Plan:** Unpack `state, opt, kind = create_train_state(...)` at the top, pass `jnp.array(1.0)` as `lr_scale` to the train_step call, and either widen the return type or drop the string-valued result rows.
**Cost:** S.

### 6. ``evaluate.py`` return-type lies  [REAL_BUG]  [STATUS: DONE]

Two functions promise `Dict[str, float]` but return `Dict[str, str | int | float]` because of an `{"error": "..."}` early-return pattern:
- `evaluate.py:262` — `return {"error": "No resource constraint equation found"}`
- `evaluate.py:265` — same function's success path returns mixed types
- `evaluate.py:156` — different function: returns `Dict[str, Array | list]` instead of declared `Dict[str, Array]`

**Plan:** Widen the return signatures (`Dict[str, float | str]` for the error-tolerant ones, `Dict[str, Array | list]` for the other) OR raise on the error path. Quick read of callers to decide.
**Cost:** M.

### 7. ``evaluate.py:437`` operator issue  [DECISION_NEEDED]  [STATUS: DONE: resolved transitively by earlier iterations (Metrics widening, compute_loss param widening). Both ty and pyright now clean on this line; no code change needed.]

`evaluate.py:437` — `0.01 * span` where `span` is `Array | tuple[Array, ...]`. Likely a `jnp.where` returning a tuple in some branch the typer can see but the runtime never hits.

**Plan:** Read the surrounding context to decide whether the type union is real or an annotation issue. If real, narrow with `assert isinstance(span, Array)`; if not, fix the producing function's return type.
**Cost:** S.

### 8. Equinox ``Linear`` / ``Module`` `__init__` typing limitation  [EQX_NOISE]  [STATUS: SUPPRESSED]

`eqx.nn.Linear(...)`, `eqx.nn.LayerNorm(...)`, etc. are typed as returning `Module` (the base class), so every `self.q_proj = eqx.nn.Linear(...)` assignment fails when the field is annotated `Linear`. **22 errors** across `networks/transformer.py` (13), `networks/lstm.py` (3), `networks/mlp.py` (the `_apply_init` invocations, 6).

**Plan:** Add `# pyright: ignore[reportAssignmentType]  # ty: ignore[invalid-assignment]  # eqx.nn.Linear is typed as Module; runtime is Linear` per assignment. Or, more sweeping: declare the eqx layer fields as `eqx.Module` (the base) at the cost of losing field-level typing. Per-line suppress is more honest.
**Cost:** M (mechanical but many sites).

### 9. Equinox ``Module.__call__`` is invisible to the type checker  [EQX_NOISE]  [STATUS: SUPPRESSED]

`policy_net(state)` where `policy_net: eqx.Module` produces `call-non-callable`. **20 errors**: `irf.py` (6), `evaluate.py` (4), `warm_start.py` (2), `composite_loss.py` and friends.

`eqx.Module` doesn't declare `__call__` on the base class — subclasses do, but the type system can't see through `state.params: eqx.Module`.

**Plan:** Suppress at call sites with `# pyright: ignore[reportCallIssue]  # ty: ignore[call-non-callable]  # eqx.Module subclasses define __call__; base class typing can't see it`. Only ~20 lines.
**Cost:** M.

### 10. ``VariableSpec.unpack_array`` typing  [JAX_NOISE]  [STATUS: DONE]

`variable_spec.py:54-56` — `values[0].ndim`, `jnp.stack(values)` complain because `values` is typed `list[object]` after a list comprehension over `arr[..., i]`. The runtime values are arrays.

Plus two `invalid-named-tuple` errors at lines 20, 25 — `NamedTuple("State", [(name, Array) for name in names])` uses a non-literal field list, which the typer can't accept.

Plus two `invalid-type-form` at 150, 171 — `callable` used as a type form (probably a typo for `Callable`).

**Plan:** For the unpack_array nepotism: type the local `values: list[Array]`. For the dynamic NamedTuple factory: suppress with rationale (already documented in the docstring as a runtime construction). For the `callable` usage at 150, 171: read and likely change `callable` to `Callable` (this is a real typo).
**Cost:** S.

### 11. ``networks/common.py:95`` operator issue  [JAX_NOISE]  [STATUS: DONE: real narrowing fix, not just suppression -- the conditional only checked input_shift but the body also dereferenced input_scale; both checks are required.]

`(x - jax.lax.stop_gradient(input_shift)) / jax.lax.stop_gradient(input_scale)` — operator on JAX-tracer types confuses ty. The runtime is correct.

**Plan:** Suppress with `# ty: ignore[unsupported-operator]  # jax.lax.stop_gradient return type confuses the type checker`.
**Cost:** S.

### 12. ``warm_start.py`` Dynare ghx/ghu CSV typing  [DECISION_NEEDED]  [STATUS: DONE: CSV-dict errors resolved transitively by earlier list-of-Array typing fixes; the remaining warm_start.py:63 EQX `.layers` access was suppressed as a drive-by.]

`warm_start.py:357, 376` — dict.get / `__getitem__` typing on dicts built from CSV columns. Requires reading the surrounding logic to decide whether the keys are statically known or really runtime-dynamic.

**Plan:** Investigate. Either tighten the dict typing (build from a known set of keys) or suppress.
**Cost:** M.

### 13. ``config.py:1151, 1155`` Pydantic dict-of-Any  [PYDANTIC_DICT]  [STATUS: SUPPRESSED]

`net_dict["hidden_sizes"] = tuple(...)` where `net_dict` is a Pydantic-derived `dict[str, Any & str]`. Two errors at one logical site.

**Plan:** Two `# pyright: ignore[reportArgumentType]  # ty: ignore[invalid-assignment]  # YAML-loaded dict; Pydantic validators check the actual types at construction time` directives.
**Cost:** S.

### 14. ``benchmark.py:67`` second `train_step` argument  [REAL_BUG]  [STATUS: DONE: rolled into item 5; benchmark.py is now clean.]

(Subset of item 5; tracked separately if item 5's edit doesn't catch all callsites.)

**Plan:** Roll into item 5.
**Cost:** S.

---

## Working order rationale

Items 1–4 collapse the most errors per LOC changed and are pure annotation tightening (the `Metrics` / `VariableSpec` precedent). Items 5–7 are real-bug fixes. Items 8–13 are framework-noise suppressions; mechanical but lower priority. Item 12 is the only one with genuine "investigate the call site" cost.

After items 1–7 we should be in single-digit territory; items 8–13 either close the gap or document why the residual count is what it is.
