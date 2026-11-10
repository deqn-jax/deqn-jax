# Reality — what a claim is checked against
The graph is a model of the work, the repo a part of its realization; named here is the third thing — what the work becomes when it runs, and how to look at it. The carrier follows the claim: behaviour is decided by the trained checkpoint and the run, not by the source that should have produced them; the content of a source or a norm is the text itself; tests are a rung of evidence, not a carrier.

Carriers were accepted by the maintainer (2026-09-03); rows added by use since are marked "proposed".

|Claim class|Canonical carrier|How to observe|Who can|
|---|---|---|---|
|"model X is solved / certified"|the **final** checkpoint on the DGX, `runs/<arm>_s<seed>/checkpoint_003000.eqx`, three seeds, fp64|`JAX_ENABLE_X64=1 uv run python scripts/cert/disaster_ss_probe.py --runs-dir runs/disaster_cert --arms <arm> --seeds 0,1,2` on the DGX host; stress grid via `scripts/cert/ewm_stress_table.py`; report the learned-block ρ, ŝ, residuals at ŝ. Falsifier: any leg of the stack failing on any seed|agent (ssh)|
|"the code is correct / tests pass"|the suite run on the DGX host, not the laptop|`ssh anna@130.223.169.108 'export PATH=$HOME/.local/bin:$PATH; cd ~/projects/<lane-dir> && make check'` (`check-fast` while iterating)|agent|
|"CI is green"|the GitHub Actions run for the PR/commit|`gh pr checks <n> --watch` / `gh run list --branch <b>`|agent|
|"a training recipe behaves"|the run directory on the DGX (checkpoints, `DONE` marker, config, TensorBoard)|`ls runs/<arm>_s<seed>/`, `logs/cert_container*.log`; launch: `scripts/dgx/run_sweep_in_container.sh` in the NGC container|agent|
|"a refactor changed no numbers"|same-seed short training on trunk and on the branch, and the bit-identical step guard|`uv run deqn-jax train <model> -n 3 --seed 0` on both trees, compare the final loss; `tests/test_step_common_guard.py` compares exactly on its recording platform (x64 on, per `tests/conftest.py`)|agent|
|"a certified checkpoint still loads"|the certified checkpoints in the DGX record, through the current loader|the probe command of the first row against `../deqn-jax/runs/disaster_cert` from a lane directory; compare with the record in `docs/dev/disaster_cert_report_2026_07_07.md`|agent (ssh)|
|"a documented command works" (proposed)|a clean install — plain `uv sync` in a fresh lane directory — running the command as the README or docs page writes it|run it there; `tests/test_documented_commands.py` where present is a rung, not the carrier|agent|
|"the picture of a policy" (proposed)|figures rendered from the final checkpoints (no-shock path, simulation, IRFs, policy slices, equation errors), seeds side by side with the first-order rule|render from the DGX checkpoints and look at the images, not at their summary statistics|agent renders; the maintainer looks|
|"the docs are live"|https://deqn-jax.github.io|`curl -sI` the page; deploy is `mkdocs gh-deploy --remote-name pages`|agent observes; maintainer deploys|
|"a config change took"|the resolved `TrainConfig` printed at run start / `config.yaml` in the run dir|`uv run deqn-jax train <model> --config <yaml> -n 1 -q` and read the resolved config|agent|

**Ceiling**: the disaster model's *true* equilibrium (no oracle exists — the research program is about certificates in its absence); Dynare comparisons (fixtures are not in the repo — `dynare/` is ignored, the Dynare tests skip); parity with external references (the EWM numbers, the RSS checkpoint) is approximate by construction and closes only by convergence of independent evidence; the user's private repositories and machines; GPU behaviour of the gate (the DGX lane environment runs CPU jaxlib outside the container).

**The table grows by use.** A session taught something the table does not hold (an unnamed carrier, an observation that turned out reachable or unreachable — then *Ceiling*, a wrong command here) — write the row then, before closing the work that taught it. Only what observation needs goes here; dated measurements and history are nodes in the graph or lines of the cert report.
