## HOW TO USE THIS FILE

- Before any task in this repository, read this compact product specification.
- Treat these specifications as required contracts and preserve them while working.
- Keep this file limited to durable, product-facing requirements; transient implementation notes, experiments, and debugging details must remain outside it.
- After any task, update this file when the user states a durable requirement, a requirement changes or is dropped, or the work reveals a durable product contract.
- Keep this file compact, accurate, and non-redundant by consolidating overlapping requirements rather than duplicating them.
- Put detailed operational guidance in the relevant reference or skill when it is necessary but too specific for this contract.

## PROJECT PURPOSE

rlab is a reproducible reinforcement-learning workbench for game-agent researchers. It carries explicit goals through training, trustworthy evaluation and ranking, replay, publication, and local or queue-backed operation while preserving traceability and comparability.

## PROJECT REQUIREMENTS

- The project must support coherent workflows for defining goals, training policies, evaluating and ranking checkpoints, replaying behavior, publishing models, operating queued work, and running comparable benchmarks.
- Goal definitions must independently specify the environment, acceptance criteria, ranking order, evaluation protocol, and release expectations.
- Goal environment definitions must explicitly cover every provider constructor argument through canonical environment fields or provider-native `env_args`; inheriting environment-provider defaults is unsupported.
- Training configurations must declare a finite resource cap, a meaningful description, and every value required for validation and execution.
- Invalid or internally inconsistent goals, training configurations, benchmarks, capacity rules, and machine settings must be rejected before execution.
- Every run must be traceable to its goal, training configuration, overrides, seed, launch time, source state, resolved settings, and runtime identity.
- Queue-backed W&B identity must keep an opaque stable run id, a seed-complete human run name, a submission-cohort batch/group, optional cross-submission campaign metadata, canonical game-family project routing, and run-id-owned artifact collections; historical project and display-name artifact references must remain readable.
- Generated outputs and secrets must remain outside tracked project content, and normal operation must not expose credentials.
- The supported application runtime is CPython 3.14; dependency resolution, provider packages, runtime images, and verification must remain installable on that runtime.
- Supported environment providers must include native Gymnasium vector environments, `ale-py`, `stable-retro-turbo`, and SuperMarioBros-NES Turbo (`supermariobrosnes-turbo`) under explicit identities that reject unknown or incompatible configuration.
- Every provider must expose a native Gymnasium `VectorEnv` with correct spaces, reset, step, observations, rewards, termination, truncation, and columnar information; scalar environments and synthetic vectorization are unsupported.
- Native providers must use disabled/manual autoreset and masked lane reset.
- A masked reset must leave every unselected lane's emulator state, RNG, observations, frame stacks, counters, and sticky-action state unchanged.
- The rlab SB3 facade must reset completed lanes before returning a vector step so policy-facing reset behavior is same-step.
- Providers may interrupt internal frame skip only for genuine engine termination.
- Vectorized task kernels must compute task events, reward shaping, task termination, truncation, outcomes, and metrics at vector-step boundaries from provider facts.
- Policy actions may use native Gymnasium spaces or a task-owned, bind-time-validated discrete lookup codec; providers must not contain policy-specific action mappings.
- Generic Gymnasium providers must support training, evaluation, and playback without ROMs, save states, or Mario-specific information fields.
- `stable-retro-turbo` must preserve ROM identity, save-state starts, observations, raw signals, and engine termination across training, evaluation, and playback.
- Atari environments must use `stable-retro-turbo`'s native `RetroVecEnv` with the packaged Stella core, not an `ale-py` vector environment constructed by rlab.
- Breakout and Ms. Pac-Man training must use disabled autoreset and masked lane reset through the same rlab facade as Mario.
- Stable Retro Atari environments must use the provider's `use_fire_reset` reset mechanic rather than task-kernel or action-codec FIRE overrides; it is disabled by default. Breakout must expose standard NOOP/FIRE/LEFT/RIGHT policy actions; life loss and level progression must not be episode boundaries.
- Official Farama Stable Retro is unsupported; `stable-retro-turbo` with rlab's native-vector lifecycle extension is the supported Retro runtime.
- SuperMarioBros-NES Turbo must match the applicable `stable-retro-turbo` public environment contract, with every provider-specific difference explicit and validated.
- Official ViZDoom Gymnasium environments are unsupported unless a provider supplies a registered native vector entry point with disabled/manual autoreset, masked reset, columnar signals, and rendering; scalar ViZDoom environments must not be synthesized into vectors.
- Provider observation layouts must be normalized to the policy's declared input contract without changing semantic content.
- Dependency installation must be reproducible, supply-chain hardened, resistant to known-bad releases, and compatible with every supported provider.
- Training must preserve durable metrics and checkpoint artifacts by default, keep result evidence distinct from job state, and use documented, unambiguous metric semantics.
- Evaluation metrics must use `eval/screen/*`, `eval/confirm/*`, or `eval/full/*` according to the producing protocol. Metrics with the same semantic quantity must keep the same semantic suffix after the training phase or evaluation-protocol prefix; protocol-specific aggregation windows may add explicit suffixes but must not rename the underlying quantity.
- Evaluation must reproduce the goal's observation, action, reward, start-state, reset, and termination contract; training metrics and results from another contract cannot establish acceptance.
- Every policy evaluation and playback must sample actions stochastically; deterministic argmax evaluation and playback are unsupported.
- Mario completion must be a clean level transition without death or life loss.
- Mario early stopping and checkpoint promotion must rely on goal-defined checkpoint evaluation, not training metrics alone.
- Checkpoint promotion must rank candidates by worst-start completion, mean per-start completion, least timesteps to the completion goal, and then evaluation reward.
- Active goals must explicitly use Modal for queue-backed checkpoint evaluation; local evaluation is an explicit per-submission fallback only.
- Playback must support local and remote model artifacts under the evaluation preprocessing contract.
- Playback's game and policy-observation viewers must render the same policy environment step; playback must not advance an independent display environment.
- Remote artifact caches must refresh when an artifact alias resolves to different content so alias advancement cannot reuse stale model files.
- Visual releases must include reproducibility metadata and a representative replay.
- Queued attempts must be isolated, use declared runtimes, preserve durable results, recover safely after interruption, separate observation from mutation, and clean unused resources without affecting active or demanded work.
- Every queued job must name exactly one registered machine; resource-class targets, automatic placement, and silent machine fallback are unsupported.
- Queue reconciliation must run from one user-session Mac launchd service that invokes bounded short-lived passes; runner machines remain SSH/Docker-only, and remote containers continue independently while the control Mac sleeps or is logged out.
- A queued job has one stable launch/container identity and is never retried automatically after execution starts; an explicit retry creates a new traceable job.
- The registered-machine, stable-container-identity, and explicit-retry requirements apply to queued training jobs. Backend-bound checkpoint-evaluation tasks are separately traceable, need not name a runner machine, and may make at most two independently recorded attempts while leaving training success unchanged.
- Machine capacity, runtime paths, scheduling limits, and operator guidance must each have an authoritative source and remain mutually consistent.
- Benchmark claims must be reproducible and compare matching provider, task-kernel, event-boundary, workload, concurrency, and host-load contracts; Stable Retro benchmarks must use `info_filter=all`, never `none`.
- Changes must pass relevant automated verification and preserve internally consistent project configuration.
