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
- Every training configuration must select one strict `training_backend` identity and backend-local configuration. `timesteps` is the universal environment-step cap; learner updates, replay samples, search simulations, retained knowledge, and generated tokens remain backend-owned state and metrics.
- The common training orchestrator owns validated configuration, environment resolution, run resources, metrics, readiness, and finalization. Each backend owns its collector, replay or retained knowledge, actor/learner queues, search, tokenization, inference scheduling, model construction, optimization, and checkpoint payload.
- `sb3.ppo` is the only runnable backend. `rlab.ppo` and `rlab.a2c`, including model selection, PopArt, RND, and pHash options, are compatibility placeholders that must fail preflight before run resources are created.
- A runnable backend must preserve the supported checkpoint evaluation, playback, and publication contract before it can run through the normal queue workflow; those pipelines remain SB3 ZIP-specific until another concrete backend requires a new evaluator boundary.
- Invalid or internally inconsistent goals, training configurations, benchmarks, capacity rules, and machine settings must be rejected before execution.
- Every run must be traceable to its goal, training configuration, overrides, seed, launch time, source state, resolved settings, and runtime identity.
- Queue-backed W&B identity must keep an opaque stable run id, a seed-complete human run name, a submission-cohort batch/group, optional cross-submission campaign metadata, canonical game-family project routing, and run-id-owned artifact collections; historical project and display-name artifact references must remain readable.
- Generated outputs and secrets must remain outside tracked project content, and normal operation must not expose credentials.
- The supported application runtime is CPython 3.14; dependency resolution, provider packages, runtime images, and verification must remain installable on that runtime.
- Supported environment providers must include native Gymnasium vector environments, `ale-py`, `stable-retro-turbo`, and SuperMarioBros-NES Turbo (`supermariobrosnes-turbo`) under explicit identities that reject unknown or incompatible configuration.
- Rlab must provide `rlab:Bandit-v0` as a deterministic, ROM-free native-vector smoke environment with disabled autoreset and masked lane reset.
- Every provider must expose a native Gymnasium `VectorEnv` with correct spaces, reset, step, observations, rewards, termination, truncation, and columnar information; scalar environments and synthetic vectorization are unsupported.
- Native providers must use disabled/manual autoreset and masked lane reset.
- A masked reset must leave every unselected lane's emulator state, RNG, observations, frame stacks, counters, and sticky-action state unchanged.
- The rlab SB3 facade must reset completed lanes before returning a vector step so policy-facing reset behavior is same-step.
- The backend-neutral batch runtime must return post-reset observations, separate termination and truncation, batched final observations, and columnar transition/reset information without constructing SB3 dictionaries. Returned hot-path buffers are borrowed until the next runtime call.
- Actor-style single-host backends must partition globally unique lane ids, and episode seeds must derive from the base seed, global lane id, and episode index so actor regrouping cannot duplicate streams.
- Backend-neutral observation and action transport must preserve numeric, text, object-array, and structured Gymnasium values so interactive LLM agents can operate in RL environments without an SB3-shaped boundary.
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
- Aggregate training success paths must place the aggregation window before the semantic suffix so minimum and mean completion metrics end in `rate/min` and `rate/mean`, matching evaluation and remaining easy to discover together.
- Evaluation must reproduce the goal's observation, action, reward, start-state, reset, and termination contract; training metrics and results from another contract cannot establish acceptance.
- Every policy evaluation and playback must sample actions stochastically; deterministic argmax evaluation and playback are unsupported.
- Mario completion must be a clean level transition without death or life loss.
- Mario early stopping and checkpoint promotion must rely on goal-defined checkpoint evaluation, not training metrics alone.
- Checkpoint promotion must rank candidates by worst-start completion, mean per-start completion, least timesteps to the completion goal, and then evaluation reward.
- Active goals must explicitly use Modal for queue-backed checkpoint evaluation. Local evaluation is an explicit per-submission fallback. Evaluation may be disabled only by an explicit per-submission `none` override for smoke or debugging; such runs must disable eval-owned early stopping and cannot establish checkpoint promotion or goal acceptance.
- Preview capture must be enabled for every normal queue-backed screen checkpoint evaluation, with the browser-safe policy-observation video stored in canonical object storage and exposed in the producing W&B run; explicit no-eval smoke/debug submissions are exempt.
- Playback must support local and remote model artifacts under the evaluation preprocessing contract.
- Playback's game and policy-observation viewers must render the same policy environment step; playback must not advance an independent display environment.
- Remote artifact caches must refresh when an artifact alias resolves to different content so alias advancement cannot reuse stale model files.
- Published policies must use the project-generated Hugging Face identity `<game-family>_<goal>_<policy-variant>_<algorithm>` under the `tsilva` profile and be grouped into one public Collection per game family; manual repository names and provider-specific repository axes are unsupported. Exact provider, environment hash, run, recipe, seed, runtime, evaluation, and artifact hashes belong in portable release metadata.
- Visual releases must pass stochastic evaluation, include a representative browser-safe replay and YouTube cross-link, and publish the standardized model card, MIT license, checkpoint, metadata, manifest, and replay file set as one verified release.
- Queued attempts must be isolated, use declared runtimes, preserve durable results, recover safely after interruption, separate observation from mutation, and clean unused resources without affecting active or demanded work.
- New queue-backed training must fail closed unless an immutable image receipt matches the exact clean source revision and the image accepts the fully materialized train payload on the named machine. Backend readiness is additionally required only when the materialized backend needs it; implicit fallback to older runtimes is unsupported.
- Backend-required queue submissions must fail closed unless the latest control-plane reconciliation pass for that backend is fresh and successful.
- A queue-backed train job is running only after both the learner and its W&B publisher are ready and the durable W&B run identity and URL are observable; container startup alone is a starting state.
- Every queued job must name exactly one registered machine; resource-class targets, automatic placement, and silent machine fallback are unsupported.
- Queue reconciliation must run from one user-session Mac launchd service that invokes bounded short-lived passes; runner machines remain SSH/Docker-only, and remote containers continue independently while the control Mac sleeps or is logged out.
- Checkpoint evaluation and artifact publication failures must be isolated per run, use bounded retries, never block unrelated runs, and remain nonterminal until promotion and required W&B projections are complete or explicitly failed.
- A queued job has one stable launch/container identity and is never retried automatically after execution starts; an explicit retry creates a new traceable job.
- The registered-machine, stable-container-identity, and explicit-retry requirements apply to queued training jobs. Backend-bound checkpoint-evaluation tasks are separately traceable, need not name a runner machine, and may make at most two independently recorded attempts while leaving training success unchanged.
- Machine capacity, runtime paths, scheduling limits, and operator guidance must each have an authoritative source and remain mutually consistent.
- Benchmark claims must be reproducible and compare matching provider, task-kernel, event-boundary, workload, concurrency, and host-load contracts; Stable Retro benchmarks must use `info_filter=all`, never `none`.
- Changes must pass relevant automated verification and preserve internally consistent project configuration.
