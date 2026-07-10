## HOW TO USE THIS FILE

- Before any task in this repository, read this compact product specification.
- Treat these specifications as required contracts and preserve them while working.
- Keep this file limited to durable, product-facing requirements; transient implementation notes, experiments, and debugging details must remain outside it.
- After any task, update this file when the user states a durable requirement, a requirement changes or is dropped, or the work reveals a durable product contract.
- Keep this file compact, accurate, and non-redundant by consolidating overlapping requirements rather than duplicating them.
- Put detailed operational guidance in the relevant reference or skill when it is necessary but too specific for this contract.

## PROJECT PURPOSE

rlab is a reproducible reinforcement-learning workbench for game-agent researchers. It must carry versioned goals through training, trustworthy evaluation and ranking, replay, publication, and local or queue-backed operation while preserving traceability and comparability.

## PROJECT REQUIREMENTS

- The project must provide coherent workflows for defining goals, training policies, evaluating and ranking checkpoints, replaying behavior, publishing models, operating queued work, and running comparable benchmarks.
- Goal definitions must specify the environment, acceptance criteria, ranking order, evaluation protocol, and release expectations independently of training configurations.
- Training configurations must declare a finite resource cap, a meaningful description, and every value required for validation and execution.
- Invalid or internally inconsistent goals, training configurations, benchmarks, capacity rules, and machine settings must be rejected before execution.
- Every run must be traceable to its goal, training configuration, overrides, seed, launch time, source revision, resolved settings, and immutable runtime identity.
- Generated outputs and secrets must remain outside versioned project content, and normal operation must never expose credentials.
- Supported environment providers must include native Gymnasium vector environments, `ale-py`, `stable-retro-turbo`, and SuperMarioBros-NES Turbo (`supermariobrosnes-turbo`) under explicit identities that reject unknown or incompatible configuration.
- Every provider must expose a natively vectorized environment that follows Gymnasium `VectorEnv` semantics for spaces, reset, step, observations, rewards, termination, truncation, and columnar information; scalar environments and synthetic vectorization are unsupported.
- Native providers must run with disabled/manual autoreset and implement masked lane reset. Unselected lanes must preserve emulator state, RNG, observations, frame stacks, counters, and sticky-action state. The rlab SB3 facade must reset completed lanes before returning the vector step so policy-facing behavior remains same-step.
- Providers may interrupt internal frame skip only for genuine engine termination. Task events, reward shaping, task termination, truncation, outcomes, and metrics must be computed by vectorized task kernels at vector-step boundaries from provider facts.
- Policy actions may pass through natively supported Gymnasium spaces or through a task-owned, bind-time-validated discrete lookup codec. Providers must not contain policy-specific action mappings.
- Generic Gymnasium providers must support training, evaluation, and playback without requiring provider-specific concepts such as ROMs, save states, or Mario information fields.
- `stable-retro-turbo` must preserve ROM identity, save-state starts, observations, raw signals, and engine termination consistently across training, evaluation, and playback. Version `1.0.1.post13` is the minimum supported release because it provides disabled autoreset and masked lane reset.
- Official Farama Stable Retro is unsupported; the supported Retro runtime is `stable-retro-turbo` with rlab's native-vector lifecycle extension.
- SuperMarioBros-NES Turbo must match the applicable `stable-retro-turbo` public environment contract, and every provider-specific difference must be explicit and validated. Version `0.2.20` is the minimum supported release because it provides disabled autoreset and masked lane reset.
- Official ViZDoom Gymnasium environments are unsupported until a provider supplies a registered native vector entry point with disabled/manual autoreset, masked reset, columnar signals, and rendering. Synthesizing vectors from scalar ViZDoom environments is not allowed.
- Provider observation layouts must be normalized to the policy's declared input contract without changing their semantic content.
- Dependency installation must be reproducible, supply-chain hardened, resistant to known-bad releases, and compatible with every supported environment provider.
- Training must preserve durable metrics and checkpoint artifacts by default, keep result evidence distinct from job state, and use documented, unambiguous metric semantics.
- Evaluation must reproduce the goal's observation, action, reward, start-state, reset, and termination contract; training metrics and results from another contract cannot establish acceptance.
- Mario completion must mean a clean level transition without death or life loss, and early stopping and promotion must rely on goal-defined checkpoint evaluation rather than training metrics alone.
- Checkpoint promotion must rank candidates by worst-start completion, mean per-start completion, least timesteps to the completion goal, and then evaluation reward.
- Playback must support local and remote model artifacts under the evaluation preprocessing contract, and visual releases must include reproducibility metadata and a representative replay.
- Queued attempts must be isolated, use immutable runtimes, preserve durable results, recover safely after interruption, separate observation from mutation, and clean unused resources without affecting active or demanded work.
- Machine capacity, runtime paths, scheduling limits, and operator guidance must each have an authoritative source and remain mutually consistent.
- Benchmark claims must be reproducible and compare matching provider, task-kernel, event-boundary, workload, concurrency, and host-load contracts.
- Changes are acceptable only when relevant automated verification passes and versioned project configuration remains internally consistent.
