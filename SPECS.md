## PROJECT PURPOSE

rlab is a reproducible reinforcement-learning workbench for game-agent researchers. It carries explicit research goals through validated training, trustworthy checkpoint evaluation and ranking, replay, publication, and local or queued operation while preserving traceability and comparability.

## PROJECT REQUIREMENTS

- A supported research goal must pass coherently through validation, its declared training and acceptance workflow, playback, publication, and local or queued operation without phase-specific reinterpretation.
- Goal definitions must independently declare the environment contract, default acceptance criteria, ranking order, evaluation protocol, and release expectations; any recipe-specific acceptance workflow and whether it requires independent evaluation must also be explicit and validated. Every launchable recipe must belong to exactly one goal, while reusable recipe defaults may be inherited. Training and evaluation environment definitions must explicitly cover every provider constructor argument rather than inherit provider defaults.
- Training configurations must declare a finite resource cap, a meaningful description, one backend identity, and every value required for validation and execution. Unsupported backends must fail before run resources are created.
- Invalid or internally inconsistent goals, training configurations, benchmarks, capacity rules, and machine settings must be rejected before execution or external mutation.
- Every run must be traceable to its goal, training configuration, overrides, seed, launch time, source state, resolved settings, execution target, and runtime identity.
- Run, cohort, campaign, and artifact identities must remain unambiguous and stable, and historical artifact references supported by the project must remain readable.
- Generated outputs and secrets must remain outside tracked project content, and normal operation must not expose credentials.
- Every environment provider must expose a native Gymnasium vector environment with correct spaces, reset, step, observation, reward, termination, truncation, and columnar-information semantics; scalar environments must not be presented as native vectors through synthetic wrapping.
- Native providers must use disabled or manual autoreset and support masked lane reset. Resetting selected lanes must not change any unselected lane's state, randomness, observations, frame history, counters, or action state, and completed lanes must be reset before the next policy-facing observation is returned.
- The runtime may request an episode boundary for any live lane after a completed vector step. Such forced resets must preserve the completed transition as a neutral truncation, record their reason, reset all lane-local task and policy history, and support either deterministic configured start sampling or an explicit next start.
- Parallel environment collection must assign unique lane identities and deterministic per-episode randomness so regrouping or concurrency changes cannot duplicate streams.
- Environment and runtime boundaries must preserve supported Gymnasium observation and action values, normalize layout only to the policy's declared input contract, and never change semantic content.
- Task events, rewards, termination, truncation, outcomes, and metrics must be derived from provider facts at the declared vector-step boundaries.
- Training, evaluation, and playback must use the same declared observation, action, reward, start-state, reset, and termination contract; evidence produced under a different contract cannot establish goal acceptance.
- Equivalent providers for one canonical environment must share one goal and recipe contract. A provider override must atomically select that provider for training, evaluation, and playback without changing the remaining contract, while preserving the selected provider in provenance.
- Provider-specific requirements must not leak into generic Gymnasium workflows, and supported ROM-free and ROM-backed environments must remain trainable, evaluable, and playable through the common workflow.
- Dependency installation must be reproducible, supply-chain hardened, resistant to known-bad releases, and compatible with every supported environment provider and execution path.
- Training must preserve durable metrics and checkpoint artifacts by default, keep result evidence distinct from job state, and use documented, unambiguous metric semantics.
- Policy evaluation and playback must sample actions stochastically. Except for an explicitly declared deterministic-search workflow, early stopping and checkpoint promotion must rely on goal-defined checkpoint evaluation rather than training metrics alone, and a run without the required evaluation cannot establish promotion or goal acceptance.
- An explicitly declared deterministic-search workflow may accept the first training episode that emits its goal success event without independent evaluation; exhausting its resource cap without that event is unsuccessful, and an accepted policy must still be published and playable.
- Playback must support local and remote artifacts under the evaluation preprocessing contract, keep all simultaneous viewers on the same environment step, and refresh cached content when a mutable artifact reference advances.
- Published policies must provide portable provenance and reproducibility metadata, verified evaluation evidence, the policy artifact, and, for visual behavior, a representative browser-safe replay.
- Queued execution must fail closed unless its source state, fully resolved configuration, runtime, backend readiness, and explicit execution target satisfy the declared run contract; silent fallback to another runtime or target is unsupported.
- Runtime reuse across source states must occur only when a complete content-addressed runtime-input contract proves the runtime inputs identical; every run must still preserve exact source and runtime provenance.
- Queued attempts must be isolated, preserve durable results, use stable traceable identities, recover safely after interruption, represent retries as distinct attempts, isolate post-training failures per run, and clean unused resources without affecting active or demanded work.
- Reported job states must correspond to observable readiness or completion of the required work and durable evidence, not merely process or container startup.
- Benchmark claims must be reproducible and compare matching environment, task, event-boundary, workload, concurrency, and host-load contracts.
- Changes must pass relevant automated verification and preserve internally consistent project configuration.
