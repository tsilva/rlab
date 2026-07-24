## PROJECT PURPOSE

rlab is a reproducible reinforcement-learning workbench for game-agent researchers. It carries explicit research goals through validated training, trustworthy evaluation and ranking, playback, publication, and local or queued execution while keeping results traceable and comparable.

## PROJECT REQUIREMENTS

### Goals and Run Contracts

- Each research goal must declare whether it is evaluated or training-only. Evaluated goals must define their environment, success criteria, ranking, evaluation, and release rules. Training-only goals must remain ineligible for goal acceptance, goal completion, checkpoint promotion, or release.
- Every launchable training configuration must belong to one goal, declare a finite resource limit and meaningful description, and resolve every choice needed for execution.
- Invalid or internally inconsistent goals, run configurations, benchmarks, capacity rules, or execution settings must be rejected before execution or external mutation.
- A run must resolve one complete declared observation, action, reward, event, start-state, and episode-boundary contract and preserve it across training, evaluation, and evidence-bearing playback.
- Training-only curricula or phase-specific behavior must be explicitly declared, and evidence produced outside the finalized evaluation contract must not establish acceptance or promotion.

### Evaluation and Evidence

- Except for an explicitly declared deterministic-search workflow, acceptance and promotion must use goal-defined checkpoint evaluation rather than training metrics or playback behavior.
- A deterministic-search workflow may accept only a policy that produces the goal’s success event within its resource limit, and the accepted policy must be published and playable.
- Recorded datasets and results from recording, playback, integrity verification, or reexecution must never establish goal acceptance or checkpoint promotion.
- Policy evaluation and playback must default to stochastic action sampling. Playback may explicitly select deterministic sampling, but must visibly preserve that choice and never use deterministic results as evaluation or promotion evidence.

### Provenance and Security

- Every run, cohort, campaign, result, and artifact must have a stable identity and enough provenance to reconstruct its goal, configuration, overrides, seed, source, runtime, environment, execution target, and launch context. Supported historical references must remain readable.
- Generated outputs and secrets must remain outside tracked source, and normal operation must not expose credentials.
- Externally supplied executable models must not run until their complete content is integrity-checked and the user has trusted their source or approved that invocation after an explicit authority-and-credential warning.
- Installation must be reproducible, supply-chain hardened, resistant to known-bad releases, and compatible with supported workflows.

### Environment Compatibility

- Supported environment providers must provide correct, isolated parallel execution with deterministic, nonduplicated episode streams. Resetting, completing, or forcing a boundary in one lane must not disturb any other lane.
- Equivalent providers must preserve the same declared observations, actions, rewards, events, and episode semantics across training, evaluation, and playback. Switching providers must change only provider identity and remain traceable.
- Provider-specific requirements must not leak into common workflows, and every supported environment must remain trainable, evaluable, and playable through those workflows.

### Training Results and Publication

- Training must durably preserve authoritative, unambiguous metrics and checkpoints by default, keep scientific evidence separate from job state and diagnostics, and prevent observability systems from throttling training or determining scientific outcomes.
- For queued runs with publication enabled, every ready periodic and final checkpoint must be published independently of evaluation and remain downloadable and playable without private infrastructure credentials.
- Published policies must include the policy artifact, portable provenance and reproducibility metadata, verified evaluation evidence, and a representative browser-safe replay for visual behavior.

### Playback and Human Control

- Playback must support local and remote artifacts under the evaluation contract, keep concurrent viewers on one trajectory, and refresh mutable references when their content changes.
- Interactive playback must provide independently arrangeable, synchronized views of game frames, policy inputs and decisions, transition facts, and bounded histories without inspection changing the trajectory or policy randomness.
- Human control must preserve declared action semantics, fail safe when focus or control is lost, and keep all human-intervened results ineligible for acceptance or promotion.

### Queued Operation and Benchmarks

- Queued execution must be explicit, fail closed, isolated, and reproducible. Attempts must preserve exact provenance and durable results across interruption and retries, report only evidence-backed states, and clean unused resources without affecting active work.
- Benchmark claims must be reproducible and compare equivalent environments, semantics, workloads, concurrency, and host-load conditions.
