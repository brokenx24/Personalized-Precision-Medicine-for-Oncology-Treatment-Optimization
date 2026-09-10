# Report 05: Orchestration & DAG Execution

## 1. Execution Directed Acyclic Graph (DAG)
The execution topology is formally specified in `DependencyGraph` with 8 sequential nodes:
1. `INPUT_VALIDATION`
2. `PREPROCESSING`
3. `ML_STAGE`
4. `DL_STAGE`
5. `NLP_STAGE`
6. `SLM_STAGE` (Depends on ML, DL, NLP)
7. `SAFETY_STAGE` (Depends on SLM)
8. `OUTPUT_FORMATTING` (Depends on Safety)

## 2. Failure Handling & Degradation
The `FailureHandler` strictly enforces **Fail-Closed** semantics. If any pipeline stage raises an unhandled exception or data corruption is detected:
- The execution status is immediately marked `STAGE_EXECUTION_FAILURE`
- Safety status transitions to `FAIL`
- `approved_for_presentation` is set to `False`
- Autonomous clinical decisions remain strictly `FORBIDDEN`
