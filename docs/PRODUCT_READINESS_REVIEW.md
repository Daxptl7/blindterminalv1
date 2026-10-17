# BlindAssist product readiness review

Reviewed 18 October 2026. This document records the baseline audit before fixes.
Subsequent implementation adds session privacy, strict private playback, explicit
GPS failure, model-specific indexes, atomic storage, cancellation, configuration
validation and deployment tooling. See [deployment instructions](DEPLOYMENT.md)
for the current behavior and hardware release checks that remain pending.

**Assessment: the repository has a substantial prototype and useful regression tests, but the current privacy, GPS, retrieval, and deployment behavior blocks a dependable customer release.**

**Evidence and limits**

- `.venv/bin/python -m unittest discover -s tests -v`: 164 tests ran, 163 passed, one skipped because pygame is absent; no failures. These results do not establish real camera, microphone, earphone routing, or Raspberry Pi performance.
- Ten additional synthetic checks reproduced unsafe audio fallback, conflicting public/private devices, public playback after an unanswered privacy prompt, unmarked simulated GPS coordinates, document loss after an embedding dimension change, incorrect vector/text pairing after an embedding failure, indexing success without storage, an offline mock answer, cancellation waiting for work to finish, and public playback of a private OCR explanation.
- `.venv/bin/python selftest.py`: exited 0 while reporting 15 feature limitations in this laptop environment. Offline speech models and local translation bundles are unavailable here, and MediaPipe is absent. Missing Linux device nodes on macOS do not establish that a physical camera is broken.
- `.venv/bin/python -m pip check`: exited 1. The environment has six missing pyobjc framework dependencies and a huggingface-hub/click version conflict. This describes the current environment, not every possible clean installation.
- No real-world GPS route, private sound-card playback, live cloud response, user study, or target-Pi benchmark was validated in this review. Model identifier availability was not checked against live provider catalogs.

Priority meanings: **P0** blocks use of the affected feature with customers; **P1** blocks a dependable release; **P2** should be addressed before expanding a pilot.

1. **P0 — Missing GPS can become a believable, incorrect location.**

   Evidence: [gps_navigator.py](../modules/gps_navigator.py), `get_location()` at line 303, falls through to `_read_gps_simulation()` when hardware and IP lookup fail. Its result has no simulation/source marker. [main.py](../main.py), `mode_gps()` at line 1430, announces a “live location” and “You are near” using that result. `get_directions()` also uses it as the route origin.

   Reproduced: hardware/IP failure returned the configured test coordinates with no indication that they were simulated.

   Fix: make simulation an explicit development-only configuration. In device mode, return a typed `NO_FIX` result instead of coordinates. Carry `source`, fix time, fix validity and available quality information through the result. Validate RMC status/GGA fix quality, support the hardware's actual talker IDs, reject stale fixes, and require a current hardware fix before routing. Label approximate IP locations separately and exclude them from turn instructions. Speak an actionable no-fix message.

   Acceptance: disconnect GPS, supply invalid/stale NMEA messages and deny network access; none may produce a simulated customer location or start a route. Explicit simulation must announce itself.

2. **P0 — Private speech can fall back to the public output.**

   Evidence: [tts.py](../modules/tts.py), `_validate_device()` at line 93, can select the speaker when it is the only available card. `_deliver()` at line 727 tries default pygame/system playback after named-device playback fails. On macOS it ignores the requested ALSA route. `PrivateAudio` in [confidential_mode.py](../modules/confidential_mode.py) does not validate successful private playback.

   Reproduced: a missing earphone resolved to the public card; failed named private playback called default pygame playback.

   Fix: attach an explicit privacy requirement to each utterance. Require a verified private endpoint distinct from the public endpoint. For private utterances, allow only players that support that endpoint; return a playback error on disconnect/failure. A generic public error tone/message may explain the problem without containing document text. Never substitute an arbitrary card. On platforms without verified private routing, report that capability as unavailable.

   Acceptance: unplug the private card before and during speech, configure identical endpoints, and force every routed backend to fail. Public/default players must never receive private content. Verify this with the physical device as well as mocks.

3. **P0 — Privacy choices are lost across follow-up actions and timeouts.**

   Evidence: `mode_ocr_scan()` in [main.py](../main.py), line 369, receives `document_mode` but does not apply it to the explanation. The document reader's `PrivateAudio` context resets to the speaker before `ask_ai()` and `_speak(answer)`. `deliver_answer()` in [ai_query.py](../modules/ai_query.py), line 481, chooses the speaker on no response and substitutes the speaker if the earphone is missing. [voice.py](../modules/voice.py), `play_with_privacy_prompt()` at line 1086, also defaults to the speaker on timeout.

   Reproduced: the synthetic document used the earphone; its explanation used the speaker. An unanswered answer-routing prompt delivered the answer publicly.

   Fix: create a privacy session for the document/question and inherit it for explanations, translations, replay, and errors. Unanswered or unavailable privacy selection must cancel content playback. Public playback requires explicit selection. Return the actual playback outcome rather than reporting “earphone” after a substituted route.

   Acceptance: select private OCR, request an explanation and replay it; all content remains private. Timeouts, missing buttons, and missing earphones produce no content on the speaker.

4. **P1 — Private audio selection does not control cloud transmission or storage.**

   Evidence: OCR can send a page to Gemini in [ocr.py](../modules/ocr.py), `GeminiOCREngine.extract_text()`, before the privacy question in `mode_ocr_scan()`. The orchestrator later logs OCR text and indexes it regardless of `document_mode`. [ai_query.py](../modules/ai_query.py), `ask_ai()` at line 732, sends context to racing providers and logs a prompt excerpt. [tts.py](../modules/tts.py), `_synthesize()` at line 488, may use gTTS for private output; `_process_queue()` at line 793 logs text. Voice recognition can use Google and logs transcripts. Voice recordings persist by default. The translation engine has a useful `privacy=True` restriction, but its speech output still uses the shared TTS path.

   Fix: select a processing policy before capture/transcription. Thread `allow_cloud`, `allow_persistence` and output privacy through OCR, STT, embeddings, retrieval, AI, translation and TTS. Use local TTS for confidential sessions; prohibit private indexing and raw-content logs unless separately requested. Make recordings ephemeral by default for private sessions, clean them on exit/error, and provide a clear retention/deletion control. Keep diagnostics to timing, error codes and counts. Apply the same rule to telemetry.

   Acceptance: run a full private capture → OCR/transcription → explanation/translation → speech session with network clients and persistence sinks instrumented. No private content may reach any prohibited sink, including TTS and logs.

5. **P1 — Cloud/local embedding fallback can erase the searchable library.**

   Evidence: [embedder.py](../services/embedder.py), `_compute_embedding()` at line 168, falls back per request from cloud embeddings to a local model. [vector_store.py](../services/vector_store.py), `add_batch()` at line 85, drops incompatible existing vectors and their text when dimensions change. `retrieve()` rejects a query of the wrong dimension before BM25 can help. Cache keys contain only text, so results from a fallback model remain cached without model identity.

   Reproduced: adding a vector with a different dimension removed the original document metadata. An existing regression test explicitly expects this deletion.

   Fix: preserve source documents independently from indexes. Identify every index/cache by provider, model version, dimension and normalization. Keep separate local and cloud indexes; prepare local embeddings in advance if offline retrieval is promised. Reject incompatible writes without deleting content. Rebuild into a separate generation and switch only after successful validation. Permit lexical retrieval when dense embedding is unavailable.

   Acceptance: ingest a library online, lose connectivity, add another page, restart, and reconnect. Original documents remain intact and retrievable through the appropriate index. Update the regression that currently endorses data loss.

6. **P1 — A failed embedding can attach another chunk's vector to the wrong text.**

   Evidence: `get_embeddings()` in [embedder.py](../services/embedder.py), line 210, removes failed/empty embeddings. `index_document()` in [rag_pipeline.py](../services/rag_pipeline.py), line 135, still passes all chunks. `add_batch()` in [vector_store.py](../services/vector_store.py) zips that original text list with the shortened vector matrix without a length check. Mixed cloud/local dimensions can also make `np.vstack()` fail.

   Reproduced: the first chunk failed to embed; the second chunk's vector was stored against the first chunk's text, and the second text was omitted.

   Fix: return embedding records containing the chunk ID, vector and model identity, or fail the entire batch. Validate exact row/text/tag counts, finite values and one consistent embedding space before mutation. Retry failed chunks by ID and publish a document only after its batch is complete.

   Acceptance: fail the first, middle and last chunks independently, and alternate model availability. No stored vector may be paired with a different text or tag record.

7. **P1 — Indexing can report success without saving, and writes are vulnerable to interruption.**

   Evidence: `index_document()` catches exceptions and returns no outcome. `index_file()` returns `True` after that call even if embeddings were empty or storage failed. `save_index()` in [vector_store.py](../services/vector_store.py), line 271, catches write failures and directly writes four related files separately. There is no transaction spanning vectors, FAISS, text and tags, nor a store-level write lock.

   Reproduced: `index_file()` returned `True` while no embedding or storage operation succeeded. Power-loss corruption is a source-level risk; a physical power-cut test was not performed.

   Fix: propagate a structured indexing/storage result to the caller. Serialize writes, write a complete new generation to temporary files, flush it, then atomically publish a manifest that selects that generation. Validate file counts, checksums and model identity on load; retain the last valid generation. Use stable document/chunk IDs to support idempotent reingestion and deletion.

   Acceptance: inject disk-full/write errors and interrupt each write phase; success is never reported for an uncommitted index, and restart recovers a complete generation. Reimporting a book does not duplicate it.

8. **P1 — Cancel can leave the user waiting for the entire operation.**

   Evidence: [processing_feedback.py](../modules/processing_feedback.py), `run_with_feedback()` at line 9, only raises cancellation after its worker completes. While waiting, cancellation merely suppresses progress speech. [tts.py](../modules/tts.py) calls gTTS without an explicit timeout; its blocking utterance wait has a timeout but does not report timeout status or cancel the utterance. Slow synthesis can hold the sole speech queue.

   Reproduced: cancelling a blocked operation did not return until the operation was released.

   Fix: use short cancellation polling, an end-to-end deadline and cancellable operations. Put non-cooperative CPU/model work in a managed process when necessary; return control promptly while ensuring that abandoned work cannot publish results, hold the camera, or overlap a new session. Bound gTTS connect/read time and provide offline prompt speech. Return explicit cancellation and playback-timeout outcomes.

   Acceptance: cancel deliberately stalled OCR, synthesis and network calls. Define and measure a cancel-to-menu target, initially at most one second for UI acknowledgement, and prove cleanup before resource reuse. Late answers must never play in a later mode.

9. **P1 — Offline behavior and startup checks do not establish advertised readiness.**

   Evidence: [gemini_agent.py](../services/gemini_agent.py), `generate_response()` at line 67, returns an `[Offline Mock Response]` when no client exists. This is the direct RAG service path; `ask_ai()` has a separate local-model fallback. [README.md](../README.md) claims production readiness and “100% offline resilience,” while actual offline models are separately provisioned. `selftest.py` checks many imports/configurations without executing end-to-end operations; a configured API key is not proof that an API request succeeds.

   Reproduced: the RAG service returned a mock answer with `client=None`. This laptop has only online speech recognition available, no configured local translation bundles, and no MediaPipe, while the unit suite still passes. Preflight exited 0 with 15 limitations under its laptop policy.

   Fix: remove mock output from customer paths. Return a clear capability-unavailable result or call a real provisioned local engine. Define required capabilities per release profile, run preflight for that profile at boot, and require actual audio/model execution where appropriate. Offer only usable modes and explain degraded capability audibly. Replace readiness claims with measured supported configurations.

   Acceptance: boot a clean target image with networking disabled and demonstrate every advertised offline feature using bundled models. Required failures must prevent the release profile from reporting ready.

10. **P1 — Installation and configuration are not reproducible.**

    Evidence: [install.sh](../install.sh) installs Claude Code rather than BlindAssist. The README uses a nested `Blindterminal/` layout absent from this checkout and links hardware instructions through an absolute path on a developer machine. Tracked `data` and `models` symlinks point to `/mnt/aet_usb/AET_USB/`. Most Python requirements are lower bounds or unpinned. Two example settings files disagree on model IDs and other defaults. Several modules independently load/cast settings at import; values are not centrally schema-validated. `vector_store_type`, `use_cohere_rerank` and the JSON Pinecone key do not control the inspected construction paths as their names suggest: store setup uses environment variables, and reranking initialization depends on key presence.

    Fix: replace or clearly relocate the unrelated installer; publish a tested device installer/image. Select a supported Python/OS pair for each platform, produce platform-specific dependency locks, and pin/checksum provisioned models. Use writable device-local data directories with explicit removable-media support. Keep one canonical config template and a validated schema; honor or remove unused settings. Centralize model IDs and test configured providers without exposing credentials. Repair README paths.

    Acceptance: a fresh Pi image installs from the documented command, passes dependency checks, provisions assets, survives reboot and starts without developer paths. Invalid configuration produces a precise setup error instead of silently removing a feature.

11. **P2 — Provider racing and background input have incomplete lifecycle control.**

    Evidence: `ask_ai()` submits Groq and OpenAI for each request and returns the first answer without cancelling queued losing futures. A running loser may continue its network call and cost. The shared executor has no shutdown in `main.shutdown()`. In GPS mode, microphone listeners and gesture workers start in background threads; menu selection does not cancel/join the initial listener, and gesture shutdown sets an event without waiting for camera release before continuing.

    Fix: use a configurable primary provider with a controlled fallback and one shared request deadline; reserve racing for measured latency needs and permitted data destinations. Cancel pending futures, bound in-flight work, and explicitly close provider resources. Give each mode a resource-owning session with cancellation and join/cleanup; allow only one microphone/camera owner at a time. A stop request must remain available during all phases.

    Acceptance: make one provider slow and the other fast over repeated questions; queue depth stays bounded and shutdown completes. Select GPS options immediately by button while a voice attempt is active, then start another voice/camera mode; no prior worker may retain the resource or consume later input.

12. **P2 — Textbook answers and quality gates need stronger evidence.**

    Evidence: [ai_query.py](../modules/ai_query.py) uses a general conversational system prompt. Retrieval can fail and the same question still goes to general providers; the result carries no source/page evidence or answer type. The direct `GeminiAgent` has a stricter prompt, but `ask_ai()` uses its own provider path. Existing OCR/translation benchmark scripts and validation documents are useful; they do not themselves demonstrate measured target-device accuracy. No completed representative device evaluation was established by this review. The OCR script's numeric metric matches unordered counts, so swapped numbers can still count as numerically correct; it also permits a release gate without numeric examples. Translation human labels are read from input rows rather than being tied to a saved, reviewed generated output.

    Fix: separate textbook-grounded answers from general conversation. Preserve document/page/chunk identities through retrieval, require sufficient evidence for a textbook answer, and announce when evidence is missing. Include a short spoken source and retain a reviewable answer record without storing confidential content. For evaluation, freeze real device inputs and version the exact generated outputs, model/configuration and native-reviewer scores. Check ordered/contextual numeric fidelity and mandatory subsets; enforce required dataset coverage in the scripts.

    Acceptance: ask unanswerable and deliberately misleading textbook questions; the app must abstain or label a general answer. Meet the documented OCR and translation gates on representative held-out material, record per-language/per-layout results and target-Pi p95 latency, and test complete user tasks with blind participants.

**Additional navigation scope issue**

`_get_route()` produces one concatenated string containing all maneuvers, and `mode_gps()` speaks it once. There is no position-tracking loop, progress-based turn triggering, deviation detection or rerouting. The destination geocoder accepts the first global match without confirmation. This is a route summary, not demonstrated live guidance.

Use a confirmed destination with locality context, a routing backend verified for pedestrian routes, fresh position updates and bounded turn prompts before describing the feature as live navigation. The URL segment `/foot/` alone does not prove that the backend uses pedestrian data: OSRM documents that the routing mode is determined when the server's data is prepared. See the [official OSRM HTTP API](https://project-osrm.org/docs/v26.5.0/http). The actual hosted backend's pedestrian behavior was not validated in this audit.

**Implementation order**

1. Resolve findings 1–4 together: explicit device/simulation separation, one inherited privacy policy, strict private audio and control over cloud/storage sinks. Add end-to-end regressions for private OCR explanations, routing failure and timeout.
2. Resolve findings 5–7: stable document identities, model-specific indexes, exact chunk/vector pairing, transactional persistence and honest outcomes. Verify offline/online transitions and interrupted writes.
3. Resolve findings 8–11: bounded cancellation, reliable offline prompts, capability-based startup, reproducible installation and mode resource ownership.
4. Resolve finding 12 and the navigation scope issue through measured device validation. Keep unvalidated features disabled in a customer release profile.

**Suggested first product scope**

Start with an accessible document reader, exact local math, and only the language directions that pass measured validation. Add source-backed educational Q&A after retrieval is reliable. Keep object recognition scoped to announcing recognized objects and horizontal position, as its setup document already does. Gate live navigation behind a separate validation effort. Provide a short repeat/help/cancel interaction, adjustable speech, resumable document reading and an audible startup health report.

For a distributable device, also maintain a tested boot service, restart/recovery policy, bounded and redacted logs, versioned update/rollback procedure, and a dependency/model/voice asset inventory with applicable licenses and attribution. These are release work items; this review does not conclude that any particular dependency is legally unsuitable.

**Release decision**

Do not use the README badge or passing mocked tests as a shipping gate. Require closure of P0/P1 findings, a clean installation on the supported Pi image, full private-session sink tests, successful offline demonstrations of advertised capabilities, interrupted-write recovery, stable repeated mode changes, and the representative accuracy/usability evidence described above. Record actual results and remaining limitations for the exact released build.
