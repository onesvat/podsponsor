# Podsponsor: Technical Overview

Podsponsor is an automated podcast ad removal and transcription pipeline. It processes podcast audio files to transcribe content, detect sponsored segments, remove them seamlessly, and keep all metadata synced. The system integrates speech-to-text models, heuristic text matching, Large Language Models (LLMs), and precise audio manipulation logic.

## Pipeline Architecture

Processing runs in two sequential phases:

1. **Phase 1 — Transcription**: All target MP3 files are transcribed first, producing `.srt` and `<episode>.podsponsor.json` segment data. Files with existing valid sidecar segments are skipped. This runs before any detection to ensure the full corpus is indexed.
2. **Phase 2 — Detection & Cutting**: Cross-file matching identifies repeated segments across the archive. Each new/unprocessed file is then sent through LLM analysis, confidence filtering, silence-snapped cutting, and transcript shifting.

A `--transcribe-only` flag allows running Phase 1 in isolation (useful for building a transcript corpus without triggering ad removal). A `--dry-run` flag runs the full detection pipeline but logs what *would* be cut without modifying any files.

## Core Features

### 1. Audio Transcription & Segment Timing
- **WhisperX Integration**: Uses the heavily optimized `whisperx` library for batched GPU transcription. It supports precise configuration (model size, compute type `float16`, batch size).
- **Segment Granularity Note**: WhisperX segment boundaries are influenced by model/VAD/chunking internals. Podsponsor exposes `whisper.chunk_size` (default `20`) to bias toward shorter/longer chunking, while still relying on WhisperX's internal VAD/segmentation behavior.
- **Sidecar Segments**: Podsponsor stores segment timestamps and text directly in `<episode>.podsponsor.json` and uses those segments as the sole pipeline state source.
- **Language Detection**: Automatically respects language settings defined in a show's `metadata.json`, preventing erroneous language hallucination.

### 2. Multi-Tier Ad Detection Engine
To maximize accuracy, ad detection is divided into multiple phases:

#### a) Cross-File Segment Matching (1-to-N)
- **Concept**: Promos, jingles, and standard ad-reads are frequently repeated across episodes. By finding text that exists in multiple files, the system algorithmically identifies non-unique content.
- **Exact & Fuzzy Matching**: Uses dictionary hashes for O(N) exact matches and `SequenceMatcher` trigram comparisons for near-duplicate fuzzy matching. A trigram pre-filter rapidly eliminates dissimilar pairs before invoking the expensive `SequenceMatcher`.
- **1-to-N Optimization**: To scale to large podcast archives, it avoids O(N²) comparisons. It only compares **new/unprocessed** episodes against the established archive. 
- **Multiprocessing**: Fuzzy text comparison is highly un-GIL friendly. On POSIX systems, Podsponsor uses a fork-based `ProcessPoolExecutor` for speed. On platforms where `fork` is unavailable (for example Windows), it falls back to in-process fuzzy matching to remain functional, with slower performance.

#### b) LLM-Assisted Analysis
- **Provider Fallback**: Uses an OpenAI-compatible API client with an ordered list of fallback providers configured in `config.yaml` to guarantee resilience against rate limits or downtime.
- **Prompt Injection & Structured Output**: Identified repeated segments are tagged with `[REPEATED]`, and very high-frequency segments (`frequency >= 3`) are tagged with `[HIGH_FREQ]` in the LLM prompt. The LLM is forced to return strict `json_schema` output outlining the `start_index`, `end_index`, and `confidence` of the ad blocks. 
- **Confidence & Duration Filtering**: LLM results are filtered by a configurable `min_confidence` threshold (default 0.70) to discard uncertain detections, and by `min_ad_duration` (default 8s) to avoid cutting brief false positives like topic transitions.
- **Episode Summary Generation**: In the same pass, the LLM generates a structured Markdown file per episode outlining the main topics, references, and takeaways.

### 3. Precision Audio Cutting
- **Silence Snapping (`ffmpeg`)**: Raw mathematical cuts often clip mid-word. Podsponsor runs `ffmpeg silencedetect` (-40dB, 1.5s default) across the file to identify conversational pauses. It mathematically snaps the LLM's ad boundaries outwards to the nearest silence boundaries.
- **Re-encoding**: Instead of fragile stream-copying, it extracts the "kept" regions and cleanly concatenates them while re-encoding the audio track (`libmp3lame`, variable bitrate) to avoid corrupt seeking or duration headers.
- **Safety Guardrail**: Any detection that calculates the ad ratio to be greater than 50% of the show automatically aborts the cutting process to prevent destructive false positives.

### 4. Transcript Mathematical Shifting
- Simply removing audio from an MP3 will instantly desynchronize the accompanying `.srt` file. 
- Podsponsor uses a mathematical piecewise mapping function (`shift_transcript`). It calculates the exact amount of time removed prior to any given word and shifts its `start` and `end` timestamps down by the exact offset, preserving transcript sync against the newly rendered "clean" MP3.
- For segment-only entries that do not have word timestamps, both segment bounds must map into kept audio. If a segment straddles a cut boundary, the current behavior drops that segment rather than splitting it.

### 5. Idempotence, State, and Safety
- **Per-Episode Sidecar (`<episode>.podsponsor.json`)**: Tracks status (`new -> transcripted -> success`), transcript segments, frequency counts, and ad blocks.
- **Ad Block Frequency Semantics**: `ad_blocks[].frequency` is recorded as the minimum segment frequency inside that block (conservative lower-bound repeat signal).
- **Backup Strategy**: Original `.mp3` files are copied to `backup/` before the cleanly cut version overwrites the primary path.
- **Memory Efficiency**: For large podcast directories, rather than repeatedly querying disk structures during 1-to-N analysis, segment JSON data is pre-loaded sequentially into a dictionary cache, severely cutting down on I/O bottlenecks.
