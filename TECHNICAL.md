# Podsponsor: Technical Overview

Podsponsor is an automated podcast ad removal and transcription pipeline. It processes podcast audio files to transcribe content, detect sponsored segments, remove them seamlessly, and keep all metadata synced. The system integrates speech-to-text models, heuristic text matching, Large Language Models (LLMs), and precise audio manipulation logic.

## Pipeline Architecture

Processing runs in two sequential phases:

1. **Phase 1 — Transcription**: All target MP3 files are transcribed first, producing `.words.json` (word-level timings) and `.srt` files. Files with existing valid transcriptions are skipped. This runs before any detection to ensure the full corpus is indexed.
2. **Phase 2 — Detection & Cutting**: Cross-file matching identifies repeated segments across the archive. Each new/unprocessed file is then sent through LLM analysis, confidence filtering, silence-snapped cutting, and transcript shifting.

A `--transcribe-only` flag allows running Phase 1 in isolation (useful for building a transcript corpus without triggering ad removal). A `--dry-run` flag runs the full detection pipeline but logs what *would* be cut without modifying any files.

## Core Features

### 1. Audio Transcription & Word-Level Timing
- **WhisperX Integration**: Uses the heavily optimized `whisperx` library for batched GPU transcription. It supports precise configuration (model size, compute type `float16`, batch size).
- **Word-Level Precision**: Standard Whisper produces segment-level timestamps. Podsponsor specifically stores word-level timing data in a `.words.json` format alongside standard `.srt` subtitles. This granularity is essential for making precise, non-abrupt audio cuts later.
- **Language Detection**: Automatically respects language settings defined in a show's `metadata.json`, preventing erroneous language hallucination.

### 2. Multi-Tier Ad Detection Engine
To maximize accuracy, ad detection is divided into multiple phases:

#### a) Cross-File Segment Matching (1-to-N)
- **Concept**: Promos, jingles, and standard ad-reads are frequently repeated across episodes. By finding text that exists in multiple files, the system algorithmically identifies non-unique content.
- **Exact & Fuzzy Matching**: Uses dictionary hashes for O(N) exact matches and `SequenceMatcher` trigram comparisons for near-duplicate fuzzy matching. A trigram pre-filter rapidly eliminates dissimilar pairs before invoking the expensive `SequenceMatcher`.
- **1-to-N Optimization**: To scale to large podcast archives, it avoids O(N²) comparisons. It only compares **new/unprocessed** episodes against the established archive. 
- **Multiprocessing**: Fuzzy text comparison is highly un-GIL friendly, so it uses a fork-based `ProcessPoolExecutor` (explicitly using `fork` context for cross-platform safety) distributing chunked unique text data across available CPU cores with minimal memory overhead.

#### b) Local and Global Pattern Stores
- The system stores confirmed ad texts into JSON files: a per-show `ads.json` and a global registry `~/.podsponsor/ads.json`.
- Newly encountered segments are checked against these databases using trigram-accelerated similarity matching for immediate tagging.

#### c) LLM-Assisted Analysis
- **Provider Fallback**: Uses an OpenAI-compatible API client with an ordered list of fallback providers configured in `config.yaml` to guarantee resilience against rate limits or downtime.
- **Prompt Injection & Structured Output**: Identified repeated segments are tagged with `[REPEATED]` within the LLM prompt. The LLM is forced to return strict `json_schema` output outlining the `start_index`, `end_index`, and `confidence` of the ad blocks. 
- **Confidence & Duration Filtering**: LLM results are filtered by a configurable `min_confidence` threshold (default 0.70) to discard uncertain detections, and by `min_ad_duration` (default 8s) to avoid cutting brief false positives like topic transitions.
- **Episode Summary Generation**: In the same pass, the LLM generates a structured Markdown file per episode outlining the main topics, references, and takeaways.

### 3. Precision Audio Cutting
- **Silence Snapping (`ffmpeg`)**: Raw mathematical cuts often clip mid-word. Podsponsor runs `ffmpeg silencedetect` (-40dB, 1.5s default) across the file to identify conversational pauses. It mathematically snaps the LLM's ad boundaries outwards to the nearest silence boundaries.
- **Re-encoding**: Instead of fragile stream-copying, it extracts the "kept" regions and cleanly concatenates them while re-encoding the audio track (`libmp3lame`, variable bitrate) to avoid corrupt seeking or duration headers.
- **Safety Guardrail**: Any detection that calculates the ad ratio to be greater than 50% of the show automatically aborts the cutting process to prevent destructive false positives.

### 4. Transcript Mathematical Shifting
- Simply removing audio from an MP3 will instantly desynchronize the accompanying `.srt` and `.words.json` files. 
- Podsponsor uses a mathematical piecewise mapping function (`shift_transcript`). It calculates the exact amount of time removed prior to any given word and shifts its `start` and `end` timestamps down by the exact offset, completely preserving transcript sync against the newly rendered "clean" MP3.
- In `save_as_clean` mode, shifted transcripts are saved alongside the clean audio as `-clean.srt` and `-clean.words.json`.

### 5. Idempotence, State, and Safety
- **Manifest Store (`.podsponsor-manifest.json`)**: Tracks processing state per directory. Using a fingerprint of file size and modified-time (`st_mtime_ns`), the system knows exactly which files succeeded, failed, or are pending. 
- **Backup Strategy**: Supports multiple configurable output methods (`output_scheme`), defaulting to `overwrite_with_backup`. The original `.mp3`, `.srt`, and `.words.json` files are safely copied to a `backup/` subfolder before the cleanly cut versions overwrite the primary paths. Backups are always freshly copied (even on `--force` reruns) to guarantee the backup reflects the true original.
- **Memory Efficiency**: For large podcast directories, rather than repeatedly querying disk structures during 1-to-N analysis, segment JSON data is pre-loaded sequentially into a dictionary cache, severely cutting down on I/O bottlenecks.
