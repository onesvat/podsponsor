import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("podsponsor")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path: Path, payload: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def get_global_patterns_path() -> Path:
    env_path = os.environ.get("PODSPONSOR_GLOBAL_PATTERNS")
    if env_path:
        return Path(env_path)
    return Path.home() / ".podsponsor" / "patterns.json"


class PodsponsorConfig:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f) or {}

        # Whisper settings
        self.w_model = self.data.get("whisper", {}).get("model", "medium")
        self.w_device = self.data.get("whisper", {}).get("device", "cuda")
        self.w_compute = self.data.get("whisper", {}).get("compute_type", "float16")

        # LLM settings
        llm = self.data.get("llm", {})
        self.summary_lang = llm.get("summary_language", "en")
        
        default_prompt = (
            "You are an expert podcast analyst. Your job is to identify sponsor ad reads and generate a detailed summary of the episode.\n"
            "The episode summary must strictly be written in {summary_language}.\n\n"
            "Here are the transcribed audio segments. Each segment has an ID. Some segments are marked with [REPEATED].\n"
            "Return the detected ad blocks with their starting and ending segment indices."
        )
        self.prompt_template = llm.get("prompt", default_prompt)
        
        self.llm_providers = llm.get("providers", [])
        if not self.llm_providers:
            # Fallback if config is outdated
            self.llm_providers = [{
                "base_url": llm.get("base_url", "http://localhost:11434/v1"),
                "model": llm.get("model", "llama3"),
                "api_key": llm.get("api_key", "dummy"),
                "timeout_seconds": float(llm.get("timeout_seconds", 600)),
            }]

        # Detection settings
        detection = self.data.get("detection", {})
        self.sim_thresh = float(detection.get("similarity_threshold", 0.80))
        self.silence_db = int(detection.get("silence_db", -40))
        self.silence_min_dur = float(detection.get("silence_min_duration", 1.5))
        self.min_confidence = float(detection.get("min_confidence", 0.70))
        self.min_ad_duration = float(detection.get("min_ad_duration", 8.0))

        # Processing (Output Scheme)
        processing = self.data.get("processing", {})
        self.output_scheme = processing.get("output_scheme", "overwrite_with_backup")
        # Legacy fallback
        if "replace_original" in processing:
            if processing.get("replace_original"):
                self.output_scheme = "overwrite_with_backup" if processing.get("backup_original") else "overwrite_no_backup"
            else:
                self.output_scheme = "save_as_clean"
                
        self.manifest_name = processing.get("manifest_name", ".podsponsor-manifest.json")
        self.local_ads_name = processing.get("local_ads_name", "ads.json")





def check_silence(input_path: Path, threshold_db: int = -40, duration_sec: float = 1.5) -> List[Tuple[float, float]]:
    """Detect silence intervals using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "warning",
        "-i",
        str(input_path),
        "-af",
        f"silencedetect=n={threshold_db}dB:d={duration_sec}",
        "-f",
        "null",
        "-",
    ]

    intervals: List[Tuple[float, float]] = []
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        start = None
        for line in res.stderr.splitlines():
            if "silence_start" in line:
                parts = line.split("silence_start: ")
                if len(parts) > 1:
                    start = float(parts[1])
            elif "silence_end" in line and start is not None:
                parts = line.split("silence_end: ")
                if len(parts) > 1:
                    end_part = parts[1].split(" |")[0]
                    end = float(end_part)
                    intervals.append((start, end))
                    start = None
    except subprocess.CalledProcessError as exc:
        logger.error("Silence detection failed: %s", exc.stderr)

    return intervals


def load_patterns(path: Path) -> List[str]:
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load pattern file %s: %s", path, exc)
        return []

    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str)]

    if isinstance(raw, dict):
        values = raw.get("patterns", [])
        if isinstance(values, list):
            return [item for item in values if isinstance(item, str)]

    logger.warning("Pattern file has unsupported format: %s", path)
    return []


class PatternStore:
    def __init__(self, show_dir: Path, local_name: str):
        self.local_path = show_dir / local_name
        self.global_path = get_global_patterns_path()

        self.local_patterns = load_patterns(self.local_path)
        self.global_patterns = load_patterns(self.global_path)

        self._local_set = set(self.local_patterns)
        self._global_set = set(self.global_patterns)

    def is_known_ad(self, text: str, threshold: float) -> bool:
        t1 = text.strip().lower()
        if len(t1) < 20:
            return False

        for pattern in self.local_patterns:
            if SequenceMatcher(None, t1, pattern.strip().lower()).ratio() > threshold:
                return True

        for pattern in self.global_patterns:
            if SequenceMatcher(None, t1, pattern.strip().lower()).ratio() > threshold:
                return True

        return False

    def add_pattern(self, text: str):
        cleaned = text.strip()
        if len(cleaned) <= 30:
            return

        local_changed = False
        global_changed = False

        if cleaned not in self._local_set:
            self._local_set.add(cleaned)
            self.local_patterns.append(cleaned)
            local_changed = True

        if cleaned not in self._global_set:
            self._global_set.add(cleaned)
            self.global_patterns.append(cleaned)
            global_changed = True

        if local_changed:
            save_json_atomic(self.local_path, self.local_patterns)

        if global_changed:
            save_json_atomic(self.global_path, self.global_patterns)


class GlobalPatternDB:
    def __init__(self):
        self.path = get_global_patterns_path()
        self.patterns = load_patterns(self.path)

    def save(self):
        save_json_atomic(self.path, self.patterns)


class ManifestStore:
    def __init__(self, show_dir: Path, manifest_name: str):
        self.path = show_dir / manifest_name
        self.data = {"version": 1, "files": {}}
        self._load()

    def _load(self):
        if not self.path.exists():
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load manifest %s: %s", self.path, exc)
            return

        if isinstance(loaded, dict) and isinstance(loaded.get("files"), dict):
            self.data = loaded
        else:
            logger.warning("Manifest has invalid format, ignoring: %s", self.path)

    def save(self):
        save_json_atomic(self.path, self.data)

    def fingerprint(self, mp3_path: Path) -> Dict[str, int]:
        stat = mp3_path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def get_entry(self, mp3_path: Path) -> dict:
        return dict(self.data.setdefault("files", {}).get(mp3_path.name, {}))

    def should_skip(self, mp3_path: Path, force: bool = False) -> bool:
        if force:
            return False

        entry = self.get_entry(mp3_path)
        if not entry:
            return False

        return entry.get("status") == "success" and entry.get("fingerprint") == self.fingerprint(mp3_path)

    def mark_processing(self, mp3_path: Path, stage: str, fingerprint: Dict[str, int], used_srt_cache: bool = False):
        files = self.data.setdefault("files", {})
        files[mp3_path.name] = {
            "status": "processing",
            "stage": stage,
            "fingerprint": fingerprint,
            "used_srt_cache": used_srt_cache,
            "last_error": "",
            "updated_at": utc_now_iso(),
        }
        save_json_atomic(self.path, self.data)

    def mark_failed(
        self,
        mp3_path: Path,
        stage: str,
        fingerprint: Dict[str, int],
        error: str,
        used_srt_cache: bool = False,
    ):
        files = self.data.setdefault("files", {})
        files[mp3_path.name] = {
            "status": "failed",
            "stage": stage,
            "fingerprint": fingerprint,
            "used_srt_cache": used_srt_cache,
            "last_error": error,
            "updated_at": utc_now_iso(),
        }
        save_json_atomic(self.path, self.data)

    def mark_success(self, mp3_path: Path, fingerprint: Dict[str, int], ad_blocks: List[Dict[str, float]] = None):
        files = self.data.setdefault("files", {})
        
        entry = {
            "status": "success",
            "stage": "done",
            "fingerprint": fingerprint,
            "last_error": "",
            "updated_at": utc_now_iso(),
        }
        
        if ad_blocks is not None:
            entry["ad_blocks"] = ad_blocks

        files[mp3_path.name] = entry
        save_json_atomic(self.path, self.data)


# --- Transcriber ---
class Transcriber:
    def __init__(self, config: PodsponsorConfig):
        self.config = config
        self._model = None

    def transcribe(self, audio_path: Path) -> dict:
        maybe_apply_runtime_lib_dir(self.config.ffmpeg_lib_dir)
        import whisperx

        if self._model is None:
            logger.info("Loading WhisperX model '%s' on %s", self.config.w_model, self.config.w_device)
            self._model = whisperx.load_model(
                self.config.w_model,
                self.config.w_device,
                compute_type=self.config.w_compute,
                threads=self.config.w_workers,
            )

        audio = whisperx.load_audio(str(audio_path))
        logger.info("Transcribing %s...", audio_path.name)
        result = self._model.transcribe(audio, batch_size=4)

        logger.info("Aligning %s at word level...", audio_path.name)
        align_model, align_metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.config.w_device,
        )
        return whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            self.config.w_device,
            return_char_alignments=False,
        )


def format_srt_ts(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    whole = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    return f"{whole // 3600:02}:{(whole % 3600) // 60:02}:{whole % 60:02},{ms:03}"


def parse_srt_ts(value: str) -> float:
    hhmmss, ms = value.strip().split(",")
    hours, minutes, seconds = hhmmss.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(ms) / 1000.0)


def save_srt(segments: List[dict], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, segment in enumerate(segments, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_ts(segment['start'])} --> {format_srt_ts(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")


def parse_srt(filepath: Path) -> List[dict]:
    try:
        content = filepath.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read SRT %s: %s", filepath, exc)
        return []

    segments: List[dict] = []
    for block in content.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        timing_idx = 0 if "-->" in lines[0] else 1
        if timing_idx >= len(lines) or "-->" not in lines[timing_idx]:
            continue

        try:
            start_raw, end_raw = [item.strip() for item in lines[timing_idx].split("-->", 1)]
            start = parse_srt_ts(start_raw)
            end = parse_srt_ts(end_raw)
        except (ValueError, IndexError):
            continue

        text_lines = lines[timing_idx + 1 :]
        text = " ".join(text_lines).strip()
        if not text:
            continue

        segments.append({"start": start, "end": end, "text": text})

    return segments


# --- Words JSON ---
def save_words_json(segments: List[dict], filepath: Path):
    """Save full segment data including word-level timings."""
    serializable = []
    for seg in segments:
        entry = {"start": seg["start"], "end": seg["end"], "text": seg.get("text", "")}
        if "words" in seg:
            entry["words"] = seg["words"]
        serializable.append(entry)
    save_json_atomic(filepath, serializable)


def load_words_json(filepath: Path) -> List[dict]:
    """Load segments with word-level timings from .words.json."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load words JSON %s: %s", filepath, exc)
        return []

    if not isinstance(data, list):
        return []

    return data


# --- Cross-File Matching ---
def find_repeated_segments(
    all_segments: Dict[Path, List[dict]],
    sim_threshold: float,
) -> Dict[Path, Set[int]]:
    """Find segments that appear in multiple files (likely ads).

    Uses exact-match hashing on normalized text for O(n) performance.
    Also runs fuzzy matching via SequenceMatcher for near-duplicates.
    """
    # Build index: normalized_text -> set of files it appears in
    text_to_files: Dict[str, Set[Path]] = defaultdict(set)
    # Also track: normalized_text -> list of (file, segment_index) for mapping back
    text_to_locations: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)

    for file_path, segments in all_segments.items():
        for idx, seg in enumerate(segments):
            normalized = seg.get("text", "").strip().lower()
            if len(normalized) < 20:
                continue
            text_to_files[normalized].add(file_path)
            text_to_locations[normalized].append((file_path, idx))

    # Collect suspicious indices per file (exact matches in ≥2 files)
    suspicious: Dict[Path, Set[int]] = defaultdict(set)
    for normalized, files in text_to_files.items():
        if len(files) >= 2:
            for file_path, idx in text_to_locations[normalized]:
                suspicious[file_path].add(idx)

    # Fuzzy matching: compare unique texts that didn't exact-match
    exact_matched = {t for t, files in text_to_files.items() if len(files) >= 2}
    unique_texts = [(t, locs) for t, locs in text_to_locations.items() if t not in exact_matched and len(locs) >= 2]

    for text, locations in unique_texts:
        # Check if this text appears in multiple files (fuzzy)
        files_for_text = set()
        for file_path, _ in locations:
            files_for_text.add(file_path)

        if len(files_for_text) < 2:
            continue

        # Compare across files
        file_groups: Dict[Path, List[Tuple[int, str]]] = defaultdict(list)
        for file_path, idx in locations:
            file_groups[file_path].append((idx, text))

        # If same normalized text in multiple files, already handled above
        # Here we handle different-but-similar texts across files
        pass

    # Fuzzy: compare segments across files using SequenceMatcher
    # Only compare unique texts from different files, limited to reasonable count
    all_unique_by_file: Dict[Path, List[Tuple[int, str]]] = defaultdict(list)
    for file_path, segments in all_segments.items():
        for idx, seg in enumerate(segments):
            normalized = seg.get("text", "").strip().lower()
            if len(normalized) < 20:
                continue
            if idx not in suspicious.get(file_path, set()):
                all_unique_by_file[file_path].append((idx, normalized))

    file_list = list(all_unique_by_file.keys())
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            file_a, file_b = file_list[i], file_list[j]
            for idx_a, text_a in all_unique_by_file[file_a]:
                for idx_b, text_b in all_unique_by_file[file_b]:
                    if abs(len(text_a) - len(text_b)) > len(text_a) * 0.3:
                        continue  # Skip very different lengths
                    if SequenceMatcher(None, text_a, text_b).ratio() > sim_threshold:
                        suspicious[file_a].add(idx_a)
                        suspicious[file_b].add(idx_b)

    total = sum(len(v) for v in suspicious.values())
    logger.info("Cross-file matching: found %d suspicious segments across %d files", total, len(suspicious))
    return dict(suspicious)


# --- LLM Client ---
def analyze_with_llm(
    config: PodsponsorConfig,
    segments: List[dict],
    suspicious_indices: Set[int],
) -> dict:
    from openai import OpenAI

    transcript_lines = []
    for i, segment in enumerate(segments):
        text = segment["text"].strip()
        if i in suspicious_indices:
            transcript_lines.append(f"[{i}] [REPEATED] {text}")
        else:
            transcript_lines.append(f"[{i}] {text}")

    transcript_text = "\n".join(transcript_lines)
    prompt = config.prompt_template.format(summary_language=config.summary_lang)
    
    # Define Structured Output Schema
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ad_detection_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ads": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_index": {"type": "integer"},
                                "end_index": {"type": "integer"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["start_index", "end_index", "confidence"],
                            "additionalProperties": False
                        }
                    },
                    "summary": {"type": "string"}
                },
                "required": ["ads", "summary"],
                "additionalProperties": False
            }
        }
    }

    last_error = None
    for provider in config.llm_providers:
        client = OpenAI(
            base_url=provider.get("base_url"),
            api_key=provider.get("api_key", "dummy"),
            timeout=provider.get("timeout_seconds", 600.0),
            max_retries=provider.get("max_retries", 2)
        )
        model = provider.get("model")
        logger.info("Calling LLM API using provider: %s (model: %s)", provider.get("base_url"), model)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript_text},
                ],
                temperature=0.0,
                response_format=schema,
            )

            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Empty response from LLM")
                
            return json.loads(response_content)
            
        except Exception as exc:
            logger.warning("LLM provider %s failed: %s", provider.get("base_url"), exc)
            last_error = exc

    logger.error("All LLM providers failed.")
    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


def parse_llm_ad_blocks(raw_ads, max_index: int, min_confidence: float) -> List[Tuple[int, int, float]]:
    """Parse LLM ad blocks and filter by confidence. Returns list of (start, end, confidence)."""
    blocks: List[Tuple[int, int, float]] = []
    if not isinstance(raw_ads, list):
        return blocks

    for item in raw_ads:
        if not isinstance(item, dict):
            continue

        try:
            start = int(item.get("start_index", -1))
            end = int(item.get("end_index", -1))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue

        if start < 0 or end < 0 or start > end or end >= max_index:
            continue

        if confidence < min_confidence:
            logger.info("Dropping low-confidence ad block [%d-%d] (%.2f < %.2f)", start, end, confidence, min_confidence)
            continue

        blocks.append((start, end, confidence))

    return blocks


def expand_ad_indices(
    ad_blocks: List[Tuple[int, int, float]],
    suspicious: Set[int],
    total_segments: int,
    expand_range: int = 2,
) -> set[int]:
    """Expand ad blocks by ±expand_range, including neighbours only if they are suspicious."""
    all_ad_indices: set[int] = set()

    for start, end, _ in ad_blocks:
        # Add all segments in the block
        for i in range(start, end + 1):
            all_ad_indices.add(i)

        # Expand backwards
        for i in range(max(0, start - expand_range), start):
            if i in suspicious:
                all_ad_indices.add(i)

        # Expand forwards
        for i in range(end + 1, min(total_segments, end + 1 + expand_range)):
            if i in suspicious:
                all_ad_indices.add(i)

    return all_ad_indices


def group_contiguous(indices: List[int]) -> List[List[int]]:
    if not indices:
        return []

    blocks: List[List[int]] = []
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            blocks.append(current)
            current = [idx]
    blocks.append(current)
    return blocks


def normalize_cut_regions(cut_regions: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    clamped: List[Tuple[float, float]] = []
    for start, end in cut_regions:
        s = max(0.0, min(total_duration, float(start)))
        e = max(0.0, min(total_duration, float(end)))
        if e <= s:
            continue
        clamped.append((s, e))

    if not clamped:
        return []

    clamped.sort(key=lambda item: item[0])
    merged: List[Tuple[float, float]] = [clamped[0]]

    for start, end in clamped[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-6:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def compute_keep_regions(cut_regions: List[Tuple[float, float]], total_duration: float) -> List[Tuple[float, float]]:
    keep_regions: List[Tuple[float, float]] = []
    last_end = 0.0

    for start, end in cut_regions:
        if start > last_end:
            keep_regions.append((last_end, start))
        last_end = max(last_end, end)

    if last_end < total_duration:
        keep_regions.append((last_end, total_duration))

    return [(start, end) for start, end in keep_regions if (end - start) > 0.02]


def get_time_mapping(t: float, keep_regions: List[Tuple[float, float]]) -> float | None:
    """Map old time `t` to new time, returning None if `t` falls in a cut region."""
    new_t = 0.0
    for k_start, k_end in keep_regions:
        if t < k_start:
            return None
        if t <= k_end:
            return new_t + (t - k_start)
        new_t += (k_end - k_start)
    return None


def shift_transcript(segments: List[dict], keep_regions: List[Tuple[float, float]]) -> List[dict]:
    """Shift timestamps of segments mathematically based on what regions were kept."""
    shifted = []
    
    for seg in segments:
        new_words = []
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
                
            w_start_new = get_time_mapping(w["start"], keep_regions)
            w_end_new = get_time_mapping(w["end"], keep_regions)
            
            # Require both bounds to be preserved inside a kept region
            if w_start_new is not None and w_end_new is not None:
                new_w = dict(w)
                new_w["start"] = w_start_new
                new_w["end"] = w_end_new
                new_words.append(new_w)
                
        if seg.get("words"):
            if not new_words:
                continue # Entire segment's words were cut
                
            new_seg = dict(seg)
            new_seg["words"] = new_words
            new_seg["start"] = new_words[0]["start"]
            new_seg["end"] = new_words[-1]["end"]
            
            if len(new_words) < len(seg["words"]):
                # Rebuild text if words were removed
                new_seg["text"] = "".join(w.get("word", "") for w in new_words)
                
            shifted.append(new_seg)
        else:
            # Segment without word-level timings
            s_start = get_time_mapping(seg["start"], keep_regions)
            s_end = get_time_mapping(seg["end"], keep_regions)
            
            if s_start is not None and s_end is not None:
                new_seg = dict(seg)
                new_seg["start"] = s_start
                new_seg["end"] = s_end
                shifted.append(new_seg)

    return shifted


def get_audio_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


# --- Processor Engine ---
class Processor:
    def __init__(self, config_path: str):
        self.config = PodsponsorConfig(config_path)
        self.transcriber = Transcriber(self.config)
        self.pattern_stores: Dict[Path, PatternStore] = {}
        self.manifests: Dict[Path, ManifestStore] = {}

    def _get_pattern_store(self, show_dir: Path) -> PatternStore:
        if show_dir not in self.pattern_stores:
            self.pattern_stores[show_dir] = PatternStore(show_dir, self.config.local_ads_name)
        return self.pattern_stores[show_dir]

    def _get_manifest(self, show_dir: Path) -> ManifestStore:
        if show_dir not in self.manifests:
            self.manifests[show_dir] = ManifestStore(show_dir, self.config.manifest_name)
        return self.manifests[show_dir]

    def ensure_transcription(self, mp3_path: Path):
        """Phase 1: Ensure .words.json (and .srt) exist for this file."""
        words_path = mp3_path.with_suffix(".words.json")
        srt_path = mp3_path.with_suffix(".srt")

        if words_path.exists():
            segments = load_words_json(words_path)
            if segments:
                logger.info("Word-level data exists, skipping transcription: %s", mp3_path.name)
                return

        # Need to transcribe (even if SRT exists, we need word-level data)
        if srt_path.exists() and not words_path.exists():
            logger.info("SRT exists but no word-level data, re-transcribing: %s", mp3_path.name)

        logger.info("Transcribing: %s", mp3_path.name)
        result = self.transcriber.transcribe(mp3_path)
        segments = result.get("segments", [])

        if not segments:
            logger.warning("No segments produced for %s", mp3_path.name)
            return

        # Save both formats
        save_words_json(segments, words_path)
        save_srt(segments, srt_path)
        logger.info("Saved word-level data and SRT for: %s", mp3_path.name)

    def load_all_segments(self, targets: List[Path]) -> Dict[Path, List[dict]]:
        """Load segments from .words.json for all target files."""
        all_segments: Dict[Path, List[dict]] = {}
        for mp3_path in targets:
            words_path = mp3_path.with_suffix(".words.json")
            if words_path.exists():
                segments = load_words_json(words_path)
                if segments:
                    all_segments[mp3_path] = segments
        return all_segments

    def get_exact_word_timing(self, segment: dict, boundary: str) -> float:
        """Get precise start/end time from WhisperX word timings when available."""
        words = segment.get("words", [])
        if not words:
            return float(segment[boundary])

        valid_words = [w for w in words if "start" in w and "end" in w]
        if not valid_words:
            return float(segment[boundary])

        if boundary == "start":
            return float(valid_words[0]["start"])
        return float(valid_words[-1]["end"])

    def process_file(self, mp3_path: Path, suspicious: Set[int], force: bool = False) -> str:
        """Phase 2: Detect ads and cut audio for a single file."""
        manifest = self._get_manifest(mp3_path.parent)
        pattern_store = self._get_pattern_store(mp3_path.parent)
        fingerprint = manifest.fingerprint(mp3_path)
        md_path = mp3_path.with_suffix(".md")
        words_path = mp3_path.with_suffix(".words.json")

        if manifest.should_skip(mp3_path, force=force):
            logger.info("Skipping already processed file: %s", mp3_path.name)
            return "skipped"

        stage = "load"

        try:
            # Load segments from .words.json (has word-level timings)
            segments = load_words_json(words_path)
            if not segments:
                raise RuntimeError(f"No word-level data found for {mp3_path.name}")

            # Combine suspicious indices with PatternStore matches
            known_ad_indices = {
                i for i, seg in enumerate(segments)
                if pattern_store.is_known_ad(seg.get("text", ""), self.config.sim_thresh)
            }
            all_suspicious = suspicious | known_ad_indices

            # LLM analysis
            stage = "llm"
            manifest.mark_processing(mp3_path, stage, fingerprint)
            try:
                llm_response = analyze_with_llm(self.config, segments, all_suspicious)
            except Exception as exc:
                manifest.mark_failed(mp3_path, stage, fingerprint, str(exc))
                logger.error("LLM failed for %s: %s", mp3_path.name, exc)
                return "failed"

            # Save summary
            summary = llm_response.get("summary", "")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info("Saved Summary: %s", md_path.name)

            # Parse ad blocks with confidence filter
            ad_blocks = parse_llm_ad_blocks(
                llm_response.get("ads", []),
                max_index=len(segments),
                min_confidence=self.config.min_confidence,
            )

            if not ad_blocks:
                logger.info("No ads detected (after confidence filter). Skipping cut.")
                manifest.mark_success(mp3_path, fingerprint)
                return "processed"

            # Expand with neighbours (only if also suspicious)
            all_ad_indices = expand_ad_indices(ad_blocks, all_suspicious, len(segments))

            # Group into contiguous blocks
            sorted_indices = sorted(all_ad_indices)
            blocks = group_contiguous(sorted_indices)

            # Build cut regions with word-level precision
            silences = check_silence(mp3_path, self.config.silence_db, self.config.silence_min_dur)
            total_duration = get_audio_duration(mp3_path)
            cut_regions: List[Tuple[float, float]] = []
            final_ad_blocks: List[Dict[str, float]] = []

            for block in blocks:
                start_seg = segments[block[0]]
                end_seg = segments[block[-1]]

                # Store ad text as pattern for future matching
                ad_text = " ".join(segments[i].get("text", "") for i in block)
                pattern_store.add_pattern(ad_text)

                w_start = self.get_exact_word_timing(start_seg, "start")
                w_end = self.get_exact_word_timing(end_seg, "end")

                # Min duration filter
                if (w_end - w_start) < self.config.min_ad_duration:
                    logger.info("Dropping short ad block (%.1fs < %.1fs): segments %d-%d",
                                w_end - w_start, self.config.min_ad_duration, block[0], block[-1])
                    continue

                final_start = w_start
                final_end = w_end

                # Snap to silence boundaries
                for silence_start, silence_end in silences:
                    if silence_end < final_start and (final_start - silence_end) < 15.0:
                        final_start = silence_end

                for silence_start, silence_end in silences:
                    if silence_start > final_end:
                        if (silence_start - final_end) < 15.0:
                            final_end = silence_start
                        break

                logger.info("Ad found at %.2fs - %.2fs (segments %d-%d)", w_start, w_end, block[0], block[-1])
                logger.info("  Snapping to silence: %.2fs - %.2fs", final_start, final_end)
                cut_regions.append((final_start, final_end))
                final_ad_blocks.append({
                    "start": final_start,
                    "end": final_end,
                    "text": ad_text
                })

            if not cut_regions:
                logger.info("No ad regions survived filtering. Skipping cut.")
                manifest.mark_success(mp3_path, fingerprint)
                return "processed"

            # Safety guardrail: if total ad time > 50% of episode, skip
            total_ad_time = sum(end - start for start, end in cut_regions)
            ad_ratio = total_ad_time / total_duration if total_duration > 0 else 0
            if ad_ratio > 0.50:
                logger.warning(
                    "SAFETY: Ad time (%.1fs) is %.0f%% of episode (%.1fs) — skipping cut for %s",
                    total_ad_time, ad_ratio * 100, total_duration, mp3_path.name,
                )
                manifest.mark_failed(mp3_path, "safety", fingerprint, f"Ad ratio too high: {ad_ratio:.2f}")
                return "failed"

            # Cut audio
            stage = "cut"
            manifest.mark_processing(mp3_path, stage, fingerprint)
            self._cut_audio(mp3_path, cut_regions, segments)
            manifest.mark_success(mp3_path, fingerprint, ad_blocks=final_ad_blocks)
            return "processed"

        except Exception as exc:
            manifest.mark_failed(mp3_path, stage, fingerprint, str(exc))
            logger.error("Failed processing %s at stage %s: %s", mp3_path.name, stage, exc)
            return "failed"

    def _cut_audio(self, mp3_path: Path, cut_regions: List[Tuple[float, float]], segments: List[dict]):
        out_path = mp3_path.with_name(f"{mp3_path.stem}-clean.mp3")
        total_duration = get_audio_duration(mp3_path)

        normalized_cut = normalize_cut_regions(cut_regions, total_duration)
        keep_regions = compute_keep_regions(normalized_cut, total_duration)
        if not keep_regions:
            raise RuntimeError("No keep regions left after cut planning")

        filter_parts = []
        concat_inputs = []
        for i, (start, end) in enumerate(keep_regions):
            filter_parts.append(f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[a{i}]")

        filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(keep_regions)}:v=0:a=1[outa]")
        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "warning",
            "-i",
            str(mp3_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(out_path),
        ]
        logger.info("Executing robust cut with re-encode...")
        subprocess.run(cmd, check=True)

        if self.config.output_scheme == "overwrite_with_backup" or self.config.output_scheme == "overwrite_no_backup":
            if self.config.output_scheme == "overwrite_with_backup":
                backup_dir = mp3_path.parent / "backup"
                backup_dir.mkdir(exist_ok=True)
                
                # Backup originals (audio + transcripts)
                for ext in [".mp3", ".srt", ".words.json"]:
                    orig_file = mp3_path.with_suffix(ext)
                    if orig_file.exists():
                        backup_file = backup_dir / orig_file.name
                        if not backup_file.exists():
                            shutil.copy2(orig_file, backup_file)

            shutil.move(out_path, mp3_path)
            
            # Shift transcripts and overwrite main instances
            shifted_segments = shift_transcript(segments, keep_regions)
            save_words_json(shifted_segments, mp3_path.with_suffix(".words.json"))
            save_srt(shifted_segments, mp3_path.with_suffix(".srt"))
            
            logger.info("Done. Clean file and updated transcripts are at %s", mp3_path)
        else:
            logger.info("Done. Clean file generated: %s", out_path.name)


def should_process_mp3(path: Path) -> bool:
    if path.suffix.lower() != ".mp3":
        return False
    if "backup" in path.parts:
        return False
    name = path.name.lower()
    return not (name.endswith("-backup.mp3") or name.endswith("-clean.mp3"))


def main():
    parser = argparse.ArgumentParser(description="Podsponsor - Podcast Ad Remover")
    parser.add_argument("path", nargs="?", help="Path to podcast MP3 or directory")
    parser.add_argument("--list-patterns", action="store_true", help="List known global ad patterns")
    parser.add_argument("--clear-patterns", action="store_true", help="Clear global pattern DB")
    parser.add_argument("--force", action="store_true", help="Reprocess files even if manifest says success")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    if args.list_patterns:
        db = GlobalPatternDB()
        print(f"Found {len(db.patterns)} known global ad patterns.")
        for i, pattern in enumerate(db.patterns):
            print(f"[{i}] {pattern[:100]}...")
        return

    if args.clear_patterns:
        db = GlobalPatternDB()
        db.patterns = []
        db.save()
        print("Global pattern DB cleared.")
        return

    if not args.path:
        parser.print_help()
        return

    path = Path(args.path)
    if not path.exists():
        print("Path not found.")
        sys.exit(1)

    processor = Processor(args.config)

    targets: List[Path] = []
    if path.is_file() and should_process_mp3(path):
        targets.append(path)
    elif path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_file() and should_process_mp3(candidate):
                targets.append(candidate)

    targets.sort()
    logger.info("Found %d files to process.", len(targets))

    # Phase 1: Transcribe all files
    logger.info("=== Phase 1: Transcription ===")
    for target in targets:
        try:
            processor.ensure_transcription(target)
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", target.name, exc)

    # Cross-file matching
    logger.info("=== Cross-File Matching ===")
    all_segments = processor.load_all_segments(targets)
    suspicious = find_repeated_segments(all_segments, processor.config.sim_thresh)

    # Phase 2: Detect & cut
    logger.info("=== Phase 2: Detection & Cutting ===")
    processed = 0
    skipped = 0
    failed = 0

    for target in targets:
        status = processor.process_file(target, suspicious=suspicious.get(target, set()), force=args.force)
        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    logger.info("Run complete. processed=%d skipped=%d failed=%d", processed, skipped, failed)


if __name__ == "__main__":
    main()
