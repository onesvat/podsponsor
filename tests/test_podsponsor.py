import json
import logging
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from podsponsor import (
    PodsponsorConfig,
    Processor,
    Transcriber,
    analyze_with_llm,
    build_arg_parser,
    compute_keep_regions,
    configure_logging,
    default_log_file_path,
    find_repeated_segments,
    normalize_cut_regions,
    parse_llm_ad_blocks,
    resolve_progress_mode,
    shift_transcript,
)


CONFIG_TEXT = """
whisper:
  model: medium
  device: cpu
  compute_type: float32
llm:
  summary_language: en
  providers:
    - base_url: http://localhost:11434/v1
      model: test
      api_key: dummy
      timeout_seconds: 5
      max_retries: 0
detection:
  similarity_threshold: 0.8
  silence_db: -40
  silence_min_duration: 1.5
  min_confidence: 0.70
  min_ad_duration: 8.0
""".strip()


def write_config(path: Path):
    path.write_text(CONFIG_TEXT, encoding="utf-8")


def close_root_handlers():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


class CutRegionTests(unittest.TestCase):
    def test_normalize_and_keep_regions(self):
        cuts = [(-1, 2), (10, 20), (15, 25), (40, 41)]
        normalized = normalize_cut_regions(cuts, total_duration=50)
        self.assertEqual([(0.0, 2.0), (10.0, 25.0), (40.0, 41.0)], normalized)

        keep = compute_keep_regions(normalized, total_duration=50)
        self.assertEqual([(2.0, 10.0), (25.0, 40.0), (41.0, 50.0)], keep)


class CrossFileMatchingTests(unittest.TestCase):
    def test_finds_repeated_segments(self):
        segments_a = [
            {"start": 0, "end": 5, "text": "Welcome to our show today"},
            {"start": 5, "end": 10, "text": "This episode is brought to you by Acme Corp visit acme.com"},
            {"start": 10, "end": 15, "text": "Now let's talk about science"},
        ]
        segments_b = [
            {"start": 0, "end": 5, "text": "Hello everyone and welcome"},
            {"start": 5, "end": 10, "text": "This episode is brought to you by Acme Corp visit acme.com"},
            {"start": 10, "end": 15, "text": "Today we discuss history"},
        ]
        all_segs = {Path("/a.mp3"): segments_a, Path("/b.mp3"): segments_b}
        suspicious = find_repeated_segments(
            targets=list(all_segs.keys()),
            new_targets=set(all_segs.keys()),
            sim_threshold=0.80,
            load_all_segments_func=lambda x: all_segs,
        )

        self.assertIn(1, suspicious.get(Path("/a.mp3"), set()))
        self.assertIn(1, suspicious.get(Path("/b.mp3"), set()))

    def test_no_repeats_across_single_file(self):
        segments = [
            {"start": 0, "end": 5, "text": "This is a unique segment number one"},
            {"start": 5, "end": 10, "text": "This is a unique segment number two"},
        ]
        all_segs = {Path("/a.mp3"): segments}
        suspicious = find_repeated_segments(
            targets=list(all_segs.keys()),
            new_targets=set(all_segs.keys()),
            sim_threshold=0.80,
            load_all_segments_func=lambda x: all_segs,
        )
        self.assertEqual(0, sum(len(v) for v in suspicious.values()))

    def test_cross_file_progress_callback_is_invoked(self):
        segments_a = [{"start": 0, "end": 1, "text": "repeated segment long enough alpha"}]
        segments_b = [{"start": 0, "end": 1, "text": "repeated segment long enough alpha"}]
        all_segs = {Path("/a.mp3"): segments_a, Path("/b.mp3"): segments_b}
        progress_payloads = []

        find_repeated_segments(
            targets=list(all_segs.keys()),
            new_targets=set(all_segs.keys()),
            sim_threshold=0.80,
            load_all_segments_func=lambda x: all_segs,
            fuzzy_progress_callback=lambda payload: progress_payloads.append(payload),
        )

        self.assertGreaterEqual(len(progress_payloads), 1)
        self.assertEqual(0, progress_payloads[0]["chunks_done"])
        self.assertIn("chunks_total", progress_payloads[0])

    def test_fallback_when_fork_context_is_unavailable(self):
        segments_a = [{"start": 0, "end": 1, "text": "this repeated segment should be detected across files"}]
        segments_b = [{"start": 0, "end": 1, "text": "this repeated segment should be detected across files"}]
        all_segs = {Path("/a.mp3"): segments_a, Path("/b.mp3"): segments_b}

        with patch("podsponsor.multiprocessing.get_context", side_effect=ValueError("fork unavailable")):
            suspicious = find_repeated_segments(
                targets=list(all_segs.keys()),
                new_targets=set(all_segs.keys()),
                sim_threshold=0.80,
                load_all_segments_func=lambda x: all_segs,
            )

        self.assertIn(0, suspicious.get(Path("/a.mp3"), set()))
        self.assertIn(0, suspicious.get(Path("/b.mp3"), set()))


class ParseLlmAdBlocksTests(unittest.TestCase):
    def test_parses_valid_blocks(self):
        raw = [
            {"start_index": 5, "end_index": 8, "confidence": 0.95},
            {"start_index": 20, "end_index": 22, "confidence": 0.85},
        ]
        blocks = parse_llm_ad_blocks(raw, max_index=30, min_confidence=0.70)
        self.assertEqual(2, len(blocks))
        self.assertEqual((5, 8, 0.95), blocks[0])

    def test_filters_low_confidence(self):
        raw = [
            {"start_index": 5, "end_index": 8, "confidence": 0.95},
            {"start_index": 20, "end_index": 22, "confidence": 0.50},
        ]
        blocks = parse_llm_ad_blocks(raw, max_index=30, min_confidence=0.70)
        self.assertEqual(1, len(blocks))


class AnalyzeWithLlmFormattingTests(unittest.TestCase):
    def test_includes_repeated_and_high_freq_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            write_config(config_path)
            config = PodsponsorConfig(str(config_path))

            captured = {}

            class FakeCompletions:
                def create(self, **kwargs):
                    captured["messages"] = kwargs["messages"]
                    parsed = {"ads": [], "summary": "ok"}
                    choice = types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(parsed)),
                        index=0,
                        logprobs=None,
                        finish_reason="stop",
                    )
                    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
                    return types.SimpleNamespace(choices=[choice], usage=usage)

            class FakeOpenAI:
                def __init__(self, **kwargs):
                    self.chat = types.SimpleNamespace(completions=FakeCompletions())

            segments = [
                {"start": 0.0, "end": 1.0, "text": "segment zero", "frequency": 2},
                {"start": 1.0, "end": 2.0, "text": "segment one", "frequency": 3},
                {"start": 2.0, "end": 3.0, "text": "segment two", "frequency": 4},
            ]

            with patch.dict("sys.modules", {"openai": types.SimpleNamespace(OpenAI=FakeOpenAI)}):
                analyze_with_llm(config, segments, suspicious_indices={0, 1})

            user_content = captured["messages"][1]["content"]
            self.assertIn("[0] [REPEATED] segment zero", user_content)
            self.assertIn("[1] [REPEATED] [HIGH_FREQ] segment one", user_content)
            self.assertIn("[2] [HIGH_FREQ] segment two", user_content)
            self.assertNotIn("[0] [REPEATED] [HIGH_FREQ]", user_content)


class WhisperTranscribeChunkSizeTests(unittest.TestCase):
    def test_transcribe_passes_default_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)
            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            config = PodsponsorConfig(str(config_path))
            transcriber = Transcriber(config)
            captured = {}

            class FakeModel:
                def transcribe(self, audio, **kwargs):
                    captured["kwargs"] = kwargs
                    return {"segments": []}

            fake_model = FakeModel()
            fake_whisperx = types.SimpleNamespace(
                load_model=lambda *args, **kwargs: fake_model,
                load_audio=lambda *args, **kwargs: [0.0],
            )

            with patch.dict("sys.modules", {"whisperx": fake_whisperx}):
                transcriber.transcribe(mp3_path)

            self.assertEqual(20, captured["kwargs"]["chunk_size"])

    def test_transcribe_falls_back_when_chunk_size_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)
            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            config = PodsponsorConfig(str(config_path))
            transcriber = Transcriber(config)

            class LegacyModel:
                def __init__(self):
                    self.calls = []

                def transcribe(self, audio, **kwargs):
                    self.calls.append(kwargs)
                    if "chunk_size" in kwargs:
                        raise TypeError("chunk_size unsupported")
                    return {"segments": []}

            legacy_model = LegacyModel()
            fake_whisperx = types.SimpleNamespace(
                load_model=lambda *args, **kwargs: legacy_model,
                load_audio=lambda *args, **kwargs: [0.0],
            )

            with patch.dict("sys.modules", {"whisperx": fake_whisperx}):
                transcriber.transcribe(mp3_path)

            self.assertEqual(2, len(legacy_model.calls))
            self.assertIn("chunk_size", legacy_model.calls[0])
            self.assertNotIn("chunk_size", legacy_model.calls[1])


class ProcessorSidecarFlowTests(unittest.TestCase):
    def test_creates_new_sidecar_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            processor = Processor(str(config_path))
            sidecar = processor._ensure_sidecar(mp3_path)

            self.assertEqual("new", sidecar["status"])
            self.assertTrue((tmp_path / "episode.podsponsor.json").exists())

    def test_new_to_transcripted_writes_srt_and_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            processor = Processor(str(config_path))
            fake_result = {
                "language": "tr",
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": "hello"},
                    {"start": 2.0, "end": 3.0, "text": "world"},
                ],
            }
            with patch.object(processor.transcriber, "transcribe", return_value=fake_result):
                processor.ensure_transcription(mp3_path)

            sidecar = processor._ensure_sidecar(mp3_path)
            self.assertEqual("transcripted", sidecar["status"])
            self.assertEqual(2, len(sidecar["original_segments"]))
            self.assertTrue((tmp_path / "episode.srt").exists())
            self.assertFalse((tmp_path / "episode.words.json").exists())

    def test_progress_events_for_transcription_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)
            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            events = []
            processor = Processor(
                str(config_path),
                progress_callback=lambda event, mp3, payload: events.append((event, mp3, payload)),
            )

            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "transcripted"
            sidecar["original_segments"] = [{"start": 0.0, "end": 1.0, "text": "already here", "frequency": 0}]
            processor._save_sidecar(mp3_path)

            processor.ensure_transcription(mp3_path)
            event_names = [event for event, _, _ in events]
            self.assertIn("transcribing", event_names)
            self.assertIn("skipped", event_names)

    def test_frequency_uses_transcripted_and_success_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_a = tmp_path / "a.mp3"
            mp3_b = tmp_path / "b.mp3"
            mp3_a.write_bytes(b"a")
            mp3_b.write_bytes(b"b")

            processor = Processor(str(config_path))

            sidecar_a = processor._ensure_sidecar(mp3_a)
            sidecar_a["status"] = "transcripted"
            sidecar_a["original_segments"] = [
                {"start": 0.0, "end": 1.0, "text": "this is a repeated long segment", "frequency": 0}
            ]
            processor._save_sidecar(mp3_a)

            sidecar_b = processor._ensure_sidecar(mp3_b)
            sidecar_b["status"] = "success"
            sidecar_b["original_segments"] = [
                {"start": 0.0, "end": 1.0, "text": "this is a repeated long segment", "frequency": 0}
            ]
            processor._save_sidecar(mp3_b)

            processor.update_segment_frequencies([mp3_a, mp3_b])
            updated = processor._ensure_sidecar(mp3_a)
            self.assertEqual(2, updated["original_segments"][0]["frequency"])

    def test_update_flag_recuts_without_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)
            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")
            srt_path = mp3_path.with_suffix(".srt")
            srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello")

            processor = Processor(str(config_path))
            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "success"
            sidecar["original_segments"] = [{"start": 0.0, "end": 1.0, "text": "hello"}]
            sidecar["ad_blocks"] = [{"start": 0.0, "end": 0.5, "confidence": 1.0}] # Manual edit or previous run
            
            # Setup backup
            backup_dir = tmp_path / "backup"
            backup_dir.mkdir()
            backup_mp3 = backup_dir / "episode.mp3"
            backup_mp3.write_bytes(b"original")
            backup_srt = backup_dir / "episode.srt"
            backup_srt.write_text("original srt")
            
            sidecar["backup_path"] = str(backup_mp3.absolute())
            sidecar["backup_srt_path"] = str(backup_srt.absolute())
            processor._save_sidecar(mp3_path)

            with patch("podsponsor.analyze_with_llm", side_effect=AssertionError("LLM should not be called")):
                with patch("podsponsor.get_audio_duration", return_value=1.0):
                    with patch("podsponsor.check_silence", return_value=[]):
                        with patch.object(processor, "_cut_audio", return_value=str(backup_mp3.absolute())) as mock_cut:
                            status = processor.process_file(mp3_path, suspicious=set(), update=True)
                            self.assertEqual("processed", status)
                            mock_cut.assert_called_once()
                            # Verify restoration (shutil.copy2 was used)
                            self.assertEqual(mp3_path.read_bytes(), b"original")
                            self.assertEqual(srt_path.read_text(), "original srt")

    def test_success_updates_sidecar_and_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            processor = Processor(str(config_path))
            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "transcripted"
            sidecar["original_segments"] = [
                {"start": 0.0, "end": 10.0, "text": "this is an ad segment repeated", "frequency": 3},
                {"start": 10.0, "end": 20.0, "text": "this is normal content long enough", "frequency": 1},
            ]
            processor._save_sidecar(mp3_path)

            llm_parsed = {"ads": [{"start_index": 0, "end_index": 0, "confidence": 0.95}], "summary": "ok"}
            llm_payload = {"parsed": llm_parsed, "provider": "x", "model": "y", "usage": {}, "choices": []}

            with patch("podsponsor.analyze_with_llm", return_value=(llm_parsed, llm_payload)):
                with patch("podsponsor.get_audio_duration", return_value=120.0):
                    with patch("podsponsor.check_silence", return_value=[]):
                        with patch.object(processor, "_cut_audio", return_value=str(tmp_path.absolute() / "backup/episode.mp3")):
                            status = processor.process_file(mp3_path, suspicious=set(), force=False)

            self.assertEqual("processed", status)
            updated = processor._ensure_sidecar(mp3_path)
            self.assertEqual("success", updated["status"])
            self.assertTrue(Path(updated["backup_path"]).is_absolute())
            self.assertEqual(1, len(updated["ad_blocks"]))

    def test_success_without_backup_keeps_sidecar_backup_fields_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)
            with config_path.open("a", encoding="utf-8") as f:
                f.write("\nbackup:\n  enabled: false\n")

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            processor = Processor(str(config_path))
            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "transcripted"
            sidecar["original_segments"] = [
                {"start": 0.0, "end": 10.0, "text": "this is an ad segment repeated", "frequency": 3},
                {"start": 10.0, "end": 20.0, "text": "this is normal content long enough", "frequency": 1},
            ]
            processor._save_sidecar(mp3_path)

            llm_parsed = {"ads": [{"start_index": 0, "end_index": 0, "confidence": 0.95}], "summary": "ok"}
            llm_payload = {"parsed": llm_parsed, "provider": "x", "model": "y", "usage": {}, "choices": []}

            with patch("podsponsor.analyze_with_llm", return_value=(llm_parsed, llm_payload)):
                with patch("podsponsor.get_audio_duration", return_value=120.0):
                    with patch("podsponsor.check_silence", return_value=[]):
                        with patch.object(processor, "_cut_audio", return_value=None):
                            status = processor.process_file(mp3_path, suspicious=set(), force=False)

            self.assertEqual("processed", status)
            updated = processor._ensure_sidecar(mp3_path)
            self.assertEqual("success", updated["status"])
            self.assertIsNone(updated["backup_path"])
            self.assertIsNone(updated["backup_srt_path"])

    def test_ad_block_frequency_uses_min_segment_frequency(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            processor = Processor(str(config_path))
            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "transcripted"
            sidecar["original_segments"] = [
                {"start": 0.0, "end": 10.0, "text": "ad block part one", "frequency": 5},
                {"start": 10.0, "end": 20.0, "text": "ad block part two", "frequency": 2},
                {"start": 20.0, "end": 30.0, "text": "normal content", "frequency": 1},
            ]
            processor._save_sidecar(mp3_path)

            llm_parsed = {"ads": [{"start_index": 0, "end_index": 1, "confidence": 0.95}], "summary": "ok"}
            llm_payload = {"parsed": llm_parsed, "provider": "x", "model": "y", "usage": {}, "choices": []}

            with patch("podsponsor.analyze_with_llm", return_value=(llm_parsed, llm_payload)):
                with patch("podsponsor.get_audio_duration", return_value=120.0):
                    with patch("podsponsor.check_silence", return_value=[]):
                        with patch.object(processor, "_cut_audio", return_value=str(tmp_path.absolute() / "backup/episode.mp3")):
                            status = processor.process_file(mp3_path, suspicious=set(), force=False)

            self.assertEqual("processed", status)
            updated = processor._ensure_sidecar(mp3_path)
            self.assertEqual(2, updated["ad_blocks"][0]["frequency"])

    def test_fail_fast_keeps_transcripted_if_cut_fails_after_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            processor = Processor(str(config_path))
            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "transcripted"
            sidecar["original_segments"] = [
                {"start": 0.0, "end": 10.0, "text": "this is an ad segment repeated", "frequency": 3},
                {"start": 10.0, "end": 20.0, "text": "this is normal content long enough", "frequency": 1},
            ]
            processor._save_sidecar(mp3_path)

            llm_parsed = {"ads": [{"start_index": 0, "end_index": 0, "confidence": 0.95}], "summary": "ok"}
            llm_payload = {"parsed": llm_parsed, "provider": "x", "model": "y", "usage": {}, "choices": []}

            with patch("podsponsor.analyze_with_llm", return_value=(llm_parsed, llm_payload)):
                with patch("podsponsor.get_audio_duration", return_value=120.0):
                    with patch("podsponsor.check_silence", return_value=[]):
                        with patch.object(processor, "_cut_audio", side_effect=RuntimeError("cut failed")):
                            with self.assertRaises(RuntimeError):
                                processor.process_file(mp3_path, suspicious=set(), force=False)

            updated = processor._ensure_sidecar(mp3_path)
            self.assertEqual("transcripted", updated["status"])
            # self.assertIsNotNone(updated["processing_info"]["llm_response"]) # Removed llm_response
            self.assertEqual(1, len(updated["ad_blocks"]))

    def test_progress_failed_event_emitted_on_llm_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.yaml"
            write_config(config_path)

            mp3_path = tmp_path / "episode.mp3"
            mp3_path.write_bytes(b"fake")

            events = []
            processor = Processor(
                str(config_path),
                progress_callback=lambda event, mp3, payload: events.append((event, mp3, payload)),
            )
            sidecar = processor._ensure_sidecar(mp3_path)
            sidecar["status"] = "transcripted"
            sidecar["original_segments"] = [
                {"start": 0.0, "end": 1.0, "text": "unique transcript segment long enough", "frequency": 1}
            ]
            processor._save_sidecar(mp3_path)

            with patch("podsponsor.analyze_with_llm", side_effect=RuntimeError("llm down")):
                status = processor.process_file(mp3_path, suspicious=set(), force=False)

            self.assertEqual("skipped", status)
            event_names = [event for event, _, _ in events]
            self.assertIn("failed", event_names)


class ShiftTranscriptTests(unittest.TestCase):
    def test_shift_transcript_adjusts_segment_times_without_words(self):
        segments = [{"start": 15.0, "end": 17.0, "text": "After the ad"}]
        keep_regions = [(0.0, 5.0), (10.0, 20.0)]
        shifted = shift_transcript(segments, keep_regions)

        self.assertEqual(1, len(shifted))
        self.assertAlmostEqual(10.0, shifted[0]["start"])
        self.assertAlmostEqual(12.0, shifted[0]["end"])


class FileFilterTests(unittest.TestCase):
    def test_should_process_mp3(self):
        from podsponsor import should_process_mp3

        self.assertTrue(should_process_mp3(Path("/some/dir/episode.mp3")))
        self.assertFalse(should_process_mp3(Path("/some/dir/episode.txt")))
        self.assertFalse(should_process_mp3(Path("/some/dir/episode-backup.mp3")))
        self.assertFalse(should_process_mp3(Path("/some/dir/episode-clean.mp3")))
        self.assertFalse(should_process_mp3(Path("/some/dir/backup/episode.mp3")))

    def test_should_process_mp3_respects_excluded_dirs(self):
        from podsponsor import should_process_mp3

        excluded = {Path("/some/dir/custom-backups").resolve()}
        self.assertFalse(
            should_process_mp3(
                Path("/some/dir/custom-backups/episode.mp3"),
                excluded_dirs=excluded,
            )
        )
        self.assertTrue(should_process_mp3(Path("/some/dir/episode.mp3"), excluded_dirs=excluded))


class ConfigTests(unittest.TestCase):
    def test_backup_location_defaults_to_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            write_config(config_path)
            config = PodsponsorConfig(str(config_path))
            self.assertEqual("backup", config.backup_location)

    def test_backup_location_reads_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            write_config(config_path)
            with config_path.open("a", encoding="utf-8") as f:
                f.write("\nbackup:\n  location: /tmp/podsponsor-backups\n")
            config = PodsponsorConfig(str(config_path))
            self.assertEqual("/tmp/podsponsor-backups", config.backup_location)

    def test_backup_enabled_reads_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            write_config(config_path)
            with config_path.open("a", encoding="utf-8") as f:
                f.write("\nbackup:\n  enabled: false\n")
            config = PodsponsorConfig(str(config_path))
            self.assertFalse(config.backup_enabled)


class CliAndLoggingTests(unittest.TestCase):
    def test_parser_accepts_progress_and_log_file(self):
        parser = build_arg_parser()
        args = parser.parse_args(["/tmp/pods", "--progress", "off", "--log-file", "/tmp/podsponsor.log"])
        self.assertEqual("/tmp/pods", args.path)
        self.assertEqual("off", args.progress)
        self.assertEqual("/tmp/podsponsor.log", args.log_file)

    def test_default_log_file_path_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = default_log_file_path(cwd=Path(tmp))
            self.assertTrue(str(path).startswith(str(Path(tmp) / "logs" / "podsponsor-")))
            self.assertTrue(str(path).endswith(".log"))
            self.assertRegex(path.name, r"^podsponsor-\d{8}-\d{6}\.log$")

    def test_resolve_progress_mode(self):
        self.assertEqual("off", resolve_progress_mode("off", stderr_is_tty=True, tqdm_available=True))
        self.assertEqual("tqdm", resolve_progress_mode("on", stderr_is_tty=False, tqdm_available=True))
        self.assertEqual("plain", resolve_progress_mode("on", stderr_is_tty=True, tqdm_available=False))
        self.assertEqual("plain", resolve_progress_mode("auto", stderr_is_tty=False, tqdm_available=True))

    def test_configure_logging_handles_tqdm_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "run.log"
            with patch("podsponsor.tqdm", None):
                returned = configure_logging(use_tqdm_console=True, log_file=log_path)
            self.assertEqual(log_path, returned)
            self.assertTrue(log_path.exists())
            close_root_handlers()
            logging.basicConfig(level=logging.INFO)

    def test_configure_logging_closes_previous_handlers(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_log = Path(tmp) / "first.log"
            second_log = Path(tmp) / "second.log"

            configure_logging(use_tqdm_console=False, log_file=first_log)
            first_file_handler = next(
                handler for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)
            )
            self.assertIsNotNone(first_file_handler.stream)
            self.assertFalse(first_file_handler.stream.closed)

            configure_logging(use_tqdm_console=False, log_file=second_log)
            self.assertTrue(first_file_handler.stream is None or first_file_handler.stream.closed)
            self.assertTrue(second_log.exists())

            close_root_handlers()
            logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    unittest.main()
