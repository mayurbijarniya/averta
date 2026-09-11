from averta.normalize import (
    digest,
    error_signature,
    is_valid_json,
    looks_like_error,
    parse_instance_id,
)


class TestDigest:
    def test_ignores_whitespace_differences(self):
        assert digest('{"cmd": "ls"}') == digest('{"cmd":   "ls"}\n')

    def test_distinguishes_different_inputs(self):
        assert digest('{"cmd": "ls"}') != digest('{"cmd": "pwd"}')

    def test_none_passes_through(self):
        assert digest(None) is None


class TestIsValidJson:
    def test_valid(self):
        assert is_valid_json('{"command": "str_replace", "path": "/a/b.py"}')

    def test_truncated_generation(self):
        assert not is_valid_json('{"command":"str_replace","path":"/workspace/get')

    def test_empty(self):
        assert not is_valid_json("")


class TestParseInstanceId:
    def test_simple(self):
        assert parse_instance_id("getmoto__moto-5321") == ("getmoto/moto", "5321")

    def test_hyphenated_owner(self):
        assert parse_instance_id("pandas-dev__pandas-1234") == ("pandas-dev/pandas", "1234")

    def test_hyphenated_repo(self):
        assert parse_instance_id("PlasmaFAIR__sdf-xarray-24") == ("PlasmaFAIR/sdf-xarray", "24")

    def test_commit_suffix(self):
        repo, suffix = parse_instance_id("pandas-dev__pandas-dbf8aaf4a3f3b41e5c")
        assert repo == "pandas-dev/pandas"
        assert suffix == "dbf8aaf4a3f3b41e5c"

    def test_unparseable(self):
        assert parse_instance_id("not-an-instance") == (None, None)


class TestLooksLikeError:
    def test_traceback_anywhere(self):
        content = "x" * 5000 + "\nTraceback (most recent call last):\n  File ..."
        assert looks_like_error(content)

    def test_leading_marker(self):
        assert looks_like_error("ERROR: The path /tmp/x does not exist.")

    def test_ignores_error_word_deep_in_file_contents(self):
        content = "here is the file:\n" + "clean line\n" * 200 + "raise ImportError('nope')"
        assert not looks_like_error(content)

    def test_nonzero_exit_code(self):
        assert looks_like_error("command failed with exit code 1")

    def test_zero_exit_code_is_not_an_error(self):
        assert not looks_like_error("finished with exit code 0")

    def test_empty(self):
        assert not looks_like_error("")


class TestErrorSignature:
    def test_collapses_varying_line_numbers(self):
        first = error_signature("ERROR: Invalid view_range: [1, 40] exceeds file of 12 lines")
        second = error_signature("ERROR: Invalid view_range: [3, 90] exceeds file of 55 lines")
        assert first == second

    def test_collapses_varying_paths(self):
        first = error_signature("ERROR: The path /tmp/abc/def.py does not exist.")
        second = error_signature("ERROR: The path /home/user/xyz/other.py does not exist.")
        assert first == second

    def test_extracts_exception_from_traceback(self):
        content = (
            "Traceback (most recent call last):\n"
            '  File "/repo/mod.py", line 42, in run\n'
            "    value = int(raw)\n"
            "ValueError: invalid literal for int() with base 10: 'abc'"
        )
        assert error_signature(content).startswith("ValueError:")

    def test_distinct_errors_stay_distinct(self):
        first = error_signature("ERROR: The path /a/b.py does not exist.")
        second = error_signature("ERROR: Permission denied while writing output")
        assert first != second

    def test_returns_none_for_clean_output(self):
        assert error_signature("ran 12 tests, all passed") is None
