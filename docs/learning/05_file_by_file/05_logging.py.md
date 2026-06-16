# logging.py — Complete Line-by-Line Explanation

**File**: `src/baligh/utils/logging.py` (93 lines)
**Purpose**: Configures loguru-based logging with console, file, and JSON output.

---

## Imports (Lines 1-7)

Line 1: Module docstring.

Line 3: `sys` for stdout.

Line 4: `Path` for log file paths.

Line 5: `loguru.logger` — the main logger object.

Line 6: Import `get_config` for default settings.

---

## setup_logging (Lines 9-53)

```python
def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    log_file: Path | None = None,
    json_logs: bool = False,
) -> None:
```

Lines 9-14: Configure loguru logger.

Lines 15-17: Docstring.

Line 18: Remove default loguru handler (so we can add our own).

Lines 20-28: Console handler:
- If json format: Use serialized JSON output
- If text format: Use colored text with timestamp, level, module, function, line

Lines 30-33: Text format details:
- Green timestamp
- Level name
- Cyan module:function:line
- Message

Lines 42-53: File handler (if log_file provided):
- Create parent directory if needed
- Rotation: 100 MB per file
- Retention: Keep 30 days
- Compression: gzip old files
- Serialization: JSON if json_logs=True

Lines 56-57: Intercept stdlib logging and redirect to loguru.

---

## InterceptHandler (Lines 60-78)

Lines 60-61: Redirect Python stdlib logging to loguru.

Lines 63-66: Get loguru level from stdlib level name.

Lines 68-72: Find the original caller (skip logging library frames).

Lines 74-77: Log with the original depth and exception info.

---

## get_logger (Lines 81-83)

```python
def get_logger(name: str):
    return logger.bind(name=name)
```

Lines 81-83: Create a logger bound to a module name. All log messages will show the module name.

---

## Auto-initialization (Lines 86-93)

Lines 86-87: Read config on import.

Lines 88-93: Call setup_logging with project defaults:
- log_level from config
- log_format from config
- log_file: training/logs/baligh.log
- json_logs: True if format is "json"
