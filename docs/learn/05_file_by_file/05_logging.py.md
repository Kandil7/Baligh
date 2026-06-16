# logging.py - Logging Setup

**Path**: `src/baligh/utils/logging.py` (93 lines)

## Purpose

Configures loguru-based logging with console, file, and JSON output.

---

## setup_logging (lines 9-53)

Configures three log handlers:
1. **Console**: Colored text format with timestamp, level, module, function, line
2. **Console JSON**: Serialized JSON format for structured logging
3. **File**: With rotation (100MB), retention (30 days), compression (gz)

Also intercepts stdlib logging via InterceptHandler.

---

## InterceptHandler (lines 60-78)

Redirects Python stdlib logging calls to loguru, preserving the original caller information.

---

## get_logger (lines 81-83)

Returns a loguru logger instance bound to a module name.

---

## Auto-initialization (lines 86-93)

On import, reads config and calls setup_logging with project defaults.
