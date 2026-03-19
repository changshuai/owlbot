---
name: test_runner
description: Simple skill to run a demo Python script via tools.
tags: [demo,script,testing]
---

This is a **demo skill** that shows how to:

- Read a local Python script using the `fileOps` tool.
- Execute the script using the `bash` tool.

The script lives at:

- `skills/general/test-runner/scripts/hello.py`

When you want to run it:

1. Optionally inspect the script:
   - Call `fileOps` with `{"action": "read", "file_path": "skills/general/test-runner/scripts/hello.py"}`.
2. Run the script:
   - Call `bash` with `{"command": "python skills/general/test-runner/scripts/hello.py"}`.

You can adapt this pattern for more complex scripts: keep the logic in `.py` files and use tools to read/execute them.

