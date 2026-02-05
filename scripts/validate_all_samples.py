#!/usr/bin/env python3
"""
Validate all .qsamples files in a directory using the validate_samples tool.

Usage:
    python validate_all_samples.py <directory> [--verbose]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_qsamples_files(directory: Path) -> list[Path]:
    """Find all .qsamples files in the directory."""
    return sorted(directory.glob("*.qsamples"))


def validate_file(validate_binary: Path, file_path: Path, verbose: bool) -> tuple[bool, str]:
    """
    Run validate_samples on a single file.

    Returns:
        (has_violations, output) tuple
    """
    cmd = [str(validate_binary), str(file_path)]
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Non-zero exit code means violations found
        has_violations = result.returncode != 0
        output = result.stdout + result.stderr

        return has_violations, output

    except subprocess.TimeoutExpired:
        return True, f"TIMEOUT after 60s"
    except Exception as e:
        return True, f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Validate all .qsamples files in a directory"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing .qsamples files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output from validate_samples"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).parent.parent / "build" / "validate_samples",
        help="Path to validate_samples binary (default: build/validate_samples)"
    )

    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory", file=sys.stderr)
        return 1

    if not args.binary.exists():
        print(f"Error: validate_samples binary not found at {args.binary}", file=sys.stderr)
        print("Build it first with: cmake --build build --target validate_samples", file=sys.stderr)
        return 1

    files = find_qsamples_files(args.directory)

    if not files:
        print(f"No .qsamples files found in {args.directory}")
        return 0

    print(f"Found {len(files)} .qsamples files in {args.directory}")
    print()

    files_with_violations = []
    total_violations = 0

    for i, file_path in enumerate(files, 1):
        if args.verbose:
            print(f"\n[{i}/{len(files)}] Checking {file_path.name}...")
        else:
            print(f"[{i}/{len(files)}] {file_path.name}...", end=" ", flush=True)

        has_violations, output = validate_file(args.binary, file_path, args.verbose)

        if has_violations:
            files_with_violations.append(file_path.name)
            if args.verbose:
                print(output)
            else:
                print("VIOLATIONS FOUND")
                if output.strip():
                    print(f"  {output.strip()}")
        else:
            if not args.verbose:
                print("OK")

    print()
    print("=" * 80)
    print(f"SUMMARY: Checked {len(files)} files")

    if files_with_violations:
        print(f"Files with violations: {len(files_with_violations)}")
        for filename in files_with_violations:
            print(f"  - {filename}")
        return 1
    else:
        print("All files passed pathfinding validation!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
