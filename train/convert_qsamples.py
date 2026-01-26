#!/usr/bin/env python3
"""
Convert .qsamples files from v1 format to v2 format.

V1 format (864 bytes per sample):
    CompactState (24 bytes)
    float policy[209] (836 bytes)
    float value (4 bytes)

V2 format (876 bytes per sample):
    CompactState (24 bytes)
    float policy[209] (836 bytes)
    uint32_t wins (4 bytes)
    uint32_t losses (4 bytes)  
    float value (4 bytes)
    uint32_t reserved (4 bytes)

Conversion logic:
    - value > 0 -> wins=1, losses=0
    - value < 0 -> wins=0, losses=1
    - value == 0 -> wins=0, losses=0 (draw, sample will be skipped during training)
"""

import argparse
import struct
import numpy as np
from pathlib import Path
import time

QSMP_MAGIC = 0x51534D50
QSMP_HEADER_SIZE = 64

V1_SAMPLE_SIZE = 876
V2_SAMPLE_SIZE = 872  # 24 + 836 + 4 + 4 + 4 + 4 = 876

V1_DTYPE = np.dtype([
    ('p1_row', 'u1'), ('p1_col', 'u1'), ('p2_row', 'u1'), ('p2_col', 'u1'),
    ('p1_fences', 'u1'), ('p2_fences', 'u1'),
    ('flags', 'u1'), ('reserved_byte', 'u1'),
    ('fences_h', '<u8'), ('fences_v', '<u8'),
    ('policy', '<f4', (209,)),
    ('wins', '<u4'),
    ('losses', '<u4'),
    ('value', '<f4'),
    ('reserved', '<u4')
])

V2_DTYPE = np.dtype([
    ('p1_row', 'u1'), ('p1_col', 'u1'), ('p2_row', 'u1'), ('p2_col', 'u1'),
    ('p1_fences', 'u1'), ('p2_fences', 'u1'),
    ('flags', 'u1'), ('reserved_byte', 'u1'),
    ('fences_h', '<u8'), ('fences_v', '<u8'),
    ('policy', '<f4', (209,)),
    ('wins', '<u4'),
    ('losses', '<u4'),
    ('value', '<f4')
])


def convert_file(input_path: str, output_path: str = None, in_place: bool = False) -> bool:
    """
    Convert a single .qsamples file from v1 to v2 format.

    Args:
        input_path: Path to input v1 file
        output_path: Path to output v2 file (None = auto-generate)
        in_place: If True, replace input file with converted output

    Returns:
        True if conversion succeeded
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False

    with open(input_path, 'rb') as f:
        header_data = f.read(QSMP_HEADER_SIZE)
        if len(header_data) < QSMP_HEADER_SIZE:
            print(f"Error: Invalid file (header too short): {input_path}")
            return False

        magic, version, flags, sample_count = struct.unpack('<IHHI', header_data[:12])
        timestamp = struct.unpack('<Q', header_data[16:24])[0]

        if magic != QSMP_MAGIC:
            print(f"Error: Invalid magic number in {input_path}")
            return False

        # if version >= 2:
        #     print(f"Skipping {input_path}: already v{version}")
        #     return True

        print(f"Converting {input_path}: {sample_count} samples, v{version} -> v2")

        v1_data = np.fromfile(f, dtype=V1_DTYPE, count=sample_count)

    if len(v1_data) != sample_count:
        print(f"Warning: Expected {sample_count} samples, got {len(v1_data)}")
        sample_count = len(v1_data)

    v2_data = np.zeros(sample_count, dtype=V2_DTYPE)

    v2_data['p1_row'] = v1_data['p1_row']
    v2_data['p1_col'] = v1_data['p1_col']
    v2_data['p2_row'] = v1_data['p2_row']
    v2_data['p2_col'] = v1_data['p2_col']
    v2_data['p1_fences'] = v1_data['p1_fences']
    v2_data['p2_fences'] = v1_data['p2_fences']
    v2_data['flags'] = v1_data['flags']
    v2_data['reserved_byte'] = v1_data['reserved']
    v2_data['fences_h'] = v1_data['fences_h']
    v2_data['fences_v'] = v1_data['fences_v']
    v2_data['policy'] = v1_data['policy']
    v2_data['value'] = v1_data['value']

    values = v1_data['value']
    v2_data['wins'] = (values > 0).astype(np.uint32)
    v2_data['losses'] = (values < 0).astype(np.uint32)

    wins_count = np.sum(v2_data['wins'])
    losses_count = np.sum(v2_data['losses'])
    draws_count = sample_count - wins_count - losses_count
    print(f"  Outcomes: {wins_count} wins, {losses_count} losses, {draws_count} draws")

    if output_path is None:
        if in_place:
            output_path = input_path
        else:
            output_path = input_path.with_suffix('.v2.qsamples')
    else:
        output_path = Path(output_path)

    new_header = bytearray(QSMP_HEADER_SIZE)
    struct.pack_into('<IHHI', new_header, 0, QSMP_MAGIC, 2, flags, sample_count)
    struct.pack_into('<Q', new_header, 16, timestamp)

    temp_path = output_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        f.write(new_header)
        v2_data.tofile(f)

    temp_path.replace(output_path)

    print(f"  Saved to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert .qsamples files from v1 to v2 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file (creates file.v2.qsamples)
  python convert_qsamples.py file.qsamples

  # Convert in-place (overwrites original)
  python convert_qsamples.py --in-place file.qsamples

  # Convert all v1 files in directory
  python convert_qsamples.py --in-place samples/*.qsamples

  # Convert to specific output
  python convert_qsamples.py -o converted.qsamples input.qsamples
"""
    )

    parser.add_argument('files', nargs='+', help='Input .qsamples file(s)')
    parser.add_argument('-o', '--output', help='Output file (only valid with single input)')
    parser.add_argument('--in-place', action='store_true', dest='in_place',
                        help='Convert files in-place (overwrite originals)')

    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        print("Error: --output can only be used with a single input file")
        return 1

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for input_file in args.files:
        output_file = args.output if len(args.files) == 1 else None
        if convert_file(input_file, output_file, args.in_place):
            success_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - start_time
    print(f"\nDone: {success_count} succeeded, {fail_count} failed in {elapsed:.1f}s")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    exit(main())
