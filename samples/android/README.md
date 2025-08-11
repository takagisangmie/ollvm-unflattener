# Android ARM64 Samples

This directory contains sample Android ARM64 binaries obfuscated with OLLVM control flow flattening.

## Files Description

- `libcff_arm64.so` - Sample ARM64 shared library with control flow flattening
- `deob_libcff_arm64.so` - Deobfuscated version of the library
- `README.md` - This file

## Usage Examples

```bash
# Deobfuscate a single function in ARM64 library
python -m unflattener -i ./samples/android/libcff_arm64.so -o ./samples/android/deob_libcff_arm64.so -t 0x12345678

# Deobfuscate with call following
python -m unflattener -i ./samples/android/libcff_arm64.so -o ./samples/android/deob_libcff_arm64.so -t 0x12345678 -a
```

## Notes

- Make sure the target address is the actual function address in the ARM64 binary
- The tool automatically detects ARM64 architecture and uses appropriate instruction handling
- ARM64 specific instructions like BL, BLR, CSEL are properly supported
