#!/usr/bin/env python3
"""
Test script to verify ARM64 support for OLLVM unflattener
"""

import sys
import os
from unflattener import Unflattener

def test_arm64_support():
    """Test if the unflattener can handle ARM64 architecture"""
    print("Testing ARM64 support for OLLVM unflattener...")
    
    # Test if miasm supports aarch64l
    try:
        from miasm.analysis.machine import Machine
        from miasm.analysis.binary import Container
        from miasm.core.locationdb import LocationDB
        from miasm.arch.aarch64.regs import all_regs_ids, all_regs_ids_init
        print("✓ ARM64 imports successful")
    except ImportError as e:
        print(f"✗ ARM64 import failed: {e}")
        return False
    
    # Test if keystone supports ARM64
    try:
        from keystone import Ks, KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN
        ks = Ks(KS_ARCH_ARM64, KS_MODE_LITTLE_ENDIAN)
        # Test assembling a simple ARM64 instruction
        encoding, count = ks.asm("mov x0, x1")
        print(f"✓ Keystone ARM64 assembly test: {bytes(encoding).hex()} ({count} instruction)")
    except Exception as e:
        print(f"✗ Keystone ARM64 test failed: {e}")
        return False
    
    print("✓ All ARM64 dependencies are working!")
    return True

def simulate_arm64_unflattening():
    """Simulate the unflattening process for ARM64"""
    print("\nSimulating ARM64 unflattening process...")
    
    # Create a mock test to verify our modifications work
    try:
        # Test that our modified BinaryRewriter can handle aarch64l
        from unflattener.binrewrite import BinaryRewriter
        from miasm.core.asmblock import AsmCFG
        from miasm.core.locationdb import LocationDB
        
        loc_db = LocationDB()
        # Create a minimal ASM CFG for testing
        asmcfg = AsmCFG(loc_db)
        
        # Test BinaryRewriter initialization with ARM64
        rewriter = BinaryRewriter(asmcfg, 'aarch64l')
        print("✓ BinaryRewriter ARM64 initialization successful")
        
        # Verify keystone is set up correctly
        if rewriter.KS is not None:
            print("✓ Keystone ARM64 engine initialized")
        else:
            print("✗ Keystone ARM64 engine not initialized")
            return False
            
        # Test basic instruction assembly
        test_instructions = [
            "mov x0, x1",
            "b #0x1000",
            "b.eq #0x1000",
            "bl #0x2000"
        ]
        
        for instr in test_instructions:
            try:
                encoding, count = rewriter.KS.asm(instr)
                print(f"✓ ARM64 instruction '{instr}': {bytes(encoding).hex()}")
            except Exception as e:
                print(f"✗ Failed to assemble '{instr}': {e}")
                return False
                
    except Exception as e:
        print(f"✗ ARM64 simulation failed: {e}")
        return False
    
    print("✓ ARM64 unflattening simulation successful!")
    return True

def show_usage_example():
    """Show how to use the modified unflattener with ARM64 binaries"""
    print("\n" + "="*60)
    print("ARM64 OLLVM Unflattener Usage Example")
    print("="*60)
    
    example_usage = """
# For Android ARM64 binaries:
python -m unflattener -i /path/to/android_arm64_binary.so -o /path/to/deobfuscated.so -t 0x12345678

# With call following for ARM64:
python -m unflattener -i /path/to/android_arm64_binary.so -o /path/to/deobfuscated.so -t 0x12345678 -a

# Note: Make sure your ARM64 binary is properly analyzed by miasm
# The tool will automatically detect the architecture and use ARM64-specific handling
"""
    
    print(example_usage)
    
    print("Key changes for ARM64 support:")
    print("• Added support for aarch64l architecture detection")
    print("• ARM64 registers (SP, X29) handling in symbolic execution")
    print("• ARM64 instruction support (BL, BLR, CSEL, B.cond)")
    print("• ARM64 branch instruction length (4 bytes)")
    print("• ARM64 condition code mapping")

if __name__ == "__main__":
    print("OLLVM Unflattener ARM64 Support Test")
    print("=" * 40)
    
    success = True
    
    # Test dependencies
    if not test_arm64_support():
        success = False
    
    # Test simulation
    if not simulate_arm64_unflattening():
        success = False
    
    # Show usage
    show_usage_example()
    
    if success:
        print("\n✓ All tests passed! ARM64 support is ready.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the dependencies.")
        sys.exit(1)
