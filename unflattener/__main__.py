import unflattener
import argparse
import logging as logger

def main() -> None:
    parser = argparse.ArgumentParser(prog='unflattener', description='Python program to unflatten binaries obfuscated by ollvm')
    parser.add_argument('-i', '--input', type=str, help='Obfuscated binary path', required=True)
    parser.add_argument('-o', '--output', type=str, help='Deobfuscated output binary path', required=True)
    parser.add_argument('-t', '--target', type=str, help='Target address (hex) to deobfuscate', required=True)
    parser.add_argument('-a', '--all', action='store_true', help='Iteratively deobfuscate all functions called by the target function')
    
    args = parser.parse_args()

    logger.basicConfig(level=logger.INFO)
    
    unflat_engine = unflattener.Unflattener(args.input)
    try:
        target_address = int(args.target, 16)
    except:
        logger.info('Target address must be a valid hex value')
        
    if not args.all:
        logger.info("Unflattening function {}".format(hex(target_address)))
        try:
            if unflat_engine.unflat(target_address):
                logger.info("Generate patch for " + hex(target_address))
                patches, dinterval = unflat_engine.generate_patch(target_address)
                unflat_engine.apply_patches([(patches, dinterval)], args.output)
                logger.info("Patch successfully. Deobfuscated binary is written to {}".format(args.output))
            else:
                logger.info("Function {} is not flattened".format(hex(target_address)))
        except Exception as e:
            logger.info("Fail to unflat function {}".format(hex(target_address)))
            logger.info('Error {}'.format(str(e)))
    else:
        unflat_engine.unflat_follow_calls(target_address, args.output)
        
if __name__ == '__main__':
    main()