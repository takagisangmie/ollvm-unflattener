import unflattener
import argparse
import logging as logger
import binrewrite
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
    
    patch_data_list = []
    if not args.all:
        logger.info("Unflattening function {}".format(hex(target_address)))
        try:
            patch, func_interval = unflat_engine.unflat(target_address)
            if patch is not None:
                logger.info("Generate patch for {} successfully".format(hex(target_address)))
                patch_data_list.append((patch, func_interval))
            else:
                logger.info("Function {} is not flattened".format(hex(target_address)))
        except Exception as e:
            logger.info("Fail to unflat function {}".format(hex(target_address)))
    else:
        patch_data_list += unflat_engine.unflat_follow_calls(target_address, args.output)
    
    if len(patch_data_list) != 0:
        unflat_engine.apply_patches(patch_data_list, args.output)
        logger.info("Patch successfully. Deobfuscated binary is written to {}".format(args.output))
    
if __name__ == '__main__':
    main()