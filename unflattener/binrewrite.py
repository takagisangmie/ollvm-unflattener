from collections import defaultdict
from miasm.core.asmblock import AsmCFG, AsmBlock
from miasm.ir.symbexec import SymbolicExecutionEngine
from keystone import Ks, KS_ARCH_X86, KS_MODE_32, KS_MODE_64

class RewriteInstruction:
    def __init__(self, instruction: str, old_offset: int, new_offset: int):
        """Construction for a rewrite instruction

        Args:
            instruction (str): string representation of the instruction
            old_offset (int): original offset
            new_offset (int): relocated offset
        """
        self.instruction: str = instruction
        self.old_offset: int = old_offset
        self.new_offset: int = new_offset
        # assemble the new instruction bytes
        instruction_encoding = BinaryRewriter.KS.asm(instruction, new_offset)[0]
        self.asm: bytes = bytes(instruction_encoding)
        
    def __str__(self) -> str:
        """Get the string representation of the rewrite instruction

        Returns:
            str: String representation of the rewrite instruction
        """
        result = hex(self.old_offset) + ': ' + self.instruction
        result += '\n\tNew offset: ' + hex(self.new_offset) 
        return result

 
class BinaryRewriter:
    JMP_INSTRUCTION_DEFAULT_LEN = 6
    KS: Ks = None
    def __init__(self, asmcfg: AsmCFG, arch: str):
        """BinaryRewriter constructor

        Args:
            asmcfg (AsmCFG): function's AsmCFG
            arch (str): binary architecture

        Raises:
            Exception: None supported architecture
        """
        
        self.asmcfg: AsmCFG = asmcfg
        # reloc offset -> RewriteInstruction 
        self.rewrite_instructions: defaultdict[int, RewriteInstruction] = defaultdict(RewriteInstruction)
        
        # original offset -> reloc offset
        self.reloc_map: defaultdict[int, int] = defaultdict(int)
        
        if arch == 'x86_32':
            BinaryRewriter.KS = Ks(KS_ARCH_X86, KS_MODE_32)
        elif arch == 'x86_64':
            BinaryRewriter.KS = Ks(KS_ARCH_X86, KS_MODE_64)
        else:
            raise Exception('Not supported architecture')
        
    def init_CFF_data(self, state_order_map: defaultdict, state_to_lockey_map: defaultdict, symbex_engine: SymbolicExecutionEngine):
        """Initialize CFF information needed to generate patch

        Args:
            state_order_map (defaultdict): State order map
            state_to_lockey_map (defaultdict): State lockey map
            symbex_engine (SymbolicExecutionEngine): miasm symbolic execution engine
        """
        self.state_order_map = state_order_map
        self.state_to_lockey_map = state_to_lockey_map
        self.symbex_engine = symbex_engine
    
    def reorder_blocks(self, target_address: int):
        """Reallocate the blocks in the fixed AsmCFG

        Args:
            target_address (int): Function address
        """
        curr_reloc_address = target_address
        
        # 1. start from the head, write all head blocks
        # queue: list of tuple (current state value, loc_key for current tail block)
        process_queue = [0]
        processed_state_val_list = []
        
        while len(process_queue) != 0:
            curr_state_val = process_queue.pop()
            while True:
                processed_state_val_list.append(curr_state_val)
                next_state_vals = self.state_order_map[curr_state_val]
                
                cond_jump_type = None
                next_state_val = 0
                # process current state val
                for index, state_block_loc in enumerate(self.state_to_lockey_map[curr_state_val]):
                    state_block: AsmBlock = self.asmcfg.loc_key_to_block(state_block_loc)
                    
                    if state_block.lines[-1].name == 'JMP':
                        # delete the jump to dispatcher
                        del state_block.lines[-1]
                    
                    if len(next_state_vals) == 2:
                        # conditional block, delete CMOV instruction
                        for index, instruction in enumerate(state_block.lines):
                            if 'CMOV' in instruction.name:
                                cond_jump_type = instruction.name.replace('CMOV', 'J')
                                del state_block.lines[index]
                                break
                      
                    for instruction in state_block.lines:
                        instruction_str = str(instruction)
                        if instruction.name == 'CALL':
                            # has to resolve the call destination from loc key
                            call_dst = int(self.symbex_engine.eval_expr(instruction.args[0]))
                            instruction_str = 'CALL {}'.format(hex(call_dst))
                        
                        # relocate the instruction
                        self.rewrite_instructions[curr_reloc_address] = RewriteInstruction(instruction_str, instruction.offset, curr_reloc_address)
                            
                        self.reloc_map[instruction.offset] = curr_reloc_address
                        if instruction_str[0] == 'J':
                            # force JMP/JCC instruction to always have length 6
                            curr_reloc_address += BinaryRewriter.JMP_INSTRUCTION_DEFAULT_LEN
                        else:
                            curr_reloc_address += len(self.rewrite_instructions[curr_reloc_address].asm)
                
                if len(next_state_vals) == 0:
                    # ret block
                    break
                
                if len(next_state_vals) == 1:
                    # only one next state (non conditional)
                    next_state_val = next_state_vals[0]
                    
                    assert next_state_val in self.state_to_lockey_map
                    next_state_head_loc = self.state_to_lockey_map[next_state_val][0]
                    jump_dst = self.asmcfg.loc_db.get_location_offset(next_state_head_loc)
                    
                    if jump_dst in self.rewrite_instructions:
                        # already written before, jump backward
                        self.rewrite_instructions[curr_reloc_address] = RewriteInstruction('JMP {}'.format(hex(jump_dst)), -1, curr_reloc_address)
                        curr_reloc_address += BinaryRewriter.JMP_INSTRUCTION_DEFAULT_LEN
                        
                    # else just write the next block directly after
                elif len(next_state_vals) == 2:
                    # processing conditional
                    true_state_val = next_state_vals[0]
                    false_state_val = next_state_vals[1]
                    assert true_state_val in self.state_to_lockey_map
                    assert false_state_val in self.state_to_lockey_map
                    
                    # processing false path
                    false_head_loc = self.state_to_lockey_map[false_state_val][0]
                    false_dst = self.asmcfg.loc_db.get_location_offset(false_head_loc)
                    # old offset is -1 here cause we create this instruction out of thin air
                    # it does not exist in the original instruction
                    self.rewrite_instructions[curr_reloc_address] = RewriteInstruction('{} {}'.format(cond_jump_type, hex(false_dst)), -1, curr_reloc_address)
                    curr_reloc_address += BinaryRewriter.JMP_INSTRUCTION_DEFAULT_LEN
                    
                    # stash the false state for later traversal
                    process_queue.append(false_state_val)
                    
                    # processing true path
                    true_head_loc = self.state_to_lockey_map[true_state_val][0]
                    true_dst = self.asmcfg.loc_db.get_location_offset(true_head_loc)
                    
                    if true_dst in self.rewrite_instructions:
                        # true destination is already written, JMP backward
                        self.rewrite_instructions[curr_reloc_address] = RewriteInstruction('JMP {}'.format(hex(true_dst)), -1, curr_reloc_address)
                        curr_reloc_address += BinaryRewriter.JMP_INSTRUCTION_DEFAULT_LEN

                    next_state_val = true_state_val
                
                # if we already processed this state, skip writing the next block
                if next_state_val in processed_state_val_list:
                    # before we terminate this path
                    # gotta add a JMP to the head block of that state
                    next_state_head_loc = self.state_to_lockey_map[next_state_val][0]
                    jump_dst = self.asmcfg.loc_db.get_location_offset(next_state_head_loc)
                    self.rewrite_instructions[curr_reloc_address] = RewriteInstruction('JMP {}'.format(hex(jump_dst)), -1, curr_reloc_address)
                    curr_reloc_address += BinaryRewriter.JMP_INSTRUCTION_DEFAULT_LEN
                    break
                curr_state_val = next_state_val
                
    def generate_patch(self) -> bytes:
        """Generate the patch for the current function

        Returns:
            bytes: Function patch data
        """
        patch = b''
        # generate a patch for all of the rewrite instructions
        for reloc_addr in sorted(self.rewrite_instructions.keys()):
            reloc_instruction = self.rewrite_instructions[reloc_addr]
            
            instruction_str = reloc_instruction.instruction
            
            # get instruction patch
            instruction_patch = None
            if reloc_instruction.instruction[0] == 'J':
                # need to recalculate destination
                # because the destination of the JMP/JCC instruction has been relocated
                jump_type, destination = instruction_str.split(' ')
                destination = int(destination, 16)
                instruction_str = '{} {}'.format(jump_type, hex(self.reloc_map[destination]))
            
                instruction_encoding = BinaryRewriter.KS.asm(instruction_str, reloc_addr)[0]
                
                instruction_patch = bytes(instruction_encoding)
                
                # Adding NOPs in case the length is not 6
                # I do not wanna deal separating JMP near and JMP short here
                instruction_patch += b'\x90' * (self.JMP_INSTRUCTION_DEFAULT_LEN - len(instruction_encoding))
            else:
                instruction_patch = reloc_instruction.asm
                
            patch += instruction_patch
        return patch