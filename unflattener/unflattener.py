from collections import defaultdict
import pprint
from typing import List
from miasm.core.locationdb import LocationDB
from miasm.analysis.binary import Container
import logging as logger
from miasm.analysis.machine import Machine
from miasm.core.asmblock import AsmCFG, AsmBlock
from miasm.ir.ir import IRCFG
from miasm.arch.ppc.regs import *
from miasm.analysis.simplifier import *
from miasm.expression.expression import *
from miasm.ir.symbexec import SymbolicExecutionEngine
from miasm.core.asmblock import asm_resolve_final, AsmConstraintTo, AsmConstraintNext, AsmConstraint
import graphviz
from miasm.arch.x86.arch import instruction_x86, mn_x86
from miasm.arch.x86.disasm import dis_x86_32
from miasm.core.interval import interval
from miasm.loader.elf_init import ELF
from miasm.loader.pe_init import PE


def calc_flattening_score(asm_graph: AsmCFG) -> float:
    """Function to calculate flatenning score

    https://gist.github.com/mrphrazer/da32217f231e1dd842986f94aa6d9d37#file-flattening_heuristic-py

    Args:
        asm_graph (AsmCFG): Function's asm CFG

    Returns:
        float: Function's flattening score
    """
    
    # init score
    score = 0.0
    # walk over all entry nodes in the graph
    for head in asm_graph.heads_iter():
        # since miasm breaks basic block into multiple ones separated by CALL instruction
        # need to move this head to the final successor whose last instruction is not a CALL instruction
        # basically the tail of this head block
        skipped_head_loc_count = 0
        while asm_graph.loc_key_to_block(head).lines[-1].name == 'CALL':
            skipped_head_loc_count += 1
            head = asm_graph.successors(head)[0]
        
        # compute dominator tree
        dominator_tree = asm_graph.compute_dominator_tree(head)
        # walk over all basic blocks
        for block in asm_graph.blocks:
            # get location key for basic block via basic block address
            block_key = asm_graph.loc_db.get_offset_location(block.lines[0].offset)
            # get all blocks that are dominated by the current block
            dominated = set(
                [block_key] + [b for b in dominator_tree.walk_depth_first_forward(block_key)])
            # check for a back edge
            if not any([b in dominated for b in asm_graph.predecessors(block_key)]):
                continue
            # calculate relation of dominated blocks to the blocks in the graph
            score = max(score, len(dominated)/(len(asm_graph.nodes()) - skipped_head_loc_count))
    return score

class Unflattener:
    """
    Class for the unflattener engine
    """
    
    def __init__(self, filename: str):
        """Constructor for the unflattener engine

        Args:
            filename (str): deobfuscated binary path
        """
        
        self.loc_db: LocationDB = LocationDB()
        self.container: Container = Container.from_stream(open(filename, 'rb'), self.loc_db)
        self.machine: Machine = Machine(self.container.arch)
        self.mdis = self.machine.dis_engine(self.container.bin_stream, loc_db=self.loc_db)
        self.original_filename: str = filename
        self.flatten_func_queue: list = []
    
    def unflat(self, target_address: int) -> bool:
        """Unflatten the CFG of a function

        Args:
            target_address (int): Target function address
        Returns:
            bool: Original CFG is recovered
        """
        
        # get text section range & binary base virtual address
        if isinstance(self.container.executable, PE):
            text_section_Shdr = self.container.executable.getsectionbyvad(target_address)
            self.binary_base_va = self.container.executable.NThdr.ImageBase + (text_section_Shdr.addr - text_section_Shdr.offset)
            self.text_section_range = {'lower': self.container.executable.NThdr.ImageBase + text_section_Shdr.addr,
                                       'upper': self.container.executable.NThdr.ImageBase + text_section_Shdr.addr + text_section_Shdr.size}
        elif isinstance(self.container.executable, ELF):
            text_section_Shdr = self.container.executable.getsectionbyvad(target_address).sh
            self.binary_base_va = text_section_Shdr.addr - text_section_Shdr.offset
            
            self.text_section_range = {'lower': text_section_Shdr.addr,
                                       'upper': text_section_Shdr.addr + text_section_Shdr.size}
        else:
            raise Exception('Unsupported binary type')

        self.asmcfg: AsmCFG = self.mdis.dis_multiblock(target_address)
        self.lifter = self.machine.lifter_model_call(self.mdis.loc_db)
        self.ircfg: IRCFG = self.lifter.new_ircfg_from_asmcfg(self.asmcfg)
        score = calc_flattening_score(self.asmcfg)
        
        if score < 0.9:
            return False
        self.recover_CFG(target_address)
        return True
    
    def unflat_follow_calls(self, target_address: int, out_filename: str):
        """Unflat the target function & all calls to unflat other obfuscated functions 

        Args:
            target_address (int): Target function address
            out_filename (str): Deobfuscated output path
        """
        self.flatten_func_queue = [target_address]
        processed_flatten_func_list = []
        
        patches_list = []
        while len(self.flatten_func_queue) != 0:
            flatten_func_addr = self.flatten_func_queue.pop()
            
            if flatten_func_addr in processed_flatten_func_list:
                # do not try to flatten the same function again
                continue
            
            logger.info("Unflattening function {}".format(hex(flatten_func_addr)))
            try:
                if self.unflat(flatten_func_addr):
                    logger.info("Generate patch for " + hex(flatten_func_addr))
                    func_patch, func_interval = self.generate_patch(flatten_func_addr)
                    patches_list.append((func_patch, func_interval))
                    processed_flatten_func_list.append(flatten_func_addr)
                else:
                    logger.info("Function {} is not flattened".format(hex(flatten_func_addr)))
            except:
                logger.info("Fail to unflat function {}".format(hex(flatten_func_addr)))
        self.apply_patches(patches_list, out_filename)
        logger.info("Patch successfully. Deobfuscated binary is written to {}".format(out_filename))
    
    def render(self, dot_filename: str, image_filename: str):
        """Render the function's CFG into a DOT and PNG file

        Args:
            dot_filename (str): DOT file path
            image_filename (str): PNG file path
        """
        with open(dot_filename, 'w') as f:
            f.write(self.asmcfg.dot())
        graphviz.render('dot', 'png', dot_filename, outfile=image_filename)
    
    def print_block(self, loc_key: LocKey):
        """Print a block at the specified location

        Args:
            loc_key (LocKey): Location key
        """
        print('{} {}'.format(str(loc_key), str(self.asmcfg.loc_key_to_block(loc_key))))

    def to_loc_key(self, expr) -> LocKey:
        """Convert an expression into a location key

        Args:
            expr : Target expression

        Returns:
            LocKey: Location key
        """
        if isinstance(expr, LocKey):
            return expr
        if isinstance(expr, ExprLoc):
            return expr.loc_key
        if isinstance(expr, ExprInt):
            return self.container.loc_db.get_offset_location(int(expr))
        if isinstance(expr, int):
            return self.container.loc_db.get_offset_location(expr)
        return None

    def find_backbone_blocks(self, predispatcher_loc: LocKey) -> List[LocKey]:
        """Find all backbone blocks (blocks with code from the original program)

        Args:
            predispatcher_loc (LocKey): predispatcher location key

        Returns:
            List[LocKey]: List of backbone location keys
        """
        backbone_blocks = []
        for block_loc in self.asmcfg.predecessors(predispatcher_loc):
            # each parent block of the predispatcher is a backbone block
            last_predecessor_loc = block_loc
            curr_predecessor_loc = self.asmcfg.predecessors(block_loc)[0]
            backbone_blocks.append(last_predecessor_loc)
            
            # traverse upward from each backbone block to find all backbone blocks above it
            # this is due to CALL instructions breaking up basic block into multiple ones
            while True:
                curr_predecessor_block = self.asmcfg.loc_key_to_block(curr_predecessor_loc)
                if curr_predecessor_block.lines[-1].name in ['JZ', 'JMP', 'JNZ']:
                    break
                backbone_blocks.append(curr_predecessor_loc)
                curr_predecessor_loc = self.asmcfg.predecessors(curr_predecessor_loc)[0]
            
        # add function's tail (block with no successor) to backbone blocks
        for block in self.asmcfg.blocks:
            if len(self.asmcfg.successors(block.loc_key)) == 0:
                last_tail_loc = block.loc_key
                backbone_blocks.append(last_tail_loc)
                
                # traverse upward from each backbone block to find all backbone blocks above it
                # this is due to CALL instructions breaking up basic block into multiple ones
                curr_predecessor_tail_loc = self.asmcfg.predecessors(last_tail_loc)[0]
                while True:
                    
                    curr_predecessor_tail_block = self.asmcfg.loc_key_to_block(curr_predecessor_tail_loc)
                    if curr_predecessor_tail_block.lines[-1].name in ['JZ', 'JMP', 'JNZ']:
                        break
                    backbone_blocks.append(curr_predecessor_tail_loc)
                    curr_predecessor_tail_loc = self.asmcfg.predecessors(curr_predecessor_tail_loc)[0]

        return backbone_blocks

    def add_jump_to_next_state(self, loc_key: LocKey, jump_type: str, jump_dest: int, del_last_insn: bool):
        curr_block = self.asmcfg.loc_key_to_block(loc_key)
        patched_jmp_insn = mn_x86.fromstring('{} {}'.format(jump_type, jump_dest), self.asmcfg.loc_db, 32)
        patched_jmp_insn.l = len(mn_x86.asm(patched_jmp_insn, self.asmcfg.loc_db)[0])
        if del_last_insn:
            patched_jmp_insn.offset = curr_block.lines[-1].offset
            curr_block.lines[-1] = patched_jmp_insn
        else:
            patched_jmp_insn.offset = curr_block.lines[-1].offset + curr_block.lines[-1].l
            curr_block.lines.append(patched_jmp_insn)

    def recover_CFG(self, target_address: int):
        """Recover the function's CFG 

        Args:
            target_address (int): Target function address
        """
        
        # predispatcher is the block with the most number of parents
        predispatcher = sorted(self.asmcfg.blocks, key=lambda key: len(self.asmcfg.predecessors(key.loc_key)), reverse=True)[0]
        predispatcher_loc = predispatcher.loc_key

        # dispatcher is the only child of the predispatcher 
        dispatcher_loc = self.asmcfg.successors(predispatcher_loc)[0]
    
        # backbone: everything that is needed in the final asmcfg (except the head)
        backbone_loc_list = self.find_backbone_blocks(predispatcher_loc)

        # state var is the seceond expr in the first instructions of the dispatcher
        dispatcher_block = self.asmcfg.loc_key_to_block(dispatcher_loc)
        state_var_expr = dispatcher_block.lines[0].get_args_expr()[1]
        logger.debug('State var: ' + str(state_var_expr))

        # symbols for symbex
        init_symbols =  {}
        for i, r in enumerate(all_regs_ids):
            init_symbols[r] = all_regs_ids_init[i]
        
        # parent loc -> [children loc]
        loc_successors_map = defaultdict(list)
        
        # exec_queue: queue containing (address/loc to exec, symbex engine symbols, current state value)
        exec_queue = []
        exec_queue.append((self.to_loc_key(target_address), init_symbols, None))
        
        # starting state val for traversal
        first_state_val = None
        
        # curr state -> [next state/states]
        state_order_map = defaultdict(list)
        
        # state value -> [loc key/loc keys]
        state_to_lockey_map = defaultdict(list)
        
        # list to track all backbone blocks encountered
        backbone_encountered_list = []

        while len(exec_queue) != 0:
            # pop a loc_key to start symbex
            curr_loc, symbols, curr_state_val = exec_queue.pop()
            symbex_engine = SymbolicExecutionEngine(self.lifter, symbols)
            
            while True:
                # if current loc is a backbone block 
                if curr_loc in backbone_loc_list:
                    if curr_loc in backbone_encountered_list:
                        # if we already process all backbones, stop symbex
                        break
                    backbone_encountered_list.append(curr_loc)
                    
                    # get the current value for the state variable
                    curr_state_val = int(symbex_engine.eval_expr(state_var_expr))
                    
                    # map state val -> [current loc]
                    if curr_loc not in state_to_lockey_map[curr_state_val]:
                        state_to_lockey_map[curr_state_val].append(curr_loc)
                    
                    # get first state val for later traversal
                    if first_state_val is None:
                        first_state_val = curr_state_val
                
                # predispatcher processing
                if curr_loc == predispatcher_loc:
                    # evaluate next state var 
                    next_state_val = int(symbex_engine.eval_expr(state_var_expr))
                    
                    # map curr state val -> next state val
                    if next_state_val not in state_order_map[curr_state_val]:
                        state_order_map[curr_state_val].append(next_state_val)

                    # reset curr state val
                    curr_state_val = None
                
                # for flatten while following calls
                # if this block ends with a CALL, extract the call destination and add to self.flatten_func_queue
                curr_block = self.asmcfg.loc_key_to_block(curr_loc)
                if curr_block is not None:
                    last_instruction = curr_block.lines[-1]
                    if last_instruction.name == 'CALL':
                        destination_loc = symbex_engine.eval_expr(last_instruction.args[0])
                        destination_loc = int(destination_loc)
                        # only follows calls that are in the .text section only (avoid library calls)
                        if self.text_section_range['lower'] <= destination_loc <= self.text_section_range['upper']:
                            self.flatten_func_queue.append(int(destination_loc))

                
                # symbex block at current loc_key
                symbex_expr_result = symbex_engine.run_block_at(self.ircfg, curr_loc)
                
                # if reach the end (ret), stop this path traversal
                if symbex_expr_result is None:
                    break

                if isinstance(symbex_expr_result, ExprCond):
                    # if we reach a conditional expression

                    # Evaluate the jump addresses if the branch is taken or not
                    cond_true  = {symbex_expr_result.cond: ExprInt(1, 32)}
                    cond_false  = {symbex_expr_result.cond: ExprInt(0, 32)}
                    addr_true = expr_simp(
                            symbex_engine.eval_expr(symbex_expr_result.replace_expr(cond_true), {}))
                    addr_false = expr_simp(
                            symbex_engine.eval_expr(symbex_expr_result.replace_expr(cond_false), {}))
                    
                    addr_true = self.to_loc_key(addr_true)
                    addr_false = self.to_loc_key(addr_false)
                    
                    # stash false path away
                    exec_queue.append((addr_false, symbex_engine.symbols.copy(), curr_state_val))
                    
                    # map curr loc -> [addr true]
                    loc_successors_map[curr_loc].append(addr_true)
                    
                    # next loc_key we're jumping to
                    next_loc = addr_true
                else:
                    # find next loc_key we're jumping to
                    next_loc = expr_simp(symbex_engine.eval_expr(symbex_expr_result))

                    # map exec states <cur loc> -> [next_loc]
                    next_loc = self.to_loc_key(next_loc)
                    if next_loc not in loc_successors_map[curr_loc]:
                        loc_successors_map[curr_loc].append(next_loc)

                # update current loc_key to the next loc_key
                curr_loc = next_loc

        # logger.info('loc_successors_map')
        # pprint.pprint(loc_successors_map)
        # logger.info('state order map')
        # pprint.pprint(state_order_map)
        # logger.info('state to loc_key_map')
        # pprint.pprint(state_to_lockey_map)
        
        # NOTE: not all backbone loc_key is relevant. Only take the ones from state_to_lockey_map
        backbone_loc_list = [loc for sublist in state_to_lockey_map.values() for loc in sublist]
        
        # add prologue blocks to backbone list
        prologue_tail_loc = None
        for block_loc in self.asmcfg.predecessors(dispatcher_loc):
            # head block is the other predecessor of dispatcher beside the predispatcher
            if block_loc == predispatcher_loc:
                continue
            
            # add head to backbone
            prologue_tail_loc = block_loc
            backbone_loc_list.append(prologue_tail_loc)
            
            # add all prologue blocks above the prologue tail
            curr_prologue_loc = prologue_tail_loc
            while len(self.asmcfg.predecessors(curr_prologue_loc)) != 0:
                prev_prologue_block = self.asmcfg.predecessors(curr_prologue_loc)[0]
                backbone_loc_list.append(prev_prologue_block)
                # re-add edge with cst_next
                self.asmcfg.del_edge(prev_prologue_block, curr_prologue_loc)
                self.asmcfg.add_edge(prev_prologue_block, curr_prologue_loc, AsmConstraintNext(curr_prologue_loc))
                curr_prologue_loc = prev_prologue_block
            break
        
        # irrelevant blocks are original blocks that are not a backbone block
        irrelevant_loc_list = [original_block.loc_key for original_block in self.asmcfg.blocks if original_block.loc_key not in backbone_loc_list]
        
        # delete all irrelevant blocks
        for loc_key in irrelevant_loc_list:
            self.asmcfg.del_block(self.asmcfg.loc_key_to_block(loc_key))

        for loc_list in state_to_lockey_map.values():
            for i in range(0, len(loc_list) - 1):
                self.asmcfg.del_edge(loc_list[i], loc_list[i + 1])
                # re-add edge with cst_next
                self.asmcfg.add_edge(loc_list[i], loc_list[i + 1], AsmConstraintNext(loc_list[i + 1]))

        # start adding edges between backbones
        # first, add edge from head -> the first state block
        # add edge from head block to the head loc_key of the state
        curr_state_blocks = state_to_lockey_map[first_state_val]
        curr_state_val = first_state_val
        state_head_loc = curr_state_blocks[0]
        state_tail_loc = curr_state_blocks[-1]

        self.asmcfg.add_edge(prologue_tail_loc, state_head_loc, AsmConstraintNext(state_head_loc))
        # get offset of the loc_key to the state head block
        jump_dst = hex(self.asmcfg.loc_db.get_location_offset(state_head_loc))
        
        # add JMP <head of next state> to the end of the head block
        self.add_jump_to_next_state(prologue_tail_loc, 'JMP', jump_dst, False)

        # queue: list of tuple (current state value, loc_key for current tail block)
        process_queue = [(first_state_val, state_tail_loc)]
        processed_state_val_list = []

        while len(process_queue) != 0:
            curr_state_val, curr_state_tail_loc = process_queue.pop()
            while True:
                # if we already processed this state, skip
                if curr_state_val in processed_state_val_list:
                    break
                processed_state_val_list.append(curr_state_val)
                
                # get the next states
                next_state_vals = state_order_map[curr_state_val]
                
                if len(next_state_vals) == 1:
                    # only one next state (non conditional)
                    next_state_val = next_state_vals[0]
                    curr_state_tail_block: AsmBlock = self.asmcfg.loc_key_to_block(curr_state_tail_loc)
                    
                    # last instruction of the tail block should be a JMP to predispatcher
                    assert curr_state_tail_block.lines[-1].name == 'JMP'

                    if next_state_val in state_to_lockey_map:
                        # first item is the loc_key to the head block of that statee
                        next_state_head_loc = state_to_lockey_map[next_state_val][0]
                        jump_dst = self.asmcfg.loc_db.get_location_offset(next_state_head_loc)
                        
                        # add an edge from the current tail to the next head
                        self.asmcfg.add_edge(curr_state_tail_loc, next_state_head_loc, AsmConstraintNext(next_state_head_loc))
                        
                        # delete the JMP <dispatcher> instruction & add a JMP instruction to the current tail
                        self.add_jump_to_next_state(curr_state_tail_loc, 'JMP', jump_dst, True)
                elif len(next_state_vals) == 2:
                    # processing conditional
                    true_state_val = next_state_vals[0]
                    false_state_val = next_state_vals[1]
                    
                    # first, remove the CMOV instruction
                    cond_block: AsmBlock = self.asmcfg.loc_key_to_block(curr_state_tail_loc)
                    
                    for i in range(len(cond_block.lines)):
                        insn = cond_block.lines[i]
                        if 'CMOV' in insn.name:
                            cond_jump_type = insn.name.replace('CMOV', 'J')
                            cond_block.lines.remove(insn)
                            break
                        
                    # last instruction of the cond block should be a JMP to predispatcher
                    assert(cond_block.lines[-1].name == 'JMP')
                    
                    if false_state_val in state_to_lockey_map:
                        false_head_loc = state_to_lockey_map[false_state_val][0]
                        false_dst = self.asmcfg.loc_db.get_location_offset(false_head_loc)
                        
                        # add edge from current tail to next false head
                        self.asmcfg.add_edge(curr_state_tail_loc, false_head_loc, AsmConstraintTo(false_head_loc))
                        
                        # delete JMP <dispatchere> & add a JCC from current tail to next false head
                        self.add_jump_to_next_state(curr_state_tail_loc, cond_jump_type, false_dst, True)
                        
                        next_state_val = false_state_val

                    if true_state_val in state_to_lockey_map:
                        true_head_loc = state_to_lockey_map[true_state_val][0]
                        true_tail_loc = state_to_lockey_map[true_state_val][-1]
                        true_dst = self.asmcfg.loc_db.get_location_offset(true_head_loc)
                        
                        # add edge from current tail to next true head
                        self.asmcfg.add_edge(curr_state_tail_loc, true_head_loc, AsmConstraintNext(true_head_loc))
                        
                        # add a JMP from current tail to next false head
                        self.add_jump_to_next_state(curr_state_tail_loc, 'JMP', true_dst, False)
                        
                        # stash the false state for later traversal
                        process_queue.append((true_state_val, true_tail_loc))
                    
                # update curr state val
                curr_state_val = next_state_val
                curr_state_tail_loc = state_to_lockey_map[curr_state_val][-1]

    def generate_patch(self, target_address: int) -> tuple[dict, interval]:
        """Generate patches for an obfuscated function based on the recovered CFG

        Args:
            target_address (int): Target function address

        Returns:
            dict: Patches (offset -> data)
            interval: function interval
        """
        self.asmcfg.sanity_check()
        head_loc = self.asmcfg.heads()[0]

        self.asmcfg.loc_db.set_location_offset(head_loc, target_address)
        
        func_interval = interval(block.get_range() for block in self.asmcfg.blocks)

        # generate patches
        patches = asm_resolve_final(self.mdis.arch, self.asmcfg)
        
        return patches, func_interval
    
    def apply_patches(self, patches_list: List, out_filename: str):
        """Apply patches to deobfuscate

        Args:
            patches_list (List): List of (func_patch, func_interval)
            out_filename (str): Deobfuscated binary path
        """
        out_file = open(out_filename, 'wb')
        in_file = open(self.original_filename, 'rb')
        out_file.write(in_file.read())
        in_file.close()
        
        for func_patch, func_interval in patches_list:
            # NOP out
            for i in range(func_interval.hull()[0], func_interval.hull()[1]):
                out_file.seek(i - self.binary_base_va)
                out_file.write(b"\xCC")
                
            # Apply patches
            for offset, data in viewitems(func_patch):
                out_file.seek(offset - self.binary_base_va)
                out_file.write(data)

        out_file.close()
    