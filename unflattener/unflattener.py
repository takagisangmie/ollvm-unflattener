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
import graphviz
from miasm.arch.x86.arch import instruction_x86, mn_x86
from miasm.arch.x86.disasm import dis_x86_32
from miasm.arch.aarch64.regs import *
from miasm.core.interval import interval
from miasm.loader.elf_init import ELF
from miasm.loader.pe_init import PE
from binrewrite import BinaryRewriter

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
        self.flatten_func_encountered: list = []
    
    def unflat(self, target_address: int) -> tuple[bytes, interval]:
        """Unflatten the CFG of a function

        Args:
            target_address (int): Target function address
        Returns:
            tuple[bytes, interval]: Function patch & function interval
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
            return (None, None)
        patch = self.recover_CFG(target_address)
        func_interval = interval(block.get_range() for block in self.asmcfg.blocks)
        return (patch, func_interval)
    
    def unflat_follow_calls(self, target_address: int, out_filename: str) -> list[tuple[bytes, interval]]:
        """Unflat the target function & all calls to unflat other obfuscated functions 

        Args:
            target_address (int): Target function address
            out_filename (str): Deobfuscated output path
        Returns:
            list[tuple[bytes, interval]]: List of function patch & function interval
        """
        self.flatten_func_queue: list[int] = [target_address]
        processed_flatten_func_list: list[int] = []
        
        patch_data_list: list[tuple[bytes, interval]] = []
        while len(self.flatten_func_queue) != 0:
            flatten_func_addr = self.flatten_func_queue.pop()
            
            if flatten_func_addr in processed_flatten_func_list:
                # do not try to flatten the same function again
                continue
            
            logger.info("Unflattening function {}".format(hex(flatten_func_addr)))
            try:
                patch, func_interval = self.unflat(flatten_func_addr)
                if patch is not None:
                    logger.info("Generate patch for {} successfully".format(hex(target_address)))
                    patch_data_list.append((patch, func_interval))
                else:
                    logger.info("Function {} is not flattened".format(hex(target_address)))
            except:
                logger.info("Fail to unflat function {}".format(hex(flatten_func_addr)))
        return patch_data_list
    
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
            # this is due to CALL/BL instructions breaking up basic block into multiple ones
            while True:
                curr_predecessor_block = self.asmcfg.loc_key_to_block(curr_predecessor_loc)
                call_instr_names = ['CALL'] if self.container.arch != 'aarch64l' else ['BL', 'BLR']
                if curr_predecessor_block.lines[-1].name in ['JZ', 'JMP', 'JNZ'] + \
                   (['B.EQ', 'B.NE', 'B'] if self.container.arch == 'aarch64l' else []):
                    break
                backbone_blocks.append(curr_predecessor_loc)
                curr_predecessor_loc = self.asmcfg.predecessors(curr_predecessor_loc)[0]
            
        # add function's tail (block with no successor) to backbone blocks
        for block in self.asmcfg.blocks:
            if len(self.asmcfg.successors(block.loc_key)) == 0:
                last_tail_loc = block.loc_key
                backbone_blocks.append(last_tail_loc)
                
                # traverse upward from each backbone block to find all backbone blocks above it
                # this is due to CALL/BL instructions breaking up basic block into multiple ones
                curr_predecessor_tail_loc = self.asmcfg.predecessors(last_tail_loc)[0]
                while True:
                    
                    curr_predecessor_tail_block = self.asmcfg.loc_key_to_block(curr_predecessor_tail_loc)
                    if curr_predecessor_tail_block.lines[-1].name in ['JZ', 'JMP', 'JNZ'] + \
                       (['B.EQ', 'B.NE', 'B'] if self.container.arch == 'aarch64l' else []):
                        break
                    backbone_blocks.append(curr_predecessor_tail_loc)
                    curr_predecessor_tail_loc = self.asmcfg.predecessors(curr_predecessor_tail_loc)[0]

        return backbone_blocks

    def symbex_block(self, symbex_engine: SymbolicExecutionEngine, loc_key: LocKey) -> Expr:
        """symbolically executing a block

        Args:
            symbex_engine (SymbolicExecutionEngine): Symbolic execution engine
            loc_key (LocKey): Location key to execute

        Returns:
            Expr: Result symbolic expression
        """
        curr_block = self.asmcfg.loc_key_to_block(loc_key)
        
        if curr_block is None:
            return symbex_engine.run_block_at(self.ircfg, loc_key)
        
        # retrieve the cmp/test instruction & cmovcc instruction
        cmp_instruction = None
        cmov_instruction = None
        
        for instruction in curr_block.lines:
            if instruction.name in ['CMP', 'TEST', 'SUBS', 'CMP']:  # ARM64 comparison instructions
                cmp_instruction = instruction
            if 'CMOV' in instruction.name or 'CSEL' in instruction.name:  # ARM64 conditional select
                cmov_instruction = instruction
                break
        
        if curr_block.lines[-1].name == 'CALL' or \
           (self.container.arch == 'aarch64l' and curr_block.lines[-1].name in ['BL', 'BLR']):
            # process call regularly but we reset RSP/RBP to old RSP/RBP instead 
            #   of an ExprMem depending on miasm's call_func_stack
            #   basically overwriting the execution result of the CALL IR instruction.
            #   Here, we assume that the CALL IR does not impact the stack pointer
            original_rsp = symbex_engine.symbols[ExprId('RSP', 64)]
            original_rbp = symbex_engine.symbols[ExprId('RBP', 64)]
            original_esp = symbex_engine.symbols[ExprId('ESP', 32)]
            original_ebp = symbex_engine.symbols[ExprId('EBP', 32)]
            # ARM64 stack pointer and frame pointer
            original_sp = None
            original_x29 = None
            if self.container.arch == 'aarch64l':
                original_sp = symbex_engine.symbols[ExprId('SP', 64)]
                original_x29 = symbex_engine.symbols[ExprId('X29', 64)]
            
            result = symbex_engine.run_block_at(self.ircfg, loc_key)
            
            if self.container.arch == 'x86_32':
                symbex_engine.symbols[ExprId('ESP', 32)] = original_esp
                symbex_engine.symbols[ExprId('EBP', 32)] = original_ebp
            elif self.container.arch == 'x86_64':
                symbex_engine.symbols[ExprId('RSP', 64)] = original_rsp
                symbex_engine.symbols[ExprId('RBP', 64)] = original_rbp
            elif self.container.arch == 'aarch64l':
                symbex_engine.symbols[ExprId('SP', 64)] = original_sp
                symbex_engine.symbols[ExprId('X29', 64)] = original_x29
            return result
         
        # is an ollvm condition block if CMP instruction is followed by CMOVCC instruction
        if cmov_instruction is not None and cmp_instruction is not None\
            and curr_block.lines.index(cmp_instruction) < curr_block.lines.index(cmov_instruction):
                curr_loc = loc_key

                while True:
                    # continue to simulate to check each IR block
                    # this is because condition-generating instructions (idiv, cmov)
                    #  split a single asm block into multiple IR blocks
                    curr_ir_block: IRBlock = self.ircfg.get_block(curr_loc)
                    if curr_ir_block is None:
                        return symbex_engine.run_block_at(self.ircfg, loc_key)
                    
                    for assign_block in curr_ir_block:
                        # once found the IR assign block for the CMOV/CSEL instruction
                        if 'CMOV' in assign_block.instr.name or 'CSEL' in assign_block.instr.name:
                            # symbex the block as normal
                            symbex_engine.run_block_at(self.ircfg, curr_loc)
                            
                            # NOTE: We don't return the condition produced by symbex_engine.run_block_at here.
                            #   This is because if the condition is deterministic(in a for loop for example)
                            #       symbex_engine.run_block_at will evaluate the cond automatically
                            #       and return ExprInt for the address
                            #   We don't want this as we want to still split the IR path into two
                            #       so we have to get the ExprCond directly from the assign block
                            cmov_cond_expr = assign_block.values()[-1]
                            
                            # Handle ARM64 CSEL conditions differently from x86 CMOV
                            if self.container.arch == 'aarch64l':
                                # ARM64 CSEL uses condition codes directly
                                return cmov_cond_expr.copy()
                            else:
                                # x86 CMOV handling
                                # example: CMOVNZ -> JNZ
                                if 'CMOVN' in cmov_instruction.name:
                                    return cmov_cond_expr.copy()
                                
                                # example: CMOVZ -> JZ 
                                # need to flip the condition src fields
                                return ExprCond(cmov_cond_expr._cond.copy(),
                                    cmov_cond_expr._src2.copy(),
                                    cmov_cond_expr._src1.copy())
                    curr_loc = symbex_engine.run_block_at(self.ircfg, curr_loc)
                    continue
        else:
            # just a regular block, symbex normally
            return symbex_engine.run_block_at(self.ircfg, loc_key) 
                    
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
                # if this block ends with a CALL/BL, extract the call destination and add to self.flatten_func_queue
                curr_block = self.asmcfg.loc_key_to_block(curr_loc)
                if curr_block is not None:
                    last_instruction = curr_block.lines[-1]
                    call_instructions = ['CALL'] if self.container.arch != 'aarch64l' else ['BL', 'BLR']
                    if last_instruction.name in call_instructions:
                        destination_loc = symbex_engine.eval_expr(last_instruction.args[0])
                        
                        if isinstance(destination_loc, ExprInt):
                            destination_loc = int(destination_loc)
                            # only follows calls that are in the .text section only (avoid library calls)
                            if self.text_section_range['lower'] <= destination_loc <= self.text_section_range['upper']:
                                if destination_loc not in self.flatten_func_encountered:
                                    self.flatten_func_queue.append(int(destination_loc))
                                    self.flatten_func_encountered.append(destination_loc)
                
                # symbex block at current loc_key
                symbex_expr_result = self.symbex_block(symbex_engine, curr_loc)
                
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
        state_order_map[0].append(first_state_val)
        prologue_tail_loc = None
        for block_loc in self.asmcfg.predecessors(dispatcher_loc):
            # head block is the other predecessor of dispatcher beside the predispatcher
            if block_loc == predispatcher_loc:
                continue
            
            # add head to backbone
            prologue_tail_loc = block_loc
            backbone_loc_list.append(prologue_tail_loc)
            state_to_lockey_map[0].append(prologue_tail_loc)
            
            # add all prologue blocks above the prologue tail
            curr_prologue_loc = prologue_tail_loc
            while len(self.asmcfg.predecessors(curr_prologue_loc)) != 0:
                prev_prologue_block = self.asmcfg.predecessors(curr_prologue_loc)[0]
                backbone_loc_list.append(prev_prologue_block)
                state_to_lockey_map[0].append(prev_prologue_block)
                curr_prologue_loc = prev_prologue_block
            break
        
        # state value 0 is associated with the prologue blocks
        
        # since we add from the prologue tail up to the prologue head
        # need to flip the order before we reorder the CFG
        state_to_lockey_map[0] = state_to_lockey_map[0][::-1]
        
        # irrelevant blocks are original blocks that are not a backbone block
        irrelevant_loc_list = [original_block.loc_key for original_block in self.asmcfg.blocks if original_block.loc_key not in backbone_loc_list]
        
        # delete all irrelevant blocks
        for loc_key in irrelevant_loc_list:
            self.asmcfg.del_block(self.asmcfg.loc_key_to_block(loc_key))
        
        # init BinaryRewriter to reorder the CFG and generate a patch for rewriting
        rewriter = BinaryRewriter(self.asmcfg, self.container.arch)
        rewriter.init_CFF_data(state_order_map, state_to_lockey_map, symbex_engine)
        rewriter.reorder_blocks(target_address)
        return rewriter.generate_patch()
    
    def apply_patches(self, patch_data_list: list[tuple[bytes, interval]], out_filename: str):
        """Applying patches to the deobfuscated output file

        Args:
            patch_data_list (list[tuple[bytes, interval]]): List of function patches & function intervals
            out_filename (str): Deobfuscated output filename

        Returns:
            bool: _description_
        """
        out_file = open(out_filename, 'wb')
        in_file = open(self.original_filename, 'rb')
        out_file.write(in_file.read())
        in_file.close()
        
        for patch_data in patch_data_list:
            patch, func_interval = patch_data
            func_start = func_interval.hull()[0]
            for i in range(func_interval.hull()[0], func_interval.hull()[1]):
                out_file.seek(i - self.binary_base_va)
                out_file.write(b"\xCC")
            out_file.seek(func_start - self.binary_base_va)
            out_file.write(patch)
        
        out_file.close()
    