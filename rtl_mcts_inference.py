import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import shutil
import subprocess
import uuid
import math
import re
import inspect
import torch.nn.functional as F
from rtl_configs import VerilogConfigs


class MCTSNode:
    def __init__(self, state, prior=1.0, parent=None):
        self.state = state  # List of generated tokens
        self.prior = prior  # P(s,a) from LLM
        self.visit_count = 0
        self.value_sum = 0.0
        self.parent = parent
        self.children = {}  # action_token → MCTSNode


class RTLCoderMCTSEval:
    def __init__(self, verilog_eval_path="~/RTL-Coder", max_retries=5, use_mcts=True, mcts_simulations=100,
                 early_stop=True, debug=False, enable_ppa=True, rollout_mode='greedy'):
        self.verilog_eval_path = os.path.expanduser(verilog_eval_path)
        self.dataset_path = os.path.join(self.verilog_eval_path, "dataset_rtlcoder")
        self.build_path = os.path.join(self.verilog_eval_path, "build_rtlcoder")
        self.max_retries = max_retries
        self.use_mcts = use_mcts
        self.mcts_simulations = mcts_simulations  # MCTS模拟次数参数
        self.early_stop = early_stop  # 是否在找到完美解决方案时提前停止
        self.debug = debug  # 是否启用调试模式
        self.enable_ppa = enable_ppa  # 是否启用PPA分析
        self.rollout_mode = rollout_mode  # 新增：rollout策略 ('greedy' or 'sampling')

        # 统计信息
        self.generation_stats = {}
        self.success_stats = {}
        self.debug_info = {}

        # 加载配置
        self.config = VerilogConfigs()
        self.modules = self.config.modules

        # MCTS参数
        self.c_puct = 1.0
        self.top_k_expand = 20
        self.rollout_top_k = 50
        self.rollout_temperature = 1.0
        self.alpha_B = 0.5
        self.workspace_base = 'mcts_tmp'

        # PPA基线（每个模块单独维护）
        self.ppa_baselines = {}  # {module_name: {'area': float, 'delay': float}}

        # 初始化注释token过滤
        self.comment_token_ids = set()
        self.mask_indices = torch.tensor([], dtype=torch.long, device='cpu')

    def setup_verilog_eval_environment(self):
        """Create VerilogEval evaluation environment"""
        print("Setting up VerilogEval environment...")

        # Create dataset directory
        os.makedirs(self.dataset_path, exist_ok=True)

        # Create problems.txt
        with open(os.path.join(self.dataset_path, "problems.txt"), "w") as f:
            for module in self.modules:
                f.write(f"{module}\n")

        # Create directories and necessary files for each module
        prompts = self.config.get_prompts()
        for module_name in self.modules:
            module_dir = os.path.join(self.dataset_path, module_name)
            os.makedirs(module_dir, exist_ok=True)

            # Create prompt file
            prompt_file = os.path.join(module_dir, f"{module_name}_prompt.txt")
            with open(prompt_file, "w") as f:
                f.write(prompts[module_name])

            # Create interface file
            self._create_interface_file(module_name, module_dir)
            # Create test file
            self._create_test_file(module_name, module_dir)
            # Create reference implementation
            self._create_reference_file(module_name, module_dir)

        print(f"VerilogEval environment created at: {self.dataset_path}")

    def _create_interface_file(self, module_name, module_dir):
        """Create module interface file"""
        interfaces = self.config.get_interfaces()
        ifc_file = os.path.join(module_dir, f"{module_name}_ifc.txt")
        with open(ifc_file, "w") as f:
            f.write(interfaces[module_name])

    def _create_test_file(self, module_name, module_dir):
        """Create test file"""
        testbenches = self.config.get_testbenches()
        test_file = os.path.join(module_dir, f"{module_name}_test.sv")
        with open(test_file, "w") as f:
            if module_name in testbenches:
                f.write(testbenches[module_name])
            else:
                f.write(self.config.get_default_testbench(module_name))

    def _create_reference_file(self, module_name, module_dir):
        """Create reference implementation file"""
        references = self.config.get_references()
        ref_file = os.path.join(module_dir, f"{module_name}_ref.sv")
        with open(ref_file, "w") as f:
            if module_name in references:
                f.write(references[module_name])
            else:
                f.write(self.config.get_default_reference(module_name))

    def load_model(self):
        """Load RTLCoder model"""
        print("Loading RTLCoder model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ishorn5/RTLCoder-Deepseek-v1.1")
            self.model = AutoModelForCausalLM.from_pretrained(
                "ishorn5/RTLCoder-Deepseek-v1.1",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",  # <-- 明确指定使用 Flash Attention 2
                low_cpu_mem_usage=True
            )
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.float16,  # <-- This is the key change
            #     bnb_4bit_use_double_quant=True,
            # )
            #
            # self.model = AutoModelForCausalLM.from_pretrained(
            #     "ishorn5/RTLCoder-Deepseek-v1.1",
            #     quantization_config=bnb_config,  # <-- Pass the config object here
            #     device_map="auto",
            #     attn_implementation="flash_attention_2",
            #     low_cpu_mem_usage=True
            # )
            self.model.eval()

            # Initialize comment token filtering
            self._setup_comment_filtering()

            vocab_size = self.tokenizer.vocab_size
            valid_ids = [tok for tok in self.comment_token_ids if 0 <= tok < vocab_size]

            device = next(self.model.parameters()).device
            self.mask_indices = torch.tensor(valid_ids, device=device, dtype=torch.long)

            print("Model loaded successfully.")
            print(inspect.signature(self.model.generate))
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _setup_comment_filtering(self):
        """Collect all token IDs corresponding to comment syntax"""
        self.comment_token_ids = set()
        for sym in ("//", "/*", "*/"):
            self.comment_token_ids |= set(
                self.tokenizer.encode(sym, add_special_tokens=False)
            )
        for char in ("/", "*"):
            self.comment_token_ids |= set(
                self.tokenizer.encode(char, add_special_tokens=False)
            )

    def _mask_comment_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask comment token logits"""
        if hasattr(self, 'mask_indices') and self.mask_indices.numel() > 0:
            # 直接在词表维度做 index_fill
            logits = logits.index_fill(
                dim=-1,
                index=self.mask_indices,
                value=-float("inf")
            )
        return logits

    def _sample_non_comment(self, logits: torch.Tensor, greedy: bool):
        """Sample non-comment token from logits"""
        logits = self._mask_comment_logits(logits)
        k = 1 if greedy else self.rollout_top_k
        while True:
            if greedy:
                top_idx = logits.topk(k).indices
                cand = [int(x) for x in top_idx if int(x) not in self.comment_token_ids]
                if cand:
                    return cand[0]
                k += 1
            else:
                probs = F.softmax(logits / self.rollout_temperature, dim=-1)
                return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def get_priors(self, state_tokens):
        """Get candidate tokens and their priors for expansion phase"""
        input_ids = torch.tensor([state_tokens], device=self.model.device)
        logits = self.model(input_ids).logits[0, -1]
        logits = self._mask_comment_logits(logits)
        probs = F.softmax(logits, dim=-1)

        k = self.top_k_expand
        while True:
            topk_probs, topk_idx = probs.topk(k)
            keep = [(i.item(), p.item())
                    for i, p in zip(topk_idx, topk_probs)
                    if int(i) not in self.comment_token_ids]
            if keep:
                break
            k *= 2

        tok_ids, priors = zip(*keep[:self.top_k_expand])
        return list(tok_ids), list(priors)

    @torch.no_grad()
    def rollout_policy(self, state_tokens, prefix_len: int, max_len: int = 512, mode: str = 'greedy'):
        """Rollout policy: sample from current state to end"""
        tokens = state_tokens.copy()

        # # 调试：打印词汇表大小，了解有效ID的范围
        # print(f"[DEBUG] Vocabulary size: {self.tokenizer.vocab_size}")

        for i in range(max_len):
            input_ids = torch.tensor([tokens], device=self.model.device)

            # 调试：在调用模型前打印输入的tokens
            # print(f"[DEBUG rollout_policy loop {i}] Input token IDs: {tokens}")

            logits = self.model(input_ids).logits[0, -1]

            if mode == 'greedy':
                next_tok = self._sample_non_comment(logits, greedy=True)
            else:
                next_tok = self._sample_non_comment(logits, greedy=False)

            # 调试：打印新生成的token
            # print(f"[DEBUG rollout_policy loop {i}] Next token ID generated: {next_tok}")

            tokens.append(next_tok)

            if next_tok == self.tokenizer.eos_token_id:
                break

            # Check if module definition is complete
            current_code = self.tokenizer.decode(tokens, skip_special_tokens=True)
            num_modules = len(re.findall(r'^\s*module\b', current_code, re.MULTILINE))
            num_endmodules = current_code.count("endmodule")

            if num_modules > 0 and num_endmodules >= num_modules:
                break

        return tokens

    def _format_verilog_code(self, file_path):
        """Formats a Verilog file in-place using verible-verilog-format."""
        try:
            # 使用 --inplace 参数直接修改文件
            result = subprocess.run(
                ["verible-verilog-format", "--inplace", file_path],
                check=True, capture_output=True, text=True, timeout=10
            )
            if self.debug:
                print(f"[FORMAT] Successfully formatted {file_path}")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            if self.debug:
                print(f"[FORMAT] Failed to format {file_path}: {e.stderr}")
            return False

    def _run_static_analysis(self, file_path, module_name):
        """
        Runs static analysis and returns a dictionary with counts of
        fatal, major, and minor errors.
        """
        error_counts = {'fatal': 0, 'major': 0, 'minor': 0}

        # 定义不同层级的错误/警告关键字
        fatal_patterns = ['%Error:', 'REALCVT', 'SYSTASK', 'UNOPTFLAT']
        major_patterns = ['WIDTH', 'LATCH', 'COMBDLY']
        minor_patterns = ['DECLFILENAME', 'UNUSED', 'EOFNEWLINE']

        try:
            # 移除 check=True，这样即使有警告也不会抛出异常，以便我们完整分析输出
            result = subprocess.run(
                ["verilator", "-sv", "-Wall", "--lint-only", file_path],
                capture_output=True, text=True, timeout=20,
                errors='ignore'
            )

            output = result.stderr
            lines = output.splitlines()

            for line in lines:
                if any(p in line for p in fatal_patterns):
                    error_counts['fatal'] += 1
                elif any(p in line for p in major_patterns):
                    error_counts['major'] += 1
                elif any(p in line for p in minor_patterns):
                    error_counts['minor'] += 1

            if self.debug:
                print(f"[LINT] Analysis complete. Counts: {error_counts}")
                if output:
                    print(output)

            return error_counts

        except subprocess.TimeoutExpired as e:
            if self.debug:
                print(f"[LINT] Static analysis timed out.")
            error_counts['fatal'] += 1  # 超时是致命错误
            return error_counts

    def compute_ppa_reward(self, generated_tokens, module_name, prompt, prefix_len: int, debug=True):
        """Calculate PPA-based reward (similar to verigen project)"""

        # Create temporary workspace
        ws = os.path.join(self.workspace_base, str(uuid.uuid4()))
        os.makedirs(ws, exist_ok=True)
        dut_path = os.path.join(ws, f'{module_name}.v')

        try:
            # Decode generated code
            suffix_tokens = generated_tokens[prefix_len:]
            suffix = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True).strip()

            # Process code (reuse previous logic)
            if suffix.strip().startswith('module'):
                full_code = suffix
            else:
                interface_match = re.search(r'(module\s+' + re.escape(module_name) + r'\s*\([^)]*\)\s*;)', prompt)
                if interface_match:
                    module_header = interface_match.group(1)
                    full_code = module_header + '\n' + suffix
                else:
                    full_code = suffix

            m = re.search(r'^\s*module\b', full_code, flags=re.MULTILINE)
            if m:
                full_code = full_code[m.start():]

            # Clean code
            num_modules_total = len(re.findall(r'^\s*module\b', full_code, re.MULTILINE))
            endmodule_positions = [m.start() for m in re.finditer(r'\bendmodule\b', full_code)]

            if num_modules_total > 0 and len(endmodule_positions) >= num_modules_total:
                end_pos = endmodule_positions[num_modules_total - 1] + len("endmodule")
                full_code_to_write = full_code[:end_pos]
            else:
                full_code_to_write = full_code
                missing_endmodules = num_modules_total - len(endmodule_positions)
                if missing_endmodules > 0:
                    full_code_to_write += "\nendmodule" * missing_endmodules

            full_code_to_write = full_code_to_write.strip()

            # Write code file
            with open(dut_path, 'w', encoding='utf-8') as f:
                f.write(full_code_to_write)

            self._format_verilog_code(dut_path)

            if debug:
                print(f"[PPA] Analyzing {module_name}...")

            error_counts = self._run_static_analysis(dut_path, module_name)

            if error_counts['fatal'] > 0:
                return "LINT_FAIL", None, None, full_code_to_write, "Static analysis found fatal errors", error_counts

            # === Stage 1: Functional verification ===
            # First verify functional correctness with simple test
            self.save_to_verilog_eval(module_name, full_code_to_write)
            success, message, debug_details = self.test_single_module(module_name, debug=False)

            if not success:
                if message.startswith("Compilation failed"):
                    return "COMPILE_FAIL", None, None, full_code_to_write, message, error_counts
                else:
                    if debug:
                        print(f"[PPA] Functional test failed: {message}")
                return "FUNC_FAIL", None, None, full_code_to_write, message, error_counts


            if debug:
                print(f"[PPA] Functional test passed!")

            # === Stage 2: PPA analysis ===
            if not self.enable_ppa:
                return "FUNC_PASSED", None, None, full_code_to_write,"Functional test passed, PPA disabled.", error_counts

            try:
                # Yosys synthesis
                abc_executable = os.path.expanduser("~/abc/abc")
                lib_file = os.path.expanduser("~/test2/NangateOpenCellLibrary_typical_ccs.lib")
                blif_file_name = "design.blif"

                # Dynamic Yosys script selection
                if 'mac' in module_name.lower():
                    # Use high-level synthesis for sequential circuits
                    yosys_script = (
                        f"read_verilog {module_name}.v; "
                        f"synth -noalumacc -top {module_name}; "
                        f"write_blif {blif_file_name}"
                    )
                else:
                    # Use manual script for combinational circuits
                    yosys_script = (
                        f"read_verilog {module_name}.v; "
                        f"proc; "
                        f"hierarchy -check -top {module_name}; "
                        f"flatten; "
                        f"opt; techmap; opt -full; "
                        f"write_blif {blif_file_name}"
                    )

                subprocess.run(
                    ["yosys", "-p", yosys_script],
                    cwd=ws,
                    check=True, capture_output=True, text=True, timeout=30
                )

                # ABC analysis
                abc_script = "\n".join([
                    f"read_lib {lib_file}",
                    f"read_blif {blif_file_name}",
                    "strash",
                    "map",
                    "print_stats",
                    "quit"
                ]) + "\n"

                abc_result = subprocess.run([abc_executable],
                                            cwd=ws, input=abc_script, check=True, capture_output=True, text=True,
                                            timeout=30)
                abc_output = abc_result.stdout

                # Parse area and delay
                area = 0.0
                delay = 0.0

                area_match = re.search(r"\barea\s*=\s*([0-9]+(?:\.[0-9]+)?)", abc_output)
                if area_match:
                    area = float(area_match.group(1))

                delay_match = re.search(r"\bdelay\s*=\s*([0-9]+(?:\.[0-9]+)?)", abc_output)
                if delay_match:
                    delay = float(delay_match.group(1))

                if area == 0 or delay == 0:
                    if debug:
                        print(f"[PPA] Invalid PPA results: area={area}, delay={delay}")
                    return "SYNTH_FAIL", None, None, full_code_to_write, "PPA analysis failed", error_counts

                if debug:
                    print(f"[PPA] Analysis successful - Area: {area}, Delay: {delay}")

                return "SUCCESS", area, delay, full_code_to_write, message, error_counts

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                if debug:
                    print(f"[PPA] Synthesis failed: {e}")
                return "SYNTH_FAIL", None, None, full_code_to_write, f"Synthesis failed: {str(e)}", error_counts

        finally:
            # Clean up workspace
            shutil.rmtree(ws, ignore_errors=True)

    def compute_reward(self, generated_tokens, module_name, prompt, prefix_len: int, debug=True):
        """Calculate reward - enhanced version with PPA support"""
        message = ""
        error_counts = {'fatal': 0, 'major': 0, 'minor': 0}
        if self.enable_ppa:
            status, area, delay, code, message, error_counts = self.compute_ppa_reward(generated_tokens, module_name, prompt, prefix_len,
                                                                debug)
        else:
            # Decode generated code (simplified version)
            suffix_tokens = generated_tokens[prefix_len:]
            suffix = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True).strip()

            if suffix.strip().startswith('module'):
                full_code = suffix
            else:
                interface_match = re.search(r'(module\s+' + re.escape(module_name) + r'\s*\([^)]*\)\s*;)', prompt)
                if interface_match:
                    module_header = interface_match.group(1)
                    full_code = module_header + '\n' + suffix
                else:
                    full_code = suffix

            m = re.search(r'^\s*module\b', full_code, flags=re.MULTILINE)
            if m:
                full_code = full_code[m.start():]

            self.save_to_verilog_eval(module_name, full_code)
            success, message, debug_details = self.test_single_module(module_name, debug=False)

            if success:
                status, area, delay, code = "SUCCESS", None, None, full_code
            else:
                if "Compilation failed" in message:
                    status = "COMPILE_FAIL"
                elif "Logic errors" in message:
                    status = "SIM_FAIL"
                else:
                    status = "OTHER_FAIL"
                area, delay, code = None, None, full_code

        # Calculate reward
        reward = 0.0

        fatal_count = error_counts.get('fatal', 0)
        major_count = error_counts.get('major', 0)
        minor_count = error_counts.get('minor', 0)

        if status == "COMPILE_FAIL" or status == "LINT_FAIL":
            fatal_count += 1

        # First, processing the failure case
        if fatal_count > 0:
            # Fatal Error: Base penalty is high and increases with quantity
            reward = -1 - (fatal_count * 0.05)
        elif major_count > 0:
            # Main error: moderate penalty, increasing with quantity
            reward = -0.7 - (major_count * 0.03)
        elif minor_count > 0:
            # Minor error: Light penalty, increasing with quantity
            reward = -0.2 - (minor_count * 0.01)

        # Processing function failed
        elif status in ["SIM_FAIL", "FUNC_FAIL"]:
            mismatch_pattern = r"Logic errors:\s*(\d+)/(\d+)\s*mismatches"
            match = re.search(mismatch_pattern, message)
            if match:
                mismatches = int(match.group(1))
                total = int(match.group(2))
                error_rate = mismatches / total
                # Score based on error rate
                reward = 0.8 - 1.6 * error_rate
            else:
                reward = -0.5  # Unable to parse, default penalty

        elif status == "SYNTH_FAIL":
            reward = -0.7  # 综合失败单独处理

        elif status == "SUCCESS":
            base_reward = 1.0
            if self.enable_ppa and area is not None and delay is not None:
                # Use PPA to calculate dynamic reward (similar to verigen)
                if module_name not in self.ppa_baselines:
                    # First success: set baseline
                    self.ppa_baselines[module_name] = {'area': area, 'delay': delay}
                    base_reward = self.alpha_B  # Base reward
                    if debug:
                        print(f"[PPA] First success! Baseline set for {module_name}: area={area}, delay={delay}")
                else:
                    # Subsequent success: calculate PPA improvement
                    baseline = self.ppa_baselines[module_name]
                    adp = area * delay
                    adp_ref = baseline['area'] * baseline['delay']
                    ppa_score = 1.0 - (adp / adp_ref)
                    base_reward = self.alpha_B + ppa_score
                    if debug:
                        print(
                            f"[PPA] PPA improvement: {ppa_score:.3f} (current ADP: {adp:.2f}, baseline: {adp_ref:.2f})")

            reward = base_reward

            total_warnings = error_counts.get('major', 0) + error_counts.get('minor', 0)
            if total_warnings == 0:
                reward += 0.2  # 零警告，额外奖励
            else:
                # 修正1：移除了错误的 reward = base_reward 覆盖操作
                pass
        else:
            reward = -0.8

        reward = max(-2.0, min(reward, 2.0))

        if debug:
            print(f"[REWARD] {module_name}: {status} -> {reward:.3f}")

            if message and reward < 0:
                print(f"[ERROR] {message}")

        return reward, status, code, message

    def uct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Calculate UCT score: UCT = Q + U"""
        Q = child.value_sum / (1 + child.visit_count)
        U = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return Q + U

    def select(self, node: MCTSNode) -> MCTSNode:
        """Selection: select from root node to unexpanded node"""
        while node.children:
            _, node = max(
                node.children.items(),
                key=lambda kv: self.uct_score(node, kv[1])
            )
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion: expand all candidate actions for current node"""
        top_tokens, top_priors = self.get_priors(node.state)
        for tok, prior in zip(top_tokens, top_priors):
            if tok in node.children:
                continue
            new_state = node.state + [tok]
            node.children[tok] = MCTSNode(new_state, prior=prior, parent=node)

        if not node.children:
            return node

        return max(node.children.values(), key=lambda n: n.prior)

    def backpropagate(self, path, reward: float):
        """Backpropagation: propagate reward back to update node statistics"""
        for n in path:
            n.visit_count += 1
            n.value_sum += reward

    def mcts_generate_with_stats(self, module_name, prompt, n_sim=None, early_stop=True):
        """Use MCTS to generate code and return actual number of simulations used"""
        if n_sim is None:
            n_sim = self.mcts_simulations

        print(f"Running MCTS for {module_name} with {n_sim} simulations (early_stop={early_stop})...")

        # Initialize
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        seed_tokens = [self.tokenizer.bos_token_id] + input_ids
        prefix_len = len(seed_tokens)

        root = MCTSNode(seed_tokens)
        best_reward = -float('inf')
        best_synthesizable_code = ""
        best_synthesizable_reward = -float('inf')

        for i in range(n_sim):
            # Show progress
            if i == 0:
                print(f"[{module_name}] Starting MCTS simulations...")
            elif (i + 1) % 10 == 0 or i == n_sim - 1:
                progress_pct = ((i + 1) / n_sim) * 100
                print(
                    f"[{module_name}] Progress: {i + 1}/{n_sim} ({progress_pct:.1f}%) - Best reward so far: {best_reward:.3f}")

            # MCTS steps
            node = self.select(root)

            if node.visit_count > 0:
                node = self.expand(node)

            rollout_tokens = self.rollout_policy(node.state, prefix_len=prefix_len, mode=self.rollout_mode)

            # Calculate reward
            reward, status, current_code, message = self.compute_reward(rollout_tokens, module_name, prompt, prefix_len, debug=self.debug)

            if self.debug:
                # 解码当前生成的代码
                current_suffix_tokens = rollout_tokens[prefix_len:]
                current_suffix = self.tokenizer.decode(current_suffix_tokens, skip_special_tokens=True).strip()
                if current_suffix.strip().startswith('module'):
                    current_code = current_suffix
                else:
                    interface_match = re.search(r'(module\s+' + re.escape(module_name) + r'\s*\([^)]*\)\s*;)',
                                                prompt)
                    if interface_match:
                        module_header = interface_match.group(1)
                        current_code = module_header + '\n' + current_suffix
                    else:
                        current_code = current_suffix

                m = re.search(r'^\s*module\b', current_code, flags=re.MULTILINE)
                if m:
                    current_code = current_code[m.start():]
                    print(current_code)
                print(f"[DEBUG-SIM{i + 1}] Reward: {reward:.3f}")
                print("-" * 40)

            # Backpropagate
            path = []
            cur = node
            while cur:
                path.append(cur)
                cur = cur.parent
            self.backpropagate(path, reward)

            # FIXED: Update best result - properly track best code
            if reward > best_reward:
                best_reward = reward
                best_tokens = rollout_tokens.copy()  # Store complete token sequence

                # Regenerate code from tokens
                suffix_tokens = rollout_tokens[prefix_len:]
                suffix = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True).strip()

                if suffix.strip().startswith('module'):
                    best_code = suffix
                else:
                    # If suffix doesn't contain complete module, extract interface from prompt
                    interface_match = re.search(r'(module\s+' + re.escape(module_name) + r'\s*\([^)]*\)\s*;)', prompt)
                    if interface_match:
                        module_header = interface_match.group(1)
                        best_code = module_header + '\n' + suffix
                    else:
                        best_code = suffix

                print(f"[{module_name}] Simulation {i + 1}: NEW BEST REWARD {reward:.3f}")

                if reward > 0:  # Only show code preview for positive rewards
                    code_preview = best_code[:100].replace('\n', ' ') + ('...' if len(best_code) > 100 else '')
                    print(f"[{module_name}] Code preview: {code_preview}")
                    # If PPA enabled, show current PPA info
                    if self.enable_ppa and module_name in self.ppa_baselines:
                        baseline = self.ppa_baselines[module_name]
                        print(
                            f"[{module_name}] PPA baseline: Area={baseline['area']:.2f}, Delay={baseline['delay']:.2f}")
            else:
                # Occasionally show current attempt results (when not best)
                if (i + 1) % 50 == 0:  # Show every 25 attempts
                    print(
                        f"[{module_name}] Simulation {i + 1}: Current reward {reward:.3f} (not better than {best_reward:.3f})")

            if status == "SUCCESS":
                # 如果这份可综合的代码比之前找到的可综合代码更好（PPA更优）
                if reward > best_synthesizable_reward:
                    best_synthesizable_reward = reward
                    best_synthesizable_code = current_code
                    print(
                        f"[{module_name}] Simulation {i + 1}: *** FOUND NEW BEST SYNTHESIZABLE SOLUTION (REWARD: {reward:.3f}) ***")

            # Optional early stopping mechanism
            if early_stop and best_synthesizable_code:
                print(f"[{module_name}] FIRST SYNTHESIZABLE SOLUTION FOUND at simulation {i + 1} - stopping early!")
                # self._clear_memory()
                break

            # self._clear_memory()

        # Final summary
        actual_simulations = i + 1  # Actual number of simulations used
        print(f"[{module_name}] MCTS completed: {actual_simulations}/{n_sim} simulations")
        print(f"[{module_name}] Final best reward: {best_reward:.3f}")

        # FIXED: Ensure best code is saved to file
        if best_synthesizable_code:
            print(f"[{module_name}] A synthesizable solution was found. Saving the best one to file.")
            self.save_to_verilog_eval(module_name, best_synthesizable_code)
            if self.debug:
                print(f"[{module_name}] Debug: Saved best synthesizable code to file")
        else:
            print(
                f"[{module_name}] No synthesizable solution was found after {actual_simulations} simulations. No file will be saved.")

        if best_synthesizable_reward > -float('inf'):
            print(
                f"[{module_name}] SUCCESS! Synthesizable solution found with best reward: {best_synthesizable_reward:.3f}")
        else:
            print(f"[{module_name}] No successful solution found")

        return best_synthesizable_code, best_reward, actual_simulations

    def generate_code_with_mcts_stats(self, task_name, prompt, attempt=1):
        """Generate code and return actual number of simulations used"""
        if self.use_mcts:
            best_code, best_reward, actual_simulations = self.mcts_generate_with_stats(task_name, prompt,
                                                                                       early_stop=self.early_stop)

            # Debug: Ensure returned code is correct
            if self.debug:
                print(f"[DEBUG] Final best code for {task_name} (reward: {best_reward:.3f}):")
                print(f"[DEBUG] Code length: {len(best_code) if best_code else 0}")
                if best_code:
                    print(f"[DEBUG] First line: {best_code.split(chr(10))[0] if best_code else 'None'}")

            return best_code, best_reward, actual_simulations
        else:
            # Traditional mode doesn't need simulation statistics
            code = self.generate_code_traditional(task_name, prompt, attempt)
            return code, 1

    def generate_code_traditional(self, task_name, prompt, attempt=1):
        """Traditional code generation method"""
        print(f"Generating code for: {task_name} (Attempt {attempt})")

        temperature = 0.5 + (attempt - 1) * 0.1
        temperature = min(temperature, 1.0)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(0)
        attention_mask = inputs.attention_mask.to(0)

        sample = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=1024,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )

        s_full = self.tokenizer.decode(sample[0])
        # Original post-processing logic
        s = s_full

        m = re.search(r'^\s*module\b', s, flags=re.MULTILINE)
        if m:
            s = s[m.start():]

        if len(s.split('endmodulemodule', 1)) == 2:
            s = s.split('endmodulemodule', 1)[0] + "\n" + "endmodule"
        else:
            s = s.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

        if s.find('top_module') != -1:
            s = s.split('top_module', 1)[0]
            s = s.rsplit('endmodule', 1)[0] + "\n" + "endmodule"

        index = s.rfind('tb_module')
        if index == -1:
            index = s.find('testbench')
        if index != -1:
            s_tmp = s[:index]
            s = s_tmp.rsplit("endmodule", 1)[0] + "\n" + "endmodule"

        if prompt in s:
            s = s.split(prompt, 1)[-1].strip()

        return s

    def save_to_verilog_eval(self, module_name, generated_code):
        """Save generated code to VerilogEval format"""
        os.makedirs(self.build_path, exist_ok=True)
        module_build_dir = os.path.join(self.build_path, module_name)
        os.makedirs(module_build_dir, exist_ok=True)

        sample_file = os.path.join(module_build_dir, f"{module_name}_sample01.sv")
        with open(sample_file, "w") as f:
            f.write(generated_code)

    def setup_build_environment(self):
        """Setup build environment"""
        print("Setting up build environment...")

        for module_name in self.modules:
            source_dir = os.path.join(self.dataset_path, module_name)
            build_dir = os.path.join(self.build_path, module_name)

            os.makedirs(build_dir, exist_ok=True)

            for file_type in ["test.sv", "ref.sv"]:
                src_file = os.path.join(source_dir, f"{module_name}_{file_type}")
                dst_file = os.path.join(build_dir, f"{module_name}_{file_type}")
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)

            for file_type in ["prompt.txt", "ifc.txt"]:
                src_file = os.path.join(source_dir, f"{module_name}_{file_type}")
                dst_file = os.path.join(self.build_path, f"{module_name}_{file_type}")
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)

    def test_single_module(self, module_name, debug=None):
        """Test single module (keep original logic)"""
        if debug is None:
            debug = self.debug

        if debug:
            print(f"=== DEBUG: Testing {module_name} ===")

        module_dir = os.path.join(self.build_path, module_name)

        if not os.path.exists(module_dir):
            if debug:
                print(f"DEBUG: Module directory not found: {module_dir}")
            return False, "Module directory not found", {}

        original_dir = os.getcwd()
        debug_details = {
            'module_dir': module_dir,
            'required_files': [],
            'compilation_output': '',
            'test_output': '',
            'full_error': ''
        }

        try:
            os.chdir(module_dir)
            if debug:
                print(f"DEBUG: Changed to directory: {module_dir}")

            required_files = [f"{module_name}_sample01.sv", f"{module_name}_test.sv", f"{module_name}_ref.sv"]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            debug_details['required_files'] = {f: os.path.exists(f) for f in required_files}

            if debug:
                print(f"DEBUG: Required files check:")
                for f in required_files:
                    exists = os.path.exists(f)
                    print(f"  {f}: {'EXISTS' if exists else 'MISSING'}")

            if missing_files:
                debug_details['full_error'] = f"Missing files: {missing_files}"
                if debug:
                    print(f"DEBUG: Missing files: {missing_files}")
                return False, f"Missing files: {missing_files}", debug_details

            exe_file = f"test_{module_name}"
            try:
                if debug:
                    print(f"DEBUG: Starting compilation with iverilog...")
                    print(
                        f"DEBUG: Command: iverilog -g2012 {module_name}_test.sv {module_name}_sample01.sv {module_name}_ref.sv -o {exe_file}")

                result = subprocess.run([
                    "timeout",
                    "60s",
                    "iverilog",
                    "-g2012",
                    f"{module_name}_test.sv",
                    f"{module_name}_sample01.sv",
                    f"{module_name}_ref.sv",
                    "-o", f"test_{module_name}"
                ], capture_output=True, text=True, timeout=30)

                debug_details['compilation_output'] = {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }

                if debug:
                    print(f"DEBUG: Compilation return code: {result.returncode}")
                    if result.stdout:
                        print(f"DEBUG: Compilation stdout:\n{result.stdout}")
                    if result.stderr:
                        print(f"DEBUG: Compilation stderr:\n{result.stderr}")

            except subprocess.TimeoutExpired:
                debug_details['full_error'] = "Compilation timed out after 30s"
                if debug:
                    print("DEBUG: Compilation timed out after 30s")
                return False, "Compilation timeout", debug_details

            if result.returncode != 0:
                error_lines = result.stderr.strip().split('\n')
                main_error = error_lines[0] if error_lines else "Compilation failed"
                debug_details['full_error'] = result.stderr.strip()
                if debug:
                    print(f"DEBUG: Compilation failed with return code {result.returncode}")
                    print(f"DEBUG: Main error: {main_error}")
                return False, f"Compilation failed: {main_error}", debug_details

            if not os.path.exists(exe_file):
                debug_details['full_error'] = f"Executable {exe_file} was not generated despite successful compilation"
                if debug:
                    print(f"DEBUG: Executable {exe_file} was not generated")
                return False, f"Executable not generated", debug_details

            if debug:
                print(f"DEBUG: Compilation successful, executable {exe_file} created")
                print(f"DEBUG: Starting test execution...")

            try:
                test_result = subprocess.run([f"./test_{module_name}"], capture_output=True, text=True, timeout=60)
                debug_details['test_output'] = {
                    'stdout': test_result.stdout,
                    'stderr': test_result.stderr,
                    'returncode': test_result.returncode
                }

                if debug:
                    print(f"DEBUG: Test execution return code: {test_result.returncode}")
                    if test_result.stdout:
                        print(f"DEBUG: Test stdout:\n{test_result.stdout}")
                    if test_result.stderr:
                        print(f"DEBUG: Test stderr:\n{test_result.stderr}")

            except subprocess.TimeoutExpired:
                debug_details['full_error'] = "Test execution timed out after 60s"
                if debug:
                    print("DEBUG: Test execution timed out after 60s")
                return False, "Test execution timeout", debug_details

            if test_result.returncode != 0:
                error_msg = test_result.stderr.strip().split('\n')[
                    0] if test_result.stderr.strip() else "Test execution failed"
                debug_details['full_error'] = test_result.stderr.strip()
                if debug:
                    print(f"DEBUG: Test execution failed with return code {test_result.returncode}")
                    print(f"DEBUG: Error message: {error_msg}")
                return False, f"Test execution failed: {error_msg}", debug_details

            output = test_result.stdout
            import re
            match = re.search(r"Mismatches:\s*(\d+)\s*in\s*(\d+)\s*samples", output)
            if match:
                mismatches = int(match.group(1))
                total = int(match.group(2))
                if debug:
                    print(f"DEBUG: Test results parsed - {mismatches} mismatches in {total} samples")
                if mismatches == 0:
                    return True, f"Perfect - 0/{total} mismatches", debug_details
                else:
                    return False, f"Logic errors: {mismatches}/{total} mismatches", debug_details
            else:
                short_output = output.strip()
                if len(short_output) > 100:
                    short_output = short_output[:97] + "..."
                debug_details['full_error'] = f"Unknown test result format. Full output: {output.strip()}"
                if debug:
                    print(f"DEBUG: Could not parse test results. Full output:\n{output}")
                return False, f"Unknown result format", debug_details

        except Exception as e:
            error_msg = str(e)
            if len(error_msg) > 80:
                short_error = error_msg[:77] + "..."
            else:
                short_error = error_msg
            debug_details['full_error'] = str(e)
            if debug:
                print(f"DEBUG: Exception occurred: {e}")
            return False, f"Test exception: {short_error}", debug_details
        finally:
            os.chdir(original_dir)
            if debug:
                print(f"DEBUG: Restored directory to: {original_dir}")

    def generate_with_retry(self, module_name, prompt):
        """Code generation with retry mechanism (now can choose to use MCTS)"""
        if module_name not in self.generation_stats:
            self.generation_stats[module_name] = 0
        if module_name not in self.debug_info:
            self.debug_info[module_name] = {
                'attempts': [],
                'final_status': '',
                'total_attempts': 0,
                'total_simulations': 0  # New: total simulation count
            }

        if self.use_mcts:
            # Use MCTS, try only once but perform multiple simulations
            print(f"\n{'=' * 60}")
            print(f"Module: {module_name} (Using MCTS)")
            print(f"{'=' * 60}")

            attempt_info = {
                'attempt_number': 1,
                'generated_code': '',
                'test_result': '',
                'debug_details': {},
                'simulations_used': 0  # New: record number of simulations used
            }

            # MCTS generation and return actual number of simulations used
            generated_code, best_reward, actual_simulations = self.generate_code_with_mcts_stats(module_name, prompt, 1)
            self.generation_stats[module_name] = actual_simulations  # Use actual simulation count
            self.debug_info[module_name]['total_simulations'] = actual_simulations
            attempt_info['generated_code'] = generated_code
            attempt_info['simulations_used'] = actual_simulations

            if not generated_code:
                print(f"MCTS: Failed to generate code")
                attempt_info['test_result'] = 'Code generation failed'
                self.debug_info[module_name]['attempts'].append(attempt_info)
                self.success_stats[module_name] = {
                    'attempts': actual_simulations,  # Show simulation count
                    'status': 'failed',
                    'message': 'Code generation failed',
                    'score': best_reward
                }
                self.debug_info[module_name]['final_status'] = 'failed'
                self.debug_info[module_name]['total_attempts'] = 1
                return False

            print("--- Generated Code ---")
            print(generated_code)
            print("--- End Code ---")

            # Test generated code directly, no need to save again
            success, message, debug_details = self.test_single_module(module_name)
            attempt_info['test_result'] = message
            attempt_info['debug_details'] = debug_details
            self.debug_info[module_name]['attempts'].append(attempt_info)

            if success:
                print(f"MCTS: SUCCESS - {message}")
                self.success_stats[module_name] = {
                    'attempts': actual_simulations,
                    'status': 'success',
                    'message': message,
                    'score': best_reward
                }
                self.debug_info[module_name]['final_status'] = 'success'
                self.debug_info[module_name]['total_attempts'] = 1
                return True
            else:
                print(f"MCTS: FAILED - {message}")
                self.success_stats[module_name] = {
                    'attempts': actual_simulations,
                    'status': 'failed',
                    'message': message,
                    'score': best_reward
                }
                self.debug_info[module_name]['final_status'] = 'failed'
                self.debug_info[module_name]['total_attempts'] = 1
                return False
        else:
            # Use original retry logic
            for attempt in range(1, self.max_retries + 1):
                print(f"\n{'=' * 60}")
                print(f"Module: {module_name} (Attempt {attempt}/{self.max_retries})")
                print(f"{'=' * 60}")

                attempt_info = {
                    'attempt_number': attempt,
                    'generated_code': '',
                    'test_result': '',
                    'debug_details': {}
                }

                generated_code = self.generate_code_traditional(module_name, prompt, attempt)
                self.generation_stats[module_name] = attempt
                attempt_info['generated_code'] = generated_code

                if not generated_code:
                    print(f"Attempt {attempt}: Failed to generate code")
                    attempt_info['test_result'] = 'Code generation failed'
                    self.debug_info[module_name]['attempts'].append(attempt_info)
                    continue

                print("--- Generated Code ---")
                print(generated_code)
                print("--- End Code ---")

                self.save_to_verilog_eval(module_name, generated_code)
                success, message, debug_details = self.test_single_module(module_name)
                attempt_info['test_result'] = message
                attempt_info['debug_details'] = debug_details
                self.debug_info[module_name]['attempts'].append(attempt_info)

                if success:
                    print(f"Attempt {attempt}: SUCCESS - {message}")
                    self.success_stats[module_name] = {
                        'attempts': attempt,
                        'status': 'success',
                        'message': message
                    }
                    self.debug_info[module_name]['final_status'] = 'success'
                    self.debug_info[module_name]['total_attempts'] = attempt
                    return True
                else:
                    print(f"Attempt {attempt}: FAILED - {message}")
                    if attempt < self.max_retries:
                        print(f"Retrying with higher diversity...")

            # All attempts failed
            print(f"Module {module_name} failed after {self.max_retries} attempts")
            final_message = message if 'message' in locals() else 'All attempts failed'
            self.success_stats[module_name] = {
                'attempts': self.max_retries,
                'status': 'failed',
                'message': final_message
            }
            self.debug_info[module_name]['final_status'] = 'failed'
            self.debug_info[module_name]['total_attempts'] = self.max_retries
            return False

    def print_final_statistics(self):
        """Print final statistics"""
        print(f"\n{'=' * 80}")
        print(f"FINAL STATISTICS")
        print(f"{'=' * 80}")

        successful_tasks = 0
        total_attempts = 0

        print(f"{'Module':<15} {'Attempts':<10} {'Status':<12} {'Details'}")
        print(f"{'-' * 65}")

        for module_name in self.modules:
            stats = self.success_stats.get(module_name, {'attempts': 0, 'status': 'not_run', 'message': 'N/A'})
            attempts = stats['attempts']
            status = stats['status']
            message = stats['message']
            score = stats.get('score', 'N/A')

            total_attempts += attempts
            if status == 'success':
                successful_tasks += 1

            score_str = f"{score:.3f}" if isinstance(score, float) else "N/A"
            print(f"{module_name:<15} {attempts:<10} {score_str:<10} {status:<12} {message}")

        print(f"{'-' * 65}")
        print(f"Summary:")
        print(
            f"   • Successful modules: {successful_tasks}/{len(self.modules)} ({successful_tasks / len(self.modules) * 100:.1f}%)")

        if self.use_mcts:
            total_simulations = sum(self.generation_stats.values())
            print(f"   • Total MCTS simulations: {total_simulations}")
            print(f"   • Method: MCTS-guided generation ({self.mcts_simulations} max simulations per module)")
        else:
            print(f"   • Total attempts: {total_attempts}")
            print(f"   • Method: Traditional retry (Max retries: {self.max_retries})")

        # Detailed analysis
        print(f"\nDetailed Analysis:")

        if self.use_mcts:
            # MCTS mode analysis
            early_success = sum(1 for stats in self.success_stats.values()
                                if stats['status'] == 'success' and stats['attempts'] <= 10)
            print(
                f"   • Early success (<=10 sims): {early_success}/{len(self.modules)} ({early_success / len(self.modules) * 100:.1f}%)")

            avg_simulations = sum(
                stats['attempts'] for stats in self.success_stats.values() if stats['status'] != 'not_run') / max(1,
                                                                                                                  len([s
                                                                                                                       for
                                                                                                                       s
                                                                                                                       in
                                                                                                                       self.success_stats.values()
                                                                                                                       if
                                                                                                                       s[
                                                                                                                           'status'] != 'not_run']))
            print(f"   • Average simulations per module: {avg_simulations:.1f}")
        else:
            # Traditional mode analysis
            first_try_success = sum(1 for stats in self.success_stats.values()
                                    if stats['status'] == 'success' and stats['attempts'] == 1)
            print(
                f"   • First-try success: {first_try_success}/{len(self.modules)} ({first_try_success / len(self.modules) * 100:.1f}%)")

        failed_modules = [module for module, stats in self.success_stats.items()
                          if stats['status'] == 'failed']
        if failed_modules:
            print(f"   • Failed modules: {', '.join(failed_modules)}")

        print(f"\nResults saved in: {self.build_path}")
        print(f"{'=' * 80}")

    def save_statistics_to_file(self):
        """Save statistics to file"""
        stats_file = os.path.join(self.build_path, "generation_statistics.json")
        debug_file = os.path.join(self.build_path, "debug_details.json")

        # Brief statistics
        stats_data = {
            'max_retries': self.max_retries,
            'use_mcts': self.use_mcts,
            'modules': self.modules,
            'generation_stats': self.generation_stats,
            'success_stats': self.success_stats,
            'summary': {
                'total_modules': len(self.modules),
                'successful_modules': len([s for s in self.success_stats.values() if s['status'] == 'success']),
                'total_attempts': sum(self.generation_stats.values()),
                'first_try_success': len([s for s in self.success_stats.values()
                                          if s['status'] == 'success' and s['attempts'] == 1])
            }
        }

        # Detailed debug information
        debug_data = {
            'debug_info': self.debug_info,
            'generation_timestamp': __import__('datetime').datetime.now().isoformat(),
            'config_info': {
                'modules': self.modules,
                'max_retries': self.max_retries,
                'use_mcts': self.use_mcts
            }
        }

        import json
        # Save brief statistics
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)

        # Save detailed debug information
        with open(debug_file, 'w') as f:
            json.dump(debug_data, f, indent=2)

        print(f"Statistics saved to: {stats_file}")
        print(f"Debug details saved to: {debug_file}")

    def show_debug_info(self, module_name=None):
        """Show detailed debug information"""
        if module_name:
            if module_name in self.debug_info:
                print(f"\nDebug info for {module_name}:")
                print("=" * 50)
                debug = self.debug_info[module_name]
                print(f"Final status: {debug['final_status']}")
                print(f"Total attempts: {debug['total_attempts']}")

                for i, attempt in enumerate(debug['attempts'], 1):
                    print(f"\nAttempt {i}:")
                    print(f"  Result: {attempt['test_result']}")

                    if attempt['debug_details'].get('full_error'):
                        print("  Full error:")
                        print(f"    {attempt['debug_details']['full_error']}")

                    if attempt['debug_details'].get('compilation_output'):
                        comp = attempt['debug_details']['compilation_output']
                        if comp['stderr']:
                            print("  Compilation stderr:")
                            print(f"    {comp['stderr']}")
            else:
                print(f"No debug info found for {module_name}")
        else:
            print("\nAvailable debug info for modules:")
            for module in self.debug_info:
                status = self.debug_info[module]['final_status']
                attempts = self.debug_info[module]['total_attempts']
                print(f"  {module}: {status} ({attempts} attempts)")
            print("\nUse show_debug_info('module_name') for detailed info")

    def run_full_pipeline(self):
        """Run complete generation and evaluation pipeline"""
        method_name = "MCTS-guided" if self.use_mcts else "Traditional retry"
        print(f"Starting RTLCoder -> VerilogEval pipeline with {method_name} mechanism...")
        if self.use_mcts:
            if self.enable_ppa:
                print(
                    f"Configuration: MCTS enabled with {self.mcts_simulations} simulations (early_stop={self.early_stop}, PPA=ON, rollout={self.rollout_mode})")
            else:
                print(
                    f"Configuration: MCTS enabled with {self.mcts_simulations} simulations (early_stop={self.early_stop}, PPA=OFF, rollout={self.rollout_mode})")
        else:
            print(f"Configuration: Max retries = {self.max_retries}")

        # 1. Setup environment
        self.setup_verilog_eval_environment()

        # 2. Load model
        self.load_model()

        # 3. Setup build environment
        self.setup_build_environment()

        # 4. Generate and test all tasks
        prompts = self.config.get_prompts()
        total_modules = len(prompts)
        print(f"\nStarting generation for {total_modules} modules...")
        print("=" * 80)

        for idx, (module_name, prompt) in enumerate(prompts.items(), 1):
            print(f"\nMODULE {idx}/{total_modules}: {module_name}")
            print("-" * 50)
            self.generate_with_retry(module_name, prompt)
            print("-" * 50)

        # 5. Print final statistics
        print("\n" + "=" * 80)
        self.print_final_statistics()

        # 6. Save statistics to file
        self.save_statistics_to_file()

        print(f"\nPipeline completed!")

        # Return success statistics for further analysis
        return self.success_stats

    def run_quick_test(self, module_names=None):
        """Quick test mode (test only specified modules)"""
        if module_names is None:
            module_names = self.modules

        method_name = "MCTS-guided" if self.use_mcts else "Traditional retry"
        total_modules = len(module_names)
        print(f"Quick test mode with {method_name} for {total_modules} modules: {module_names}")

        # Setup environment
        self.setup_verilog_eval_environment()
        self.load_model()
        self.setup_build_environment()

        # Generate only specified modules
        all_prompts = self.config.get_prompts()
        selected_prompts = {k: v for k, v in all_prompts.items() if k in module_names}

        print("=" * 60)
        for idx, (module_name, prompt) in enumerate(selected_prompts.items(), 1):
            print(f"\nMODULE {idx}/{total_modules}: {module_name}")
            print("-" * 40)
            self.generate_with_retry(module_name, prompt)
            print("-" * 40)

        print("\n" + "=" * 60)
        self.print_final_statistics()
        return self.success_stats

    def _clear_memory(self):
        """Force garbage collection and clear PyTorch cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




if __name__ == "__main__":
    random.seed(50)
    torch.manual_seed(50)

    # 启用PPA分析的评估器
    evaluator_ppa = RTLCoderMCTSEval(
        verilog_eval_path="~/RTL-Coder",
        use_mcts=True,
        mcts_simulations=1000,
        early_stop=True,
        debug=True,
        enable_ppa=True,
        rollout_mode='sampling'
    )

    print("Testing with PPA analysis enabled...")
    results_ppa = evaluator_ppa.run_quick_test([
            "hypotenuse",
            "sine_lut"
    ])

    print("\n === Generate finished ===")
    for module, stats in results_ppa.items():
        print(f"{module}: {stats['status']}")

