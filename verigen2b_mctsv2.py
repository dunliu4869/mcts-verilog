import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import random, torch
import uuid
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import math
import subprocess
import json
import time
from tqdm import trange
import matplotlib.pyplot as plt
from problems import all_problems



class MCTSNode:
    def __init__(self, state, prior=1.0, parent=None):
        self.state = state          # List of generated tokens
        self.prior = prior          # P(s,a) from LLM
        self.visit_count = 0
        self.value_sum = 0.0
        self.parent = parent
        self.children = {}          # action_token → MCTSNode

class MCTSSearcher:
    def __init__(
        self,
        model_name: str,
        hf_token: str = None,
        device: str = None,
        c_puct: float = 1.0,
        top_k_expand: int = 20,
        rollout_top_k: int = 50,
        rollout_temperature: float = 1.0,
        alpha_B: float = 50,
        area_ref: float = 100.0,
        delay_ref: float = 10.0,
        workspace_base: str = 'mcts_tmp',
    ):
        # initialization
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hf_token = hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token
        ).to(self.device)
        # eval mode
        self.model.eval()

        # MCTS hyperparameters
        self.c_puct = c_puct
        self.top_k_expand = top_k_expand        # Expansion phase Top-k
        self.rollout_top_k = rollout_top_k      # Rollout phase Top-k
        self.rollout_temperature = rollout_temperature
        self.alpha_B = alpha_B
        self.area_ref = area_ref
        self.delay_ref = delay_ref
        self.workspace_base = workspace_base

    @torch.no_grad()
    def get_priors(self, state_tokens):
        """Return the candidate tokens for the expansion phase and their prior probabilities."""
        input_ids = torch.tensor([state_tokens], device=self.device)
        logits = self.model(input_ids).logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k_expand)
        return topk_idx.tolist(), topk_probs.tolist()

    @torch.no_grad()
    def rollout_policy(
        self,
        state_tokens,
        prefix_len: int,
        max_len: int = 256,
        mode: str = 'greedy',
    ):
        """
        Rollout：Generate from the current state sample until the end
        mode='greedy' greedy sampling; otherwise, Top-k sampling
        """
        tokens = state_tokens.copy()
        for _ in range(max_len):
            input_ids = torch.tensor([tokens], device=self.device)
            logits = self.model(input_ids).logits[0, -1]
            if mode == 'greedy':
                # Greedy, choose the token with the highest probability
                next_token = int(logits.argmax())
            else:
                # Top-k sampling
                probs = F.softmax(logits / self.rollout_temperature, dim=-1)
                topk_probs, topk_idx = probs.topk(self.rollout_top_k)
                topk_probs = topk_probs / topk_probs.sum()
                idx = torch.multinomial(topk_probs, num_samples=1).item()
                next_token = int(topk_idx[idx])

            tokens.append(next_token)
            # --- NEW: Simplified and More Robust Termination Condition ---
            # also check if the token is an EOS token.
            if next_token == self.tokenizer.eos_token_id:
                break

            suffix = self.tokenizer.decode(tokens[prefix_len:], skip_special_tokens=True)
            if 'endmodule' in suffix:
                break

        return tokens

    def uct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Calculate UCT score: Q + U"""
        Q = child.value_sum / (1 + child.visit_count)
        U = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        return Q + U

    def select(self, node: MCTSNode) -> MCTSNode:
        """Selection：select from the root node to the unexpanded node"""
        while node.children:
            _, node = max(
                node.children.items(),
                key=lambda kv: self.uct_score(node, kv[1])
            )
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion：expand all candidate actions for the current node."""
        top_tokens, top_priors = self.get_priors(node.state)
        for tok, prior in zip(top_tokens, top_priors):
            if tok not in node.children:
                new_state = node.state + [tok]
                node.children[tok] = MCTSNode(new_state, prior=prior, parent=node)
        # Select the child with the highest prior probability for rollout.
        return max(node.children.values(), key=lambda n: n.prior)

    def compute_reward(self, generated_tokens, problem: dict, prefix_len: int):
        """
        1. Decode generated tokens -> write DUT (dut.v)
        2. Write testbench (sim.v) that `include "dut.v"`
        3. iverilog+vvp([dut.v, sim.v]) -> compile & functional check
        4. yosys script (only synthesize dut.v) -> area / (fake) delay
        5. reward = alpha_B + max(0, 1 - (area*delay)/(area_ref*delay_ref))
        """

        # create fresh workspace
        ws = os.path.join(self.workspace_base, str(uuid.uuid4()))
        os.makedirs(ws, exist_ok=True)
        dut_path = os.path.join(ws, 'dut.v')

        # # --- For debugging, always decode the full text ---
        full_generated_text_for_debug = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # --- logic to get the suffix ---
        suffix_tokens = generated_tokens[prefix_len:]
        suffix = self.tokenizer.decode(suffix_tokens, skip_special_tokens=True).strip()

        full_code_to_write = problem["prompt"] + "\n" + suffix
        if "endmodule" not in suffix:
            full_code_to_write += "\nendmodule"

        if not suffix or re.search(r"^\s*module\s+\w+", suffix, re.MULTILINE):
            print(
                f"--- EVALUATION FAILED: COMPILE_FAIL (Bad Suffix) ---\nCode:\n{full_generated_text_for_debug}\n----------------\n")
            return "COMPILE_FAIL", None, None, full_code_to_write

        with open(dut_path, 'w', encoding='utf-8') as f:
            f.write(full_code_to_write)

        # Get testbench from the problem definition
        with open(os.path.join(ws, 'sim.v'), 'w', encoding='utf-8') as f:
            f.write(problem["testbench"])

        try:
            # Compilation Check: Verify Both Return Code and Standard Error Output
            p1 = subprocess.run(['iverilog', '-g2005-sv', '-o', 'out.vvp', 'sim.v'], cwd=ws, capture_output=True, text=True, timeout=30)
            if p1.returncode != 0 or "error" in p1.stderr.lower():
                print(
                    f"--- EVALUATION FAILED: COMPILE_FAIL (iverilog) ---\nCode:\n{full_code_to_write}\nError:\n{p1.stderr}\n----------------\n")
                return "COMPILE_FAIL", None, None, full_code_to_write

            # Simulation check
            p2 = subprocess.run(['vvp', 'out.vvp'], cwd=ws, capture_output=True, text=True, timeout=30)
            if p2.returncode != 0:
                print(f"--- EVALUATION FAILED: SIM_FAIL (vvp) ---\nCode:\n{full_code_to_write}\n----------------\n")
                print("--- Verilog Testbench Output ---")
                print(p2.stdout)
                if p2.stderr:
                    print("--- Simulator Errors ---")
                    print(p2.stderr)
                print("--------------------------------")
                return "SIM_FAIL", None, None, full_code_to_write

            # Comprehensive Check: Simultaneously verify return codes and standard error output
            stat_json = 'stat.json'
            script_path = os.path.join(ws, 'script.ys')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(textwrap.dedent(f"""\
                            read_verilog dut.v
                            hierarchy -top {problem['module_name']}
                            synth
                            tee -o {stat_json} stat -json {problem['yosys_module_name']}
                            delay -max -json delay.json
                        """))

            p3 = subprocess.run(['yosys', '-s', 'script.ys'], cwd=ws, capture_output=True, text=True, timeout=30)
            if p3.returncode != 0 or "error" in p3.stderr.lower():
                print(f"DEBUG: yosys failed. stderr:\n{p3.stderr}")
                return "SYNTH_FAIL", None, None, full_code_to_write

        except subprocess.TimeoutExpired:
            # If any process times out, return a new failure status
            return "TIMEOUT_FAIL", None, None, full_code_to_write

        stat_path = os.path.join(ws, stat_json)
        if not os.path.exists(stat_path):
            return "SYNTH_FAIL", None, None, full_code_to_write

        try:
            with open(stat_path, 'r', encoding='utf-8') as jf:
                stats = json.load(jf)
            area = stats.get('modules', {}).get(problem['yosys_module_name'], {}).get('num_cells', 0)
            delay = 1.0
            if area == 0:
                print(
                    f"--- EVALUATION FAILED: EMPTY_FAIL (Area is 0) ---\nCode:\n{full_code_to_write}\n----------------\n")
                return "EMPTY_FAIL", None, None, full_code_to_write

            print(f"--- EVALUATION SUCCEEDED --- \nCode:\n{full_code_to_write}\nArea: {area}\n----------------\n")
            return "SUCCESS", area, delay, full_code_to_write

        except (json.JSONDecodeError, KeyError):
            print(f"--- EVALUATION FAILED: JSON_FAIL ---\nCode:\n{full_code_to_write}\n----------------\n")
            return "SYNTH_FAIL", None, None, full_code_to_write


    def backpropagate(self, path, reward: float):
        """Backpropagation：propagate reward back to update node statistics"""
        for n in path:
            n.visit_count += 1
            n.value_sum += reward

    def search(self, problem: dict,
               output_dir: str,
               n_sim: int = 100,
               rollout_mode: str = 'greedy',
               debug_print_interval: int = 2000):
        """
        MCTS process：
        1) Selection
        2) Expansion
        3) Rollout
        4) Reward
        5) Backpropagation
        """
        # Get prompt from problem and set it for the searcher instance
        self.prompt_str = problem["prompt"]
        input_ids = self.tokenizer.encode(self.prompt_str, add_special_tokens=False)
        seed_tokens = [self.tokenizer.bos_token_id] + input_ids
        prefix_len = len(seed_tokens)

        root = MCTSNode(seed_tokens)
        start_time = time.perf_counter()
        reward_history = []

        # --- Tracking the best solutions encountered ---
        best_reward_so_far = -float('inf')
        best_tokens_so_far = []
        best_code_so_far = ""  # Text for storing optimal code

        # --- Dynamic baseline variables, reset for each problem ---
        dynamic_area_ref = None
        dynamic_delay_ref = None

        with trange(n_sim, desc=f"MCTS for {problem['name']}") as t:
            for i in t:
                node = self.select(root)
                MAX_SEQUENCE_LENGTH = 256
                if len(node.state) >= MAX_SEQUENCE_LENGTH:
                    if node.parent:
                        rollout_tokens = self.rollout_policy(node.parent.state, prefix_len=prefix_len, mode=rollout_mode)
                    else:
                        rollout_tokens = self.rollout_policy(node.state, prefix_len=prefix_len, mode=rollout_mode)
                else:
                    if node.visit_count > 0:
                        node = self.expand(node)
                    rollout_tokens = self.rollout_policy(node.state, prefix_len=prefix_len, mode=rollout_mode)

                # Get evaluation status and data
                status, area, delay, evaluated_code = self.compute_reward(rollout_tokens, problem, prefix_len=prefix_len)



                # Dynamic reward calculation logic
                reward = 0.0  # Initialize reward for this step
                if status == "SUCCESS":
                    if dynamic_area_ref is None:  # This is the FIRST successful run
                        # Set the baseline for this experiment
                        dynamic_area_ref = area
                        dynamic_delay_ref = delay
                        # The reward for the first success is just the base reward
                        reward = self.alpha_B
                        t.write(
                            f"  Simulation #{i + 1}: First success! Baseline set: area={area}, delay={delay}. Reward={reward:.2f}")
                    else:  # This is a subsequent successful run
                        adp = area * delay
                        adp_ref = dynamic_area_ref * dynamic_delay_ref
                        ppa_score = 1.0 - (adp / adp_ref)
                        reward = self.alpha_B + ppa_score
                else:  # Any kind of failure
                    if status == "COMPILE_FAIL":
                        reward = -1.0
                    else:  # SIM_FAIL, SYNTH_FAIL, EMPTY_FAIL
                        reward = -0.1

                reward_history.append(reward)

                # If the code functions correctly and the score exceeds the current highest score
                if status == "SUCCESS" and reward > best_reward_so_far:
                    # Save this improved code, overwriting the old file
                    filename = f"{problem['name'].replace(' ', '_')}.v"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(evaluated_code)

                    # print information
                    t.write(f"  >>> New best code saved to {filepath} (Reward: {reward:.2f}, Area: {area})")

                    # Update the optimal code text for the final report
                    best_code_so_far = evaluated_code

                # Check if it's a new high score and update the record
                is_new_best = False
                if reward > best_reward_so_far:
                    is_new_best = True
                    best_reward_so_far = reward
                    best_tokens_so_far = rollout_tokens

                # Check if the print interval has been reached
                is_periodic_print = ((i + 1) % debug_print_interval == 0)

                # If it's a new high score or the print interval is reached, print a detailed report
                if is_new_best or is_periodic_print:
                    # Using t.write() can prevent disrupting the progress bar display.
                    t.write("\n" + "=" * 20 + f" REPORT (Simulation #{i + 1}) " + "=" * 20)

                    # Display different titles based on the triggering reason
                    if is_new_best:
                        # For a new best score, also announce the first success if it is one
                        if status == "SUCCESS" and dynamic_area_ref == area and dynamic_delay_ref == delay:
                            t.write(f"First success! Baseline set: area={area}, delay={delay}.")
                        t.write(f"New Best Reward Found: {reward:.2f}")
                    else:
                        t.write(
                            f"Status: Periodic Report. Current Reward: {reward:.2f} (Best so far: {best_reward_so_far:.2f})")

                    t.write(f"Evaluation Status: {status}")
                    if status == "SUCCESS":
                        t.write(f"Metrics: Area={area}, Delay={delay}")

                    t.write("--- Evaluated Code ---")
                    t.write(evaluated_code)
                    t.write("=" * 62 + "\n")

                # Construct the path from the leaf node back to the root
                path = []
                cur = node
                while cur:
                    path.append(cur)
                    cur = cur.parent
                # Update statistics for all nodes on the path
                self.backpropagate(path, reward)

        elapsed = time.perf_counter() - start_time
        print(f"MCTS for {problem['name']} completed {n_sim} simulations in {elapsed:.2f}s")

        # Directly return the recorded best token sequence
        print(f"Search complete. Best reward found during search: {best_reward_so_far:.2f}")
        return best_tokens_so_far, best_code_so_far, reward_history


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    # Folder for storing the final generated code
    output_dir = "generated_rtl"
    os.makedirs(output_dir, exist_ok=True)

    # MCTS searcher
    token = os.getenv('HF_TOKEN', None)
    searcher = MCTSSearcher(
        model_name='shailja/fine-tuned-codegen-2B-Verilog',
        hf_token=token,
        c_puct=1.38,
        top_k_expand=20,
        rollout_top_k=50,
        rollout_temperature=1.0,
        alpha_B=0.5
    )

    results = {}

    # Execute each test task in a loop
    for problem in all_problems:
        print(f"\n==============================================")
        print(f"Starting test for: {problem['name']}")
        print(f"==============================================")

        # Perform operations on a copy to avoid modifying the original problem definition
        current_problem = problem.copy()

        # Check if the current issue has declared dependencies
        if "depends_on" in current_problem:
            dependency_name = current_problem["depends_on"]

            # Check whether the dependencies have been successfully executed and generated the code
            if dependency_name in results and results[dependency_name]["status"] == "Success":
                # Retrieve the generated dependency code from the results dictionary
                dependency_code = results[dependency_name]["code"]

                # Inject the dependency code into the prompt template to generate the final prompt
                final_prompt = current_problem["prompt_template"].format(dependency_code=dependency_code)
                current_problem["prompt"] = final_prompt  # Update the prompt to be used for the current task

                print(f"Injecting generated code from '{dependency_name}' into the prompt.")
            else:
                print(
                    f"Skipping '{current_problem['name']}' because its dependency '{dependency_name}' failed or was not found.")
                results[current_problem['name']] = {"status": "Skipped (Dependency Failed)"}
                continue  # Skip the current task and proceed to the next one


            # 运行MCTS搜索，它会返回最佳的完整token序列
        final_tokens, final_code, reward_history = searcher.search(
            problem=current_problem,
            output_dir=output_dir,  # 传入路径
            n_sim=2000,
            rollout_mode='sample'
        )
        # try:
        #     prompt_ids = searcher.tokenizer.encode(current_problem["prompt"], add_special_tokens=False)
        #     prefix_len = len(prompt_ids) + 1  # +1 for the BOS token
        #
        #     implementation_tokens = final_tokens[prefix_len:]
        #     implementation_text = searcher.tokenizer.decode(implementation_tokens, skip_special_tokens=True).strip()
        #
        #     final_code = current_problem["prompt"] + "\n" + implementation_text
        #
        #     # 确保代码以 endmodule 结尾
        #     if 'endmodule' not in implementation_text:
        #         final_code += "\nendmodule"
        #
        #     # 将代码保存到文件中
        #     filename = f"{problem['name'].replace(' ', '_')}.v"
        #     filepath = os.path.join(output_dir, filename)
        #     with open(filepath, 'w', encoding='utf-8') as f:
        #         f.write(final_code)
        #     print(f"Successfully generated code saved to: {filepath}")
        #
        #     # 存储结果用于报告
        #     results[problem['name']] = {
        #         "status": "Success",
        #         "code": final_code,
        #         "best_reward": max(reward_history) if reward_history else "N/A"
        #     }

        if final_code:  # if any successful code is found
            print(f"Best code for {problem['name']} has been saved during the search.")
            results[problem['name']] = {
                "status": "Success",
                "code": final_code,  # Directly use the best code returned
                "best_reward": max(reward_history) if reward_history else "N/A"
            }
        else:
            print(f"No functional code found for {problem['name']} after {len(reward_history)} simulations.")
            results[problem['name']] = {
                "status": "Failed",
                "code": "N/A",
                "best_reward": "N/A"
            }
        if reward_history:
            plt.figure()  # Create a new figure
            plt.plot(range(1, len(reward_history) + 1), reward_history)
            plt.xlabel('Simulation #')
            plt.ylabel('Reward')
            plt.title(f"Reward vs. Simulation Count for {problem['name']}")
            plt.ylim(-1.5, 2.0)  # adjust based on alpha_b
            plt.tight_layout()

            # Use the task name to name the image file to avoid overwriting
            filename = f"reward_curve_{problem['name'].replace(' ', '_')}.png"
            plt.savefig(filename, dpi=300)
            plt.close()  # Close the figure to prevent continuous display in environments like Jupyter
            print(f"Saved reward curve to {filename}")
            # ---------------------------------------------

        # except Exception as e:
        #     print(f"An error occurred during MCTS for {problem['name']}: {e}")
        #     results[problem['name']] = {"status": f"Failed with error: {e}"}

    # print final report
    print("\n\n===== BATCH TEST SUMMARY =====")
    for name, result in results.items():
        print(f"\n----- Task: {name} -----")
        print(f"Status: {result['status']}")
        if result['status'] == 'Success':
            print(f"Final Reward: {result['best_reward']:.2f}")
            print("Generated Code:")
            print(result['code'])
    print("============================")


