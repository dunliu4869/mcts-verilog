#!/usr/bin/env python3
"""
将MCTS生成的RTL代码适配到VerilogEval评估框架 - 完整测试版本
"""

import os
import shutil
import subprocess
import sys
import glob
import time
from rtl_mcts_inference import RTLCoderMCTSEval
from rtl_configs import VerilogConfigs

class MCTSVerilogEvalAdapter:
    def __init__(self, verilogeval_path=".", 
                 mcts_simulations=100,
                 early_stop=True,
                 debug=False,
                 enable_ppa=False):
        self.verilogeval_path = os.path.expanduser(verilogeval_path)
        self.mcts_simulations = mcts_simulations

        # 初始化您的MCTS评估器
        self.mcts_evaluator = RTLCoderMCTSEval(
            verilog_eval_path=self.verilogeval_path,
            use_mcts=True,
            mcts_simulations=mcts_simulations,  # 可调整
            early_stop=early_stop,
            debug=debug,
            enable_ppa=enable_ppa
        )

        # 设置正确的数据集和构建路径
        self.mcts_evaluator.dataset_path = os.path.join(self.verilogeval_path, "dataset_spec-to-rtl")
        self.mcts_evaluator.build_path = os.path.join(self.verilogeval_path, "build_mcts_temp")
        
        # 创建临时构建目录
        os.makedirs(self.mcts_evaluator.build_path, exist_ok=True)

        # 加载模型
        print("Loading model...")
        self.mcts_evaluator.load_model()
        print("Model loaded successfully!\n")
        print(f"MCTS simulations set to: {mcts_simulations}\n")

    def generate_for_verilogeval(self, task="spec-to-rtl", 
                             num_samples=20, 
                             num_problems=None,
                             start_from=0):
        """为VerilogEval生成代码"""

        # 1. 创建build目录
        build_name = f"build_mcts_{task}_n{num_samples}_full"
        build_path = os.path.join(self.verilogeval_path, build_name)
        os.makedirs(build_path, exist_ok=True)

        # 2. 读取VerilogEval的问题
        dataset_path = os.path.join(self.verilogeval_path, f"dataset_{task}")

        # 从dataset目录中提取问题列表
        import glob
        prompt_files = glob.glob(os.path.join(dataset_path, "*_prompt.txt"))
        problems = []
        for pf in prompt_files:
            basename = os.path.basename(pf)
            if basename.endswith("_prompt.txt"):
                problem_name = basename[:-11]
                problems.append(problem_name)

        problems.sort()

        # 限制问题数量
        if num_problems is not None:
            problems = problems[start_from:start_from + num_problems]
        else:
            problems = problems[start_from:]

        print(f"Will generate code for {len(problems)} problems")
        print(f"Samples per problem: {num_samples}")
        print(f"Total samples to generate: {len(problems) * num_samples}")
        print("=" * 50)

        # 记录进度
        progress_file = os.path.join(build_path, "progress.txt")
        completed_problems = set()

        # 读取已完成的问题（断点续传）
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                completed_problems = set(line.strip() for line in f)
            print(f"Found {len(completed_problems)} already completed problems")

        # 3. 为每个问题生成代码
        start_time = time.time()

        for idx, problem in enumerate(problems, 1):
            # 跳过已完成的问题
            if problem in completed_problems:
                print(f"\n[{idx}/{len(problems)}] {problem} - Already completed, skipping...")
                continue

            problem_start_time = time.time()
            print(f"\n[{idx}/{len(problems)}] Generating for {problem}")

            prompt_file = os.path.join(dataset_path, f"{problem}_prompt.txt")

            if not os.path.exists(prompt_file):
                print(f"  Warning: {prompt_file} not found, skipping...")
                continue

            # 读取prompt
            with open(prompt_file, 'r') as f:
                prompt = f.read()

            # ===== 关键修复：在生成代码前复制测试文件到MCTS的build目录 =====
            mcts_problem_dir = os.path.join(self.mcts_evaluator.build_path, problem)
            os.makedirs(mcts_problem_dir, exist_ok=True)

            # 复制测试和参考文件到MCTS build目录
            for file_type in ["test.sv", "ref.sv"]:
                src = os.path.join(dataset_path, f"{problem}_{file_type}")
                dst = os.path.join(mcts_problem_dir, f"{problem}_{file_type}")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    if self.mcts_evaluator.debug:
                        print(f"  Copied {os.path.basename(src)} to MCTS build dir")
            # ================================================================

            # 创建输出目录（用于最终结果）
            output_dir = os.path.join(build_path, problem)
            os.makedirs(output_dir, exist_ok=True)

            # 同时也复制测试文件到最终输出目录（用于VerilogEval评估）
            for file_type in ["test.sv", "ref.sv"]:
                src = os.path.join(dataset_path, f"{problem}_{file_type}")
                dst = os.path.join(output_dir, f"{problem}_{file_type}")
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            # 生成多个样本
            successful_samples = 0
            for sample_id in range(1, num_samples + 1):
                sample_start = time.time()
                print(f"  Sample {sample_id}/{num_samples}...", end='', flush=True)

                try:
                    # 使用MCTS生成代码
                    code, reward, sims = self.mcts_evaluator.mcts_generate_with_stats(
                        problem, prompt,
                        n_sim=self.mcts_simulations
                    )

                    # 保存代码到最终输出目录
                    sample_file = os.path.join(output_dir, f"{problem}_sample{sample_id:02d}.sv")
                    with open(sample_file, 'w') as f:
                        if code:
                            f.write(code)
                            successful_samples += 1
                            print(f" ✓ (reward: {reward:.2f}, sims: {sims}, time: {time.time()-sample_start:.1f}s)")
                        else:
                            f.write(f"module TopModule();\nendmodule\n")
                            print(f" ✗ (no code generated, time: {time.time()-sample_start:.1f}s)")

                except Exception as e:
                    print(f" ✗ (error: {str(e)[:50]}...)")
                    sample_file = os.path.join(output_dir, f"{problem}_sample{sample_id:02d}.sv")
                    with open(sample_file, 'w') as f:
                        f.write(f"module TopModule();\nendmodule\n")

            # 记录完成的问题
            with open(progress_file, 'a') as f:
                f.write(f"{problem}\n")

            problem_time = time.time() - problem_start_time
            print(f"  Problem completed: {successful_samples}/{num_samples} successful samples in {problem_time:.1f}s")

            # 估算剩余时间
            elapsed_time = time.time() - start_time
            problems_done = idx - len(completed_problems)
            if problems_done > 0:
                avg_time_per_problem = elapsed_time / problems_done
                remaining_problems = len(problems) - idx
                estimated_remaining = avg_time_per_problem * remaining_problems
                print(f"  Estimated time remaining: {estimated_remaining/60:.1f} minutes")

        # 创建makefile配置文件
        self._create_makefiles(build_path, problems, num_samples)

        total_time = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Generation completed in {total_time/60:.1f} minutes")

        return build_path



    def _create_makefiles(self, build_path, problems, num_samples):
        """创建VerilogEval需要的makefile配置文件"""
        
        with open(os.path.join(build_path, "problems.mk"), 'w') as f:
            f.write("#=========================================================================\n")
            f.write("# problems.mk\n")
            f.write("#=========================================================================\n")
            f.write("problems = \\\n")
            for problem in problems:
                f.write(f"  {problem} \\\n")
            f.write("\n")
        
        with open(os.path.join(build_path, "samples.mk"), 'w') as f:
            f.write("#=========================================================================\n")
            f.write("# samples.mk\n")
            f.write("#=========================================================================\n")
            for problem in problems:
                f.write(f"{problem}_num_samples = {num_samples}\n")
            f.write("\n")
    
    def run_evaluation(self, build_path):
        """运行VerilogEval的评估"""
        
        print(f"\nRunning evaluation in {build_path}")
        
        makefile_in = os.path.join(self.verilogeval_path, "Makefile.in")
        if os.path.exists(makefile_in):
            shutil.copy2(makefile_in, os.path.join(build_path, "Makefile.in"))
        
        original_dir = os.getcwd()
        os.chdir(build_path)
        
        try:
            # 获取样本数
            with open("samples.mk", 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "_num_samples" in line:
                        num_samples = int(line.split("=")[1].strip())
                        break
            
            # 配置
            configure_cmd = [
                "../configure",
                "--with-task=spec-to-rtl",
                "--with-model=mcts",
                f"--with-samples={num_samples}"
            ]
            subprocess.run(configure_cmd, check=True)
            
            # 运行评估（使用更多并行进程）
            print("Running tests with iverilog...")
            subprocess.run(["make", "-j8"], check=True)
            
            # 读取结果
            if os.path.exists("summary.txt"):
                with open("summary.txt", 'r') as f:
                    results = f.read()
                    print("\n=== Evaluation Results ===")
                    print(results)
                    
                # 也保存一份到上级目录
                shutil.copy2("summary.txt", "../mcts_evaluation_results.txt")
            else:
                print("No summary.txt found")
                
        except subprocess.CalledProcessError as e:
            print(f"Error during evaluation: {e}")
        finally:
            os.chdir(original_dir)
    
    def run_full_test(self, num_samples=20, num_problems=None):
        """
        运行完整测试
        
        Args:
            num_samples: 每个问题的样本数（建议1或20）
            num_problems: 要测试的问题数量（None表示全部156个）
        """
        print("Starting MCTS-VerilogEval Full Integration Test")
        print("=" * 50)
        
        # 生成代码
        build_path = self.generate_for_verilogeval(
            task="spec-to-rtl",
            num_samples=num_samples,
            num_problems=num_problems
        )
        
        # 运行评估
        self.run_evaluation(build_path)
        
        print("\n" + "=" * 50)
        print("Full evaluation completed!")
        print(f"Results saved in: {build_path}/summary.txt")
        print(f"Also copied to: mcts_evaluation_results.txt")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCTS VerilogEval Adapter')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of samples per problem (1 or 20)')
    parser.add_argument('--problems', type=int, default=None,
                       help='Number of problems to test (default: all 156)')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='Number of MCTS simulations')
    parser.add_argument('--no-early-stop', action='store_true',
                       help='Disable early stopping in MCTS')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--enable-ppa', action='store_true',
                       help='Enable PPA analysis')
    
    args = parser.parse_args()
    
    # 创建适配器
    adapter = MCTSVerilogEvalAdapter(
        mcts_simulations=args.mcts_sims,
        early_stop=not args.no_early_stop,
        debug=args.debug,
        enable_ppa=args.enable_ppa
    )
    
    # 运行完整测试
    adapter.run_full_test(
        num_samples=args.samples,
        num_problems=args.problems
    )
