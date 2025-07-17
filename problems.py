import textwrap

# ==============================================================================
# FILE: problems.py
# This file contains the definitions for all Verilog generation problems.
# ==============================================================================


# --- Problem: 4-bit Adder (The Building Block) ---
# Dependency module for larger designs.
# The testbench has been upgraded to be exhaustive, testing all 512 possible input combinations.
adder4_problem = {
    "name": "4-bit Adder",
    "module_name": "adder4",
    "yosys_module_name": "\\adder4",
    "prompt": textwrap.dedent("""\
        // Design a 4-bit adder.
        // It has two 4-bit inputs, a and b, and a 1-bit carry-in, cin.
        // The outputs are a 4-bit value, sum, and a single carry-out bit, cout.
        module adder4(
            input  [3:0] a,
            input  [3:0] b,
            input        cin,
            output [3:0] sum,
            output       cout
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [3:0] a, b;
          reg       cin;
          wire [3:0] sum;
          wire      cout;
          integer   i, j, k;
          integer   error_count;
          reg [4:0] expected_result;

          adder4 UUT(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));

          initial begin
            error_count = 0;
            $display("Starting exhaustive test for 4-bit adder...");

            // Exhaustive test for all 2^(4+4+1) = 512 combinations.
            // This is the most robust way to verify a small module.
            for (i = 0; i < 16; i = i + 1) begin
              for (j = 0; j < 16; j = j + 1) begin
                for (k = 0; k < 2; k = k + 1) begin
                  a = i;
                  b = j;
                  cin = k;
                  expected_result = a + b + cin;
                  #10; // Allow time for combinatorial logic to settle.
                  if ({cout, sum} !== expected_result) begin
                    $display("TEST FAILED: a=%h, b=%h, cin=%b", a, b, cin);
                    $display("  GOT: {cout,sum} = %h, EXPECTED: %h", {cout, sum}, expected_result);
                    error_count = error_count + 1;
                  end
                end
              end
            end

            if (error_count == 0) begin
              $display("All 512 tests passed for 4-bit adder!");
            end else begin
              $fatal(1, "Found %0d errors in 4-bit adder test.", error_count);
            end

            $finish(0);
          end
        endmodule
        """)
}

# --- Problem 1: 8-bit Adder (Dependency - Must be defined first, as it is depended upon by other issues) ---
adder8_problem = {
    "name": "8-bit Adder",
    "module_name": "adder8",
    "yosys_module_name": "\\adder8",
    "prompt": textwrap.dedent("""\
        // Design an 8-bit adder.
        // It has two 8-bit inputs, a and b, and a 1-bit carry-in, cin.
        // The outputs are an 8-bit value, sum, and a single carry-out bit, cout.
        // The module should perform addition on the inputs to produce the sum and cout values.
        module adder8(
            input  [7:0] a,
            input  [7:0] b,
            input        cin,
            output [7:0] sum,
            output       cout
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [7:0] a, b; reg cin;
          wire [7:0] sum; wire cout;
          integer i, j, k, error_count;
          adder8 UUT(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
          initial begin
            error_count = 0;
            // Exhaustive test for all 2^(8+8+1) = 131,072 combinations.
            for (i = 0; i < 256; i = i + 1) for (j = 0; j < 256; j = j + 1) for (k = 0; k < 2; k = k + 1) begin
              a = i; b = j; cin = k; #10;
              if ({cout, sum} !== a + b + cin) begin
                $display("ERROR [8-bit Adder]: a=%h, b=%h, cin=%b -> GOT: %h, EXPECTED: %h", a, b, cin, {cout, sum}, a + b + cin);
                error_count = error_count + 1;
              end
            end
            if (error_count == 0) $display("All 131,072 tests passed for 8-bit adder!");
            else $fatal(1, "Found %0d errors in 8-bit adder test.", error_count);
            $finish(0);
          end
        endmodule
        """)
}

# --- Problem: 16-bit Adder (Modular Design) ---
# Depends on the 4-bit adder.
# Testbench focuses on inter-module carry propagation and other critical corner cases.
adder16_problem = {
    "name": "16-bit Adder (Modular)",
    "module_name": "adder16",
    "yosys_module_name": "\\adder16",
    "depends_on": "4-bit Adder",  # Declares dependency
    "prompt_template": textwrap.dedent("""\
        // Here is a pre-generated, optimized 4-bit adder module.
        // You can instantiate it in your design.
        // START of provided 4-bit adder context:
        {dependency_code}
        // END of provided 4-bit adder context.

        // Now, using the 'adder4' module, design a 16-bit adder named 'adder16'.
        // It should be composed of 4 instances of the 'adder4' module.
        module adder16(
            input  [15:0] a,
            input  [15:0] b,
            input         cin,
            output [15:0] sum,
            output        cout
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [15:0] a, b;
          reg        cin;
          wire [15:0] sum;
          wire       cout;
          integer    error_count;

          adder16 UUT(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));

          // Reusable task to check results and report errors.
          task check;
            input [15:0] cur_a;
            input [15:0] cur_b;
            input        cur_cin;
            input [16:0] expected;
            input string message;
            begin
              a = cur_a;
              b = cur_b;
              cin = cur_cin;
              #10;
              if ({cout, sum} !== expected) begin
                $display("-----------------------------------------");
                $display("TEST FAILED: %s", message);
                $display("  INPUTS:  a=0x%h, b=0x%h, cin=%b", a, b, cin);
                $display("  RESULTS: GOT=0x%h, EXPECTED=0x%h", {cout, sum}, expected);
                $display("-----------------------------------------");
                error_count = error_count + 1;
              end
            end
          endtask

          initial begin
            error_count = 0;
            $display("Starting corner-case test for 16-bit modular adder...");

            // Test Group 1: Zero and Max Value Corner Cases
            check(16'h0000, 16'h0000, 1'b0, 17'h00000, "Zero Add");
            check(16'h0000, 16'h0000, 1'b1, 17'h00001, "Zero Add with Cin");
            check(16'hFFFF, 16'h0001, 1'b0, 17'h10000, "Max Value Carry Out");
            check(16'hFFFF, 16'hFFFF, 1'b0, 17'h1FFFE, "Max Values, No Cin");
            check(16'hFFFF, 16'hFFFF, 1'b1, 17'h1FFFF, "Max Values with Cin");

            // Test Group 2: Carry Propagation Across All Module Boundaries (CRITICAL TEST)
            check(16'h000F, 16'h0001, 1'b0, 17'h00010, "Carry propagates across 1st boundary (bit 3->4)");
            check(16'h00FF, 16'h0001, 1'b0, 17'h00100, "Carry propagates across 2nd boundary (bit 7->8)");
            check(16'h0FFF, 16'h0001, 1'b0, 17'h01000, "Carry propagates across 3rd boundary (bit 11->12)");
            check(16'hFFFF, 16'h0001, 1'b0, 17'h10000, "Full carry propagation from LSB to Cout");
            check(16'h00FF, 16'hFF01, 1'b0, 17'h10000, "Complex full carry propagation");

            // Test Group 3: Alternating Bit Patterns
            check(16'hAAAA, 16'h5555, 1'b0, 17'h0FFFF, "Alternating Bits (AAAA+5555)");
            check(16'hAAAA, 16'h5555, 1'b1, 17'h10000, "Alternating Bits with Cin");

            // Test Group 4: Regression tests from original file
            check(16'd1000, 16'd2000, 1'b1, 17'd3001,  "Regression Test 1 (1000+2000+1)");
            check(16'hABCD, 16'h1234, 1'b1, 17'h0BE02, "Regression Test 2 (ABCD+1234+1)");

            if (error_count == 0) begin
              $display("All tests passed for 16-bit adder!");
            end else begin
              $fatal(1, "Found %0d errors in 16-bit adder test.", error_count);
            end

            $finish(0);
          end
        endmodule
        """)
}


# --- Problem: 32-bit Adder (Standalone, Non-Modular) ---
# Testbench uses critical corner cases and large random numbers.
adder32_problem = {
    "name": "32-bit Adder",
    "module_name": "adder32",
    "yosys_module_name": "\\adder32",
    "prompt": textwrap.dedent("""\
        // Design a 32-bit adder.
        module adder32(input [31:0] a, input [31:0] b, input cin, output [31:0] sum, output cout);"""),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [31:0] a, b; reg cin;
          wire [31:0] sum; wire cout;
          integer error_count;
          adder32 UUT(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));

          task check;
            input [31:0] cur_a, cur_b; input cur_cin; input string message;
            begin
              a = cur_a; b = cur_b; cin = cur_cin; #10;
              if ({cout, sum} !== a + b + cin) begin
                $display("-----------------------------------------");
                $display("TEST FAILED [32-bit Adder]: %s", message);
                $display("  INPUTS:  a=0x%h, b=0x%h, cin=%b", a, b, cin);
                $display("  RESULTS: GOT=0x%h, EXPECTED=0x%h", {cout, sum}, a + b + cin);
                $display("-----------------------------------------");
                error_count = error_count + 1;
              end
            end
          endtask

          initial begin
            error_count = 0;
            // Test Group 1: Corner Cases
            check(32'd0, 32'd0, 1'b0, "Zero Add");
            check(32'hFFFF_FFFF, 32'd1, 1'b0, "Max Value Carry Out");
            check(32'hFFFF_FFFF, 32'hFFFF_FFFF, 1'b1, "Max Values with Cin");

            // Test Group 2: Carry Propagation
            check(32'h0000_FFFF, 32'd1, 1'b0, "Carry propagates past bit 15");
            check(32'hFFFF_FFFF, 32'd1, 1'b0, "Full carry propagation");

            // Test Group 3: Alternating Bit Patterns
            check(32'hAAAA_AAAA, 32'h5555_5555, 1'b0, "Alternating Bits (no cin)");
            check(32'hAAAA_AAAA, 32'h5555_5555, 1'b1, "Alternating Bits (with cin)");

            // Test Group 4: Random Large Numbers
            for (int i = 0; i < 100; i=i+1) begin
                check($random, $random, $random % 2, "Randomized Test");
            end

            if (error_count == 0) $display("All tests passed for 32-bit adder!");
            else $fatal(1, "Found %0d errors in 32-bit adder test.", error_count);
            $finish(0);
          end
        endmodule
        """)
}


# --- Target: 64-bit Adder, which USES the 8-bit adder ---
adder64_problem = {
    "name": "64-bit Adder (Modular)",
    "module_name": "adder64",
    "yosys_module_name": "\\adder64",
    # Statement of dependency on the "8-bit Adder" issue
    "depends_on": "8-bit Adder",
    # Using prompt templates with placeholders
    "prompt_template": textwrap.dedent("""\
        // Here is a pre-generated, optimized 8-bit adder module.
        // You can instantiate it in your design.
        // START of provided 8-bit adder context:
        {dependency_code}
        // END of provided 8-bit adder context.

        // Now, using the 'adder8' module, design a 64-bit adder named 'adder64'.
        // It should be composed of eight instances of the 'adder8' module.
        module adder64(
            input  [63:0] a,
            input  [63:0] b,
            input         cin,
            output [63:0] sum,
            output        cout
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [63:0] a, b; reg cin;
          wire [63:0] sum; wire cout;
          integer error_count;
          adder64 UUT(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
          
          task check;
            input [63:0] cur_a, cur_b; input cur_cin; input string message;
            begin
              a = cur_a; b = cur_b; cin = cur_cin; #10;
              if ({cout, sum} !== a + b + cin) begin
                $display("-----------------------------------------");
                $display("TEST FAILED [64-bit Adder]: %s", message);
                $display("  INPUTS:  a=0x%h, b=0x%h, cin=%b", a, b, cin);
                $display("  RESULTS: GOT=0x%h, EXPECTED=0x%h", {cout, sum}, a + b + cin);
                $display("-----------------------------------------");
                error_count = error_count + 1;
              end
            end
          endtask

          initial begin
            error_count = 0;
            // Test Group 1: Corner Cases
            check(64'd0, 64'd0, 1'b0, "Zero Add");
            check(64'hFFFF_FFFF_FFFF_FFFF, 64'd1, 1'b0, "Max Value Carry Out");
            check(64'hFFFF_FFFF_FFFF_FFFF, 64'hFFFF_FFFF_FFFF_FFFF, 1'b1, "Max Values with Cin");

            // Test Group 2: Carry Propagation Across All 8-bit Boundaries (CRITICAL)
            check(64'h00000000_000000FF, 64'd1, 1'b0, "Carry propagates past bit 7");
            check(64'h00000000_0000FFFF, 64'd1, 1'b0, "Carry propagates past bit 15");
            check(64'h00000000_FFFFFF, 64'd1, 1'b0, "Carry propagates past bit 23");
            check(64'h00000000_FFFFFFFF, 64'd1, 1'b0, "Carry propagates past bit 31");
            check(64'h0000FFFF_FFFFFFFF, 64'd1, 1'b0, "Carry propagates past bit 47");
            check(64'hFFFFFFFF_FFFFFFFF, 64'd1, 1'b0, "Full carry propagation");

            // Test Group 3: Alternating Bit Patterns
            check(64'hAAAA_AAAA_AAAA_AAAA, 64'h5555_5555_5555_5555, 1'b0, "Alternating Bits (no cin)");
            check(64'hAAAA_AAAA_AAAA_AAAA, 64'h5555_5555_5555_5555, 1'b1, "Alternating Bits (with cin)");

            // Test Group 4: Random Large Numbers
            check(64'hDEAD_BEEF_CAFE_F00D, 64'h1234_5678_90AB_CDEF, 1'b1, "Random large numbers");

            if (error_count == 0) $display("All tests passed for 64-bit adder!");
            else $fatal(1, "Found %0d errors in 64-bit adder test.", error_count);
            $finish(0);
          end
        endmodule
        """)
}



# --- Problem: 4-bit Multiplier ---
multiplier4_problem = {
    "name": "4-bit Multiplier",
    "module_name": "multiplier4",
    "yosys_module_name": "\\multiplier4",
    "prompt": textwrap.dedent("""\
        // Design a 4-bit multiplier.
        module multiplier4(input [3:0] a, input [3:0] b, output [7:0] p);"""),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [3:0] a, b; wire [7:0] p; integer i, j, error_count;
          multiplier4 UUT(.a(a), .b(b), .p(p));
          initial begin
            error_count = 0;
            for (i = 0; i < 16; i = i + 1) for (j = 0; j < 16; j = j + 1) begin
              a = i; b = j; #10;
              if (p !== a * b) begin
                $display("ERROR [4-bit Multiplier]: a=%d, b=%d -> GOT: %d, EXPECTED: %d", a, b, p, a * b);
                error_count = error_count + 1;
              end
            end
            if (error_count == 0) $display("All 256 tests passed for 4-bit multiplier!");
            else $fatal(1, "Found %0d errors.", error_count);
            $finish(0);
          end
        endmodule
        """)
}

# --- Problem: 8-bit Multiplier ---
multiplier8_problem = {
    "name": "8-bit Multiplier",
    "module_name": "multiplier8",
    "yosys_module_name": "\\multiplier8",
    "prompt": textwrap.dedent("""\
        // Design an 8-bit multiplier.
        module multiplier8(input [7:0] a, input [7:0] b, output [15:0] p);"""),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [7:0] a, b; wire [15:0] p; integer i, j, error_count;
          multiplier8 UUT(.a(a), .b(b), .p(p));
          initial begin
            error_count = 0;
            for (i = 0; i < 256; i = i + 1) for (j = 0; j < 256; j = j + 1) begin
              a = i; b = j; #10;
              if (p !== a * b) begin
                $display("ERROR [8-bit Multiplier]: a=%d, b=%d -> GOT: %d, EXPECTED: %d", a, b, p, a * b);
                error_count = error_count + 1;
              end
            end
            if (error_count == 0) $display("All 65,536 tests passed for 8-bit multiplier!");
            else $fatal(1, "Found %0d errors.", error_count);
            $finish(0);
          end
        endmodule
        """)
}

# --- Problem: 16-bit Multiplier ---
multiplier16_problem = {
    "name": "16-bit Multiplier",
    "module_name": "multiplier16",
    "yosys_module_name": "\\multiplier16",
    "prompt": textwrap.dedent("""\
        // Design a 16-bit multiplier.
        module multiplier16(input [15:0] a, input [15:0] b, output [31:0] p);"""),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [15:0] a, b; wire [31:0] p; integer error_count;
          multiplier16 UUT(.a(a), .b(b), .p(p));
          task check; input [15:0] ca, cb; input string msg; begin
            a=ca; b=cb; #10; if(p !== a*b) begin
              $display("ERROR [16-bit]: %s (a=%h, b=%h)", msg, a, b); error_count=error_count+1; end
          end endtask
          initial begin
            error_count = 0;
            check(16'd0, 16'hFFFF, "Multiply by zero");
            check(16'hFFFF, 16'd0, "Multiply by zero");
            check(16'd1, 16'hABCD, "Multiply by one");
            check(16'hFFFF, 16'hFFFF, "Max values");
            for (int i=0; i<500; i=i+1) check($random, $random, "Randomized test");
            if (error_count == 0) $display("All tests passed for 16-bit multiplier!");
            else $fatal(1, "Found %0d errors.", error_count);
            $finish(0);
          end
        endmodule
        """)
}

# --- Problem: 32-bit Multiplier ---
multiplier32_problem = {
    "name": "32-bit Multiplier",
    "module_name": "multiplier32",
    "yosys_module_name": "\\multiplier32",
    "prompt": textwrap.dedent("""\
        // Design a 32-bit multiplier.
        module multiplier32(input [31:0] a, input [31:0] b, output [63:0] p);"""),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [31:0] a, b; wire [63:0] p; integer error_count;
          multiplier32 UUT(.a(a), .b(b), .p(p));
          task check; input [31:0] ca, cb; input string msg; begin
            a=ca; b=cb; #10; if(p !== a*b) begin
              $display("ERROR [32-bit]: %s (a=%h, b=%h)", msg, a, b); error_count=error_count+1; end
          end endtask
          initial begin
            error_count = 0;
            check(32'd0, 32'hFFFF_FFFF, "Multiply by zero");
            check(32'd1, 32'hDEAD_BEEF, "Multiply by one");
            check(32'hFFFF_FFFF, 32'hFFFF_FFFF, "Max values");
            for (int i=0; i<1000; i=i+1) check($random, $random, "Randomized test");
            if (error_count == 0) $display("All tests passed for 32-bit multiplier!");
            else $fatal(1, "Found %0d errors.", error_count);
            $finish(0);
          end
        endmodule
        """)
}

# --- Problem: 64-bit Multiplier ---
multiplier64_problem = {
    "name": "64-bit Multiplier",
    "module_name": "multiplier64",
    "yosys_module_name": "\\multiplier64",
    "prompt": textwrap.dedent("""\
        // Design a 64-bit multiplier.
        module multiplier64(input [63:0] a, input [63:0] b, output [127:0] p);"""),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        module tb;
          reg [63:0] a, b; wire [127:0] p; integer error_count;
          multiplier64 UUT(.a(a), .b(b), .p(p));
          task check; input [63:0] ca, cb; input string msg; begin
            a=ca; b=cb; #10; if(p !== a*b) begin
              $display("ERROR [64-bit]: %s (a=%h, b=%h)", msg, a, b); error_count=error_count+1; end
          end endtask
          initial begin
            error_count = 0;
            check(64'd0, -1, "Multiply by zero");
            check(1, 64'hDEAD_BEEF_CAFE_F00D, "Multiply by one");
            check(-1, -1, "Max values");
            for (int i=0; i<2000; i=i+1) check({$random, $random}, {$random, $random}, "Randomized test");
            if (error_count == 0) $display("All tests passed for 64-bit multiplier!");
            else $fatal(1, "Found %0d errors.", error_count);
            $finish(0);
          end
        endmodule
        """)
}

# ---  MAC units ---
# --- Problem: 4-bit MAC Unit ---
mac4_problem = {
    "name": "4-bit MAC Unit",
    "module_name": "mac4",
    "yosys_module_name": "\\mac4",
    "prompt": textwrap.dedent("""\
        // Design a 4-bit Multiply-Accumulate (MAC) unit with a synchronous, active-high reset.
        //
        // On the positive edge of the clock (clk), the module should behave as follows:
        // - If 'reset' is high (1), the output accumulator 'p' should be cleared to 0.
        // - If 'reset' is low (0), 'p' should be updated with the value of: p + (a * b).
        //
        // Module: mac4
        // Inputs:
        //   - clk:   1-bit clock signal
        //   - reset: 1-bit active-high reset
        //   - a:     4-bit input
        //   - b:     4-bit input
        // Outputs:
        //   - p:     8-bit registered output accumulator
        //
        // All state changes should occur on the rising edge of the clock.
        // The implementation should be synthesizable Verilog.
        module mac4(
            input        clk,
            input        reset,
            input  [3:0] a,
            input  [3:0] b,
            output reg [7:0] p
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"

        module tb;
          localparam WIDTH = 4;
          localparam OUTW  = 2 * WIDTH;

          reg clk, reset;
          reg [WIDTH-1:0] a, b;
          wire [OUTW-1:0] p;
          reg [OUTW-1:0] expected;
          integer errors;

          mac4 UUT(.clk(clk), .reset(reset), .a(a), .b(b), .p(p));

          initial clk = 0;
          always #5 clk = ~clk;

          task check;
            input string message;
            #1; // Delay for stable signal checking
            if (p !== expected) begin
              $error("TEST FAILED [%s]: a=%h, b=%h => GOT: %h, EXPECTED: %h", message, a, b, p, expected);
              errors = errors + 1;
            end
          endtask

          initial begin
            errors = 0; expected = 0;

            // --- 1. Synchronous Reset Test ---
            $display("-> Performing Synchronous Reset Test...");
            reset = 1;
            a = 'X; // 在复位期间将输入设为不定态
            b = 'X;
            @(posedge clk);
            check("During Reset"); // 检查p是否为0
            reset = 0;
            a = 0; // <-- 在下一个时钟沿前，为a提供确定值
            b = 0; // <-- 在下一个时钟沿前，为b提供确定值
            @(posedge clk);
            check("After de-asserting reset");
            $display("-> Synchronous Reset Test Passed.");

            // --- 2. Exhaustive Functional Test ---
            $display("-> Starting exhaustive test...");
            for (integer i = 0; i < (1<<WIDTH); i = i + 1) begin
              for (integer j = 0; j < (1<<WIDTH); j = j + 1) begin
                a = i; b = j;
                expected = expected + a * b;
                @(posedge clk);
                check($sformatf("Exhaustive test i=%0d, j=%0d", i, j));
              end
            end
            $display("-> Exhaustive Test Passed.");
            
            // --- 3. Asynchronous Reset Trap Test ---
            $display("-> Performing Asynchronous Reset Trap Test...");
            a = 0;
            b = 0;
            // The goal is to pulse reset *between* clock edges.
            // A synchronous design should ignore this pulse. An asynchronous one will fail.
            expected = expected + (a * b);
            @(posedge clk);
            check("State after neutralizing inputs");
            #1; // Move slightly past the clock edge
            reset = 1;
            #3; // Pulse width is 3ns, less than the 5ns to the next edge
            reset = 0;
            // 'expected' should NOT change, as a sync reset would be ignored.
            // 'p' should also not have changed.
            check("State immediately after reset pulse"); // Check if p has been incorrectly zeroed out
            // Prepare for the next clock cycle; although a and b are 0, the expected value remains unchanged
            expected = expected + (a*b);
            @(posedge clk);
            check("State one cycle after reset pulse"); // Double-check that p is not affected by the pulse
            $display("-> Asynchronous Reset Trap Test Passed (pulse was correctly ignored).");
            
            // --- 4. Final Result ---
            if (errors == 0) $display("PASS: All tests passed for mac4!");
            else $fatal(1, "FAIL: Found %0d errors in mac4 test.", errors);
            $finish;
          end
        endmodule
        """)
}

# --- Problem: 8-bit MAC Unit ---
# --- Problem: 8-bit MAC Unit (Corrected) ---
mac8_problem = {
    "name": "8-bit MAC Unit",
    "module_name": "mac8",
    "yosys_module_name": "\\mac8",
    "prompt": textwrap.dedent("""\
        // Design an 8-bit Multiply-Accumulate (MAC) unit with a synchronous, active-high reset.
        //
        // On the positive edge of the clock (clk), the module must behave as follows:
        // - If 'reset' is high (1), the output accumulator 'p' should be cleared to 0.
        // - If 'reset' is low (0), 'p' must be updated with the value of: p + (a * b).
        //
        // All state changes should occur on the rising edge of the clock.
        // The implementation should be synthesizable Verilog.
        module mac8(
            input         clk,
            input         reset,
            input   [7:0] a,
            input   [7:0] b,
            output reg [15:0] p
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"
        
        module tb_mac8;
          // 参数定义
          localparam WIDTH = 8;
          localparam OUTW  = 2*WIDTH;
        
          // 信号声明
          reg               clk;
          reg               reset;
          reg  [WIDTH-1:0]  a, b;
          wire [OUTW-1:0]   p;
          reg  [OUTW-1:0]   expected;
          reg  [OUTW-1:0]   p_posedge;  // 存储上一个 posedge 的 p
          integer           errors;
        
          // DUT 实例
          mac8 UUT (
            .clk   (clk),
            .reset (reset),
            .a     (a),
            .b     (b),
            .p     (p)
          );
        
          // 时钟生成：周期 10ns
          initial clk = 0;
          always #5 clk = ~clk;
        
          // ====== 关键：时序边沿对比 ======
          // 在 posedge 时存下 p
          always @(posedge clk) begin
            p_posedge <= p;
          end
        
          // 在 negedge 时，先等待一个极小的延时，再对比
          always @(negedge clk) begin
            #1; // 增加 #1 延时以避免竞争冒险
            if (p !== p_posedge) begin
              $error("NEGEDGE ERROR: p changed between edges! posedge p=0x%h, now p=0x%h", p_posedge, p);
              errors = errors + 1;
            end
          end
        
          // ====== 功能检查任务 ======
          task check(input string msg);
            begin
              // 等待 1 个时间步，保证 posedge 后信号稳定
              #1;
              if (p !== expected) begin
                $error("TEST FAILED [%s]: a=%h, b=%h → GOT p=%h, EXPECTED p=%h",
                       msg, a, b, p, expected);
                errors = errors + 1;
              end
            end
          endtask
        
          // ====== 测试流程 ======
          initial begin
            errors   = 0;
            expected = 0;
        
            // 1) 同步复位测试
            $display("-> Sync Reset Test");
            reset = 1; a = 'bx; b = 'bx;
            @(posedge clk);
            expected = 0;
            check("During Reset");
            reset = 0; a = 0; b = 0;
            @(posedge clk);
            expected = 0;
            check("After Reset Release");
            $display("   Sync Reset Passed\n");
        
            // 2) 边界情况测试
            $display("-> Corner Cases");
            a = 8'h00; b = 8'hFF; expected = expected + a*b; @(posedge clk); check("0 x FF");
            a = 8'hFF; b = 8'h01; expected = expected + a*b; @(posedge clk); check("FF x 1");
            a = 8'hFF; b = 8'hFF; expected = expected + a*b; @(posedge clk); check("FF x FF");
            $display("   Corner Cases Passed\n");
        
            // 3) 随机功能测试
            $display("-> Random Tests");
            for (integer i = 0; i < 1000; i = i + 1) begin
              a        = $urandom & 8'hFF;
              b        = $urandom & 8'hFF;
              expected = expected + a*b;
              @(posedge clk);
              check($sformatf("Random #%0d", i));
            end
            $display("   Random Tests Passed\n");
        
            // 4) 异步复位脉冲陷阱
            $display("-> Async Reset Trap");
            a        = 0; b = 0;
            expected = expected + a*b;
            @(posedge clk); check("Pre-async-reset");
            #1; reset = 1;
            #3; reset = 0;
            check("During-async-reset");
            @(posedge clk);
            expected = expected + a*b;
            check("Post-async-reset");
            $display("   Async Reset Trap Passed\n");
        
            // 5) 最终结果
            if (errors == 0) begin
              $display("PASS: All tests passed for mac8!");
            end else begin
              $fatal(1, "FAIL: %0d errors detected in mac8 testbench.", errors);
            end
        
            $finish;
          end
        
        endmodule
        """)
}

# --- Problem: 16-bit MAC Unit ---
# --- Problem: 16-bit MAC Unit (Corrected) ---
mac16_problem = {
    "name": "16-bit MAC Unit",
    "module_name": "mac16",
    "yosys_module_name": "\\mac16",
    "prompt": textwrap.dedent("""\
        // Design a 16-bit Multiply-Accumulate (MAC) unit with a synchronous, active-high reset.
        //
        // On the positive edge of the clock (clk), the module must behave as follows:
        // - If 'reset' is high (1), the output accumulator 'p' should be cleared to 0.
        // - If 'reset' is low (0), 'p' must be updated with the value of: p + (a * b).
        //
        // All state changes should occur on the rising edge of the clock.
        // The implementation should be synthesizable Verilog.
        module mac16(
            input          clk,
            input          reset,
            input   [15:0] a,
            input   [15:0] b,
            output reg [31:0] p
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"

        module tb;
          localparam WIDTH = 16;
          localparam OUTW  = 2 * WIDTH;

          reg clk, reset;
          reg [WIDTH-1:0] a, b;
          wire [OUTW-1:0] p;
          reg [OUTW-1:0] expected;
          integer errors;

          mac16 UUT(.clk(clk), .reset(reset), .a(a), .b(b), .p(p));

          initial clk = 0;
          always #5 clk = ~clk;

          task check;
            input string message;
            #1;
            if (p !== expected) begin
              $error("TEST FAILED [%s]: a=%h, b=%h => GOT: %h, EXPECTED: %h", message, a, b, p, expected);
              errors = errors + 1;
            end
          endtask

          initial begin
            errors = 0; expected = 0;

            // --- 1. Synchronous Reset Test ---
            $display("-> Performing Synchronous Reset Test...");
            reset = 1;
            a = 'X; 
            b = 'X;
            @(posedge clk);
            check("During Reset");
    
            reset = 0;
            a = 0; 
            b = 0; 
            @(posedge clk); 
            check("First cycle after reset"); 
            $display("-> Synchronous Reset Test Passed.");

            // --- 2. Corner Case Tests ---
            $display("-> Performing Corner Case Tests...");
            a = 16'h0; b = 16'hFFFF; expected = expected + a*b; @(posedge clk); check("Corner: 0 x Max");
            a = 16'hFFFF; b = 16'h1; expected = expected + a*b; @(posedge clk); check("Corner: Max x 1");
            a = 16'hFFFF; b = 16'hFFFF; expected = expected + a*b; @(posedge clk); check("Corner: Max x Max");
            $display("-> Corner Case Tests Passed.");
            
            // --- 3. Randomized Functional Test ---
            $display("-> Starting 2000 randomized tests...");
            for (integer i = 0; i < 2000; i = i + 1) begin
              a = $urandom & 16'hFF; $urandom & 16'hFF;
              expected = expected + a * b;
              @(posedge clk);
              check($sformatf("Random test #%0d", i));
            end
            $display("-> Randomized Tests Passed.");

            // --- 4. Asynchronous Reset Trap Test ---
            $display("-> Performing Asynchronous Reset Trap Test...");
            a = 0;
            b = 0;
            // The goal is to pulse reset *between* clock edges.
            // A synchronous design should ignore this pulse. An asynchronous one will fail.
            expected = expected + (a * b);
            @(posedge clk);
            check("State after neutralizing inputs");
            #1; // Move slightly past the clock edge
            reset = 1;
            #3; // Pulse width is 3ns, less than the 5ns to the next edge
            reset = 0;
            // 'expected' should NOT change, as a sync reset would be ignored.
            // 'p' should also not have changed.
            check("State immediately after reset pulse"); // Check if p has been incorrectly zeroed out
            // Prepare for the next clock cycle; although a and b are 0, the expected value remains unchanged
            expected = expected + (a*b);
            @(posedge clk);
            check("State one cycle after reset pulse"); // Double-check that p is not affected by the pulse
            $display("-> Asynchronous Reset Trap Test Passed (pulse was correctly ignored).");

            // --- 5. Final Result ---
            if (errors == 0) $display("PASS: All tests passed for mac16!");
            else $fatal(1, "FAIL: Found %0d errors in mac16 test.", errors);
            $finish;
          end
        endmodule
        """)
}

# --- Problem: 32-bit MAC Unit ---
# --- Problem: 32-bit MAC Unit (Corrected) ---
mac32_problem = {
    "name": "32-bit MAC Unit",
    "module_name": "mac32",
    "yosys_module_name": "\\mac32",
    "prompt": textwrap.dedent("""\
        // Design a 32-bit Multiply-Accumulate (MAC) unit with a synchronous, active-high reset.
        //
        // On the positive edge of the clock (clk), the module must behave as follows:
        // - If 'reset' is high (1), the output accumulator 'p' should be cleared to 0.
        // - If 'reset' is low (0), 'p' must be updated with the value of: p + (a * b).
        //
        // All state changes should occur on the rising edge of the clock.
        // The implementation should be synthesizable Verilog.
        module mac32(
            input          clk,
            input          reset,
            input   [31:0] a,
            input   [31:0] b,
            output reg [63:0] p
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"

        module tb;
          localparam WIDTH = 32;
          localparam OUTW  = 2 * WIDTH;

          reg clk, reset;
          reg [WIDTH-1:0] a, b;
          wire [OUTW-1:0] p;
          reg [OUTW-1:0] expected;
          integer errors;

          mac32 UUT(.clk(clk), .reset(reset), .a(a), .b(b), .p(p));

          initial clk = 0;
          always #5 clk = ~clk;

          task check;
            input string message;
            #1;
            if (p !== expected) begin
              $error("TEST FAILED [%s]: a=%h, b=%h => GOT: %h, EXPECTED: %h", message, a, b, p, expected);
              errors = errors + 1;
            end
          endtask

          initial begin
            errors = 0; expected = 0;

            // --- 1. Synchronous Reset Test ---
            $display("-> Performing Synchronous Reset Test...");
            reset = 1;
            a = 'X; 
            b = 'X;
            @(posedge clk);
            check("During Reset");
    
            reset = 0;
            a = 0; 
            b = 0;
            @(posedge clk); 
            check("First cycle after reset"); 
            $display("-> Synchronous Reset Test Passed.");

            // --- 2. Corner Case Tests ---
            $display("-> Performing Corner Case Tests...");
            a = 32'h0; b = 32'hFFFFFFFF; expected = expected + a*b; @(posedge clk); check("Corner: 0 x Max");
            a = 32'hFFFFFFFF; b = 32'h1; expected = expected + a*b; @(posedge clk); check("Corner: Max x 1");
            $display("-> Corner Case Tests Passed.");
            
            // --- 3. Randomized Functional Test ---
            $display("-> Starting 5000 randomized tests...");
            for (integer i = 0; i < 5000; i = i + 1) begin
              a = $urandom;
              b = $urandom;
              expected = expected + a * b;
              @(posedge clk);
              check($sformatf("Random test #%0d", i));
            end
            $display("-> Randomized Tests Passed.");

            // --- 4. Asynchronous Reset Trap Test ---
            $display("-> Performing Asynchronous Reset Trap Test...");
            a = 0;
            b = 0;
            // The goal is to pulse reset *between* clock edges.
            // A synchronous design should ignore this pulse. An asynchronous one will fail.
            expected = expected + (a * b);
            @(posedge clk);
            check("State after neutralizing inputs");
            #1; // Move slightly past the clock edge
            reset = 1;
            #3; // Pulse width is 3ns, less than the 5ns to the next edge
            reset = 0;
            // 'expected' should NOT change, as a sync reset would be ignored.
            // 'p' should also not have changed.
            check("State immediately after reset pulse"); // Check if p has been incorrectly zeroed out
            // Prepare for the next clock cycle; although a and b are 0, the expected value remains unchanged
            expected = expected + (a*b);
            @(posedge clk);
            check("State one cycle after reset pulse"); // Double-check that p is not affected by the pulse
            $display("-> Asynchronous Reset Trap Test Passed (pulse was correctly ignored).");

            // --- 5. Final Result ---
            if (errors == 0) $display("PASS: All tests passed for mac32!");
            else $fatal(1, "FAIL: Found %0d errors in mac32 test.", errors);
            $finish;
          end
        endmodule
        """)
}

# --- Problem: 64-bit MAC Unit ---
# --- Problem: 64-bit MAC Unit (Corrected) ---
mac64_problem = {
    "name": "64-bit MAC Unit",
    "module_name": "mac64",
    "yosys_module_name": "\\mac64",
    "prompt": textwrap.dedent("""\
        // Design a 64-bit Multiply-Accumulate (MAC) unit with a synchronous, active-high reset.
        //
        // On the positive edge of the clock (clk), the module must behave as follows:
        // - If 'reset' is high (1), the output accumulator 'p' should be cleared to 0.
        // - If 'reset' is low (0), 'p' must be updated with the value of: p + (a * b).
        //
        // All state changes should occur on the rising edge of the clock.
        // The implementation should be synthesizable Verilog.
        module mac64(
            input          clk,
            input          reset,
            input   [63:0] a,
            input   [63:0] b,
            output reg [127:0] p
        );
        """),
    "testbench": textwrap.dedent("""\
        `timescale 1ns/1ps
        `include "dut.v"

        module tb;
          localparam WIDTH = 64;
          localparam OUTW  = 2 * WIDTH;

          reg clk, reset;
          reg [WIDTH-1:0] a, b;
          wire [OUTW-1:0] p;
          reg [OUTW-1:0] expected;
          integer errors;

          mac64 UUT(.clk(clk), .reset(reset), .a(a), .b(b), .p(p));

          initial clk = 0;
          always #5 clk = ~clk;

          task check;
            input string message;
            #1;
            if (p !== expected) begin
              $error("TEST FAILED [%s]: a=%h, b=%h => GOT: %h, EXPECTED: %h", message, a, b, p, expected);
              errors = errors + 1;
            end
          endtask

          initial begin
            errors = 0; expected = 0;

            // --- 1. Synchronous Reset Test ---
            $display("-> Performing Synchronous Reset Test...");
            reset = 1;
            a = 'X; 
            b = 'X;
            @(posedge clk);
            check("During Reset");
    
            reset = 0;
            a = 0; // 
            b = 0; // 
            @(posedge clk); 
            check("First cycle after reset"); 
            $display("-> Synchronous Reset Test Passed.");

            // --- 2. Corner Case Tests ---
            $display("-> Performing Corner Case Tests...");
            a = 64'h0; b = {~0, ~0}; expected = expected + a*b; @(posedge clk); check("Corner: 0 x Max");
            a = {~0, ~0}; b = 64'h1; expected = expected + a*b; @(posedge clk); check("Corner: Max x 1");
            $display("-> Corner Case Tests Passed.");
            
            // --- 3. Randomized Functional Test ---
            $display("-> Starting 10000 randomized tests...");
            for (integer i = 0; i < 10000; i = i + 1) begin
              a = {$urandom(), $urandom()};
              b = {$urandom(), $urandom()};
              expected = expected + a * b;
              @(posedge clk);
              check($sformatf("Random test #%0d", i));
            end
            $display("-> Randomized Tests Passed.");

            // --- 4. Asynchronous Reset Trap Test ---
            $display("-> Performing Asynchronous Reset Trap Test...");
            a = 0;
            b = 0;
            // The goal is to pulse reset *between* clock edges.
            // A synchronous design should ignore this pulse. An asynchronous one will fail.
            expected = expected + (a * b);
            @(posedge clk);
            check("State after neutralizing inputs");
            #1; // Move slightly past the clock edge
            reset = 1;
            #3; // Pulse width is 3ns, less than the 5ns to the next edge
            reset = 0;
            // 'expected' should NOT change, as a sync reset would be ignored.
            // 'p' should also not have changed.
            check("State immediately after reset pulse"); // Check if p has been incorrectly zeroed out
            // Prepare for the next clock cycle; although a and b are 0, the expected value remains unchanged
            expected = expected + (a*b);
            @(posedge clk);
            check("State one cycle after reset pulse"); // Double-check that p is not affected by the pulse
            $display("-> Asynchronous Reset Trap Test Passed (pulse was correctly ignored).");

            // --- 5. Final Result ---
            if (errors == 0) $display("PASS: All tests passed for mac64!");
            else $fatal(1, "FAIL: Found %0d errors in mac64 test.", errors);
            $finish;
          end
        endmodule
        """)
}

# mac_bit_widths = [4, 8, 16, 32, 64]
# mac_problems = []
#
# for N in mac_bit_widths:
#     # 乘积和累加器的位宽会自动计算
#     P_WIDTH = 2 * N
#
#     # 为每个位宽创建一个详细的、指导性的 Prompt
#     mac_prompt = textwrap.dedent(f"""\
#         // Design a {N}-bit Multiply-Accumulate (MAC) unit with synchronous reset.
#         //
#         // Detailed requirements:
#         // 1. The operation should be triggered on the positive edge of the clock (posedge clk).
#         // 2. The reset signal is active-high. When reset is 1, the accumulator 'p' should be cleared to 0.
#         // 3. If reset is 0, the accumulator should update with the operation: p = p + (a * b).
#         // 4. Use non-blocking assignments (<=) for all assignments inside the always block.
#         //
#         module mac{N}(
#             input               clk,
#             input               reset,
#             input      [{N - 1}:0]  a,
#             input      [{N - 1}:0]  b,
#             output reg [{P_WIDTH - 1}:0] p
#         );
#
#         // Please implement the logic inside a single always_ff block.
#         // The structure should be:
#         // always_ff @(posedge clk) begin
#         //   if (reset) begin
#         //     // handle reset logic here
#         //   end else begin
#         //     // handle accumulation logic here
#         //   end
#         // end
#
#         endmodule
#         """)
#
#     # 为每个位宽创建一个统一的、健壮的 Testbench
#     mac_testbench = textwrap.dedent(f"""\
#         `timescale 1ns/1ps
#         `include "dut.v"
#         module tb;
#           reg clk, reset;
#           reg [{N - 1}:0] a, b;
#           wire [{P_WIDTH - 1}:0] p;
#           reg [{P_WIDTH - 1}:0] expected_p;
#           integer error_count;
#           integer i;
#
#           mac{N} UUT(.clk(clk), .reset(reset), .a(a), .b(b), .p(p));
#
#           always #5 clk = ~clk;
#
#           task check;
#             input [{P_WIDTH - 1}:0] expected; input string msg;
#             begin
#                 #10; // Wait for the next clock edge for the result to propagate
#                 if (p !== expected) begin
#                     $display("ERROR [{N}-bit MAC]: %s", msg);
#                     $display("  -> Inputs: a=%h, b=%h", a, b);
#                     $display("  -> Previous accumulator value: %h", expected - (a*b));
#                     $display("  -> GOT: %h, EXPECTED: %h", p, expected);
#                     error_count = error_count + 1;
#                 end
#             end
#           endtask
#
#           initial begin
#             clk = 0; reset = 1; error_count = 0; expected_p = 0;
#             a = 0; b = 0;
#             #10; // Assert reset
#             reset = 0;
#             check(0, "Initial state after reset");
#
#             // --- Fixed Test Sequence ---
#             a = {{(({N}>8)?{N}-8:0){{1'b0}}}}, 8'h0A}}; b = {{(({N}>8)?{N}-8:0){{1'b0}}}}, 8'h05}}; // a=10, b=5
#             expected_p = expected_p + (a*b);
#             check(expected_p, "First accumulation: 10 * 5");
#
#             a = {{(({N}>8)?{N}-8:0){{1'b0}}}}, 8'h0F}}; b = {{(({N}>8)?{N}-8:0){{1'b0}}}}, 8'h02}}; // a=15, b=2
#             expected_p = expected_p + (a*b);
#             check(expected_p, "Second accumulation: 15 * 2");
#
#             // --- Random Test Sequence ---
#             for (i=0; i<100; i=i+1) begin
#                 a = $random;
#                 b = $random;
#                 expected_p = expected_p + (a*b);
#                 check(expected_p, $sformatf("Random test #%0d", i));
#             end
#
#             if (error_count == 0) $display("All tests passed for {N}-bit MAC!");
#             else $fatal(1, "Found %0d errors in {N}-bit MAC test.", error_count);
#             $finish(0);
#           end
#         endmodule
#         """)
#
#     # 创建问题定义字典
#     mac_problem = {
#         "name": f"{N}-bit MAC Unit",
#         "module_name": f"mac{N}",
#         "yosys_module_name": f"\\mac{N}",
#         "prompt": mac_prompt,
#         "testbench": mac_testbench
#     }
#     mac_problems.append(mac_problem)


# --- Define a list containing all the questions to be tested, and the main file will import this list ---
all_problems = [
    adder4_problem,
    # adder8_problem,
    # adder16_problem,
    # adder32_problem,
    # adder64_problem,
    # multiplier4_problem,
    # multiplier8_problem,
    # multiplier16_problem,
    # multiplier32_problem,
    # multiplier64_problem,
    # mac4_problem,
    # mac8_problem,
    # mac16_problem,
    # mac32_problem,
    # mac64_problem
]
# all_problems.extend(mac_problems)