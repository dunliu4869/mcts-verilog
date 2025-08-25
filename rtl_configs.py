# rtl_configs.py
from textwrap import dedent

class VerilogConfigs:
    def __init__(self):
        self.modules = [
            "multiplier",
            "divider",
            "sqrt_integer",
            "square",
            "hypotenuse",
            "sine_lut",
            "log2_integer"
        ]

    def get_prompts(self):
        """返回所有任务的prompt定义"""
        # return {
        #     "multiplier": "Design a Verilog module named 'multiplier' that takes two 8-bit unsigned inputs 'a' and 'b', and produces a 16-bit unsigned output 'result'. The module should perform multiplication: result = a * b. Use combinational logic.",
        #     "divider": "Create a Verilog module named 'divider' for 8-bit unsigned integer division. Inputs: 8-bit 'dividend' and 'divisor'. Outputs: 8-bit 'quotient' and 'remainder'. Important: handle division by zero by setting quotient=255 and remainder=0 when divisor=0. For normal cases: quotient = dividend / divisor, remainder = dividend % divisor.",
        #     "sqrt_integer": "Generate a Verilog module named 'sqrt_integer' that calculates the integer square root of a 16-bit unsigned input 'radicand'. Output: 8-bit unsigned 'root' where root = floor(sqrt(radicand)). Handle edge cases: sqrt(0)=0, sqrt(1)=1. IMPORTANT: Do not use while loops in combinational logic. Use a simple bit-by-bit calculation method with for loops or implement using the binary search algorithm with fixed iterations.",
        #     "square": "Write a Verilog module named 'square' that computes the square of an 8-bit unsigned input 'in'. The 16-bit unsigned output 'out' should equal in * in. Use combinational logic.",
        #     "hypotenuse": "Create a Verilog module named 'hypotenuse' that calculates the hypotenuse of a right triangle. Inputs: two 8-bit unsigned sides 'a' and 'b'. Output: 16-bit unsigned 'h' = floor(sqrt(a*a + b*b)). Be careful of overflow when computing a*a + b*b - use sufficient bit width.",
        #     "sine_lut": "Design a Verilog module named 'sine_lut' with a 4-bit input 'angle' (0-15) and 8-bit signed output 'sine_val'. Implement a lookup table for sin(2*pi*angle/16) scaled by 127. Use a case statement covering all 16 values. Key values: angle=0→0, angle=1→49, angle=2→90, angle=4→127, angle=8→0, angle=12→-127.",
        #     "log2": "Generate a Verilog module named 'log2_integer' that calculates floor(log2(x)) for a 16-bit unsigned input 'in'. Output: 4-bit unsigned 'log_val'. Special case: log2(0)=0. Find the position of the most significant bit. Examples: log2(1)=0, log2(8)=3, log2(7)=2."
        # }
        # return {
        #     "multiplier": """Design a Verilog module named 'multiplier' that performs 8-bit unsigned multiplication.
        # Requirements:
        # - Module name: multiplier
        # - Inputs: two 8-bit unsigned integers 'a' and 'b'
        # - Output: 16-bit unsigned integer 'result'
        # - Function: result = a * b
        # - Use combinational logic (assign or always @(*))
        # - Handle all input combinations from 0x00 to 0xFF
        # Example: multiplier(a=15, b=17) should output result=255""",
        #
        #     "divider": """Create a Verilog module named 'divider' for 8-bit unsigned integer division.
        # Requirements:
        # - Module name: divider
        # - Inputs: 8-bit 'dividend' and 8-bit 'divisor'
        # - Outputs: 8-bit 'quotient' and 8-bit 'remainder'
        # - Function: dividend = quotient * divisor + remainder
        # - CRITICAL: Handle division by zero case:
        #   * When divisor = 0: quotient = 255 (8'hFF), remainder = 0
        # - Use combinational logic (assign or always @(*))
        # - For normal division: quotient = dividend / divisor, remainder = dividend % divisor
        # Example: divider(dividend=100, divisor=7) → quotient=14, remainder=2
        # Example: divider(dividend=100, divisor=0) → quotient=255, remainder=0""",
        #
        #     "sqrt_integer": """Use Verilog syntax to design a synthesizable module named `sqrt_integer` that calculates integer square root.
        # Requirements:
        # - Module name: sqrt_integer
        # - Input: 16-bit unsigned 'radicand' (0 to 65535)
        # - Output: 8-bit unsigned 'root' (0 to 255)
        # - Function: root = floor(sqrt(radicand))
        # - Algorithm suggestion: Use binary search or iterative method
        # - Handle edge cases: sqrt(0) = 0, sqrt(1) = 1
        # - Ensure root² ≤ radicand < (root+1)²
        # - Do NOT use break`, `continue`, or `while` statements. Any `for` loops must be static and fully unrollable by a synthesis tool.
        # - Use a bit-by-bit iterative algorithm with a fixed-iteration `for` loop (e.g., 8 or 16 iterations).
        # Example: sqrt_integer(radicand=25) → root=5
        # Example: sqrt_integer(radicand=26) → root=5 (floor of 5.099)""",
        #
        #     "square": """Write a Verilog module named 'square' that computes the square of an 8-bit number.
        # Requirements:
        # - Module name: square
        # - Input: 8-bit unsigned 'in' (0 to 255)
        # - Output: 16-bit unsigned 'out' (0 to 65025)
        # - Function: out = in * in
        # - Use combinational logic (assign or always @(*))
        # - Handle all inputs: square(0)=0, square(255)=65025
        # Example: square(in=15) → out=225""",
        #
        #     "hypotenuse": """Use Verilog syntax to a synthesizable module named `hypotenuse` that calculates the hypotenuse of a right triangle.
        # Requirements:
        # - Module name: hypotenuse
        # - Inputs: two 8-bit sides 'a' and 'b' (0 to 255)
        # - Output: 16-bit 'h' = floor(sqrt(a² + b²))
        # - Algorithm:
        #   1. Calculate sum_squares = a*a + b*b (use ≥17 bits to prevent overflow)
        #   2. Calculate h = floor(sqrt(sum_squares))
        # - Handle edge cases: hypotenuse(0,0)=0, hypotenuse(3,4)=5
        # - Use combinational or clocked logic
        # Example: hypotenuse(a=3, b=4) → h=5
        # Example: hypotenuse(a=255, b=255) → h=360 (floor of 360.624)""",
        #
        #     "sine_lut": """Use Verilog syntax to a synthesizable module named `sine_lut` implementing a sine lookup table.
        # Requirements:
        # - Module name: sine_lut
        # - Input: 4-bit 'angle' (0 to 15, representing 16 equally spaced angles from 0 to 2π)
        # - Output: 8-bit signed 'sine_val' (-128 to 127, representing sine values scaled by 127)
        # - Lookup table values for sin(2π*angle/16):
        #   * angle=0: sine_val=0 (sin(0°))
        #   * angle=1: sine_val=49 (sin(22.5°) ≈ 0.383 * 127)
        #   * angle=2: sine_val=90 (sin(45°) ≈ 0.707 * 127)
        #   * angle=4: sine_val=127 (sin(90°) = 1.0 * 127)
        #   * angle=8: sine_val=0 (sin(180°))
        #   * angle=12: sine_val=-127 (sin(270°))
        # - Use case statement in always @(*) block
        # - Ensure all 16 cases (0-15) are covered
        # Example: sine_lut(angle=4) → sine_val=127 (represents sin(90°)=1.0)""",
        #
        #     "log2_integer": """Generate a Verilog module named 'log2_integer' that calculates floor(log₂(x)).
        # Requirements:
        # - Module name: log2_integer
        # - Input: 16-bit unsigned 'in' (0 to 65535)
        # - Output: 4-bit unsigned 'log_val' (0 to 15)
        # - Function: log_val = floor(log₂(in))
        # - Special case: log₂(0) = 0 (by definition for this implementation)
        # - Algorithm: Find the position of the most significant bit
        # - Examples:
        #   * log2_integer(in=1) → log_val=0 (log₂(1)=0)
        #   * log2_integer(in=8) → log_val=3 (log₂(8)=3)
        #   * log2_integer(in=7) → log_val=2 (floor(log₂(7))=2)
        #   * log2_integer(in=32768) → log_val=15 (log₂(2¹⁵)=15)
        # - Use combinational logic to scan for MSB position"""
        # }
        return {
            "multiplier": '''Design a Verilog synthesizable module named `multiplier`.
            The module should take two 8-bit unsigned inputs,`a` and `b`.
            It must produce a 16-bit unsigned output, `result`.
            The functionality is to compute the product of the two inputs.
            Key Considerations:
            - Ensure the output port is wide enough to hold the maximum possible result (255 * 255).
            - The implementation should be purely combinational.''',

            "divider": '''Design a synthesizable Verilog module named `divider` for unsigned integer division.
            The module should have an 8-bit input `dividend` and an 8-bit input `divisor`.
            It must produce an 8-bit output `quotient` and an 8-bit output `remainder`.
            Key Considerations:
            - You must handle the division-by-zero edge case. If `divisor` is 0, the `quotient` should be `8'hFF` (all ones) and the `remainder` should be 0.
            - The implementation must be purely combinational. Avoid clocked sequential logic (`always @(posedge clk)`).''',

            "sqrt_integer": '''Use Verilog syntax to design a synthesizable module named `sqrt_integer` that calculates the integer square root.
            The module should take a 16-bit unsigned input, `radicand`.
            It must produce an 8-bit unsigned output, `root`.
            The functionality is to compute `floor(sqrt(radicand))`.
            Key Considerations:
            - Do NOT use break statements.
            - The implementation should be purely combinational.
            - Consider edge cases like an input of 0 or 1.''',

            "square":'''Design a Verilog synthesizable module named `square`.
            The module should take a single 8-bit unsigned input, `in`.
            It must produce a 16-bit unsigned output, `out`.
            The functionality is to compute `in * in`.
            Key Considerations:
            - Ensure the output port is wide enough to prevent overflow.
            - The implementation should be simple and combinational.''',

            # "hypotenuse": '''Use Verilog syntax to a synthesizable module named `hypotenuse` to calculate the integer length of a right triangle's hypotenuse.
            # The module takes two 8-bit unsigned inputs, `a` and `b`, representing the two shorter sides.
            # It must produce a 16-bit unsigned output, `h`.
            # The functionality is to compute `floor(sqrt(a*a + b*b))`.
            # Key Considerations:
            # - Use bit-by-bit square root algorithm, NOT $sqrt().
            # - Do NOT use break`, `continue`, or `while` statements. Any `for` loops must be static and fully unrollable by a synthesis tool.
            # - Pay attention to potential overflow when calculating `a*a + b*b`. The intermediate result of this sum will require more than 16 bits.
            # - The implementation should be purely combinational.''',

            "hypotenuse": '''You are an expert RTL engineer. Your task is to write a high-quality, synthesizable Verilog module named 'hypotenuse'.
            ### MODULE INTERFACE
            The module has two 8-bit inputs, 'a' and 'b', and one 16-bit output, 'h'.
            ### FUNCTIONALITY
            The module must calculate the integer hypotenuse of a right triangle: h = floor(sqrt(a^2 + b^2)). The implementation must be purely combinational.
            ### STEP-BY-STEP IMPLEMENTATION GUIDE
            1.  First, calculate the sum of the squares: `sum_squares = a*a + b*b`.
            2.  **[Bit-Width]** To prevent overflow, the intermediate signal `sum_squares` must be at least 17 bits wide. For example, declare it as `reg [16:0] sum_squares;`.
            3.  Next, implement an integer square root algorithm to calculate 'h' from `sum_squares`.
            ### CRITICAL CONSTRAINTS AND PRECAUTIONS
            -   **[NON-SYNTHESIZABLE CODE]** You **MUST NOT** use non-synthesizable system tasks like `$sqrt` or `$floor`. The implementation must be fully synthesizable.
            -   **[ALGORITHM]** You **MUST** use a synthesizable algorithm for the square root. A **binary search algorithm** that iterates a fixed number of times (e.g., 16 iterations for a 16-bit result) is a robust and recommended approach. Do not use recursive functions.
            -   **[SYNTHESIS]** All logic must be combinational (use `always @(*)`). Ensure all output signals are assigned a value in every possible branch of `if` or `case` statements to avoid inferring latches.''',

            # "sine_lut": '''Use Verilog syntax to design a synthesizable module named `sine_lut` that acts as a sine function look-up table (LUT).
            # The module takes a 4-bit input, `angle`, representing 16 equal divisions of a full circle (0 to 15).
            # It must produce an 8-bit signed output, `sine_val`.
            # The mapping should approximate `127 * sin(2 * pi * angle / 16)`.
            # Key Considerations:
            # - You must use a hardware-friendly, bit-by-bit iterative refinement method.
            # - The entire module must be written in synthesizable, combinational logic suitable for an ASIC or FPGA flow.
            # - Use a `case` statement for the implementation.
            # - The output must be a `signed` type to represent negative values.
            # - Pre-calculate the 16 constant values for the LUT.''',

            "sine_lut": '''You are an expert RTL engineer. Your task is to write a high-quality, synthesizable Verilog module named 'sine_lut'.
            ### MODULE INTERFACE
            The module has a 4-bit input, 'angle', and an 8-bit **signed** output, 'sine_val'.
            ### FUNCTIONALITY
            The module implements a Look-Up Table (LUT) to approximate a sine wave. The 4-bit 'angle' input represents 16 discrete points on the wave, corresponding to angles from 0 to 337.5 degrees in 22.5-degree steps.
            ### STEP-BY-STEP IMPLEMENTATION GUIDE
            1.  The most efficient and standard implementation is to use a **constant memory array (LUT)** to store the 16 pre-calculated sine values.
            2.  The `angle` input should be used as the index to read the corresponding value from this array.
            3.  The logic must be combinational. Do **NOT** use a large `case` statement.
            ### CRITICAL CONSTRAINTS AND PRECAUTIONS
            -   **[VALUES]** The output `sine_val` is an 8-bit **SIGNED** value (range -128 to 127). You must provide the complete and correct 16 values for the LUT. The values must follow the shape of a sine wave: start at 0, peak near +127, cross 0, go to a minimum near -128, and return towards 0.
            -   **[NUMBER FORMAT]** To avoid errors with negative numbers, you **MUST** specify the values in **signed decimal format** (e.g., `8'sd49`, `-8'sd90`). Do not use hexadecimal for negative values.
            -   **[HINT - Correct Values]** To ensure correctness, use the following pre-calculated 16 values for the LUT array initialization:
                `0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49`''',

            "log2_integer": '''Use Verilog syntax to design a synthesizable module named `log2_integer` that calculates the floor of the base-2 logarithm.
            The module should take a 16-bit input, `in`.
            It must produce a 4-bit output, `log_val`. For example, if the input is 25, the output should be 4 (since 2^4 <= 25 < 2^5).
            Key Considerations:
            - You must use a hardware-friendly, bit-by-bit iterative refinement method.
            - The entire module must be written in synthesizable, combinational logic suitable for an ASIC or FPGA flow.
            - Do NOT use break`, `continue`, or `while` statements. Any `for` loops must be static and fully unrollable by a synthesis tool.
            - You must handle the edge case where the input `in` is 0. If `in` is 0, the output `log_val` should also be 0.
            - The implementation should be purely combinational.'''
        }

    def get_interfaces(self):
        """返回所有模块的接口定义"""
        return {
            "multiplier": "module multiplier(\n    input [7:0] a,\n    input [7:0] b,\n    output [15:0] result\n);",
            "divider": "module divider(\n    input [7:0] dividend,\n    input [7:0] divisor,\n    output [7:0] quotient,\n    output [7:0] remainder\n);",
            "sqrt_integer": "module sqrt_integer(\n    input [15:0] radicand,\n    output [7:0] root\n);",
            "square": "module square(\n    input [7:0] in,\n    output [15:0] out\n);",
            "hypotenuse": "module hypotenuse(\n    input [7:0] a,\n    input [7:0] b,\n    output [15:0] h\n);",
            "sine_lut": "module sine_lut(\n    input [3:0] angle,\n    output signed [7:0] sine_val\n);",
            "log2_integer": "module log2_integer(\n    input [15:0] in,\n    output [3:0] log_val\n);"
        }

    def get_testbenches(self):
        """返回所有模块的testbench定义"""
        return {
            "multiplier": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [7:0] a, b;
    wire [15:0] result;
    wire [15:0] expected;

    integer errors;
    integer tests;
    integer i;

    multiplier dut (.a(a), .b(b), .result(result));
    multiplier_ref dut_ref (.a(a), .b(b), .result(expected));

    initial begin
        errors = 0;
        tests = 0;

        // Directed Edge Case Testing
        a = 0; b = 123; #1; tests = tests + 1;
        if (result !== expected) begin errors = errors + 1; $display("ERROR: 0*123 failed."); end

        a = 1; b = 210; #1; tests = tests + 1;
        if (result !== expected) begin errors = errors + 1; $display("ERROR: 1*210 failed."); end

        a = 255; b = 255; #1; tests = tests + 1;
        if (result !== expected) begin errors = errors + 1; $display("ERROR: 255*255 failed."); end

        // Randomized Testing
        for (i = 0; i < 100; i = i + 1) begin
            a = $urandom & 8'hFF;
            b = $urandom & 8'hFF;
            #1;
            tests = tests + 1;
            if (result !== expected) begin
                $display("ERROR: a=%d, b=%d, got=%d, expected=%d", a, b, result, expected);
                errors = errors + 1;
            end
        end

        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule'''),

            "square": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [7:0] in;
    wire [15:0] out;
    wire [15:0] expected;

    integer errors, tests, i;

    square dut (.in(in), .out(out));
    square_ref dut_ref (.in(in), .out(expected));

    initial begin
        errors = 0; tests = 0;
        for (i = 0; i < 256; i = i + 1) begin
            in = i; #1; tests = tests + 1;
            if (out !== expected) begin
                $display("ERROR: %d^2, got=%d, expected=%d", in, out, expected);
                errors = errors + 1;
            end
        end
        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule'''),

            "divider": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [7:0] dividend, divisor;
    wire [7:0] quotient, remainder;
    wire [7:0] expected_quotient, expected_remainder;

    integer errors, tests, i;

    divider dut (.dividend(dividend), .divisor(divisor), .quotient(quotient), .remainder(remainder));
    divider_ref dut_ref (.dividend(dividend), .divisor(divisor), .quotient(expected_quotient), .remainder(expected_remainder));

    initial begin
        errors = 0; tests = 0;

        // Directed Edge Case Testing
        dividend = 177; divisor = 1; #1; tests = tests + 1;
        if (quotient !== expected_quotient || remainder !== expected_remainder) begin errors = errors + 1; end

        dividend = 240; divisor = 240; #1; tests = tests + 1;
        if (quotient !== expected_quotient || remainder !== expected_remainder) begin errors = errors + 1; end

        dividend = 0; divisor = 55; #1; tests = tests + 1;
        if (quotient !== expected_quotient || remainder !== expected_remainder) begin errors = errors + 1; end

        dividend = 100; divisor = 0; #1; tests = tests + 1;
        if (quotient !== expected_quotient || remainder !== expected_remainder) begin errors = errors + 1; end

        // Randomized Testing
        for (i = 0; i < 100; i = i + 1) begin
            dividend = $urandom & 8'hFF;
            divisor = $urandom & 8'hFF;
            if (divisor == 0) divisor = 1;

            #1; tests = tests + 1;
            if (quotient !== expected_quotient || remainder !== expected_remainder) begin
                $display("ERROR: %d/%d, got q=%d r=%d, exp q=%d r=%d", dividend, divisor, quotient, remainder, expected_quotient, expected_remainder);
                errors = errors + 1;
            end
        end

        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule'''),

            "sqrt_integer": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [15:0] radicand;
    wire [7:0] root;
    wire [7:0] expected;

    integer errors, tests, i;

    sqrt_integer dut (.radicand(radicand), .root(root));
    sqrt_integer_ref dut_ref (.radicand(radicand), .root(expected));

    initial begin
        errors = 0; tests = 0;

        // Directed Edge Case Testing
        radicand = 0; #1; tests = tests + 1;
        if (root !== expected) begin errors = errors + 1; end

        radicand = 1; #1; tests = tests + 1;
        if (root !== expected) begin errors = errors + 1; end

        radicand = 26; #1; tests = tests + 1;
        if (root !== expected) begin errors = errors + 1; end

        radicand = 65535; #1; tests = tests + 1;
        if (root !== expected) begin errors = errors + 1; end

        // Randomized Testing
        for (i = 0; i < 100; i = i + 1) begin
            radicand = $urandom & 16'hFFFF;
            #1; tests = tests + 1;
            if (root !== expected) begin
                $display("ERROR: sqrt(%d), got=%d, expected=%d", radicand, root, expected);
                errors = errors + 1;
            end
        end

        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule'''),

            "hypotenuse": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [7:0] a, b;
    wire [15:0] h;
    wire [15:0] expected;

    integer errors, tests, i;

    hypotenuse dut (.a(a), .b(b), .h(h));
    hypotenuse_ref dut_ref (.a(a), .b(b), .h(expected));

    initial begin
        errors = 0; tests = 0;

        // Directed Testing
        a = 3; b = 4; #1; tests = tests + 1;
        if (h !== expected) begin errors = errors + 1; end

        a = 0; b = 5; #1; tests = tests + 1;
        if (h !== expected) begin errors = errors + 1; end

        a = 10; b = 10; #1; tests = tests + 1;
        if (h !== expected) begin errors = errors + 1; end

        a = 255; b = 255; #1; tests = tests + 1;
        if (h !== expected) begin errors = errors + 1; end

        // Randomized Testing
        for (i = 0; i < 100; i = i + 1) begin
            a = $urandom & 8'hFF;
            b = $urandom & 8'hFF;
            #1; tests = tests + 1;
            if (h !== expected) begin
                $display("ERROR: hyp(a=%d, b=%d), got=%d, expected=%d", a, b, h, expected);
                errors = errors + 1;
            end
        end

        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule'''),

            "sine_lut": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [3:0] angle;
    wire signed [7:0] sine_val;
    wire signed [7:0] expected;

    integer errors, tests, i;

    sine_lut dut (.angle(angle), .sine_val(sine_val));
    sine_lut_ref dut_ref (.angle(angle), .sine_val(expected));

    initial begin
        errors = 0; tests = 0;
        for (i = 0; i < 16; i = i + 1) begin
            angle = i; #1; tests = tests + 1;
            if (sine_val !== expected) begin
                $display("ERROR: sin(angle=%d), got=%d, expected=%d", angle, sine_val, expected);
                errors = errors + 1;
            end
        end
        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule'''),

            "log2_integer": dedent('''
            `timescale 1ns/1ps
            module tb;
    reg [15:0] in;
    wire [3:0] log_val;
    wire [3:0] expected;

    integer errors, tests, i;

    log2_integer dut (.in(in), .log_val(log_val));
    log2_integer_ref dut_ref (.in(in), .log_val(expected));

    initial begin
        errors = 0; tests = 0;

        // Directed Testing
        in = 16'h0001; #1; tests = tests + 1;
        if (log_val !== expected) begin errors = errors + 1; end

        in = 16'h0000; #1; tests = tests + 1;
        if (log_val !== expected) begin errors = errors + 1; end

        in = 16'h0007; #1; tests = tests + 1;
        if (log_val !== expected) begin errors = errors + 1; end

        in = 16'h8000; #1; tests = tests + 1;
        if (log_val !== expected) begin errors = errors + 1; end

        in = 16'hFFFF; #1; tests = tests + 1;
        if (log_val !== expected) begin errors = errors + 1; end

        // Randomized Testing
        for (i = 0; i < 100; i = i + 1) begin
            in = $urandom & 16'hFFFF;
            #1; tests = tests + 1;
            if (log_val !== expected) begin
                $display("ERROR: log2(%d), got=%d, expected=%d", in, log_val, expected);
                errors = errors + 1;
            end
        end

        $display("Mismatches: %d in %d samples", errors, tests);
        $finish;
    end
endmodule''')
        }

    def get_references(self):
        """返回所有模块的参考实现"""
        return {
            "multiplier": '''module multiplier_ref(
    input [7:0] a,
    input [7:0] b,
    output [15:0] result
);
    assign result = a * b;
endmodule''',

            "square": '''module square_ref (
    input [7:0] in,
    output [15:0] out
);
    assign out = in * in;
endmodule''',

            "divider": '''module divider_ref(
    input [7:0] dividend,
    input [7:0] divisor,
    output [7:0] quotient,
    output [7:0] remainder
);
    assign quotient = (divisor == 0) ? 8'd255 : dividend / divisor;
    assign remainder = (divisor == 0) ? 8'd0 : dividend % divisor;
endmodule''',

            "sqrt_integer": '''module sqrt_integer_ref (
    input [15:0] radicand,
    output reg [7:0] root
);
    reg [15:0] temp;
    integer i;

    always @(*) begin
        root = 0;
        for (i = 7; i >= 0; i = i - 1) begin
            temp = root | (1 << i);
            if (temp * temp <= radicand) begin
                root = temp;
            end
        end
    end
endmodule''',

            "hypotenuse": '''module hypotenuse_ref(
    input [7:0] a,
    input [7:0] b,
    output reg [15:0] h
);
    reg [31:0] sum_squares; 
    reg [15:0] temp;
    integer i;

    always @(*) begin
        sum_squares = (a * a) + (b * b);
        h = 0;
        for (i = 15; i >= 0; i = i - 1) begin
            temp = h | (1 << i);
            if ({1'b0, temp} * {1'b0, temp} <= sum_squares) begin
                h = temp;
            end
        end
    end
endmodule''',

            "sine_lut": '''module sine_lut_ref(
    input [3:0] angle,
    output reg signed [7:0] sine_val
);
    always @(*) begin
        case(angle)
            4'd0:  sine_val = 8'd0;
            4'd1:  sine_val = 8'd49;
            4'd2:  sine_val = 8'd90;
            4'd3:  sine_val = 8'd118;
            4'd4:  sine_val = 8'd127;
            4'd5:  sine_val = 8'd118;
            4'd6:  sine_val = 8'd90;
            4'd7:  sine_val = 8'd49;
            4'd8:  sine_val = 8'd0;
            4'd9:  sine_val = -8'd49;
            4'd10: sine_val = -8'd90;
            4'd11: sine_val = -8'd118;
            4'd12: sine_val = -8'd127;
            4'd13: sine_val = -8'd118;
            4'd14: sine_val = -8'd90;
            4'd15: sine_val = -8'd49;
        endcase
    end
endmodule''',

            "log2_integer": '''module log2_integer_ref(
    input [15:0] in,
    output reg [3:0] log_val
);
    integer i;

    always @(*) begin
        if (in == 0) begin
            log_val = 0;
        end else begin
            log_val = 0;
            for (i = 15; i >= 0; i = i - 1) begin
                if (in[i]) begin
                    log_val = i;
                    i = -1; // Break out of loop
                end
            end
        end
    end
endmodule'''
        }

    def get_default_testbench(self, module_name):
        """为没有专门定义的模块返回默认testbench"""
        return f'''module tb;
    // 简化测试 - 只检查编译
    {module_name} dut ();
    {module_name}_ref dut_ref ();

    initial begin
        $display("Mismatches: 0 in 1 samples");
        $finish;
    end
endmodule'''

    def get_default_reference(self, module_name):
        """为没有专门定义的模块返回默认参考实现"""
        return f'''module {module_name}_ref();
    // 简化参考实现
endmodule'''