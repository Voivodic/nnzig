const std = @import("std");

// Tree with all modules
const Tree = struct {
    params: *std.Build.Module,
    eigen_wrapper: *std.Build.Module,
    activation: *std.Build.Module,
    loss: *std.Build.Module,
    norms: *std.Build.Module,
    io: *std.Build.Module,
    mlp: *std.Build.Module,
    nnzig: *std.Build.Module,
};

// Creates a tree with all modules
fn createTree(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    errors: *std.Build.Module,
    eigen: *std.Build.Dependency,
    params_path: []const u8,
    precision_flag: []const u8,
) Tree {
    // Params file module (the ZON file)
    const params_file = b.createModule(.{
        .root_source_file = b.path(params_path),
        .target = target,
        .optimize = optimize,
    });

    // Params module (parses the ZON into typed constants)
    const params = b.createModule(.{
        .root_source_file = b.path("src/core/params.zig"),
        .target = target,
        .optimize = optimize,
    });
    params.addImport("paramsFile", params_file);

    // Eigen wrapper module
    const eigen_wrapper = b.createModule(.{
        .root_source_file = b.path("src/eigen/wrappers.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    eigen_wrapper.addImport("params", params);
    eigen_wrapper.addCSourceFiles(.{
        .root = b.path("src/eigen"),
        .files = &.{ "linalg.cpp", "activations.cpp", "losses.cpp", "normalizations.cpp" },
        .flags = &.{ "-O3", "-fPIC", precision_flag },
    });
    eigen_wrapper.addIncludePath(eigen.path("./"));

    // Activation module
    const activation = b.createModule(.{
        .root_source_file = b.path("src/cpu/activations.zig"),
        .target = target,
        .optimize = optimize,
    });
    activation.addImport("params", params);
    activation.addImport("errors", errors);
    activation.addImport("eigen", eigen_wrapper);

    // Loss module
    const loss = b.createModule(.{
        .root_source_file = b.path("src/cpu/losses.zig"),
        .target = target,
        .optimize = optimize,
    });
    loss.addImport("params", params);
    loss.addImport("errors", errors);
    loss.addImport("eigen", eigen_wrapper);

    // Norms module
    const norms = b.createModule(.{
        .root_source_file = b.path("src/cpu/normalizations.zig"),
        .target = target,
        .optimize = optimize,
    });
    norms.addImport("params", params);
    norms.addImport("errors", errors);
    norms.addImport("eigen", eigen_wrapper);

    // IO module
    const io = b.createModule(.{
        .root_source_file = b.path("src/io/binary.zig"),
        .target = target,
        .optimize = optimize,
    });
    io.addImport("params", params);
    io.addImport("errors", errors);

    // MLP module
    const mlp = b.createModule(.{
        .root_source_file = b.path("src/layers/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlp.addImport("eigen", eigen_wrapper);
    mlp.addImport("act", activation);
    mlp.addImport("loss", loss);
    mlp.addImport("errors", errors);
    mlp.addImport("params", params);

    // Main NN module
    const nnzig = b.createModule(.{
        .root_source_file = b.path("src/nnzig.zig"),
        .target = target,
        .optimize = optimize,
    });
    nnzig.addImport("errors", errors);
    nnzig.addImport("act", activation);
    nnzig.addImport("loss", loss);
    nnzig.addImport("eigen", eigen_wrapper);
    nnzig.addImport("mlp", mlp);
    nnzig.addImport("params", params);
    nnzig.addImport("norms", norms);
    nnzig.addImport("io", io);

    // io needs nnzig (circular reference resolved via late import)
    io.addImport("nnzig", nnzig);

    return .{
        .params = params,
        .eigen_wrapper = eigen_wrapper,
        .activation = activation,
        .loss = loss,
        .norms = norms,
        .io = io,
        .mlp = mlp,
        .nnzig = nnzig,
    };
}

pub fn build(b: *std.Build) void {
    // Use the standard target and optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Shared errors module (does not depend on params)
    const errors = b.createModule(.{
        .root_source_file = b.path("src/core/errors.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Shared Eigen dependency
    const eigen = b.dependency("eigen", .{
        .target = target,
        .optimize = optimize,
    });

    // Precision flags derived at comptime from each ZON file
    const precision_flag_main = std.fmt.comptimePrint("-DFLOAT_PRECISION={d}", .{@import("params.zon").precision});
    const precision_flag_test = std.fmt.comptimePrint("-DFLOAT_PRECISION={d}", .{@import("tests/params.zon").precision});
    const precision_flag_bench = std.fmt.comptimePrint("-DFLOAT_PRECISION={d}", .{@import("benchmarks/params.zon").precision});

    // Create a separate module tree for each context
    const tree_main = createTree(b, target, optimize, errors, eigen, "params.zon", precision_flag_main);
    const tree_test = createTree(b, target, optimize, errors, eigen, "tests/params.zon", precision_flag_test);
    const tree_bench = createTree(b, target, optimize, errors, eigen, "benchmarks/params.zon", precision_flag_bench);

    // --- Test step (uses tests/params.zon) ---

    const test_step = b.step("test", "Run tests");

    const run_test_params = b.addRunArtifact(b.addTest(.{ .name = "params", .root_module = tree_test.params }));
    test_step.dependOn(&run_test_params.step);

    const run_test_eigen = b.addRunArtifact(b.addTest(.{ .name = "eigen", .root_module = tree_test.eigen_wrapper }));
    test_step.dependOn(&run_test_eigen.step);

    const run_test_io = b.addRunArtifact(b.addTest(.{ .name = "io", .root_module = tree_test.io }));
    test_step.dependOn(&run_test_io.step);

    const run_test_norms = b.addRunArtifact(b.addTest(.{ .name = "norms", .root_module = tree_test.norms }));
    test_step.dependOn(&run_test_norms.step);

    const run_test_activation = b.addRunArtifact(b.addTest(.{ .name = "activation", .root_module = tree_test.activation }));
    test_step.dependOn(&run_test_activation.step);

    const run_test_loss = b.addRunArtifact(b.addTest(.{ .name = "loss", .root_module = tree_test.loss }));
    test_step.dependOn(&run_test_loss.step);

    const run_test_mlp = b.addRunArtifact(b.addTest(.{ .name = "mlp", .root_module = tree_test.mlp }));
    test_step.dependOn(&run_test_mlp.step);

    const run_test_nnzig = b.addRunArtifact(b.addTest(.{ .name = "nnzig", .root_module = tree_test.nnzig }));
    test_step.dependOn(&run_test_nnzig.step);

    // Integration tests
    const test_mod = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("nnzig", tree_test.nnzig);
    test_mod.addImport("params", tree_test.params);

    const run_tests = b.addRunArtifact(b.addTest(.{ .name = "main", .root_module = test_mod }));
    test_step.dependOn(&run_tests.step);

    // --- Documentation step (uses params.zon) ---

    const docs_step = b.step("docs", "Generate documentation");
    const docs_test = b.addTest(.{ .name = "nnzig_docs", .root_module = tree_main.nnzig });
    docs_step.dependOn(&b.addInstallDirectory(.{
        .source_dir = docs_test.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    }).step);

    // --- Benchmark step (uses benchmarks/params.zon) ---

    const benchmark = b.createModule(.{
        .root_source_file = b.path("benchmarks/runs/run_nnzig.zig"),
        .target = target,
        .optimize = optimize,
    });
    benchmark.addImport("nnzig", tree_bench.nnzig);
    benchmark.addImport("params", tree_bench.params);
    benchmark.addImport("io", tree_bench.io);

    const exe_benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_module = benchmark,
    });
    b.installArtifact(exe_benchmark);

    const run_benchmark = b.addRunArtifact(exe_benchmark);

    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);
}
