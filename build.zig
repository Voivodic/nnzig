const std = @import("std");

pub fn build(b: *std.Build) void {
    // Use the standard target and optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // --- Eigen wrapper ---

    // Add the Eigen dependency from build.zig.zon
    const eigen = b.dependency("eigen", .{
        .target = target,
        .optimize = optimize,
    });

    // Create the Eigen wrapper module
    const eigen_wrapper = b.createModule(.{
        .root_source_file = b.path("src/eigen/wrappers.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    eigen_wrapper.addCSourceFiles(.{
        .root = b.path("src/eigen"),
        .files = &.{
            "linalg_f32.cpp",
            "linalg_f64.cpp",
            "linalg_f16.cpp",
        },
        .flags = &.{
            "-O3",
            "-fPIC",
        },
    });
    eigen_wrapper.addIncludePath(eigen.path("./"));

    // Add the tests for the Eigen wrapper module
    const test_eigen = b.addTest(.{
        .name = "eigen",
        .root_module = eigen_wrapper,
    });
    const run_test_eigen = b.addRunArtifact(test_eigen);

    // --- Core modules ---

    // Add the errors module
    const errors = b.createModule(.{
        .root_source_file = b.path("src/core/errors.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create a module with the params from params.zon
    const params_file = b.addModule("paramsFile", .{
            .root_source_file = b.path("params.zon"),
            .target = target,
            .optimize = optimize,
    });

    // Add the params module
    const params = b.createModule(.{
        .root_source_file = b.path("src/core/params.zig"),
        .target = target,
        .optimize = optimize,
    });
    params.addImport("paramsFile", params_file);

    // Add the tests for the params module
    const test_params = b.addTest(.{
        .name = "params",
        .root_module = params,
    });
    const run_test_params = b.addRunArtifact(test_params);

    // --- IO modules ---

    // Add the io module
    const io = b.createModule(.{
        .root_source_file = b.path("src/io/binary.zig"),
        .target = target,
        .optimize = optimize,
    });
    io.addImport("params", params);
    io.addImport("errors", errors);

    // Add the tests for the io module
    const test_io = b.addTest(.{
        .name = "io",
        .root_module = io,
    });
    const run_test_io = b.addRunArtifact(test_io);

    // --- OPS modules ---

    // Add the activation module
    const activation = b.createModule(.{
        .root_source_file = b.path("src/cpu/activations.zig"),
        .target = target,
        .optimize = optimize,
    });
    activation.addImport("params", params);
    activation.addImport("errors", errors);

    // Add the tests for the activation module
    const test_activation = b.addTest(.{
        .name = "activation",
        .root_module = activation,
    });
    const run_test_activation = b.addRunArtifact(test_activation);

    // Add the loss module
    const loss = b.createModule(.{
        .root_source_file = b.path("src/cpu/losses.zig"),
        .target = target,
        .optimize = optimize,
    });
    loss.addImport("params", params);
    loss.addImport("errors", errors);

    // Add the tests for the loss module
    const test_loss = b.addTest(.{
        .name = "loss",
        .root_module = loss,
    });
    const run_test_loss = b.addRunArtifact(test_loss);

    // Add the norms module
    const norms = b.createModule(.{
        .root_source_file = b.path("src/core/normalizations.zig"),
        .target = target,
        .optimize = optimize,
    });
    norms.addImport("params", params);
    norms.addImport("errors", errors);

    // Add the tests for the norms module
    const test_norms = b.addTest(.{
        .name = "norms",
        .root_module = norms,
    });
    const run_test_norms = b.addRunArtifact(test_norms);

    // --- Layers modules ---

    // Add the mlp module
    const mlp = b.createModule(.{
        .root_source_file = b.path("src/layers/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });
    mlp.addImport("eigen", eigen_wrapper);
    mlp.addImport("act", activation);
    mlp.addImport("errors", errors);
    mlp.addImport("params", params);

    // Add the tests for the mlp module
    const test_mlp = b.addTest(.{
        .name = "mlp",
        .root_module = mlp,
    });
    const run_test_mlp = b.addRunArtifact(test_mlp);

    // --- Main NN module ---

    // Add the nnzig module
    const nnzig_mod = b.addModule("nnzig", .{
        .root_source_file = b.path("src/nnzig.zig"),
        .target = target,
        .optimize = optimize,
    });
    nnzig_mod.addImport("errors", errors);
    nnzig_mod.addImport("act", activation);
    nnzig_mod.addImport("loss", loss);
    nnzig_mod.addImport("eigen", eigen_wrapper);
    nnzig_mod.addImport("mlp", mlp);
    nnzig_mod.addImport("params", params);
    nnzig_mod.addImport("norms", norms);
    nnzig_mod.addImport("io", io);

    const test_nnzig = b.addTest(.{
        .name = "nnzig",
        .root_module = nnzig_mod,
    });
    const run_test_nnzig = b.addRunArtifact(test_nnzig);

    // Import nnzig to modules that need it
    io.addImport("nnzig", nnzig_mod);

    // --- Test modules ---

    // Add the test module
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("nnzig", nnzig_mod);
    test_mod.addImport("params", params);

    // Add the main tests
    const tests = b.addTest(.{
        .name = "main",
        .root_module = test_mod,
    });
    const run_tests = b.addRunArtifact(tests);

    // Create the test step
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);
    test_step.dependOn(&run_test_params.step);
    test_step.dependOn(&run_test_eigen.step);
    test_step.dependOn(&run_test_io.step);
    test_step.dependOn(&run_test_norms.step);
    test_step.dependOn(&run_test_activation.step);
    test_step.dependOn(&run_test_loss.step);
    test_step.dependOn(&run_test_mlp.step);
    test_step.dependOn(&run_test_nnzig.step);

    // --- Documentation step ---

    const docs_step = b.step("docs", "Generate documentation");
    docs_step.dependOn(&b.addInstallDirectory(.{
        .source_dir = test_nnzig.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    }).step);
}
