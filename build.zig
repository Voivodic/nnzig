const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const eigen = b.dependency("eigen", .{
        .target = target,
        .optimize = optimize,
    });

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
            "eigen.cpp",
        },
        .flags = &.{
            "-O3",
            "-fPIC",
        },
    });
    eigen_wrapper.addIncludePath(eigen.path("./"));

    const errors = b.createModule(.{
        .root_source_file = b.path("src/core/errors.zig"),
        .target = target,
        .optimize = optimize,
    });

    const params = b.createModule(.{
        .root_source_file = b.path("src/core/params.zig"),
        .target = target,
        .optimize = optimize,
    });

    const activation = b.createModule(.{
        .root_source_file = b.path("src/cpu/activations.zig"),
        .target = target,
        .optimize = optimize,
    });
    activation.addImport("params", params);
    activation.addImport("errors", errors);

    const loss = b.createModule(.{
        .root_source_file = b.path("src/cpu/losses.zig"),
        .target = target,
        .optimize = optimize,
    });
    loss.addImport("params", params);
    loss.addImport("errors", errors);

    params.addImport("act", activation);
    params.addImport("loss", loss);

    const norms = b.createModule(.{
        .root_source_file = b.path("src/core/normalizations.zig"),
        .target = target,
        .optimize = optimize,
    });
    norms.addImport("params", params);
    norms.addImport("errors", errors);

    const MLP = b.createModule(.{
        .root_source_file = b.path("src/layers/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });
    MLP.addImport("eigen", eigen_wrapper);
    MLP.addImport("act", activation);
    MLP.addImport("errors", errors);
    MLP.addImport("params", params);

    const nnzig_mod = b.addModule("nnzig", .{
        .root_source_file = b.path("src/nnzig.zig"),
        .target = target,
        .optimize = optimize,
    });
    nnzig_mod.addImport("errors", errors);
    nnzig_mod.addImport("act", activation);
    nnzig_mod.addImport("loss", loss);
    nnzig_mod.addImport("eigen", eigen_wrapper);
    nnzig_mod.addImport("mlp", MLP);
    nnzig_mod.addImport("params", params);
    nnzig_mod.addImport("norms", norms);

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("nnzig", nnzig_mod);
    test_mod.addImport("params", params);

    const tests = b.addTest(.{
        .root_module = test_mod,
    });

    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);
}
