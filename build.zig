const std = @import("std");

// Path to the params.json file
const params_path = "params.json";

// Main function to build the zig project
pub fn build(b: *std.Build) void {
    // Define the target
    const target = b.standardTargetOptions(.{});

    // Define the optimization
    const optimize = b.standardOptimizeOption(.{});

    // Add eigen as a dependency
    const eigen = b.dependency("eigen", .{
        .target = target,
        .optimize = optimize,
    });

    // Create the wrapper module
    const eigen_wrapper = b.createModule(.{
        .root_source_file = b.path("src/eigen/wrappers.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    // Add C++ source files to the wrapper module
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

    // Create the module with the core structures
    const errors = b.createModule(.{
        .root_source_file = b.path("src/core/errors.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define the file with the parameters
    const params = b.createModule(.{
        .root_source_file = b.path("src/core/params.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Give the params.json file to the params module
    const config_options = b.addOptions();
    config_options.addOptionPath("params_path", b.path(params_path));
    params.addOptions("config", config_options);

    // Create the module with the activation functions
    const activation = b.createModule(.{
        .root_source_file = b.path("src/cpu/activations.zig"),
        .target = target,
        .optimize = optimize,
    });
    activation.addImport("params", params);
    activation.addImport("errors", errors);

    // Create the module with the loss functions
    const loss = b.createModule(.{
        .root_source_file = b.path("src/cpu/losses.zig"),
        .target = target,
        .optimize = optimize,
    });
    loss.addImport("params", params);
    loss.addImport("errors", errors);

    // Add activation and loss types to the paraams modules.
    params.addImport("act", activation);
    params.addImport("loss", loss);

    // Create the module with the normalizations
    const norms = b.createModule(.{
        .root_source_file = b.path("src/core/normalizations.zig"),
        .target = target,
        .optimize = optimize,
    });
    norms.addImport("params", params);
    norms.addImport("errors", errors);

    // Create the module to the MLP
    const MLP = b.createModule(.{
        .root_source_file = b.path("src/layers/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });
    MLP.addImport("eigen", eigen_wrapper);
    MLP.addImport("act", activation);
    MLP.addImport("errors", errors);
    MLP.addImport("params", params);

    // Create the NNzig library
    const nnzig = b.createModule(.{
        .root_source_file = b.path("src/nnzig.zig"),
        .target = target,
        .optimize = optimize,
    });
    nnzig.addImport("errors", errors);
    nnzig.addImport("act", activation);
    nnzig.addImport("loss", loss);
    nnzig.addImport("eigen", eigen_wrapper);
    nnzig.addImport("mlp", MLP);
    nnzig.addImport("params", params);
    nnzig.addImport("norms", norms);

    // Create the main module
    const main_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    main_mod.addImport("nnzig", nnzig);

    // Declare the main executable
    const main_exe = b.addExecutable(.{
        .name = "main",
        .root_module = main_mod,
    });

    // Install the zig executable
    b.installArtifact(main_exe);

    // Create the run step
    const run_cmd = b.addRunArtifact(main_exe);

    // Make the run step depends on the installation
    run_cmd.step.dependOn(b.getInstallStep());

    // Pass the arguments to the zig exe
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // Create the run option
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
