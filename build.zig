const std = @import("std");

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

    // Compile the cpp static library
    const eigen_cpp = b.addStaticLibrary(.{
        .name = "eigen_cpp",
        .target = target,
        .optimize = optimize,
    });
    eigen_cpp.addCSourceFiles(.{
        .root = b.path("src"),
        .files = &.{
            "c_wrapper.cpp",
        },
        .flags = &.{
            "-O3",
            "-fPIC",
        },
    });
    eigen_cpp.addIncludePath(eigen.path("./"));
    eigen_cpp.linkLibCpp();

    // Create the wrapper module
    const eigen_wrapper = b.createModule(.{
        .root_source_file = b.path("src/c_wrapper.zig"),
        .target = target,
        .optimize = optimize,
    });
    eigen_wrapper.linkLibrary(eigen_cpp);

    // Create the module with the core structures
    const core = b.createModule(.{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create the module with the activation functions
    const activation = b.createModule(.{
        .root_source_file = b.path("src/activation.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create the module with the loss functions
    const loss = b.createModule(.{
        .root_source_file = b.path("src/loss.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define the file with the parameters
    const params = b.createModule(.{
        .root_source_file = b.path("./params.zig"),
        .target = target,
        .optimize = optimize,
    });
    params.addImport("act", activation);
    params.addImport("loss", loss);

    // Create the module to the MLP
    const MLP = b.createModule(.{
        .root_source_file = b.path("src/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });
    MLP.addImport("eigen", eigen_wrapper);
    MLP.addImport("act", activation);
    MLP.addImport("core", core);

    // Create the NNzig library
    const nnzig = b.createModule(.{
        .root_source_file = b.path("src/nnzig.zig"),
        .target = target,
        .optimize = optimize,
    });
    nnzig.addImport("core", core);
    nnzig.addImport("act", activation);
    nnzig.addImport("loss", loss);
    nnzig.addImport("eigen", eigen_wrapper);
    nnzig.addImport("mlp", MLP);
    nnzig.addImport("params", params);

    // Declare the main executable
    const main_exe = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add the import
    main_exe.root_module.addImport("nnzig", nnzig);

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
