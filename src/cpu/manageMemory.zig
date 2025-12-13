//! This module allocates and frees memory in the cpu

// Import the modules use
const std = @import("std");

/// Allocates memory on the cpu
pub fn alloc(allocator: std.mem.Allocator, comptime T: type, size: usize) ![]T {
    return try allocator.alloc(T, size);
}

/// Free memory on the cpu
pub fn free(allocator: std.mem.Allocator, comptime T: type, slice: *[]T) void {
    allocator.free(slice.*);
}
