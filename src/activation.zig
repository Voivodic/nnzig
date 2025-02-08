// Define the possible activation functions
pub const Activation = enum(u8) {
    none,
    relu,
    tanh,
    sigmoid,
};

// Compute the derivative for no activation
fn none(comptime T: type, df: []T) void {
    for (df) |*d| {
        d.* = 1.0;
    }
}

// Apply the relu activation funciton
fn relu(comptime T: type, slice: []T, df: []T) void {
    for (slice, df) |*elem, *d| {
        if (elem.* < 0.0) {
            elem.* = 0.0;
            d.* = 0.0;
        } else {
            d.* = 1.0;
        }
    }
}

// Apply the tanh activation function
fn tanh(comptime T: type, slice: []T, df: []T) void {
    for (slice, df) |*elem, *d| {
        const exp = @exp(elem.*);
        const invexp = 1.0 / exp;

        elem.* = (exp * exp - 1.0) / (exp * exp + 1.0);
        d.* = 4.0 / (exp + invexp) * (exp + invexp);
    }
}

// Apply the sigmoid activation function
fn sigmoid(comptime T: type, slice: []T, df: []T) void {
    for (slice, df) |*elem, *d| {
        const sigma = 1.0 / (1.0 + @exp(-elem.*));

        elem.* = sigma;
        d.* = sigma * (1.0 - sigma);
    }
}

// Apply the activation function to each element of the slice
pub fn activateElements(comptime T: type, input: []T, df: []T, act: Activation) void {
    // Check the activation function to be used
    switch (act) {
        Activation.none => none(T, df),
        Activation.relu => relu(T, input, df),
        Activation.tanh => tanh(T, input, df),
        Activation.sigmoid => sigmoid(T, input, df),
    }
}
