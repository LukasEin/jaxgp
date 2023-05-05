## Notes testing

- Only the `eval` methods in the pure (non-combined) kernels need `asserts` for the shape. The gradients and "hessians" as well as any combined kernel `eval` methods call functions that already have asserts.