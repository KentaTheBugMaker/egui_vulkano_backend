---
name: Bug report
about: Create a report to help us improve
title: "[Bug]"
labels: bug, wontfix assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
minimal code here

```rust

```

**Crash log**

```
thread 'main' panicked at 'assertion failed: inner.offset < inner.buffer.size()', /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/vulkano-0.21.0/src/command_buffer/sys.rs:432:9
stack backtrace:
   0: rust_begin_unwind
             at /rustc/cb75ad5db02783e8b0222fee363c5f63f7e2cf5b/library/std/src/panicking.rs:493:5
   1: core::panicking::panic_fmt
             at /rustc/cb75ad5db02783e8b0222fee363c5f63f7e2cf5b/library/core/src/panicking.rs:92:14
   2: core::panicking::panic
             at /rustc/cb75ad5db02783e8b0222fee363c5f63f7e2cf5b/library/core/src/panicking.rs:50:5
   3: vulkano::command_buffer::sys::UnsafeCommandBufferBuilder<P>::bind_index_buffer
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/vulkano-0.21.0/src/command_buffer/sys.rs:432:9
   4: <vulkano::command_buffer::synced::commands::<impl vulkano::command_buffer::synced::base::SyncCommandBufferBuilder<P>>::bind_index_buffer::Cmd<B> as vulkano::command_buffer::synced::base::Command<P>>::send
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/vulkano-0.21.0/src/command_buffer/synced/commands.rs:181:17
   5: vulkano::command_buffer::synced::base::SyncCommandBufferBuilder<P>::build
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/vulkano-0.21.0/src/command_buffer/synced/base.rs:760:17
   6: vulkano::command_buffer::auto::AutoCommandBufferBuilder<P>::build
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/vulkano-0.21.0/src/command_buffer/auto.rs:577:20
   7: egui_vulkano_backend::EguiVulkanoRenderPass::create_command_buffer
             at ./egui_vulkano_backend/src/lib.rs:328:9
   8: example::main::{{closure}}
             at ./example/src/main.rs:222:38
   9: winit::platform_impl::platform::sticky_exit_callback
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/winit-0.24.0/src/platform_impl/linux/mod.rs:736:5
  10: winit::platform_impl::platform::wayland::event_loop::EventLoop<T>::run_return
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/winit-0.24.0/src/platform_impl/linux/wayland/event_loop/mod.rs:388:13
  11: winit::platform_impl::platform::wayland::event_loop::EventLoop<T>::run
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/winit-0.24.0/src/platform_impl/linux/wayland/event_loop/mod.rs:191:9
  12: winit::platform_impl::platform::EventLoop<T>::run
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/winit-0.24.0/src/platform_impl/linux/mod.rs:652:56
  13: winit::event_loop::EventLoop<T>::run
             at /home/rustacean/.cargo/registry/src/github.com-1ecc6299db9ec823/winit-0.24.0/src/event_loop.rs:154:9
  14: example::main
             at ./example/src/main.rs:138:5
  15: core::ops::function::FnOnce::call_once
             at /home/rustacean/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ops/function.rs:227:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.

```

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Enviroment:**

- OS: [e.g. iOS]
- GPU:[e.g. Radeon RX 580]
- build:[e.g. release]

**Additional context**
Add any other context about the problem here.
