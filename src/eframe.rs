use crate::*;
use std::sync::{Arc, Mutex};

use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::GpuFuture;
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;

use vulkano::command_buffer::CommandBufferUsage;
use vulkano::device::physical::PhysicalDevice;
struct RequestRepaintEvent;

struct VulkanoRepaintSignal(
    std::sync::Mutex<winit::event_loop::EventLoopProxy<RequestRepaintEvent>>,
);

impl epi::RepaintSignal for VulkanoRepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(RequestRepaintEvent).ok();
    }
}
struct VulkanoDependents {
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface: Arc<Surface<winit::window::Window>>,
    swap_chain: Arc<Swapchain<winit::window::Window>>,
    images: Vec<Arc<SwapchainImage<winit::window::Window>>>,
}
fn create_display(
    window_builder: winit::window::WindowBuilder,
    event_loop: &winit::event_loop::EventLoop<RequestRepaintEvent>,
) -> VulkanoDependents {
    // The start of this examples is exactly the same as `triangle`. You should read the
    // `triangle` examples if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_0, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name,
        physical.properties().device_type
    );

    let surface = window_builder
        .build_vk_surface(event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        assert!(caps.supported_formats.contains(&(
            vulkano::format::Format::R8G8B8A8_SRGB,
            vulkano::swapchain::ColorSpace::SrgbNonLinear
        )));
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .format(vulkano::format::Format::R8G8B8A8_SRGB)
            .dimensions(dimensions)
            .composite_alpha(alpha)
            .num_images(caps.min_image_count)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .clipped(true)
            .color_space(vulkano::swapchain::ColorSpace::SrgbNonLinear)
            .build()
            .unwrap()
    };
    VulkanoDependents {
        device,
        queue,
        surface,
        swap_chain: swapchain,
        images,
    }
}

// ----------------------------------------------------------------------------

use egui::epaint::TessellationOptions;
use egui_winit::winit;

/// Run an egui app
///
/// Experimentally we use multi thread tesselation to provide performance boost in some CPU bound scenario.
pub fn run(app: Box<dyn epi::App>, native_options: &epi::NativeOptions) -> ! {
    let persistence = egui_winit::epi::Persistence::from_app_name(app.name());
    let window_settings = persistence.load_window_settings();
    let window_builder =
        egui_winit::epi::window_builder(native_options, &window_settings).with_title(app.name());
    let event_loop = winit::event_loop::EventLoop::with_user_event();
    let VulkanoDependents {
        device,
        queue,
        surface,
        mut swap_chain,
        images,
    } = create_display(window_builder, &event_loop);

    let repaint_signal = std::sync::Arc::new(VulkanoRepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    let mut painter = crate::Painter::new(device.clone(), queue, swap_chain.format());
    painter.create_frame_buffers(&images);
    let mut integration = egui_winit::epi::EpiIntegration::new(
        "egui_vulkano_backend",
        surface.window(),
        &mut painter,
        repaint_signal,
        persistence,
        app,
    );
    let ctx = integration.egui_ctx.clone();
    let mut is_focused = true;
    let (sender, receiver) = std::sync::mpsc::channel();
    let sender = Arc::new(Mutex::new(sender));
    let tess_threads = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .thread_name(|id| format!("egui Tesselation thread {}", id))
        .build()
        .unwrap();

    event_loop.run(move |event, _, control_flow| {
        let mut redraw = || {
            if !is_focused {
                // On Mac, a minimized Window uses up all CPU: https://github.com/emilk/egui/issues/325
                // We can't know if we are minimized: https://github.com/rust-windowing/winit/issues/208
                // But we know if we are focused (in foreground). When minimized, we are not focused.
                // However, a user may want an egui with an animation in the background,
                // so we still need to repaint quite fast.
                std::thread::sleep(std::time::Duration::from_millis(10));
            }

            let (needs_repaint, shapes) = integration.update(surface.window(), &mut painter);
            //let clipped_meshes = integration.egui_ctx.tessellate(shapes);
            let sender = sender.clone();
            let ctx = ctx.clone();
            let mut tess_options = ctx.memory().options.tessellation_options;
            tess_options.pixels_per_point =ctx.pixels_per_point();
            tess_options.aa_size = 1.0/ctx.pixels_per_point();
            tess_threads.spawn_fifo(move || {
                sender
                    .lock()
                    .unwrap()
                    .send(egui::epaint::tessellator::tessellate_shapes(
                        shapes,
                        tess_options,
                        [ctx.texture().width, ctx.texture().height],
                    ))
                    .unwrap()
            });

            if let Ok(clipped_meshes) = receiver.try_recv() {
                let mut previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                previous_frame_end.as_mut().unwrap().cleanup_finished();
                let (index, future) = match swapchain::acquire_next_image(swap_chain.clone(), None)
                {
                    Ok((_, true, _)) => return,
                    Err(AcquireError::OutOfDate) => return,
                    Ok((index, false, future)) => (index, future),
                    Err(err) => {
                        panic!("Unrecoverable error occurred : {}", err)
                    }
                };
                let _color = integration.app.clear_color();
                let mut command_buffer_builder =
                    vulkano::command_buffer::AutoCommandBufferBuilder::primary(
                        device.clone(),
                        device.active_queue_families().next().unwrap(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();

                painter.request_upload_egui_texture(&integration.egui_ctx.texture());
                painter
                    .create_command_buffer(
                        &mut command_buffer_builder,
                        RenderTarget::FrameBufferIndex(index),
                        clipped_meshes,
                        &ScreenDescriptor {
                            physical_width: surface.window().inner_size().width,
                            physical_height: surface.window().inner_size().height,
                            scale_factor: integration.egui_ctx.pixels_per_point() as f32,
                        },
                    )
                    .unwrap();
                let command_buffer = command_buffer_builder.build().unwrap();
                painter.present_to_screen(command_buffer, future);
            }

            {
                *control_flow = if integration.should_quit() {
                    winit::event_loop::ControlFlow::Exit
                } else if needs_repaint {
                    surface.window().request_redraw();
                    winit::event_loop::ControlFlow::Poll
                } else {
                    winit::event_loop::ControlFlow::Wait
                };
            }

            integration.maybe_autosave(swap_chain.surface().window());
        };

        match event {
            // Platform-dependent event handlers to workaround a winit bug
            // See: https://github.com/rust-windowing/winit/issues/987
            // See: https://github.com/rust-windowing/winit/issues/1619
            winit::event::Event::RedrawEventsCleared if cfg!(windows) => redraw(),
            winit::event::Event::RedrawRequested(_) if !cfg!(windows) => redraw(),

            winit::event::Event::WindowEvent { event, .. } => {
                if let winit::event::WindowEvent::Focused(new_focused) = event {
                    is_focused = new_focused;
                }

                if let winit::event::WindowEvent::Resized(physical_size) = event {
                    let dimensions: [u32; 2] = physical_size.into();
                    let (new_swapchain, new_images) =
                        match swap_chain.recreate().dimensions(dimensions).build() {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };
                    swap_chain = new_swapchain;
                    painter.create_frame_buffers(&new_images);
                }

                integration.on_event(&event);
                if integration.should_quit() {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }
                surface.window().request_redraw()
            }
            winit::event::Event::LoopDestroyed => {
                integration.on_exit(surface.window());
            }
            winit::event::Event::UserEvent(_) => {
                surface.window().request_redraw();
            }
            _ => (),
        }
    })
}
