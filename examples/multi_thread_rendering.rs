use vulkano::device::{Device, DeviceExtensions};

use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, PhysicalDevice};

use vulkano::swapchain::{
    ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};

use vulkano::sync::GpuFuture;

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use chrono::Timelike;
use egui::FontDefinitions;
use egui_vulkano_backend::{RenderingDispatcher, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use epi::App;

use std::time::Instant;

use vulkano::format::Format;

/// A custom event type for the winit app.
enum EguiEvent {
    RequestRedraw,
}

/// This is the repaint signal type that egui needs for requesting a repaint from another thread.
/// It sends the custom RequestRedraw event to the winit event loop.
struct ExampleRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<EguiEvent>>);

impl epi::RepaintSignal for ExampleRepaintSignal {
    fn request_repaint(&self) {
        self.0
            .lock()
            .unwrap()
            .send_event(EguiEvent::RequestRedraw)
            .ok();
    }
}
fn main() {
    // The start of this examples is exactly the same as `triangle`. You should read the
    // `triangle` examples if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );
    // create rendering request channel

    let event_loop = EventLoop::with_user_event();
    let surface = WindowBuilder::new()
        .with_title("Egui Vulkano Backend sample")
        .build_vk_surface(&event_loop, instance.clone())
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

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        //     let alpha = CompositeAlpha::PreMultiplied;
        assert!(&caps
            .supported_formats
            .contains(&(Format::B8G8R8A8Srgb, ColorSpace::SrgbNonLinear)));
        let format = Format::B8G8R8A8Srgb;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            ImageUsage::color_attachment(),
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };

    let mut multi_thread_renderer =
        RenderingDispatcher::build(device.clone(), queue.clone(), &images);

    //init egui
    let repaint_signal = std::sync::Arc::new(ExampleRepaintSignal(std::sync::Mutex::new(
        event_loop.create_proxy(),
    )));

    // We use the egui_winit_platform crate as the platform.
    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: swapchain.dimensions()[0],
        physical_height: swapchain.dimensions()[1],
        scale_factor: surface.window().scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    // Display the demo application that ships with egui.
    let mut demo_app = egui_demo_lib::WrapApp::default();
    //we want to initialize all framebuffers so we check it true
    let mut recreate_swapchain = false;
    let http = std::sync::Arc::new(epi_http::EpiHttp {});
    let start_time = Instant::now();
    let mut previous_frame_time = None;
    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared | Event::RedrawRequested(_) => {
                platform.update_time(start_time.elapsed().as_secs_f64());

                // Begin to draw the UI frame.
                let egui_start = Instant::now();
                platform.begin_frame();
                let mut app_output = epi::backend::AppOutput::default();

                let mut frame = epi::backend::FrameBuilder {
                    info: epi::IntegrationInfo {
                        web_info: None,
                        cpu_usage: previous_frame_time,
                        seconds_since_midnight: Some(seconds_since_midnight()),
                        native_pixels_per_point: Some(surface.window().scale_factor() as _),
                    },
                    tex_allocator: &mut multi_thread_renderer,
                    http: http.clone(),
                    output: &mut app_output,
                    repaint_signal: repaint_signal.clone(),
                }
                .build();

                // Draw the demo application.
                demo_app.update(&platform.context(), &mut frame);

                // End the UI frame. We could now handle the output and draw the UI with the backend.
                let (output, paint_commands) = platform.end_frame();
                let paint_jobs = platform.context().tessellate(paint_commands);

                let frame_time = (Instant::now() - egui_start).as_secs_f64() as f32;
                previous_frame_time = Some(frame_time);
                let mut previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate_with_dimensions(dimensions) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::UnsupportedDimensions) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };
                    swapchain = new_swapchain;
                    multi_thread_renderer.create_frame_buffers(&new_images);
                    recreate_swapchain = false;
                }

                let screen_descriptor = ScreenDescriptor {
                    physical_width: surface.window().inner_size().width,
                    physical_height: surface.window().inner_size().height,
                    scale_factor: surface.window().scale_factor() as f32,
                };

                multi_thread_renderer.upload_egui_texture(&platform.context().texture());

                println!("call mtr render");
                recreate_swapchain = multi_thread_renderer.render(paint_jobs, screen_descriptor);

                let epi::backend::AppOutput { quit, window_size } = app_output;
                *control_flow = if quit {
                    ControlFlow::Exit
                } else if output.needs_repaint {
                    surface.window().request_redraw();
                    ControlFlow::Poll
                } else {
                    ControlFlow::Wait
                };

                *control_flow = ControlFlow::Poll
            }
            Event::UserEvent(_) => {
                surface.window().request_redraw();
            }
            _ => (),
        }
    });
}

/// Time of day as seconds since midnight. Used for clock in demo app.
pub fn seconds_since_midnight() -> f64 {
    let time = chrono::Local::now().time();
    time.num_seconds_from_midnight() as f64 + 1e-9 * (time.nanosecond() as f64)
}
