use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::Timelike;
use egui::{ClippedMesh, FontDefinitions};
use egui_winit_platform::{Platform, PlatformDescriptor};
use epi::App;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::GpuFuture;
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow::{Poll, Wait};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use egui_vulkano_backend::{RenderTarget, ScreenDescriptor};

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
    let instance = Instance::new(
        None,
        Version::V1_0,
        &required_extensions,
        None,
    )
    .unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let physical_properties = physical.properties().clone();
    let gpu_name = physical_properties.device_name.unwrap();
    let gpu_type = physical_properties.device_type.unwrap();
    let heap_infos: Vec<String> = physical
        .memory_heaps()
        .map(|heap| {
            format!(
                "Bytes : {}\nDevice local : {} \nMulti instance : {} \n",
                heap.size(),
                heap.is_device_local(),
                heap.is_multi_instance()
            )
        })
        .collect();
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
        assert!(caps.supported_formats.contains(&(
            vulkano::format::Format::R8G8B8A8Srgb,
            vulkano::swapchain::ColorSpace::SrgbNonLinear
        )));
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        Swapchain::start(device.clone(), surface.clone())
            .format(vulkano::format::Format::R8G8B8A8Srgb)
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

    //create renderer
    let mut egui_render_pass =
        egui_vulkano_backend::EguiVulkanoRenderPass::new(device.clone(), queue, swapchain.format());
    // initialize framebuffer
    egui_render_pass.create_frame_buffers(&images);
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

    let start_time = Instant::now();
    let mut previous_frame_time = None;
    /*
    tessellating thread
    */

    let (work_tx, work_rx): (
        Sender<Box<dyn FnOnce() -> Vec<ClippedMesh> + Send + Sync>>,
        Receiver<Box<dyn FnOnce() -> Vec<ClippedMesh> + Send + Sync>>,
    ) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();

    let shared_sender = Arc::new(Mutex::new(work_tx));
    std::thread::spawn(move || {
        for work in work_rx.iter() {
            let result: Vec<ClippedMesh> = work();
            result_tx.send(result).unwrap();
        }
    });
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
                event: WindowEvent::Resized(size),
                ..
            } => {
                let dimensions: [u32; 2] = size.into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate().dimensions(dimensions).build() {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                swapchain = new_swapchain;
                egui_render_pass.create_frame_buffers(&new_images);
            }
            Event::RedrawEventsCleared => {
                platform.update_time(start_time.elapsed().as_secs_f64());

                // Begin to draw the UI frame.
                let egui_start = Instant::now();
                platform.begin_frame();
                let mut app_output = epi::backend::AppOutput::default();

                let mut frame = epi::backend::FrameBuilder {
                    info: epi::IntegrationInfo {
                        web_info: None,
                        prefer_dark_mode: None,
                        cpu_usage: previous_frame_time,
                        seconds_since_midnight: Some(seconds_since_midnight()),
                        native_pixels_per_point: Some(surface.window().scale_factor() as _),
                    },
                    tex_allocator: &mut egui_render_pass,
                    output: &mut app_output,
                    repaint_signal: repaint_signal.clone(),
                }
                .build();

                // Draw the demo application.
                demo_app.update(&platform.context(), &mut frame);
                egui::Window::new("Your Hardware").show(&platform.context(), |ui| {
                    ui.label(format!("GPU type : {:?}", gpu_type));
                    ui.label(format!("GPU name : {}", gpu_name));
                    ui.label("Heaps");
                    let _: Vec<()> = heap_infos
                        .iter()
                        .map(|info| {
                            ui.label(info);
                        })
                        .collect();
                });
                // End the UI frame. We could now handle the output and draw the UI with the backend.
                let (output, paint_commands) = platform.end_frame();
                let frame_time = (Instant::now() - egui_start).as_secs_f64() as f32;
                previous_frame_time = Some(frame_time);
                if output.needs_repaint {
                    *control_flow = Poll;
                } else {
                    *control_flow = Wait;
                }
                println!("number of shapes : {}", paint_commands.len());
                let tess_start = Instant::now();

                let ctx = platform.context().clone();
                shared_sender
                    .lock()
                    .unwrap()
                    .send(Box::new(move || ctx.tessellate(paint_commands))
                        as Box<dyn FnOnce() -> Vec<ClippedMesh> + Send + Sync>)
                    .unwrap();
                println!("send to tessellator");
                if let Ok(paint_jobs) = result_rx.recv() {
                    println!("Tessellating time : {:?} ", tess_start.elapsed());

                    let mut previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    let (image_num, suboptimal, acquire_future) =
                        match swapchain::acquire_next_image(swapchain.clone(), None) {
                            Ok(r) => r,
                            Err(AcquireError::OutOfDate) => {
                                return;
                            }
                            Err(e) => panic!("Failed to acquire next image: {:?}", e),
                        };

                    if suboptimal {
                        return;
                    }

                    let screen_descriptor = ScreenDescriptor {
                        physical_width: surface.window().inner_size().width,
                        physical_height: surface.window().inner_size().height,
                        scale_factor: surface.window().scale_factor() as f32,
                    };
                    egui_render_pass.request_upload_egui_texture(&platform.context().texture());

                    let render_target = RenderTarget::FrameBufferIndex(image_num);
                    let render_command = egui_render_pass.create_command_buffer(
                        render_target,
                        paint_jobs,
                        &screen_descriptor,
                    );

                    egui_render_pass.present_to_screen(render_command, acquire_future);
                }
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
