use egui::TextureId;

use vulkano::device::{Device, DeviceExtensions};
use vulkano::image::{AttachmentImage, ImageUsage};
use vulkano::instance::Instance;
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::GpuFuture;
use vulkano::{swapchain, Version};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use egui_vulkano_backend::{EguiVulkanoBackend, ScreenDescriptor};

use crate::renderer::TeapotRenderer;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::physical::PhysicalDevice;

mod model;
mod renderer;

fn main() {
    // The start of this examples is exactly the same as `triangle`. You should read the
    // `triangle` examples if you haven't done so yet.

    let required_extensions = vulkano_win::required_extensions();
    static INSTANCE: OnceCell<Arc<Instance>> = OnceCell::new();
    INSTANCE
        .set(Instance::new(None, Version::V1_0, &required_extensions, None).unwrap())
        .unwrap();
    let physical = PhysicalDevice::enumerate(INSTANCE.get().unwrap())
        .next()
        .unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.properties().device_name,
        physical.properties().device_type
    );

    let event_loop: EventLoop<()> = EventLoop::with_user_event();
    let surface = WindowBuilder::new()
        .with_title("Egui Vulkano Backend sample")
        .build_vk_surface(&event_loop, INSTANCE.get().unwrap().clone())
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

    let mut egui = EguiVulkanoBackend::new(
        surface.clone(),
        device.clone(),
        queue.clone(),
        swapchain.format(),
    );

    egui.create_frame_buffers(images.as_slice());
    //init egui
    // create relation between TextureID and render target
    let (texture_id, mut render_target) = egui
        .painter_mut()
        .init_vulkano_image_with_dimensions([1280, 720])
        .unwrap();
    let size = surface.window().inner_size();

    //create renderer
    let mut teapot_renderer = TeapotRenderer::new(device.clone(), queue);
    //set render target
    teapot_renderer.set_render_target(render_target.clone());
    let mut screen_descriptor = ScreenDescriptor {
        physical_width: size.width,
        physical_height: size.height,
        scale_factor: surface.window().scale_factor() as f32,
    };
    let mut rotate = 0.0;
    let mut height_percent = 0.7;
    let mut image_size = [size.width as f32, size.height as f32 * height_percent];
    let mut needs_to_resize_teapot_rt = false;
    event_loop.run(move |event, _, control_flow| {
        let mut redraw = || {
            // Begin to draw the UI frame.

            egui.begin_frame();

            egui::CentralPanel::default().show(egui.ctx(), |ui| {
                ui.vertical(|ui| {
                    ui.image(texture_id, image_size);
                    ui.horizontal(|ui| {
                        //add Model view adjuster
                        if ui
                            .add(egui::widgets::Slider::new(&mut height_percent, 0.1..=0.9))
                            .changed()
                        {
                            needs_to_resize_teapot_rt = true;
                        }
                        //add rotation control
                        if ui
                            .add(egui::Slider::new(
                                &mut rotate,
                                -std::f32::consts::PI..=std::f32::consts::PI,
                            ))
                            .changed()
                        {
                            teapot_renderer.set_rotate(rotate)
                        }
                    });
                })
            });
            // End the UI frame. We could now handle the output and draw the UI with the backend.
            let (needs_repaint, shapes) = egui.end_frame();
            if needs_repaint {
                surface.window().request_redraw();
                winit::event_loop::ControlFlow::Poll
            } else {
                winit::event_loop::ControlFlow::Wait
            };

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
            if needs_to_resize_teapot_rt {
                needs_to_resize_teapot_rt = false;
                let size = surface.window().inner_size();
                image_size = [size.width as f32, size.height as f32 * height_percent];
                render_target =
                    teapot_rt_resize(size.into(), texture_id, height_percent, egui.painter_mut());
                teapot_renderer.set_render_target(render_target.clone());
            }
            teapot_renderer.draw();

            let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue_family,
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            egui.paint(image_num, &mut command_buffer_builder, shapes);
            egui.painter_mut()
                .present_to_screen(command_buffer_builder.build().unwrap(), acquire_future);
        };

        match event {
            winit::event::Event::RedrawEventsCleared if cfg!(windows) => redraw(),
            winit::event::Event::RedrawRequested(_) if !cfg!(windows) => redraw(),
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                match swapchain.recreate().dimensions(size.into()).build() {
                    Ok(r) => {
                        swapchain = r.0;
                        egui.create_frame_buffers(&r.1);
                    }
                    Err(SwapchainCreationError::UnsupportedDimensions) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                }
                //resize render_target
                needs_to_resize_teapot_rt = true;
                //set screen descriptor
                screen_descriptor.physical_height = size.height;
                screen_descriptor.physical_width = size.width;
            }

            winit::event::Event::WindowEvent { event, .. } => {
                if egui.is_quit_event(&event) {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                }

                egui.on_event(&event);

                surface.window().request_redraw(); // TODO: ask egui if the events warrants a repaint instead
            }

            _ => (),
        }
    });
}

fn teapot_rt_resize(
    size: [u32; 2],
    texture_id: TextureId,
    height_percent: f32,
    painter: &mut egui_vulkano_backend::painter::Painter,
) -> Arc<AttachmentImage> {
    painter
        .recreate_vulkano_texture_with_dimensions(
            texture_id,
            [size[0], (size[1] as f32 * height_percent) as u32],
        )
        .unwrap()
}
