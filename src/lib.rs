//! [egui](https://docs.rs/egui) rendering integration for vulkano
//!
//! This crate have some features
//! * background user texture upload
//! * map vulkano texture to user texture
//! * lower memory usage
//! * parallel data uploading
//crate import

use epi::egui;

//egui
use egui::Color32;
use egui::TextureId;
//vulkano
use vulkano::device::{Device, Queue};
use vulkano::image::view::ImageViewCreationError;
use vulkano::image::{ImageAccess, ImageCreationError, ImageViewAbstract};
use vulkano::render_pass::FramebufferAbstract;

use crate::painter::Painter;
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::swapchain::Surface;

pub mod painter;
pub(crate) mod render_pass;
pub(crate) mod shader;

/// Glium backend like
pub struct EguiVulkanoBackend {
    egui_winit: egui_winit::State,
    egui_ctx: egui::CtxRef,
    painter: Painter,
    surface: Arc<Surface<winit::window::Window>>,
}
impl EguiVulkanoBackend {
    pub fn new(
        surface: Arc<Surface<winit::window::Window>>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_target_format: vulkano::format::Format,
    ) -> Self {
        Self {
            egui_ctx: Default::default(),
            egui_winit: egui_winit::State::new(surface.window()),
            painter: crate::painter::Painter::new(device, queue, render_target_format),
            surface,
        }
    }
    pub fn request_redraw(&self) {
        self.surface.window().request_redraw();
    }
    pub fn ctx(&self) -> &egui::CtxRef {
        &self.egui_ctx
    }

    pub fn painter_mut(&mut self) -> &mut crate::Painter {
        &mut self.painter
    }

    pub fn ctx_and_painter_mut(&mut self) -> (&egui::CtxRef, &mut crate::Painter) {
        (&self.egui_ctx, &mut self.painter)
    }

    pub fn pixels_per_point(&self) -> f32 {
        self.egui_winit.pixels_per_point()
    }

    pub fn egui_input(&self) -> &egui::RawInput {
        self.egui_winit.egui_input()
    }

    pub fn on_event(&mut self, event: &winit::event::WindowEvent<'_>) {
        self.egui_winit.on_event(self.egui_ctx.as_ref(), event);
    }

    /// Is this a close event or a Cmd-Q/Alt-F4 keyboard command?
    pub fn is_quit_event(&self, event: &winit::event::WindowEvent<'_>) -> bool {
        self.egui_winit.is_quit_event(event)
    }

    pub fn begin_frame(&mut self) {
        let raw_input = self.take_raw_input();
        self.begin_frame_with_input(raw_input);
    }

    pub fn begin_frame_with_input(&mut self, raw_input: egui::RawInput) {
        self.egui_ctx.begin_frame(raw_input);
    }

    /// Prepare for a new frame. Normally you would call [`Self::begin_frame`] instead.
    pub fn take_raw_input(&mut self) -> egui::RawInput {
        self.egui_winit.take_egui_input(self.surface.window())
    }

    /// Returns `needs_repaint` and shapes to draw.
    pub fn end_frame(&mut self) -> (bool, Vec<egui::epaint::ClippedShape>) {
        let (egui_output, shapes) = self.egui_ctx.end_frame();
        let needs_repaint = egui_output.needs_repaint;
        self.handle_output(egui_output);
        (needs_repaint, shapes)
    }

    pub fn handle_output(&mut self, output: egui::Output) {
        self.egui_winit
            .handle_output(self.surface.window(), &self.egui_ctx, output);
    }

    pub fn paint(
        &mut self,
        image_number: usize,
        command_buffer_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        shapes: Vec<egui::epaint::ClippedShape>,
    ) {
        //upload updated egui system texture
        self.painter
            .request_upload_egui_texture(&self.egui_ctx.texture());

        let clipped_meshes = self.egui_ctx.tessellate(shapes);
        let sf = self.egui_winit.pixels_per_point();
        let screen_desc = {
            if let Some(fb) = self.painter.frame_buffers.get(image_number) {
                ScreenDescriptor {
                    physical_width: fb.width(),
                    physical_height: fb.height(),
                    scale_factor: sf,
                }
            } else {
                return;
            }
        };
        let target = RenderTarget::FrameBufferIndex(image_number);
        self.painter.create_command_buffer(
            command_buffer_builder,
            target,
            clipped_meshes,
            &screen_desc,
        )
    }
    /// I extended to any image to write compiz sample
    ///
    pub fn create_frame_buffers<I: ImageAccess + 'static>(&mut self, swap_chain_images: &[Arc<I>]) {
        self.painter.create_frame_buffers(swap_chain_images)
    }
}

/// same as [egui_wgpu_backend::ScreenDescriptor](https://docs.rs/egui_wgpu_backend/0.8.0/egui_wgpu_backend/struct.ScreenDescriptor.html)
#[derive(Debug, Copy, Clone)]
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub physical_width: u32,
    /// Height of the window in physical pixel.
    pub physical_height: u32,
    /// HiDPI scale factor.
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    fn logical_size(&self) -> (u32, u32) {
        let logical_width = self.physical_width as f32 / self.scale_factor;
        let logical_height = self.physical_height as f32 / self.scale_factor;
        (logical_width as u32, logical_height as u32)
    }
}
impl epi::TextureAllocator for EguiVulkanoBackend {
    fn alloc_srgba_premultiplied(
        &mut self,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) -> TextureId {
        self.painter.alloc_srgba_premultiplied(size, srgba_pixels)
    }

    fn free(&mut self, id: TextureId) {
        self.painter.free(id)
    }
}

impl epi::NativeTexture for EguiVulkanoBackend {
    type Texture = Arc<dyn ImageViewAbstract>;

    fn register_native_texture(&mut self, native: Self::Texture) -> TextureId {
        self.painter.register_vulkano_image_view(native)
    }

    fn replace_native_texture(&mut self, id: TextureId, replacing: Self::Texture) {
        self.painter.replace_native_texture(id, replacing)
    }
}

#[derive(Debug, Clone)]
pub enum RecreateErrors {
    TextureIdIsEgui,
    TextureIdNotFound(TextureId),
    ImageViewCreationError(ImageViewCreationError),
    ImageCreationError(ImageCreationError),
}

#[derive(Debug, Clone)]
pub enum InitRenderAreaError {
    ImageViewCreationError(ImageViewCreationError),
    ImageCreationError(ImageCreationError),
}

pub enum RenderTarget {
    /// render to texture or any other use
    ColorAttachment(Arc<dyn ImageViewAbstract>),
    /// render to own framebuffer
    FrameBufferIndex(usize),
}
