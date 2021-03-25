mod render_pass;
mod shader;

extern crate vulkano;

use crate::render_pass::EguiRenderPassDesc;
use epi::egui;
use epi::egui::{ClippedMesh, Color32, Texture, TextureId};

use std::sync::Arc;
use vulkano::buffer::{BufferSlice, BufferUsage, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBuffer, DynamicState, SubpassContents};
use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSetsPool, PersistentDescriptorSet};
use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, Queue};
use vulkano::format::Format::R8G8B8A8Srgb;
use vulkano::framebuffer::{FramebufferAbstract, RenderPass};
use vulkano::image::{Dimensions, ImageViewAccess, MipmapsCount, SwapchainImage};

use crate::shader::create_pipeline;
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::framebuffer::Framebuffer;

use iter_vec::ExactSizedIterVec;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::{Scissor, Viewport};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::swapchain::SwapchainAcquireFuture;
use vulkano::sync::GpuFuture;

#[derive(Default, Debug, Copy, Clone)]
struct EguiVulkanoVertex {
    a_pos: [f32; 2],
    a_tex_coord: [f32; 2],
    a_color: u32,
}
struct UniformBuffer {
    _u_screen_size: [f32; 2],
}
vulkano::impl_vertex!(EguiVulkanoVertex, a_pos, a_tex_coord, a_color);
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
type Pipeline = GraphicsPipeline<
    SingleBufferDefinition<EguiVulkanoVertex>,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Arc<RenderPass<EguiRenderPassDesc>>,
>;
/// egui rendering command builder
pub struct EguiVulkanoRenderPass {
    pipeline: Arc<Pipeline>,
    egui_texture_descriptor_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
    egui_texture_version: Option<u64>,
    user_textures: Vec<Option<UserTexture>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    sampler: Arc<Sampler>,
    frame_buffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    uniform_buffer_pool: CpuBufferPool<UniformBuffer>,
    vertex_buffer_pool: CpuBufferPool<EguiVulkanoVertex>,
    index_buffer_pool: CpuBufferPool<u32>,
    descriptor_set_0_pool: FixedSizeDescriptorSetsPool,
    dynamic: DynamicState,
}

impl EguiVulkanoRenderPass {
    ///create command builder
    ///
    /// if render target format incompatible with SwapChain format may cause color glitch
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_target_format: vulkano::format::Format,
    ) -> Self {
        let pipeline = create_pipeline(device.clone(), render_target_format);
        let sampler = Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        .unwrap();
        let uniform_buffer_pool = CpuBufferPool::uniform_buffer(device.clone());
        let vertex_buffer_pool = CpuBufferPool::vertex_buffer(device.clone());
        let index_buffer_pool = CpuBufferPool::new(device.clone(), BufferUsage::index_buffer());
        let descriptor_set_0_pool = FixedSizeDescriptorSetsPool::new(
            pipeline.layout().descriptor_set_layout(0).unwrap().clone(),
        );

        Self {
            pipeline,
            egui_texture_descriptor_set: None,
            egui_texture_version: None,
            user_textures: vec![],
            device,
            queue,

            sampler,
            frame_buffers: vec![],
            uniform_buffer_pool,
            vertex_buffer_pool,
            index_buffer_pool,
            descriptor_set_0_pool,
            dynamic: Default::default(),
        }
    }

    /// you must call when SwapChain or render target resize or before first create_command_buffer call
    pub fn create_frame_buffers<Wnd: Send + Sync + 'static>(
        &mut self,
        image_views: &[Arc<SwapchainImage<Wnd>>],
    ) {
        self.frame_buffers = image_views
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(self.pipeline.clone())
                        .add(image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<dyn FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>();
    }
    /// execute command and present to screen
    pub fn present_to_screen<Wnd>(
        &self,
        command: AutoCommandBuffer<StandardCommandPoolAlloc>,
        acquire_future: SwapchainAcquireFuture<Wnd>,
    ) {
        let swap_chain = acquire_future.swapchain().clone();
        let image_id = acquire_future.image_id();

        if let Ok(ok) = acquire_future
            .then_execute(self.queue.clone(), command)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), swap_chain, image_id)
            .then_signal_fence_and_flush()
        {
            if ok.wait(None).is_ok() {};
        }
    }

    /// translate egui rendering request to vulkano AutoCommandBuffer
    ///
    /// color attachment is render target.
    ///
    /// If color_attachment is none then render to self framebuffer and image num must Some
    ///
    /// If color attachment is some then render to any image e.g. AttachmentImage StorageImage
    pub fn create_command_buffer(
        &mut self,
        color_attachment: Option<Arc<dyn ImageViewAccess + Send + Sync>>,
        image_num: Option<usize>,
        paint_jobs: &[ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
    ) -> AutoCommandBuffer<StandardCommandPoolAlloc> {
        let mut exact_sized_iter_vec_vertices = ExactSizedIterVec::new();
        let mut exact_sized_iter_vec_indices = ExactSizedIterVec::new();
        let framebuffer = if let Some(color_attachment) = color_attachment {
            Arc::new(
                vulkano::framebuffer::Framebuffer::start(self.pipeline.render_pass().clone())
                    .add(color_attachment)
                    .unwrap()
                    .build()
                    .expect("failed to create frame buffer"),
            )
        } else {
            self.frame_buffers[image_num.unwrap()].clone()
        };
        let logical = screen_descriptor.logical_size();
        let sub_buffer = self
            .uniform_buffer_pool
            .next(UniformBuffer {
                _u_screen_size: [logical.0 as f32, logical.1 as f32],
            })
            .unwrap();
        let descriptor_set_0 = Arc::new(
            self.descriptor_set_0_pool
                .next()
                .add_buffer(sub_buffer)
                .unwrap()
                .add_sampler(self.sampler.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let mut pass = vulkano::command_buffer::AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .expect("failed to create command buffer builder");
        pass.begin_render_pass(
            framebuffer,
            SubpassContents::Inline,
            vec![[0.0, 0.0, 0.0, 0.0].into()],
        )
        .unwrap();

        let physical_height = screen_descriptor.physical_height;
        let physical_width = screen_descriptor.physical_width;

        self.dynamic.viewports = Some(vec![Viewport {
            origin: [0.0; 2],
            dimensions: [physical_width as f32, physical_height as f32],
            depth_range: Default::default(),
        }]);

        let mut all_mesh_range = Vec::with_capacity(paint_jobs.len());
        let clip_rectangles = paint_jobs.iter().map(|x| x.0);
        let scissors = skip_by_clip(screen_descriptor, clip_rectangles);
        let mut indices_from = 0;
        let mut vertices_from = 0;
        #[cfg(debug_assertions)]
        println!("start frame");
        for ((_i, egui::ClippedMesh(_, mesh)), scissor) in
            paint_jobs.iter().enumerate().zip(scissors.clone())
        {
            #[cfg(debug_assertions)]
            println!(
                "mesh {} vertices {} indices {}",
                _i,
                mesh.vertices.len(),
                mesh.indices.len()
            );
            if mesh.vertices.is_empty() || mesh.indices.is_empty() | scissor.is_none() {
                all_mesh_range.push(None);
                #[cfg(debug_assertions)]
                println!("Detect glitch mesh");
                continue;
            }
            exact_sized_iter_vec_indices.push(mesh.indices.as_slice());
            #[allow(clippy::transmute_ptr_to_ptr)]
            exact_sized_iter_vec_vertices
                .push(unsafe { std::mem::transmute(mesh.vertices.as_slice()) });

            let indices_len = mesh.indices.len();
            let vertices_len = mesh.vertices.len();
            let indices_end = indices_from + indices_len;
            let vertices_end = vertices_from + vertices_len;
            all_mesh_range.push(Some((
                indices_from..indices_end,
                vertices_from..vertices_end,
            )));
            indices_from = indices_end;
            vertices_from = vertices_end;
        }
        // put data to buffer
        let index_buffer = self
            .index_buffer_pool
            .chunk(exact_sized_iter_vec_indices.copied())
            .unwrap();
        let vertex_buffer = self
            .vertex_buffer_pool
            .chunk(exact_sized_iter_vec_vertices.copied())
            .unwrap();
        #[cfg(debug_assertions)]
        println!("end frame");
        for ((egui::ClippedMesh(_, mesh), range), scissor) in
            paint_jobs.iter().zip(all_mesh_range.iter()).zip(scissors)
        {
            if let Some(range) = range {
                let index_range = range.0.clone();
                let vertex_range = range.1.clone();
                //mesh to index and vertex buffer

                let texture_id = mesh.texture_id;
                let texture_descriptor = self.get_descriptor_set(texture_id);
                //emit valid texturing
                if scissor.is_some() {
                    if let Some(texture_desc_set) = texture_descriptor {
                        let index_buffer =
                            BufferSlice::from_typed_buffer_access(index_buffer.clone())
                                .slice(index_range)
                                .unwrap();
                        let vertex_buffer =
                            BufferSlice::from_typed_buffer_access(vertex_buffer.clone())
                                .slice(vertex_range)
                                .unwrap();
                        {
                            self.dynamic.scissors = Some(vec![scissor.unwrap()]);
                            pass.draw_indexed(
                                self.pipeline.clone(),
                                &self.dynamic,
                                vertex_buffer,
                                index_buffer,
                                (descriptor_set_0.clone(), texture_desc_set),
                                (),
                                vec![],
                            )
                            .unwrap();
                        }
                    }
                }
            }
        }
        pass.end_render_pass().unwrap();
        pass.build().unwrap()
    }
    /// Update egui system texture
    /// You must call before every  create_command_buffer
    pub fn upload_egui_texture(&mut self, texture: &Texture) {
        //no change
        if self.egui_texture_version == Some(texture.version) {
            return;
        }
        let format = vulkano::format::Format::R8G8B8A8Srgb;
        let it: Vec<[u8; 4]> = texture
            .pixels
            .iter()
            .map(|x| Color32::from_white_alpha(*x).to_array())
            .collect();

        let image = vulkano::image::ImmutableImage::from_iter(
            it.iter().cloned(),
            Dimensions::Dim2d {
                width: texture.width as u32,
                height: texture.height as u32,
            },
            MipmapsCount::One,
            format,
            self.queue.clone(),
        )
        .unwrap();
        image.1.flush().unwrap();
        let image_view = image.0 as Arc<dyn ImageViewAccess + Sync + Send>;
        let pipeline = self.pipeline.clone();
        self.egui_texture_descriptor_set =
            Some(Self::create_descriptor_set_from_view(pipeline, image_view));
        self.egui_texture_version = Some(texture.version);
    }
    fn alloc_user_texture(&mut self) -> TextureId {
        for (i, tex) in self.user_textures.iter_mut().enumerate() {
            if tex.is_none() {
                return TextureId::User(i as u64);
            }
        }
        let id = TextureId::User(self.user_textures.len() as u64);
        self.user_textures.push(None);
        id
    }
    fn free_user_texture(&mut self, id: TextureId) {
        if let TextureId::User(id) = id {
            #[cfg(debug_assertions)]
            println!("free {}", id);
            self.user_textures
                .get_mut(id as usize)
                .and_then(|option: &mut Option<UserTexture>| option.take());
        }
    }
    /// egui use lazy texture allocating so you must call before every create_command_buffer.
    pub fn upload_pending_textures(&mut self) {
        let pipeline = self.pipeline.clone();
        for user_texture in &mut self.user_textures {
            if let Some(user_texture) = user_texture {
                //when uploaded image is none
                if user_texture.descriptor_set.is_none() {
                    //get data
                    let pixels = std::mem::take(&mut user_texture.pixels);
                    //skip upload invalid texture and avoid clash
                    if pixels.is_empty() {
                        break;
                    }
                    let (image, future) = vulkano::image::ImmutableImage::from_iter(
                        pixels.iter().cloned(),
                        Dimensions::Dim2d {
                            width: user_texture.size[0],
                            height: user_texture.size[1],
                        },
                        MipmapsCount::One,
                        R8G8B8A8Srgb,
                        self.queue.clone(),
                    )
                    .unwrap();
                    future.flush().unwrap();
                    let image_view = image as Arc<dyn ImageViewAccess + Sync + Send>;
                    user_texture.descriptor_set = Some(Self::create_descriptor_set_from_view(
                        pipeline.clone(),
                        image_view,
                    ));
                }
            }
        }
    }
    fn create_descriptor_set_from_view(
        pipeline: Arc<Pipeline>,
        image_view: Arc<dyn ImageViewAccess + Sync + Send>,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        Arc::new(
            PersistentDescriptorSet::start(
                pipeline.layout().descriptor_set_layout(1).unwrap().clone(),
            )
            .add_image(image_view)
            .unwrap()
            .build()
            .unwrap(),
        )
    }
    fn get_descriptor_set(
        &self,
        texture_id: TextureId,
    ) -> Option<Arc<dyn DescriptorSet + Send + Sync>> {
        if let egui::TextureId::User(id) = texture_id {
            self.user_textures
                .get(id as usize)
                .unwrap()
                .as_ref()
                .unwrap()
                .descriptor_set
                .clone()
        } else {
            self.egui_texture_descriptor_set.clone()
        }
    }
    pub fn set_user_texture(
        &mut self,
        id: TextureId,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) {
        #[cfg(debug_assertions)]
        println!("new texture arrived {:?}", id);
        if let TextureId::User(id) = id {
            // test Texture slot allocated
            if let Some(slot) = self.user_textures.get_mut(id as usize) {
                //pre alloc pixels
                let mut pixels = Vec::with_capacity(size.0 * size.1 * 4);
                //unpack pixels
                for pixel in srgba_pixels {
                    pixels.extend(pixel.to_array().iter())
                }
                //write to slot
                *slot = Some(UserTexture {
                    pixels,
                    size: [size.0 as u32, size.1 as u32],
                    descriptor_set: None,
                })
            }
        }
    }
    /// register vulkano image view as egui texture
    ///
    /// Usable for render to image rectangle

    pub fn register_vulkano_texture(
        &mut self,
        image_view: Arc<dyn ImageViewAccess + Sync + Send>,
    ) -> TextureId {
        let id = self.alloc_user_texture();
        if let egui::TextureId::User(id) = id {
            let dimension = image_view.dimensions();
            let pipeline = self.pipeline.clone();
            if let Some(slot) = self.user_textures.get_mut(id as usize) {
                *slot = Some(UserTexture {
                    pixels: vec![],
                    size: [dimension.width(), dimension.height()],
                    descriptor_set: Some(Self::create_descriptor_set_from_view(
                        pipeline, image_view,
                    )),
                });
            }
        }
        id
    }
}
impl epi::TextureAllocator for EguiVulkanoRenderPass {
    fn alloc_srgba_premultiplied(
        &mut self,
        size: (usize, usize),
        srgba_pixels: &[Color32],
    ) -> TextureId {
        let id = self.alloc_user_texture();
        self.set_user_texture(id, size, srgba_pixels);
        id
    }
    fn free(&mut self, id: TextureId) {
        self.free_user_texture(id)
    }
}
fn skip_by_clip(
    screen_desc: &ScreenDescriptor,
    clip_rectangles: impl Iterator<Item = egui::Rect> + Clone,
) -> impl Iterator<Item = Option<Scissor>> + Clone {
    let scale_factor = screen_desc.scale_factor;
    let physical_width = screen_desc.physical_width;
    let physical_height = screen_desc.physical_height;
    clip_rectangles.map(move |clip_rect| {
        // Transform clip rect to physical pixels.
        let clip_min_x = scale_factor * clip_rect.min.x;
        let clip_min_y = scale_factor * clip_rect.min.y;
        let clip_max_x = scale_factor * clip_rect.max.x;
        let clip_max_y = scale_factor * clip_rect.max.y;

        // Make sure clip rect can fit within an `u32`.
        let clip_min_x = clip_min_x.clamp(0.0, physical_width as f32);
        let clip_min_y = clip_min_y.clamp(0.0, physical_height as f32);
        let clip_max_x = clip_max_x.clamp(clip_min_x, physical_width as f32);
        let clip_max_y = clip_max_y.clamp(clip_min_y, physical_height as f32);

        let clip_min_x = clip_min_x.round() as u32;
        let clip_min_y = clip_min_y.round() as u32;
        let clip_max_x = clip_max_x.round() as u32;
        let clip_max_y = clip_max_y.round() as u32;

        let width = (clip_max_x - clip_min_x).max(1);
        let height = (clip_max_y - clip_min_y).max(1);
        // clip scissor rectangle to target size
        let x = clip_min_x.min(physical_width);
        let y = clip_min_y.min(physical_height);
        let width = width.min(physical_width - x);
        let height = height.min(physical_height - y);
        // skip rendering with zero-sized clip areas
        if width == 0 || height == 0 {
            None
        } else {
            Some(Scissor {
                origin: [x as i32, y as i32],
                dimensions: [width, height],
            })
        }
    })
}
#[derive(Default)]
struct UserTexture {
    pixels: Vec<u8>,
    size: [u32; 2],
    descriptor_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}
