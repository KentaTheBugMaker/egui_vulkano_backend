mod render_pass;

extern crate vulkano;
extern crate vulkano_shaders;

use crate::render_pass::EguiRenderPassDesc;
use epi::egui;
use epi::egui::{ClippedMesh, Color32, Texture, TextureId};
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{CommandBuffer, DynamicState, SubpassContents};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
use vulkano::device::{Device, Queue};
use vulkano::format::Format::R8G8B8A8Srgb;
use vulkano::framebuffer::{LoadOp, RenderPass, RenderPassDesc, Subpass};
use vulkano::image::{AttachmentImage, Dimensions, ImageViewAccess, MipmapsCount};
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Scissor;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::sync::GpuFuture;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
#[derive(Default, Debug, Copy, Clone)]
struct EguiVulkanoVertex {
    position: [f32; 2],
    tex_coord: [f32; 2],
    color: u32,
}
vulkano::impl_vertex!(EguiVulkanoVertex, position, tex_coord, color);
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub physical_width: u32,
    /// Height of the window in physical pixel.
    pub physical_height: u32,
    /// HiDPI scale factor.
    pub scale_factor: f32,
}

type Pipeline = GraphicsPipeline<
    SingleBufferDefinition<EguiVulkanoVertex>,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Arc<RenderPass<EguiRenderPassDesc>>,
>;
type UniformBinding = ([f32; 2], Sampler);
pub struct EguiVulkanoRenderPass {
    pipeline: Arc<Pipeline>,
    vertex_buffers: Vec<Arc<CpuAccessibleBuffer<[EguiVulkanoVertex]>>>,
    index_buffers: Vec<Arc<CpuAccessibleBuffer<[u32]>>>,
    egui_texture_descriptor_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
    egui_texture_version: Option<u64>,
    user_textures: Vec<Option<UserTexture>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    sampler: Arc<Sampler>,
    uniform_buffer_vs: Arc<CpuAccessibleBuffer<vs::ty::UniformBuffer>>,
    descriptor_set_0: Arc<dyn DescriptorSet + Send + Sync>,
}

impl EguiVulkanoRenderPass {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        render_target_format: vulkano::format::Format,
    ) -> Self {
        let fs = fs::Shader::load(device.clone()).unwrap();
        let vs = vs::Shader::load(device.clone()).unwrap();

        let render_pass = Arc::new(
            EguiRenderPassDesc {
                color: (render_target_format, 0),
            }
            .build_render_pass(device.clone())
            .unwrap(),
        );
        let pipeline = Arc::new(
            vulkano::pipeline::GraphicsPipeline::start()
                .render_pass(Subpass::from(render_pass, 0).unwrap())
                .cull_mode_back()
                .depth_stencil_disabled()
                .fragment_shader(fs.main_entry_point(), ())
                .vertex_input_single_buffer()
                .vertex_shader(vs.main_entry_point(), ())
                .primitive_topology(PrimitiveTopology::TriangleList)
                .front_face_clockwise()
                .polygon_mode_fill()
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::OneMinusSrcAlpha,
                    alpha_op: BlendOp::Add,
                    alpha_source: BlendFactor::OneMinusDstAlpha,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .build(device.clone())
                .unwrap(),
        );
        let sampler = Sampler::new(
            device.clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Nearest,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        .unwrap();
        let uniform_buffer_vs = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::uniform_buffer(),
            false,
            vs::ty::UniformBuffer {
                u_screen_size: [1.0, 1.0],
            },
        )
        .unwrap();

        let descriptor_set_0 = Arc::new(
            PersistentDescriptorSet::start(
                pipeline.layout().descriptor_set_layout(0).unwrap().clone(),
            )
            .add_buffer(uniform_buffer_vs.clone())
            .unwrap()
            .add_sampler(sampler.clone())
            .unwrap()
            .build()
            .unwrap(),
        ) as Arc<dyn DescriptorSet + Send + Sync>;

        Self {
            pipeline,
            vertex_buffers: vec![],
            index_buffers: vec![],
            egui_texture_descriptor_set: None,
            egui_texture_version: None,
            user_textures: vec![],
            device,
            queue,
            sampler,
            uniform_buffer_vs,
            descriptor_set_0,
        }
    }
    pub fn execute(
        &mut self,
        color_attachment: Arc<AttachmentImage>,
        paint_jobs: &[ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
        clear_color: Option<[f32; 4]>,
    ) {
        let load_op = if let Some(_color) = clear_color {
            vulkano::framebuffer::LoadOp::Clear
        } else {
            vulkano::framebuffer::LoadOp::Load
        };
        let frame_buffer = Arc::new(
            vulkano::framebuffer::Framebuffer::start(self.pipeline.render_pass().clone())
                .add(color_attachment)
                .unwrap()
                .build()
                .expect("failed to create frame buffer"),
        );

        let mut pass = vulkano::command_buffer::AutoCommandBufferBuilder::new(
            self.device.clone(),
            self.queue.family(),
        )
        .expect("failed to create command buffer builder");
        if load_op == LoadOp::Clear {
            pass.begin_render_pass(
                frame_buffer,
                SubpassContents::Inline,
                vec![clear_color.unwrap().into()],
            )
            .unwrap();
        } else {
            pass.begin_render_pass(frame_buffer, SubpassContents::Inline, vec![])
                .unwrap();
        }
        let scale_factor = screen_descriptor.scale_factor;
        let physical_height = screen_descriptor.physical_height;
        let physical_width = screen_descriptor.physical_width;
        let mut dynamic = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
            compare_mask: None,
            write_mask: None,
            reference: None,
        };
        for ((egui::ClippedMesh(clip_rect, mesh), vertex_buffer), index_buffer) in paint_jobs
            .iter()
            .zip(self.vertex_buffers.iter())
            .zip(self.index_buffers.iter())
        {
            let texture_id = mesh.texture_id;
            let texture_descriptor = self.get_descriptor(texture_id);
            //emit valid texturing
            if let Some(texture_desc_set) = texture_descriptor {
                // Transform clip rect to physical pixels.
                let clip_min_x = scale_factor * clip_rect.min.x;
                let clip_min_y = scale_factor * clip_rect.min.y;
                let clip_max_x = scale_factor * clip_rect.max.x;
                let clip_max_y = scale_factor * clip_rect.max.y;

                // Make sure clip rect can fit within an `u32`.
                let clip_min_x = egui::clamp(clip_min_x, 0.0..=physical_width as f32);
                let clip_min_y = egui::clamp(clip_min_y, 0.0..=physical_height as f32);
                let clip_max_x = egui::clamp(clip_max_x, clip_min_x..=physical_width as f32);
                let clip_max_y = egui::clamp(clip_max_y, clip_min_y..=physical_height as f32);

                let clip_min_x = clip_min_x.round() as u32;
                let clip_min_y = clip_min_y.round() as u32;
                let clip_max_x = clip_max_x.round() as u32;
                let clip_max_y = clip_max_y.round() as u32;

                let width = (clip_max_x - clip_min_x).max(1);
                let height = (clip_max_y - clip_min_y).max(1);
                {
                    // clip scissor rectangle to target size
                    let x = clip_min_x.min(physical_width);
                    let y = clip_min_y.min(physical_height);
                    let width = width.min(physical_width - x);
                    let height = height.min(physical_height - y);

                    // skip rendering with zero-sized clip areas
                    if width == 0 || height == 0 {
                        continue;
                    }
                    dynamic.scissors = Some(vec![Scissor {
                        origin: [x as i32, y as i32],
                        dimensions: [width, height],
                    }]);
                    pass.draw_indexed(
                        self.pipeline.clone(),
                        &dynamic,
                        vertex_buffer.clone(),
                        index_buffer.clone(),
                        texture_desc_set,
                        (),
                    )
                    .unwrap();
                }
            }
        }
        pass.end_render_pass().unwrap();
        let command_buffer = pass.build().unwrap();
        command_buffer
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
    }
    pub fn upload_egui_texture(&mut self, texture: &Texture) {
        //no change
        if self.egui_texture_version == Some(texture.version) {
            return;
        }
        let format = vulkano::format::R8G8B8A8Srgb;
        let image = vulkano::image::ImmutableImage::from_iter(
            texture.pixels.iter().cloned(),
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
            Some(Self::create_texture_binding_from_view(pipeline, image_view));
        self.egui_texture_version = Some(texture.version);
    }
    fn alloc_user_texture(&mut self) -> TextureId {
        for (i, tex) in self.user_textures.iter_mut().enumerate() {
            if tex.is_none() {
                *tex = Some(Default::default());
                return TextureId::User(i as u64);
            }
        }
        let id = TextureId::User(self.user_textures.len() as u64);
        self.user_textures.push(Some(Default::default()));
        id
    }
    fn free_user_texture(&mut self, id: TextureId) {
        if let TextureId::User(id) = id {
            self.user_textures
                .get_mut(id as usize)
                .and_then(|option| option.take());
        }
    }
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
                    user_texture.descriptor_set = Some(Self::create_texture_binding_from_view(
                        pipeline.clone(),
                        image_view,
                    ));
                }
            }
        }
    }

    fn create_texture_binding_from_view(
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
    fn get_descriptor(
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
        //pre alloc area
        if let TextureId::User(id) = id {
            let mut pixels = Vec::with_capacity(size.0 * size.1 * 4);
            //unpack pixels
            for pixel in srgba_pixels {
                pixels.extend(pixel.to_array().iter())
            }
            let texture = UserTexture {
                pixels,
                size: [size.0 as u32, size.1 as u32],
                descriptor_set: None,
            };
            self.user_textures.insert(id as usize, Some(texture))
        }
    }
    /// mark vulkano image view as egui texture id
    /// enables fast and easy off-screen rendering

    pub fn vulkano_texture_as_egui(
        &mut self,
        image_view: Arc<dyn ImageViewAccess + Sync + Send>,
    ) -> TextureId {
        let id = self.alloc_user_texture();
        if let egui::TextureId::User(id) = id {
            let dimension = image_view.dimensions();
            let pipeline = self.pipeline.clone();
            self.user_textures.insert(
                id as usize,
                Some(UserTexture {
                    pixels: vec![],
                    size: [dimension.width(), dimension.height()],
                    descriptor_set: Some(Self::create_texture_binding_from_view(
                        pipeline, image_view,
                    )),
                }),
            )
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

#[derive(Default)]
struct UserTexture {
    pixels: Vec<u8>,
    size: [u32; 2],
    descriptor_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}
mod fs {
    vulkano_shaders::shader! {
    ty:"fragment",
    path:"src/shaders/fragment.glsl"
    }
}
mod vs {
    vulkano_shaders::shader! {
    ty:"vertex",
    path:"src/shaders/vertex.glsl"
    }
}
