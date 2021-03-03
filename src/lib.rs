mod render_pass;

extern crate vulkano_shaders;
extern crate vulkano;

use std::sync::Arc;
use vulkano::image::{ImageViewAccess, MipmapsCount, Dimensions};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::blend::{BlendFactor, BlendOp, AttachmentBlend};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::framebuffer::{Subpass, RenderPass, RenderPassDesc};
use crate::render_pass::EguiRenderPassDesc;
use vulkano::device::{Queue, Device};
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::format::Format::R8G8B8A8Srgb;
use epi::egui::{Texture, TextureId, Color32};


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

type Pipeline = GraphicsPipeline<
    SingleBufferDefinition<EguiVulkanoVertex>,
    Box<dyn PipelineLayoutAbstract + Send + Sync>,
    Arc<RenderPass<EguiRenderPassDesc>>,
>;
pub struct EguiVulkanoRenderPass {
    pipeline: Pipeline,
    vertex_buffer: Option<Arc<CpuAccessibleBuffer<[EguiVulkanoVertex]>>>,
    index_buffer: Option<Arc<CpuAccessibleBuffer<[u32]>>>,
    egui_texture: Option<Arc<vulkano::image::ImmutableImage<vulkano::format::R8G8B8A8Srgb>>>,
    egui_texture_version: Option<u64>,
    user_textures: Vec<Option<UserTexture>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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
        let pipeline = vulkano::pipeline::GraphicsPipeline::start()
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
            .unwrap();

        Self {
            pipeline,
            vertex_buffer: None,
            index_buffer: None,
            egui_texture: None,
            egui_texture_version: None,
            user_textures: vec![],
            device,
            queue,
        }
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
        self.egui_texture = Some(image.0);
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
        for user_texture in &mut self.user_textures {
            if let Some(user_texture) = user_texture {
                //when uploaded image is none
                if user_texture.vulkano_image_view.is_none() {
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
                    let image_view = image as Arc<dyn ImageViewAccess>;
                    user_texture.vulkano_image_view = Some(image_view)
                }
            }
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
                vulkano_image_view: None,
            };
            self.user_textures.insert(id as usize, Some(texture))
        }
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
    vulkano_image_view: Option<Arc<dyn vulkano::image::ImageViewAccess>>,
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
