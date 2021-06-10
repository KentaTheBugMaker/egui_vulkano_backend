use crate::model::{Normal, Vertex};

use std::borrow::Cow;
use std::ffi::CString;
use std::sync::Arc;
use vulkano::descriptor::descriptor::{
    DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages,
};
use vulkano::descriptor::pipeline_layout::{PipelineLayoutDesc, PipelineLayoutDescPcRange};
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageLayout, ImageUsage};
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterfaceDef, ShaderInterfaceDefEntry, ShaderModule,
};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::render_pass::{
    AttachmentDesc, Framebuffer, FramebufferAbstract, LoadOp, RenderPass, RenderPassDesc, StoreOp,
    Subpass, SubpassDesc,
};

// shader interface definition
struct VSInput;
unsafe impl ShaderInterfaceDef for VSInput {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        /*
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
        */
        Box::new(
            [
                ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("position")),
                },
                ShaderInterfaceDefEntry {
                    location: 1..2,
                    format: Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("normal")),
                },
            ]
            .iter()
            .cloned(),
        )
    }
}
//VS->FS interface
struct VSOutput;
unsafe impl ShaderInterfaceDef for VSOutput {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        /*layout(location = 0) out vec3 v_normal;*/
        Box::new(
            [ShaderInterfaceDefEntry {
                location: 0..1,
                format: Format::R32G32B32Sfloat,
                name: Some(Cow::Borrowed("v_normal")),
            }]
            .iter()
            .cloned(),
        )
    }
}

struct FSOutput;
unsafe impl ShaderInterfaceDef for FSOutput {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        /*layout(location = 0) out vec4 f_color;*/
        Box::new(
            [ShaderInterfaceDefEntry {
                location: 0..1,
                format: Format::R32G32B32A32Sfloat,
                name: Some(Cow::Borrowed("f_color")),
            }]
            .iter()
            .cloned(),
        )
    }
}
//render pass
fn create_renderpass() -> RenderPassDesc {
    let color = AttachmentDesc {
        format: Format::R8G8B8A8Srgb,
        samples: 1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Load,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::ColorAttachmentOptimal,
        final_layout: ImageLayout::ColorAttachmentOptimal,
    };
    let depth = AttachmentDesc {
        format: Format::D32Sfloat,
        samples: 1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Load,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::DepthStencilAttachmentOptimal,
        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
    };
    let sub_pass_desc = SubpassDesc {
        color_attachments: vec![(0, ImageLayout::ColorAttachmentOptimal)],
        depth_stencil: Some((1, ImageLayout::DepthStencilAttachmentOptimal)),
        input_attachments: vec![],
        resolve_attachments: vec![],
        preserve_attachments: vec![],
    };
    RenderPassDesc::new(vec![color, depth], vec![sub_pass_desc], vec![])
}
#[derive(Default, Debug, Copy, Clone)]
struct PipelineLayout;
unsafe impl PipelineLayoutDesc for PipelineLayout {
    fn num_sets(&self) -> usize {
        1
    }

    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        if set == 0 {
            Some(1)
        } else {
            None
        }
    }

    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        if set == 0 && binding == 0 {
            Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: None,
                    storage: false,
                }),
                array_count: 1,
                stages: ShaderStages {
                    vertex: true,
                    tessellation_control: false,
                    tessellation_evaluation: false,
                    geometry: false,
                    fragment: false,
                    compute: false,
                },
                readonly: true,
            })
        } else {
            None
        }
    }

    fn num_push_constants_ranges(&self) -> usize {
        0
    }

    fn push_constants_range(&self, _num: usize) -> Option<PipelineLayoutDescPcRange> {
        None
    }
}
fn create_pipeline(
    device: Arc<Device>,
) -> Arc<
    GraphicsPipeline<
        TwoBuffersDefinition<Vertex, Normal>,
        Box<dyn PipelineLayoutAbstract + Send + Sync>,
    >,
> {
    let vs = unsafe { ShaderModule::new(device.clone(), include_bytes!("vert.spv")) }.unwrap();
    let fs = unsafe { ShaderModule::new(device.clone(), include_bytes!("frag.spv")) }.unwrap();
    let cstring = CString::new("main").unwrap();
    let vs_entry = unsafe {
        vs.graphics_entry_point(
            &cstring,
            VSInput,
            VSOutput,
            PipelineLayout,
            GraphicsShaderType::Vertex,
        )
    };
    let fs_entry = unsafe {
        fs.graphics_entry_point(
            &cstring,
            VSOutput,
            FSOutput,
            PipelineLayout,
            GraphicsShaderType::Fragment,
        )
    };
    let render_pass = Arc::new(RenderPass::new(device.clone(), create_renderpass()).unwrap());
    Arc::new(
        GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<Vertex, Normal>::new())
            .vertex_shader(vs_entry, ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs_entry, ())
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .build(device.clone())
            .unwrap(),
    )
}
//Renderer
struct TeapotRenderer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    render_pass: Arc<RenderPass>,
    frame_buffer: Option<Arc<dyn FramebufferAbstract + Send + Sync>>,
}
impl TeapotRenderer {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> TeapotRenderer {
        let pipeline = create_pipeline(device.clone());
        let render_pass = pipeline.render_pass().clone();
        Self {
            device,
            queue,
            pipeline: pipeline as Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
            render_pass,
            frame_buffer: None,
        }
    }
    pub fn set_render_target(&mut self, rt: Arc<AttachmentImage>) {
        //create depth
        let depth = AttachmentImage::with_usage(
            self.device.clone(),
            rt.dimensions(),
            Format::D32Sfloat,
            ImageUsage::depth_stencil_attachment(),
        )
        .unwrap();
        //create framebuffer
        let frame_buffer = Framebuffer::start(self.render_pass.clone())
            .add(ImageView::new(rt).unwrap())
            .unwrap()
            .add(ImageView::new(depth).unwrap())
            .unwrap()
            .build()
            .unwrap() as Arc<dyn FramebufferAbstract + Send + Sync>;
        self.frame_buffer.replace(frame_buffer);
    }
    pub fn draw(&mut self){
        if let Some(frame_buffer)=self.frame_buffer.as_ref(){

        }else {
            panic!("Frame buffer is blank")
        }
    }
}
