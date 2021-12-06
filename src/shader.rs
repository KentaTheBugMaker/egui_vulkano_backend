use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;

use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};

use vulkano::render_pass::{RenderPass, Subpass};

use crate::painter::{PushConstants, WrappedEguiVertex};
use crate::render_pass::render_pass_desc_from_format;
use log::debug;

use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
    ColorComponents,
};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;

use vulkano::pipeline::graphics::rasterization::{
    CullMode, FrontFace, PolygonMode, RasterizationState,
};

use vulkano::descriptor_set::layout::DescriptorType;
use vulkano::image::view::ImageViewType;
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::layout::PipelineLayoutPcRange;
use vulkano::pipeline::{GraphicsPipeline, StateMode};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};
use vulkano::shader::{
    DescriptorRequirements, EntryPointInfo, ShaderExecution, ShaderInterface, ShaderInterfaceEntry,
    ShaderInterfaceEntryType, ShaderModule, ShaderScalarType, ShaderStages,
};
use vulkano::Version;

use vulkano::pipeline::graphics::vertex_input::{
    BuffersDefinition, VertexInputAttributeDescription, VertexInputState,
};

pub(crate) fn create_pipeline(
    device: Arc<Device>,
    render_target_format: Format,
) -> Arc<GraphicsPipeline> {
    //this is safe because we use offline compiled shader binary and shipped with this backend
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
    .expect("failed to create sampler");

    let vs_module =
        unsafe { ShaderModule::from_bytes(device.clone(), include_bytes!("shaders/vert.spv")) }
            .unwrap();

    let fs_module =
        unsafe { ShaderModule::from_bytes(device.clone(), include_bytes!("shaders/frag.spv")) }
            .unwrap();

    let render_pass = RenderPass::new(
        device.clone(),
        render_pass_desc_from_format(render_target_format),
    )
    .unwrap();
    debug!("renderpass created");
    let pipeline = {
        vulkano::pipeline::GraphicsPipeline::start()
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .fragment_shader(fs_module.entry_point("main").unwrap(), ())
            .vertex_input_state(BuffersDefinition::new().vertex::<WrappedEguiVertex>())
            .vertex_shader(vs_module.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleList),
            )
            .rasterization_state(
                RasterizationState::new()
                    .cull_mode(CullMode::None)
                    .front_face(FrontFace::CounterClockwise)
                    .polygon_mode(PolygonMode::Fill),
            )
            .depth_stencil_state(DepthStencilState::disabled())
            .color_blend_state(ColorBlendState {
                logic_op: None,
                attachments: vec![ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::OneMinusSrcAlpha,
                        alpha_op: BlendOp::Add,
                        alpha_source: BlendFactor::OneMinusDstAlpha,
                        alpha_destination: BlendFactor::One,
                    }),
                    color_write_mask: ColorComponents {
                        r: true,
                        g: true,
                        b: true,
                        a: true,
                    },
                    color_write_enable: StateMode::Fixed(true),
                }],
                blend_constants: StateMode::Fixed([1.0, 1.0, 1.0, 1.0]),
            })
            .with_auto_layout(device, |x| {
                x[0].set_immutable_samplers(0, [sampler]);
            })
            .unwrap()
    };
    debug!("pipeline created");
    pipeline
}
