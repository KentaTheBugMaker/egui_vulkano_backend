use std::ffi::CStr;
use std::sync::Arc;

use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::layout::PipelineLayoutPcRange;
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterface, ShaderInterfaceEntry, ShaderModule, ShaderStages,
    SpecializationConstants,
};
use vulkano::render_pass::{RenderPass, Subpass};

use crate::painter::{EguiVulkanoVertex, Pipeline, PushConstants};
use crate::render_pass::render_pass_desc_from_format;
use log::debug;
use vulkano::descriptor_set::layout::{
    DescriptorDesc, DescriptorDescTy, DescriptorImageDesc, DescriptorImageDescArray,
    DescriptorImageDescDimensions, DescriptorSetDesc,
};

pub(crate) fn create_pipeline(device: Arc<Device>, render_target_format: Format) -> Arc<Pipeline> {
    let fs=fs::Shader::load(device.clone()).unwrap();
    let vs=vs::Shader::load(device.clone()).unwrap();
    let render_pass = Arc::new(
        RenderPass::new(
            device.clone(),
            render_pass_desc_from_format(render_target_format),
        )
        .unwrap(),
    );
    debug!("renderpass created");
    let pipeline = Arc::new({
        vulkano::pipeline::GraphicsPipeline::start()
            .viewports_scissors_dynamic(1)
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .fragment_shader(fs.main_entry_point(), ())
            .vertex_input_single_buffer::<EguiVulkanoVertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .primitive_topology(PrimitiveTopology::TriangleList)
            .front_face_clockwise()
            .polygon_mode_fill()
            .depth_stencil_disabled()
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
            .build(device)
            .unwrap()
    });
    debug!("pipeline created");
    pipeline
}
mod vs{
    vulkano_shaders::shader!{
        ty: "vertex",
        path:"./src/shaders/shader.vert"
    }
}
mod fs{
    vulkano_shaders::shader!{
        ty: "fragment",
        path:"./src/shaders/shader.frag"
    }
}