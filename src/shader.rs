use crate::render_pass::EguiRenderPassDesc;
use crate::Pipeline;
use std::ffi::CString;
use std::sync::Arc;
use vulkano::descriptor::descriptor::{
    DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, DescriptorImageDesc,
    DescriptorImageDescArray, DescriptorImageDescDimensions, ShaderStages,
};
use vulkano::descriptor::pipeline_layout::{PipelineLayoutDesc, PipelineLayoutDescPcRange};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::framebuffer::{RenderPassDesc, Subpass};
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::shader::{
    GraphicsShaderType, ShaderInterfaceDef, ShaderInterfaceDefEntry, ShaderModule,
};

struct VSInterfaceIn;
struct VSInterfaceOut;
struct FSInterfaceIn;
struct FSInterfaceOut;
unsafe impl ShaderInterfaceDef for VSInterfaceIn {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        Box::new(
            [
                ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: Format::R32G32Sfloat,
                    name: Some(std::borrow::Cow::Borrowed("a_pos")),
                },
                ShaderInterfaceDefEntry {
                    location: 1..2,
                    format: Format::R32G32Sfloat,
                    name: Some(std::borrow::Cow::Borrowed("a_tex_coord")),
                },
                ShaderInterfaceDefEntry {
                    location: 2..3,
                    format: Format::R32Uint,
                    name: Some(std::borrow::Cow::Borrowed("a_color")),
                },
            ]
            .iter()
            .cloned(),
        )
    }
}
unsafe impl ShaderInterfaceDef for VSInterfaceOut {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        Box::new(
            [
                ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: Format::R32G32Sfloat,
                    name: Some(std::borrow::Cow::Borrowed("v_tex_coord")),
                },
                ShaderInterfaceDefEntry {
                    location: 1..2,
                    format: Format::R32G32B32A32Sfloat,
                    name: Some(std::borrow::Cow::Borrowed("v_color")),
                },
            ]
            .iter()
            .cloned(),
        )
    }
}
unsafe impl ShaderInterfaceDef for FSInterfaceIn {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        Box::new(
            [
                ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: Format::R32G32Sfloat,
                    name: Some(std::borrow::Cow::Borrowed("v_tex_coord")),
                },
                ShaderInterfaceDefEntry {
                    location: 1..2,
                    format: Format::R32G32B32A32Sfloat,
                    name: Some(std::borrow::Cow::Borrowed("v_color")),
                },
            ]
            .iter()
            .cloned(),
        )
    }
}
unsafe impl ShaderInterfaceDef for FSInterfaceOut {
    type Iter = Box<dyn ExactSizeIterator<Item = ShaderInterfaceDefEntry>>;

    fn elements(&self) -> Self::Iter {
        Box::new(
            [ShaderInterfaceDefEntry {
                location: 0..1,
                format: Format::R32G32B32A32Sfloat,
                name: Some(std::borrow::Cow::Borrowed("f_color")),
            }]
            .iter()
            .cloned(),
        )
    }
}
#[derive(Default, Debug, Copy, Clone)]
struct PipelineLayout;
unsafe impl PipelineLayoutDesc for PipelineLayout {
    fn num_sets(&self) -> usize {
        2
    }

    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        if set == 1 {
            Some(1)
        } else if set == 0 {
            Some(1)
        } else {
            None
        }
    }

    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        if set == 0 {
            if binding == 0 {
                Some(DescriptorDesc {
                    ty: DescriptorDescTy::Sampler,
                    array_count: 1,
                    stages: ShaderStages {
                        vertex: false,
                        tessellation_control: false,
                        tessellation_evaluation: false,
                        geometry: false,
                        fragment: true,
                        compute: false,
                    },
                    readonly: true,
                })
            }else {
                None
            }
        } else if set == 1 {
            if binding == 0 {
                Some(DescriptorDesc {
                    ty: DescriptorDescTy::Image(DescriptorImageDesc {
                        sampled: true,
                        dimensions: DescriptorImageDescDimensions::TwoDimensional,
                        format: None,
                        multisampled: false,
                        array_layers: DescriptorImageDescArray::NonArrayed,
                    }),
                    array_count: 1,
                    stages: ShaderStages {
                        vertex: false,
                        tessellation_control: false,
                        tessellation_evaluation: false,
                        geometry: false,
                        fragment: true,
                        compute: false,
                    },
                    readonly: true,
                })
            }else{
                None
            }
        } else {
            None
        }
    }

    fn num_push_constants_ranges(&self) -> usize {
        1
    }

    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        if num==0{
            Some(PipelineLayoutDescPcRange{
                offset: 0,
                size: 12,
                stages: ShaderStages {
                    vertex: true,
                    tessellation_control: false,
                    tessellation_evaluation: false,
                    geometry: false,
                    fragment: true,
                    compute: false
                }
            })
        }else{
            None
        }
    }
}
pub(crate) fn create_pipeline(device: Arc<Device>, render_target_format: Format) -> Arc<Pipeline> {
    let vs_module =
        unsafe { ShaderModule::new(device.clone(), include_bytes!("shaders/vert.spv")) }.unwrap();
    let fs_module =
        unsafe { ShaderModule::new(device.clone(), include_bytes!("shaders/frag.spv")) }.unwrap();
    let main = CString::new("main").unwrap();
    let vs_entry = unsafe {
        vs_module.graphics_entry_point(
            &main,
            VSInterfaceIn,
            VSInterfaceOut,
            PipelineLayout,
            GraphicsShaderType::Vertex,
        )
    };
    let fs_entry = unsafe {
        fs_module.graphics_entry_point(
            &main,
            FSInterfaceIn,
            FSInterfaceOut,
            PipelineLayout,
            GraphicsShaderType::Fragment,
        )
    };
    let render_pass = Arc::new(
        EguiRenderPassDesc {
            color: (render_target_format, 0),
        }
        .build_render_pass(device.clone())
        .unwrap(),
    );

    let pipeline = Arc::new(
        vulkano::pipeline::GraphicsPipeline::start()
            .viewports_scissors_dynamic(1)
            .render_pass(Subpass::from(render_pass, 0).unwrap())
            .depth_stencil_disabled()
            .fragment_shader(fs_entry, ())
            .vertex_input_single_buffer()
            .vertex_shader(vs_entry, ())
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
            .build(device)
            .unwrap(),
    );
    pipeline
}
