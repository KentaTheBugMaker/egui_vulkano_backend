use vulkano::format::Format;
use vulkano::image::{ImageLayout, SampleCount};
use vulkano::render_pass::{AttachmentDesc, LoadOp, RenderPassDesc, StoreOp, SubpassDesc};

#[derive(Copy, Clone)]
pub struct EguiRenderPassDesc {
    pub color: (Format, u32),
}

pub(crate) fn render_pass_desc_from_format(format: Format) -> RenderPassDesc {
    let attachment_desc = AttachmentDesc {
        format,
        samples: SampleCount::Sample1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Clear,
        stencil_store: StoreOp::Store,
        initial_layout: ImageLayout::ColorAttachmentOptimal,
        final_layout: ImageLayout::ColorAttachmentOptimal,
    };
    let depth_attachment = AttachmentDesc {
        format: vulkano::format::Format::D32Sfloat,
        samples: SampleCount::Sample1,
        load: LoadOp::Clear,
        store: StoreOp::Store,
        stencil_load: LoadOp::Clear,
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
    RenderPassDesc::new(
        vec![attachment_desc, depth_attachment],
        vec![sub_pass_desc],
        vec![],
    )
}
