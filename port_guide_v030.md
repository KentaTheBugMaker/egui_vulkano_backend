# Port guide for egui_vulkano_backend 0.3.0

## EguiVulkanoRenderPass::create_command_buffer

### if you want render to texture

#### before

 ```rust
let format = vulkano::format::Format::R8G8B8A8Srgb;
let dimensions = [640,480];
let image = AttachmentImage::new(device.clone(),dimensions,format).unwrap() as Arc<dyn ImageViewAccess +Send+Sync>;
let command_buffer = render_pass.create_command_buffer(Some(image),None,&paint_jobs,&screen_descriptor);
```

#### after

```rust
let format = vulkano::format::Format::R8G8B8A8Srgb;
let dimensions= [640,480];
let image = AttachmentImage::new(device.clone(),dimensions,format).unwrap();
let image_view = ImageView::new(image.clone()).unwrap() as Arc<dyn ImageViewAbstract +Send+Sync>;
let render_target = RenderTarget::ColorAttachment(image_view)
let command_buffer = render_pass.create_command_buffer(render_target,&paint_jobs,&screen_descriptor);
```

### if you want to render to Swapchain

#### before

 ```rust
 render_pass.create_frame_buffers(&images);
 let command_buffer = render_pass.create_command_buffer(None,Some(image_num),&paint_jobs,&screen_descriptor);
```

#### after

 ```rust
 render_pass.create_frame_buffers(&images);
 let render_target = RenderTarget::FrameBufferIndex(image_num,&paint_jobs,&screen_descriptor);
 let command_buffer = render_pass.create_command_buffer(render_target,&paint_jobs,&screen_descriptor);
```