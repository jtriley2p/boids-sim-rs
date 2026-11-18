//! GPU boids: `cargo run --release -- [count] [--novsync]`

use std::{fmt, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const MAX_CELLS: u32 = 1 << 21;
const WG: u32 = 256;
const SCAN_BLOCK: u32 = WG * 16;

#[derive(Debug)]
enum Error {
    EventLoop(winit::error::EventLoopError),
    Window(winit::error::OsError),
    Surface(wgpu::CreateSurfaceError),
    Adapter(wgpu::RequestAdapterError),
    Device(wgpu::RequestDeviceError),
    SurfaceUnsupported,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::EventLoop(e) => write!(f, "event loop: {e}"),
            Error::Window(e) => write!(f, "window: {e}"),
            Error::Surface(e) => write!(f, "surface: {e}"),
            Error::Adapter(e) => write!(f, "adapter: {e}"),
            Error::Device(e) => write!(f, "device: {e}"),
            Error::SurfaceUnsupported => write!(f, "surface unsupported by adapter"),
        }
    }
}

/// Runtime-tunable sim parameters, mirrored in the shader's `Params`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct Tune {
    sep_r: f32, // separation radius, fraction of r
    w_sep: f32,
    w_ali: f32,
    w_coh: f32,
    noise: f32,   // heading diffusion, rad / sqrt(s)
    max_acc: f32, // steering cap, 0 = off
    min_speed: f32,
    max_speed: f32,
}

impl Default for Tune {
    fn default() -> Self {
        Tune { sep_r: 0.4, w_sep: 2.0, w_ali: 4.0, w_coh: 4.0, noise: 0.0, max_acc: 0.0, min_speed: 0.08, max_speed: 0.25 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    world: [f32; 2],
    cell: [f32; 2],
    grid: [u32; 2],
    dt: f32,
    r: f32,
    tune: Tune,
}

impl Params {
    /// `scale` multiplies the default perception radius.
    fn new(width: u32, height: u32, n: u32, scale: f32, tune: Tune) -> Self {
        let world = [width.max(1) as f32 / height.max(1) as f32, 1.0];
        // perception radius (world height = 1), scaled to keep neighbor counts constant
        // capped at 1/5 of the shorter axis so a search row (<= 5 cells) never wraps onto itself
        let radius = (scale * 0.012 * (100_000.0 / n as f32).sqrt()).min(0.2 * world[0].min(world[1]));
        // largest grid with cells >= radius / 2, capped so the buffer fits
        let mut r = radius / 2.0;
        let grid = loop {
            let g = world.map(|w| (w / r) as u32);
            if g[0] * g[1] < MAX_CELLS { break g; }
            r *= 1.25;
        };
        let cell = [world[0] / grid[0] as f32, world[1] / grid[1] as f32];
        Params { world, cell, grid, dt: 0.0, r: radius, tune }
    }
}

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    params: Params,
    scale: f32,
    resets: u64,
    params_buf: wgpu::Buffer,
    boids_buf: wgpu::Buffer,
    cells: [wgpu::Buffer; 2],
    compute: [[wgpu::BindGroup; 6]; 2], // per parity: cells[parity] is scanned, cells[1 - parity] counted by sim
    parity: usize,
    recount: bool,
    draw: wgpu::BindGroup,
    passes: [wgpu::ComputePipeline; 6],
    render: wgpu::RenderPipeline,
    egui: egui::Context,
    egui_state: egui_winit::State,
    egui_rend: egui_wgpu::Renderer,
    n: u32,
    last: Instant,
    fps_t: Instant,
    frames: u32,
}

impl Gpu {
    fn new(window: Arc<Window>, n: u32, vsync: bool) -> Result<Self, Error> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).map_err(Error::Surface)?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(Error::Adapter)?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(Error::Device)?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or(Error::SurfaceUnsupported)?;
        config.present_mode = if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync };
        surface.configure(&device, &config);

        // stay within dispatch and buffer limits
        let lim = device.limits();
        let n = n.min(lim.max_compute_workgroups_per_dimension * WG).min(u32::MAX / 3).min((lim.max_storage_buffer_binding_size / 16) as u32);
        let params = Params::new(config.width, config.height, n, 1.0, Tune::default());
        let boids = seed(n, params.world, 0);
        let storage = wgpu::BufferUsages::STORAGE;
        let buf = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        let params_buf = buf("params", size_of::<Params>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let boids_buf = buf("boids", boids.len() as u64 * 4, storage | wgpu::BufferUsages::COPY_DST);
        let sorted = buf("sorted", boids.len() as u64 * 4, storage);
        let cells = ["cells0", "cells1"].map(|l| buf(l, (MAX_CELLS as u64 + 1) * 4, storage | wgpu::BufferUsages::COPY_DST));
        let blocks = buf("blocks", MAX_CELLS.div_ceil(SCAN_BLOCK) as u64 * 4, storage);
        let rank = buf("rank", n as u64 * 4, storage);
        queue.write_buffer(&boids_buf, 0, bytemuck::cast_slice(&boids));

        let module = device.create_shader_module(wgpu::include_wgsl!("boids.wgsl"));

        let passes = ["count", "reduce", "scan_blocks", "scan", "scatter", "sim"].map(|ep| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: None,
                module: &module,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            })
        });

        let render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let bind = |layout: &wgpu::BindGroupLayout, bufs: &[(u32, &wgpu::Buffer)]| {
            let entries: Vec<_> = bufs
                .iter()
                .map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
        };
        let compute = [0, 1].map(|i| {
            let (cur, next) = (&cells[i], &cells[1 - i]);
            let entries = [
                &[(0, &params_buf), (1, &boids_buf), (3, cur), (9, &rank)][..],
                &[(0, &params_buf), (7, cur), (8, &blocks)],
                &[(0, &params_buf), (7, cur), (8, &blocks)],
                &[(0, &params_buf), (7, cur), (8, &blocks)],
                &[(0, &params_buf), (1, &boids_buf), (2, &sorted), (5, cur), (9, &rank)],
                &[(0, &params_buf), (1, &boids_buf), (3, next), (5, cur), (6, &sorted), (9, &rank)],
            ];
            std::array::from_fn(|k| bind(&passes[k].get_bind_group_layout(0), entries[k]))
        });
        let draw = bind(&render.get_bind_group_layout(0), &[(0, &params_buf), (4, &boids_buf)]);

        let egui = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            Some(lim.max_texture_dimension_2d as usize),
        );
        let egui_rend = egui_wgpu::Renderer::new(&device, config.format, Default::default());

        let now = Instant::now();
        Ok(Gpu {
            window, surface, device, queue, config, params, scale: 1.0, resets: 0, params_buf, boids_buf, cells, compute,
            parity: 0, recount: true, draw, passes, render, egui, egui_state, egui_rend, n, last: now, fps_t: now, frames: 0,
        })
    }

    /// Rebuild the grid (window size or radius changed).
    fn regrid(&mut self) {
        self.params = Params::new(self.config.width, self.config.height, self.n, self.scale, self.params.tune);
        self.recount = true; // sim counted with the old grid
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.regrid();
    }

    /// Re-seed all boids with a fresh random state.
    fn reset(&mut self) {
        self.resets += 1;
        let boids = seed(self.n, self.params.world, self.resets);
        self.queue.write_buffer(&self.boids_buf, 0, bytemuck::cast_slice(&boids));
        self.recount = true;
    }

    /// Run the tuning panel and apply any changes; returns egui's output for painting.
    fn ui(&mut self) -> egui::FullOutput {
        let raw = self.egui_state.take_egui_input(&self.window);
        let (mut t, mut scale, mut reset) = (self.params.tune, self.scale, false);
        let out = self.egui.run_pass(raw, |ctx| panel(ctx, &mut t, &mut scale, &mut reset));
        self.params.tune = t;
        if scale != self.scale {
            self.scale = scale;
            self.regrid();
        }
        if reset { self.reset(); }
        out
    }

    fn frame(&mut self) {
        let now = Instant::now();
        self.params.dt = (now - self.last).as_secs_f32().min(1.0 / 30.0);
        self.last = now;
        self.frames += 1;
        let elapsed = (now - self.fps_t).as_secs_f32();
        if elapsed >= 1.0 {
            self.window.set_title(&format!("boids — {} — {:.0} fps", self.n, self.frames as f32 / elapsed));
            self.fps_t = now;
            self.frames = 0;
        }

        let tex = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = tex.texture.create_view(&Default::default());

        // after acquiring the texture, so egui's texture deltas are never dropped
        let ui = self.ui();
        self.egui_state.handle_platform_output(&self.window, ui.platform_output);
        let jobs = self.egui.tessellate(ui.shapes, ui.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui.pixels_per_point,
        };
        for (id, deltas) in &ui.textures_delta.set {
            for delta in deltas {
                self.egui_rend.update_texture(&self.device, &self.queue, *id, delta);
            }
        }

        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        let mut enc = self.device.create_command_encoder(&Default::default());
        let cells = self.params.grid[0] * self.params.grid[1];
        let used = Some((cells as u64 + 1) * 4);
        let cur = self.parity;
        if self.recount { enc.clear_buffer(&self.cells[cur], 0, used); }
        enc.clear_buffer(&self.cells[1 - cur], 0, used);
        let groups = self.n.div_ceil(WG);
        let blocks = cells.div_ceil(SCAN_BLOCK);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            let skip = if self.recount { 0 } else { 1 };
            let work = [groups, blocks, 1, blocks, groups, groups];
            for ((pipeline, bg), g) in self.passes.iter().zip(&self.compute[cur]).zip(work).skip(skip) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(g, 1, 1);
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.01, g: 0.01, b: 0.02, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.render);
            pass.set_bind_group(0, &self.draw, &[]);
            pass.draw(0..3 * self.n, 0..1);
        }
        let extra = self.egui_rend.update_buffers(&self.device, &self.queue, &mut enc, &jobs, &screen);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_rend.render(&mut pass, &jobs, &screen);
        }
        self.queue.submit(extra.into_iter().chain([enc.finish()]));
        for id in &ui.textures_delta.free {
            self.egui_rend.free_texture(id);
        }
        self.parity = 1 - cur;
        self.recount = false;
        self.window.pre_present_notify();
        self.queue.present(tex);
    }
}

/// Tuning panel, top right. Sets `reset` when the reset button is clicked.
fn panel(ctx: &egui::Context, t: &mut Tune, scale: &mut f32, reset: &mut bool) {
    use egui::Slider;
    egui::Window::new("params").anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0]).resizable(false).show(ctx, |ui| {
        ui.add(Slider::new(&mut t.w_sep, 0.0..=10.0).text("separation"));
        ui.add(Slider::new(&mut t.w_ali, 0.0..=10.0).text("alignment"));
        ui.add(Slider::new(&mut t.w_coh, 0.0..=10.0).text("cohesion"));
        ui.add(Slider::new(&mut t.sep_r, 0.05..=1.0).text("sep radius"));
        ui.add(Slider::new(scale, 0.25..=4.0).logarithmic(true).text("radius ×"));
        ui.add(Slider::new(&mut t.noise, 0.0..=5.0).text("noise"));
        ui.add(Slider::new(&mut t.max_acc, 0.0..=5.0).text("max accel (0 = off)"));
        let hi = t.max_speed;
        ui.add(Slider::new(&mut t.min_speed, 0.0..=hi).text("min speed"));
        let lo = t.min_speed;
        ui.add(Slider::new(&mut t.max_speed, lo.max(0.01)..=1.0).text("max speed"));
        ui.horizontal(|ui| {
            *reset |= ui.button("reset").clicked();
            if ui.button("defaults").clicked() {
                *t = Tune::default();
                *scale = 1.0;
            }
        });
    });
}

/// Random positions and headings from a xorshift PRNG; returns flat [px, py, vx, vy] per boid.
fn seed(n: u32, world: [f32; 2], k: u64) -> Vec<f32> {
    let mut s = 0x9e3779b97f4a7c15u64.wrapping_mul(k + 1); // odd constant: nonzero for any k
    let mut rand = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n)
        .flat_map(|_| {
            let a = rand() * std::f32::consts::TAU;
            [rand() * world[0], rand() * world[1], a.cos() * 0.15, a.sin() * 0.15]
        })
        .collect()
}

struct App {
    n: u32,
    vsync: bool,
    gpu: Option<Gpu>,
    err: Option<Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gpu.is_some() { return; }
        let attrs = Window::default_attributes().with_title("boids");
        let res = el.create_window(attrs).map_err(Error::Window).and_then(|w| Gpu::new(Arc::new(w), self.n, self.vsync));
        match res {
            Ok(gpu) => self.gpu = Some(gpu),
            Err(e) => {
                self.err = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(gpu) = &mut self.gpu else { return };
        let consumed = gpu.egui_state.on_window_event(&gpu.window, &event).consumed;
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key: Key::Named(NamedKey::Escape), state: ElementState::Pressed, .. },
                ..
            } if !consumed => el.exit(),
            WindowEvent::Resized(s) => gpu.resize(s.width, s.height),
            WindowEvent::RedrawRequested => {
                gpu.frame();
                gpu.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let mut app = App { n: 1_000_000, vsync: true, gpu: None, err: None };
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--novsync" => app.vsync = false,
            s => match s.parse::<u32>() {
                Ok(n) if n > 0 => app.n = n,
                _ => eprintln!("ignoring arg: {s}"),
            },
        }
    }
    let res = EventLoop::new().map_err(Error::EventLoop).and_then(|el| el.run_app(&mut app).map_err(Error::EventLoop));
    if let Some(e) = res.err().or(app.err) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
