// GPU boids with a spatial-hash grid rebuilt every frame by counting sort:
// reduce -> scan_blocks -> scan -> scatter -> sim. Toroidal world [0, world).
// sim counts boids into the next frame's cells and records each boid's rank within its cell;
// count does the same, only needed when there is no previous sim (start, resize).

struct Params {
    world: vec2f, // world size (height = 1)
    cell: vec2f,  // cell size, >= r / 2 on both axes
    grid: vec2u,  // cells per axis, >= 10
    dt: f32,
    r: f32,       // perception radius
    // runtime-tunable (see `Tune` in main.rs)
    sep_r: f32,   // separation radius, fraction of r
    w_sep: f32,
    w_ali: f32,
    w_coh: f32,
    noise: f32,   // heading diffusion, rad / sqrt(s)
    max_acc: f32, // steering cap, 0 = off
    min_speed: f32,
    max_speed: f32,
}

const WG = 256u;
const K = 16u;         // cells per thread in a scan block
const B = WG * K;      // cells per scan block

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> boids: array<vec4f>;  // xy = pos, zw = vel
@group(0) @binding(2) var<storage, read_write> sorted: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> cells: array<atomic<u32>>; // counts, grid.x * grid.y + 1
@group(0) @binding(4) var<storage, read> draw: array<vec4f>;          // render-only view of boids
@group(0) @binding(5) var<storage, read> starts: array<u32>;          // scanned cells: [start, end) = starts[c], starts[c + 1]
@group(0) @binding(6) var<storage, read> nbr: array<vec4f>;           // sim-only view of sorted
@group(0) @binding(7) var<storage, read_write> cnt: array<u32>;       // scan-only view of cells
@group(0) @binding(8) var<storage, read_write> blocks: array<u32>;    // per-block sums
@group(0) @binding(9) var<storage, read_write> rank: array<u32>;      // per boid: index within its cell

fn cell_of(pos: vec2f) -> u32 {
    let c = min(vec2u(max(pos / p.cell, vec2f(0.0))), p.grid - 1u);
    return c.y * p.grid.x + c.x;
}

@compute @workgroup_size(WG)
fn count(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&boids) { return; }
    rank[id.x] = atomicAdd(&cells[cell_of(boids[id.x].xy)], 1u);
}

// Exclusive scan of cell counts in three steps: block sums, scan of block sums, block-local scans.
var<workgroup> sums: array<u32, WG>;

// Hillis-Steele inclusive scan of `sums`.
fn scan_sums(t: u32) {
    for (var o = 1u; o < WG; o <<= 1u) {
        var v = 0u;
        if t >= o { v = sums[t - o]; }
        workgroupBarrier();
        sums[t] += v;
        workgroupBarrier();
    }
}

@compute @workgroup_size(WG)
fn reduce(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) t: u32) {
    let n = p.grid.x * p.grid.y;
    var s = 0u;
    for (var k = 0u; k < K; k++) {
        let i = wid.x * B + k * WG + t;
        if i < n { s += cnt[i]; }
    }
    sums[t] = s;
    workgroupBarrier();
    for (var o = WG / 2u; o > 0u; o >>= 1u) {
        if t < o { sums[t] += sums[t + o]; }
        workgroupBarrier();
    }
    if t == 0u { blocks[wid.x] = sums[0]; }
}

// Single workgroup: exclusive scan of block sums, each thread owning a contiguous chunk.
@compute @workgroup_size(WG)
fn scan_blocks(@builtin(local_invocation_index) t: u32) {
    let n = p.grid.x * p.grid.y;
    let nb = (n + B - 1u) / B;
    let per = (nb + WG - 1u) / WG;
    let lo = min(t * per, nb);
    let hi = min(lo + per, nb);
    var s = 0u;
    for (var i = lo; i < hi; i++) { s += blocks[i]; }
    sums[t] = s;
    workgroupBarrier();
    scan_sums(t);
    var acc = sums[t] - s;
    for (var i = lo; i < hi; i++) {
        let v = blocks[i];
        blocks[i] = acc;
        acc += v;
    }
    if t == 0u { cnt[n] = sums[WG - 1u]; }
}

// Each thread scans its own K contiguous cells (one cache line) in registers.
@compute @workgroup_size(WG)
fn scan(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) t: u32) {
    let n = p.grid.x * p.grid.y;
    let base = wid.x * B + t * K;
    var v: array<u32, K>;
    var s = 0u;
    for (var k = 0u; k < K; k++) {
        v[k] = select(0u, cnt[base + k], base + k < n);
        s += v[k];
    }
    sums[t] = s;
    workgroupBarrier();
    scan_sums(t);
    var acc = blocks[wid.x] + sums[t] - s;
    for (var k = 0u; k < K; k++) {
        if base + k < n { cnt[base + k] = acc; }
        acc += v[k];
    }
}

@compute @workgroup_size(WG)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= arrayLength(&boids) { return; }
    let b = boids[id.x];
    sorted[starts[cell_of(b.xy)] + rank[id.x]] = b;
}

fn hash(x: u32) -> u32 { // pcg
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

struct Acc { sep: vec2f, ali: vec2f, coh: vec2f, n: f32 }

// `pos` is pre-shifted by the wrap offset of the visited cells, so no per-neighbor wrap.
fn visit(s: u32, e: u32, pos: vec2f, acc: ptr<function, Acc>) {
    let r2 = p.r * p.r;
    let inv_sr = 1.0 / (p.r * p.sep_r);
    for (var j = s; j < e; j++) {
        let q = nbr[j];
        let d = q.xy - pos;
        let d2 = dot(d, d);
        if d2 >= r2 || d2 < 1e-12 { continue; }
        (*acc).n += 1.0;
        (*acc).ali += q.zw;
        (*acc).coh += d;
        let inv = inverseSqrt(d2);
        (*acc).sep -= d * (inv * max(1.0 - d2 * inv * inv_sr, 0.0));
    }
}

@compute @workgroup_size(WG)
fn sim(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if i >= arrayLength(&nbr) { return; }
    let b = nbr[i];
    var pos = b.xy;
    var vel = b.zw;

    // Each row of cells overlapping the perception circle is clipped to the circle's chord,
    // leaving one contiguous run in `sorted`, plus one more where it wraps in x.
    let g = vec2i(p.grid);
    let r = p.r;
    let inv = 1.0 / p.cell;
    let y0 = i32(floor((pos.y - r) * inv.y));
    let y1 = i32(floor((pos.y + r) * inv.y));
    var acc = Acc();
    for (var ry = y0; ry <= y1; ry++) {
        let band = vec2f(f32(ry), f32(ry + 1)) * p.cell.y;
        let dy = max(0.0, max(band.x - pos.y, pos.y - band.y));
        let hw = sqrt(max(r * r - dy * dy, 0.0));
        let x0 = i32(floor((pos.x - hw) * inv.x));
        let x1 = i32(floor((pos.x + hw) * inv.x));
        let wrap = select(0, 1, ry < 0) - select(0, 1, ry >= g.y);
        let row = u32(ry + wrap * g.y) * p.grid.x;
        let q = pos + vec2f(0.0, f32(wrap) * p.world.y);
        let lo = u32(max(x0, 0));
        let hi = u32(min(x1, g.x - 1));
        visit(starts[row + lo], starts[row + hi + 1u], q, &acc);
        if x0 < 0 {
            visit(starts[row + u32(x0 + g.x)], starts[row + p.grid.x], q + vec2f(p.world.x, 0.0), &acc);
        }
        if x1 >= g.x {
            visit(starts[row], starts[row + u32(x1 - g.x + 1)], q - vec2f(p.world.x, 0.0), &acc);
        }
    }

    var a = p.w_sep * acc.sep;
    if acc.n > 0.0 {
        a += p.w_ali * (acc.ali / acc.n - vel) + p.w_coh * (acc.coh / acc.n);
    }
    if p.max_acc > 0.0 {
        a *= min(1.0, p.max_acc / max(length(a), 1e-6));
    }
    vel += a * p.dt;
    if p.noise > 0.0 {
        // random turn, uniform in +-noise * sqrt(dt); seeded by index and position, which change every frame
        let h = hash(i ^ hash(bitcast<u32>(pos.x) ^ hash(bitcast<u32>(pos.y))));
        let th = p.noise * sqrt(p.dt) * (f32(h) * 4.656613e-10 - 1.0);
        let cs = cos(th);
        let sn = sin(th);
        vel = vec2f(cs * vel.x - sn * vel.y, sn * vel.x + cs * vel.y);
    }
    let sp = length(vel);
    vel *= clamp(sp, p.min_speed, p.max_speed) / max(sp, 1e-6);
    pos += vel * p.dt;
    pos -= p.world * floor(pos / p.world);
    boids[i] = vec4f(pos, vel);
    rank[i] = atomicAdd(&cells[cell_of(pos)], 1u);
}

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
}

const HUE_COS = vec3f(1.0, -0.4818, -0.4818); // cos(2pi * (0, 0.33, 0.67))
const HUE_SIN = vec3f(0.0, 0.8763, -0.8763);  // sin(2pi * (0, 0.33, 0.67))

@vertex
// non-instanced (3 vertices per boid): tiny instances are slow on some GPUs
fn vs(@builtin(vertex_index) v: u32) -> VOut {
    let b = draw[v / 3u];
    let dir = b.zw / max(length(b.zw), 1e-6);
    var shape = array(vec2f(1.0, 0.0), vec2f(-0.6, 0.45), vec2f(-0.6, -0.45));
    let s = shape[v % 3u] * p.r * 0.25;
    let w = b.xy + dir * s.x + vec2f(-dir.y, dir.x) * s.y;
    var out: VOut;
    out.pos = vec4f(w / p.world * 2.0 - 1.0, 0.0, 1.0);
    // hue = heading: cos(angle + phase) expanded so no trig is needed
    out.color = 0.55 + 0.45 * (dir.x * HUE_COS - dir.y * HUE_SIN);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
