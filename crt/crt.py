import moderngl
import numpy as np

VERTEX_SHADER = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex;
uniform vec4 SourceSize;
uniform vec4 OutputSize;
in vec2 v_uv;
out vec4 f_color;

#define SCANLINE_WEIGHT 1.2
#define SCANLINE_GAP_BRIGHTNESS 0.65
#define BLOOM_FACTOR 1.25
#define INPUT_GAMMA 2.2
#define OUTPUT_GAMMA 2.2
#define BLUR_OFFSET 0.75 

void main() {
    vec2 uv = v_uv;

    // 1. CRT Curvature Distortion
    vec2 cc = uv - 0.5;
    float dist = dot(cc, cc) * 0.04; // Slightly intensified to emphasize retro curve
    uv = (cc + cc * dist) + 0.5;

    // Hard clip tube boundary edges
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) {
        f_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float texel_x = 1.0 / SourceSize.x; 
    float texel_y = 1.0 / SourceSize.y; 

    // 2. Fetch native RGB vectors (Center + Cross Pattern Neighbors)
    vec3 col       = texture(tex, uv).rgb;
    vec3 left_col  = texture(tex, uv - vec2(texel_x * BLUR_OFFSET, 0.0)).rgb;
    vec3 right_col = texture(tex, uv + vec2(texel_x * BLUR_OFFSET, 0.0)).rgb;
    vec3 up_col    = texture(tex, uv - vec2(0.0, texel_y * BLUR_OFFSET)).rgb;
    vec3 down_col  = texture(tex, uv + vec2(0.0, texel_y * BLUR_OFFSET)).rgb;

    // Symmetric 5-tap cross blur filter
    // Center gets ~33.4% weight, surrounding directions split the remaining ~66.6%
    col = (col * 0.334) + 
          (left_col * 0.1665) + (right_col * 0.1665) + 
          (up_col * 0.1665)   + (down_col * 0.1665);
    
    // Linearize colorspace
    col = pow(col, vec3(INPUT_GAMMA));

    // 3. Render Scanlines
    float pos_y = uv.y * SourceSize.y;
    float delta_y = abs(fract(pos_y) - 0.5);
    float scanline = mix(1.0, SCANLINE_GAP_BRIGHTNESS, smoothstep(0.0, 0.5, delta_y * SCANLINE_WEIGHT));

    col *= scanline;
    col *= BLOOM_FACTOR;

    // 4. Output with proper display gamma correction
    f_color = vec4(pow(col, vec3(1.0 / OUTPUT_GAMMA)), 1.0);
}
"""


class CRTProcessor:
    def __init__(self, internal_res, output_res):
        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )

        # Safely assign system uniforms
        if "SourceSize" in self.prog:
            self.prog["SourceSize"].value = (
                internal_res[0],
                internal_res[1],
                1.0 / internal_res[0],
                1.0 / internal_res[1],
            )
        if "OutputSize" in self.prog:
            self.prog["OutputSize"].value = (
                output_res[0],
                output_res[1],
                1.0 / output_res[0],
                1.0 / output_res[1],
            )
        if "tex" in self.prog:
            self.prog["tex"].value = 0

        # Geometry Vertex Arrays Setup
        vertices = np.array(
            [
                -1.0,
                1.0,
                0.0,
                0.0,  # Top-Left
                -1.0,
                -1.0,
                0.0,
                1.0,  # Bottom-Left
                1.0,
                1.0,
                1.0,
                0.0,  # Top-Right
                1.0,
                -1.0,
                1.0,
                1.0,  # Bottom-Right
            ],
            dtype="f4",
        )

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_vert", "in_uv")]
        )

        # Build 4-channel texture target to match surface bit depth
        self.texture = self.ctx.texture(internal_res, 4)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # FIX: Align texture mapping directly with Pygame's internal BGRA buffer stream
        self.texture.swizzle = "BGRA"

    def render(self, surface):
        # Write clean byte stream directly to GPU VRAM
        self.texture.write(surface.get_view("2"))

        self.texture.use(location=0)
        self.vao.render(moderngl.TRIANGLE_STRIP)
