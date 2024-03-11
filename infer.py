import cv2
import textwrap
import requests
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.shaders import *
from glfw.GLFW import *
import onnxruntime as ort

vertex_shader_source = textwrap.dedent("""\
    precision highp float;

    uniform vec2 imgSize;
    uniform vec2 minMaxZ;
    
    attribute vec3 vPos;
    attribute vec3 normal;
    
    varying vec3 fPos;
    varying vec3 fNormal;
    varying vec2 texCoords;

    void main() {
        float xDiv = imgSize.x / 2.0;
        float yDiv = imgSize.y / 2.0;

        vec3 pos;
        pos.x = (vPos.x / xDiv) - 1.0;
        pos.y = (-vPos.y / yDiv) + 1.0;
        pos.z = (vPos.z - minMaxZ.x) / (minMaxZ.y - minMaxZ.x + 1.0);

        fPos  = pos;
        texCoords = vec2((pos.x + 1.0) / 2.0, -(pos.y - 1.0) / 2.0);

        vec3 correctedNormal = normalize(normal);
        correctedNormal = vec3(-correctedNormal.x, correctedNormal.y, -correctedNormal.z);
        fNormal = correctedNormal;

        gl_Position = vec4(pos, 1.0);
    }
    """)

fragment_shader_source = textwrap.dedent("""\
    precision highp float;

    uniform vec3 lightColor;                                      
    uniform vec3 lightPos;
    uniform sampler2D texSampler;
    uniform int textureLighting;
    uniform float lightIntensity;

    varying vec3 fPos;
    varying vec3 fNormal;
    varying vec2 texCoords;

    vec3 normal;

    void main() {
        normal = normalize(fNormal);
        vec3 lightDir = fPos - lightPos;
        float distance = length(lightDir);
        distance = distance * distance;
        lightDir = normalize(lightDir);
    
        float lambertian = max(dot(lightDir, normal), 0.0);
        vec3 diffuse = lightColor * lambertian / distance;
        float specularCoeff = 0.0;
        if (lambertian > 0.0) {
            vec3 viewDir = normalize(fPos);
            vec3 halfDir = normalize(lightDir + viewDir);          
            specularCoeff = max(dot(halfDir, normal), 0.0);
        }
        
        vec3 specular = specularCoeff * lightColor / distance;
        vec3 color = diffuse + specular;
        vec4 texColor = texture2D(texSampler, texCoords);
        gl_FragColor = texColor + vec4(lightIntensity * texColor.xyz * color, 0.0);
        
    }
    """)


def CreateMeshFromImage(image_array, step_size):

    height, width = image_array.shape

    s = step_size - step_size // 2
    image_array = cv2.copyMakeBorder(image_array, s, 2 * step_size, s,
                                     2 * step_size, cv2.BORDER_REPLICATE)

    x_step = width // step_size
    y_step = height // step_size

    if width % step_size != 0:
        x_step = x_step + 1

    if height % step_size != 0:
        y_step = y_step + 1

    mesh = []
    mesh_lt = []
    mesh_rt = []
    mesh_lb = []
    mesh_rb = []
    minZ = 255
    maxZ = 0

    for i in range(y_step):
        for j in range(x_step):
            left, top = j * step_size, i * step_size
            right, bottom = (j + 1) * step_size, (i + 1) * step_size

            vertices = [
                (left, top),
                (right, top),
                (left, bottom),
                (right, top),
                (left, bottom),
                (right, bottom),
            ]

            z_values = [image_array[y + s, x + s] for x, y in vertices]
            z_values_lt = [image_array[y, x] for x, y in vertices]
            z_values_rt = [image_array[y, x + 2 * s] for x, y in vertices]
            z_values_lb = [image_array[y + 2 * s, x] for x, y in vertices]
            z_values_rb = [
                image_array[y + 2 * s, x + 2 * s] for x, y in vertices
            ]

            minZ = min(minZ, np.min(z_values))
            maxZ = max(maxZ, np.max(z_values))

            mesh.extend([(x, y, z) for (x, y), z in zip(vertices, z_values)])
            mesh_lt.extend([(x - s, y - s, z)
                            for (x, y), z in zip(vertices, z_values_lt)])
            mesh_rt.extend([(x + s, y - s, z)
                            for (x, y), z in zip(vertices, z_values_rt)])
            mesh_lb.extend([(x - s, y + s, z)
                            for (x, y), z in zip(vertices, z_values_lb)])
            mesh_rb.extend([(x + s, y + s, z)
                            for (x, y), z in zip(vertices, z_values_rb)])

    mesh = np.array(mesh, dtype=np.float32)
    mesh_lt = np.array(mesh_lt, dtype=np.float32)
    mesh_rt = np.array(mesh_rt, dtype=np.float32)
    mesh_lb = np.array(mesh_lb, dtype=np.float32)
    mesh_rb = np.array(mesh_rb, dtype=np.float32)

    v1 = mesh_rt - mesh
    v2 = mesh_lt - mesh
    v3 = mesh_lb - mesh
    v4 = mesh_rb - mesh

    cp1 = np.cross(v1, v2)
    cp2 = np.cross(v2, v3)
    cp3 = np.cross(v3, v4)
    cp4 = np.cross(v4, v1)

    normals = (cp1 + cp2 + cp3 + cp4) / 4.0

    return mesh, normals, minZ, maxZ, width, height


def load_texture(img_data):

    h, w, _ = img_data.shape
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE,
                 img_data)

    return texture


def download_model(url, filename):
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (downloaded_size / total_size) * 100
                        # Print the progress, using '\r' to overwrite the line
                        sys.stdout.write(
                            f"\rDownloading model... {progress:.2f}%")
                        sys.stdout.flush()
            print("\nModel downloaded successfully")
        else:
            raise ("Failed to download file:", response.status_code)


class Model():

    def __init__(self) -> None:

        glfwInit()
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE)
        self.window = glfwCreateWindow(1024, 1024, "Hello World", None, None)
        glfwMakeContextCurrent(self.window)
        program = compileProgram(
            compileShader(vertex_shader_source, GL_VERTEX_SHADER),
            compileShader(fragment_shader_source, GL_FRAGMENT_SHADER))

        self.positionAttr = glGetAttribLocation(program, "vPos")
        self.normalAttr = glGetAttribLocation(program, "normal")

        self.imgSizeUnif = glGetUniformLocation(program, 'imgSize')
        self.minMaxZUnif = glGetUniformLocation(program, 'minMaxZ')
        self.lightPos = glGetUniformLocation(program, 'lightPos')
        self.texSampler = glGetUniformLocation(program, 'texSampler')
        self.lightIntensity = glGetUniformLocation(program, 'lightIntensity')
        self.lightColor = glGetUniformLocation(program, 'lightColor')
        self.lightPos_value = np.array([0.5, 0.5, -1], dtype=np.float32)
        self.lightColor_value = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.lightIntensity_value = 0.5
        self.image_width = 0
        self.image_height = 0

        glUseProgram(program)
        glEnableVertexAttribArray(self.positionAttr)
        glEnableVertexAttribArray(self.normalAttr)
        glfwMakeContextCurrent(None)

        if not os.path.exists('./depth_anything_vitb14.onnx'):

            download_model(
                'https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vitb14.onnx',
                './depth_anything_vitb14.onnx')

        self.session = ort.InferenceSession(
            './depth_anything_vitb14.onnx',
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def showScreen(self):
        glClearColor(1, 0, 0, 1)

        glEnable(GL_DEPTH_TEST)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glVertexAttribPointer(self.positionAttr, 3, GL_FLOAT, GL_FALSE, 0,
                              self.mesh)
        glVertexAttribPointer(self.normalAttr, 3, GL_FLOAT, GL_FALSE, 0,
                              self.normals)

        glUniform2fv(self.imgSizeUnif, 1, self.src_size)
        glUniform2fv(self.minMaxZUnif, 1, self.z_info)
        glUniform3fv(self.lightPos, 1, self.lightPos_value)
        glUniform3fv(self.lightColor, 1, self.lightColor_value)
        glUniform1i(self.texSampler, 0)
        glUniform1f(self.lightIntensity, self.lightIntensity_value)
        glDrawArrays(GL_TRIANGLES, 0, len(self.mesh))

    def render_image(self, x, y, z, r, g, b, power):

        self.lightPos_value = np.array([x, y, z], dtype=np.float32)
        self.lightColor_value = np.array([r, g, b], dtype=np.float32)
        self.lightIntensity_value = power
        w = self.image_width
        h = self.image_height
        glViewport(0, 0, w, h)
        self.showScreen()
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        image = cv2.flip(image, 0)

        return image

    def add_light(self, x, y, z, r, g, b, power):

        glfwMakeContextCurrent(self.window)
        render_array = self.render_image(x, y, z, r, g, b, power)
        glfwMakeContextCurrent(None)
        return render_array

    def preprocess(self, image_data):
        h, w, _ = image_data.shape
        image = cv2.resize(image_data, (518, 518),
                           interpolation=cv2.INTER_CUBIC)
        image = image / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225])
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0), w, h

    def resize_image(self, img):
        height, width = img.shape[:2]

        if height <= 1024 and width <= 1024:
            return img

        ratio = max(height, width) / 1024
        new_height = int(height / ratio)
        new_width = int(width / ratio)
        resized_img = cv2.resize(img, (new_width, new_height))

        return resized_img

    def inference_image(self, image_data, x, y, z, r, g, b, power):

        image_data = self.resize_image(image_data)

        glfwMakeContextCurrent(self.window)
        texture = load_texture(image_data)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        source_image, self.image_width, self.image_height = self.preprocess(
            image_data)
        depth = self.session.run(None, {"image": source_image})[0]
        depth = cv2.resize(depth[0, 0], (self.image_width, self.image_height),
                           interpolation=cv2.INTER_LINEAR)
        depth = (depth.max() - depth) / (depth.max() - depth.min()) * 255.0
        depth_array = np.array(depth).astype(np.uint8)
        self.mesh, self.normals, minZ, maxZ, self.image_width, self.image_height = CreateMeshFromImage(
            depth_array, 3)
        self.z_info = np.array([minZ, maxZ], dtype=np.float32)
        self.src_size = np.array([self.image_width, self.image_height],
                                 dtype=np.float32)

        render_array = self.render_image(x, y, z, r, g, b, power)

        glfwMakeContextCurrent(None)

        return cv2.cvtColor(depth_array, cv2.COLOR_GRAY2BGR), render_array
