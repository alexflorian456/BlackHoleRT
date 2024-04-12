#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <cuda_fp16.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <GLFW/glfw3.h>

#define PI 3.14159265358979323846f
#define GRAVITATIONAL_CONSTANT 6.674e-3f // real gravitational constant is 6.674e-11
#define RAY_TIME_RESOLUTION 1 // time between two ray positions when computing gravitational lensing
#define ESCAPE_SPHERE_ITERATIONS 10
#define ESCAPE_SPHERE_RADIUS 5

//TODO: change dynamically based on distance to closest BH
#define RAY_MAX_ITERATIONS 100 // maximum amount of ray positions computed per pixel

void cudaHandleError(cudaError_t cudaResult){ // TODO?: convert to macro
    if (cudaResult != cudaSuccess) {
        fprintf(stderr, "CUDA handle error: %s: %s\n", cudaGetErrorName(cudaResult), cudaGetErrorString(cudaResult));
        exit(1);
    }
}

__device__ __host__ float
radians_to_degrees(float radians){
    return radians * 180.0f / PI;
}

__device__ __host__ float
degrees_to_radians(float degrees){
    return degrees * PI / 180.0f;
}

__device__ __host__ float
clip(float x, float a, float b){
    return (x < a) ? a : ((x > b) ? b : x);
}

void
initOpenGL(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width, 0, height, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
}

class 
Vector{ // TODO: change to float or half? - maybe template

    public:
        float x;
        float y;
        float z;

        __device__ __host__
        Vector(float x, float y, float z):
            x(x),
            y(y),
            z(z)
        {}

        __device__ __host__ float
        length_2() const{
            return x * x + y * y + z * z;
        }

        __device__ __host__ float
        length() const{
            return std::sqrt(length_2());
        }

        __device__ __host__ Vector
        normalize() const{
            float norm = length();
            if(norm > 0){
                return Vector(x / norm, y / norm, z / norm);
            }
            return Vector(x, y, z);
        }

        // projection of vector b onto vector a
        __device__ __host__ static Vector
        projection(Vector a, Vector b){
            return b * ((a * b) / b.length()); 
        }

        // Rodrigues' rotation formula
        __device__ __host__ Vector
        rotate(float angle, Vector axis) const{ // angle is in radians
            return *this * std::cosf(angle) + (axis ^ *this) * std::sinf(angle) + axis * (axis * *this) * (1 - std::cosf(angle));
        }

        // angle between vectors in degrees - between 0 and 180
        __device__ __host__ float
        angle(Vector b) const{
            float dot_product = *this * b;
            float length_a = this->length();
            float length_b = b.length();

            return radians_to_degrees(std::acos(dot_product / (length_a * length_b)));
        }

        __device__ __host__ float
        azimuth_angle_on_xOz() const{ // TODO: low priority, implement azimuth for any plane
            Vector projection_on_xOz = Vector(x, 0, z);
            float projection_north_angle = projection_on_xOz.angle(Vector::North());
            return projection_on_xOz.z > 0 ?
                   projection_north_angle :
            (360 - projection_north_angle);
        }

        __device__ __host__ Vector
        operator+(const Vector& b) const{
            return Vector(x + b.x, y + b.y, z + b.z);
        }

        __device__ __host__ Vector
        operator-(const Vector& b) const{
            return Vector(x - b.x, y - b.y, z - b.z);
        }

        __device__ __host__ float
        operator*(const Vector& b) const{
            return x * b.x + y * b.y + z * b.z;
        }

        __device__ __host__ Vector
        operator^(const Vector& b) const{
            return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
        }

        __device__ __host__ Vector
        operator*(float k) const{
            return Vector(x * k, y * k, z * k);
        }

        __device__ __host__ Vector
        operator/(float k) const{
            return Vector(x / k, y / k, z / k);
        }

        __device__ __host__ Vector
        operator+=(const Vector& b){
            *this = *this + b;
            return *this;
        }

        __device__ __host__ Vector
        operator-=(const Vector& b){
            *this = *this - b;
            return *this;
        }

        __device__ __host__ static Vector
        Zero(){
            return Vector(0, 0, 0);
        }

        __device__ __host__ static Vector
        Up(){
            return Vector(0, 1, 0);
        }

        __device__ __host__ static Vector
        Down(){
            return Vector(0, -1, 0);
        }

        __device__ __host__ static Vector
        North(){
            return Vector(1, 0, 0);
        }

        __device__ __host__ static Vector
        South(){
            return Vector(-1, 0, 0);
        }

        __device__ __host__ static Vector
        East(){
            return Vector(0, 0, 1);
        }

        __device__ __host__ static Vector
        West(){
            return Vector(0, 0, -1);
        }
};

class
Color{ // TODO: change to float or half? - maybe template
    public:
        unsigned int red;
        unsigned int green;
        unsigned int blue;
        unsigned int alpha;

        __device__
        Color(  unsigned int red, unsigned int green,
                unsigned int blue, unsigned int alpha):
            red(red),
            green(green),
            blue(blue),
            alpha(alpha)
        {}

        __device__ static Color
        Red(){
            return Color(255, 0, 0, 255);
        }

        __device__ static Color
        Green(){
            return Color(0, 255, 0, 255);
        }

        __device__ static Color
        Blue(){
            return Color(0, 0, 255, 255);
        }

        __device__ static Color
        Black(){
            return Color(0, 0, 0, 255);
        }

        __device__ Color
        operator+(const Color& b) const{
            return Color(red + b.red, green + b.green, blue + b.blue, alpha + b.alpha);
        }
};

class
Camera{

    public:
        Vector position;
        Vector direction;
        Vector up;

        float view_plane_distance;
        float view_plane_width;
        float view_plane_height;

        float front_plane_distance;
        float back_plane_distance;

        //TODO: init in constructor
        float elevation_angle; // between 0 and 180; 0 <=> straight up, 180 <=> straight down
        float azimuth_angle; // between 0 and 360

        Camera( Vector position, Vector direction, Vector up,
                float view_plane_distance, float view_plane_width, float view_plane_height,
                float front_plane_distance, float back_plane_distance):

            position(position), direction(direction.normalize()), up(up.normalize()),
            view_plane_distance(view_plane_distance), view_plane_width(view_plane_width), view_plane_height(view_plane_height),
            front_plane_distance(front_plane_distance), back_plane_distance(back_plane_distance)
        {
            elevation_angle = Vector::Up().angle(direction);
            azimuth_angle = direction.azimuth_angle_on_xOz();
        }

        Camera( int output_width, int output_height, int field_of_view,
                Vector position, Vector direction, Vector up,
                float front_plane_distance, float back_plane_distance):
            
            position(position), direction(direction.normalize()), up(up.normalize()),
            front_plane_distance(front_plane_distance), back_plane_distance(back_plane_distance)
        {
            view_plane_distance = 1;
            float theta = 90.f - float(field_of_view) / 2.f;
            view_plane_width = 2.f * view_plane_distance / std::tan(degrees_to_radians(theta));
            float aspect_ratio = (float)output_width / (float)output_height;
            view_plane_height = view_plane_width / aspect_ratio;

            elevation_angle = Vector::Up().angle(direction);
            azimuth_angle = direction.azimuth_angle_on_xOz();
        }

        void update_direction(float value_x, float value_y){
            azimuth_angle += value_x;
            azimuth_angle = (float)((int)azimuth_angle % 360);
            float azimuth_angle_radians = degrees_to_radians(azimuth_angle);

            elevation_angle += value_y;
            elevation_angle = clip(elevation_angle, 0, 180);
            
            direction = Vector::Up().rotate(degrees_to_radians(elevation_angle), Vector::West()).normalize();
            up = Vector::Up().rotate(degrees_to_radians(elevation_angle - 90.f), Vector::West()).normalize();

            direction = direction.rotate(degrees_to_radians(azimuth_angle), Vector::Up()).normalize();
            up = up.rotate(degrees_to_radians(azimuth_angle), Vector::Up()).normalize();
        }
};

class
BlackHole {

    public:
        float mass;
        Vector position;

        BlackHole(float mass, Vector position):
            mass(mass),
            position(position)
        {}
};

__device__ void
set_image_pixel(unsigned char * output_pixels, int i, int j, int output_width, Color color){
    output_pixels[i * output_width * 3 + j * 3 + 0] = color.red;
    output_pixels[i * output_width * 3 + j * 3 + 1] = color.green;
    output_pixels[i * output_width * 3 + j * 3 + 2] = color.blue;
}

__device__ float
image_to_view_plane(int n, int img_size, float view_plane_size){
    return - n * view_plane_size / img_size + view_plane_size / 2;
}

__device__ Color
extract_texture_color(cudaTextureObject_t texture_object, float i, float j, int skybox_width){
    return Color(
        tex2D<float4>(texture_object, j, i).x, // TODO: try with memcpy
        tex2D<float4>(texture_object, j, i).y,
        tex2D<float4>(texture_object, j, i).z,
        0
    );
}

__global__ void // TODO: recycle variables, use scope operators to conserve register space
ray(
Camera camera,
cudaTextureObject_t skybox_texture_object, unsigned char * output_pixels,
int output_width, int output_height,
int skybox_width, int skybox_height,
int grid_index, int block_size, // currently, even if ray is called as a "remainder" kernel with eg. <<<1920, 128>>>,
                                // block_size is still passed as the block size of a "non-remainder" kernel, eg. 896
                                // in order for the image coordinate arithmetic to be correct
BlackHole * black_holes, int num_black_holes,
bool anti_aliasing, bool dynamic_ray_resolution
){
    
    int i = blockIdx.x;
    int j = threadIdx.x + grid_index * block_size;
    
    // initial ray direction
    Vector view_parallel = (camera.up ^ camera.direction).normalize();
    Vector camera_to_view_plane = camera.direction * camera.view_plane_distance;
    float image_to_view_plane_width    = image_to_view_plane(j, output_width , camera.view_plane_width);
    float image_to_view_plane_height   = image_to_view_plane(i, output_height, camera.view_plane_height);
    Vector ray_direction = camera_to_view_plane + view_parallel * image_to_view_plane_width - camera.up * image_to_view_plane_height;

    // gravitational lensing computation
    Vector old_position = camera.position;
    Vector old_velocity = ray_direction.normalize(); // TODO?: try normalize
    Vector new_position = Vector::Zero();
    Vector new_velocity = Vector::Zero();
    constexpr float gravitational_constant = GRAVITATIONAL_CONSTANT;
    constexpr float ray_time_resolution = RAY_TIME_RESOLUTION;
                                                                // to paint a pixel black, the ray has to be stuck
    constexpr int escape_sphere_radius = ESCAPE_SPHERE_RADIUS;  // in a sphere of radius = escape_sphere_radius
    Vector escape_sphere_center = old_position;                 // centered in escape_sphere_center
    int escape_sphere_iterations = ESCAPE_SPHERE_ITERATIONS;    // for escape_sphere_iterations iterations

    float min_angle_to_black_hole = ray_direction.angle(black_holes[0].position - camera.position);
    for(int black_hole_index = 1; black_hole_index<num_black_holes; black_hole_index++){
        float angle_to_black_hole = ray_direction.angle(black_holes[black_hole_index].position - camera.position);
        if(angle_to_black_hole < min_angle_to_black_hole){
            min_angle_to_black_hole = angle_to_black_hole;
        }
    }
    float ray_resolution_multiplier = 1.f / ((min_angle_to_black_hole / 25.f) * (min_angle_to_black_hole / 25.f) + 1.f); // f(x) = 1/((x/25)^2 + 1) bell-like function
    int ray_iterations;

    if(dynamic_ray_resolution){  // rays that are passing closer to black holes will be computed with a higher resolution (steps)
        ray_iterations = (int)(ray_resolution_multiplier * (float)RAY_MAX_ITERATIONS);
    }
    else{
        ray_iterations = RAY_MAX_ITERATIONS;
    }

    for(int iter=0; iter<ray_iterations; iter++){ // TODO: replace with ray_iterations for better performance but decreased quality
        Vector resultant_force = Vector::Zero();
        for(int black_hole_index=0; black_hole_index<num_black_holes; black_hole_index++){ // TO STUDY?: extract to separate kernel
            Vector black_hole_position = black_holes[black_hole_index].position;
            float black_hole_mass = black_holes[black_hole_index].mass;
            float r_squared = (black_hole_position - old_position).length_2();
            Vector r_hat = (black_hole_position - old_position).normalize();
            resultant_force += r_hat * gravitational_constant * black_hole_mass / r_squared;
        }
        new_velocity = (old_velocity + resultant_force * ray_time_resolution).normalize() * old_velocity.length();
        new_position = old_position + new_velocity * ray_time_resolution;

        if((new_position - escape_sphere_center).length() > escape_sphere_radius){
            // the ray has escaped the sphere, reset the iteration counter and set
            // sphere center to new_position
            escape_sphere_center = new_position;
            escape_sphere_iterations = ESCAPE_SPHERE_ITERATIONS;
        }
        else{
            escape_sphere_iterations--;
            if(escape_sphere_iterations < 0){
                set_image_pixel(output_pixels, i, j, output_width, Color::Black());
                return;
            }
        }

        old_velocity = new_velocity;
        old_position = new_position;
    }
    ray_direction = new_velocity;
    
    // skybox rendering
    float elevation_angle = Vector::Up().angle(ray_direction); // 0 degrees <=> straight up, 180 degress <=> straight down
    float azimuth_angle = ray_direction.azimuth_angle_on_xOz();

    // TO STUDY: why did i need printf("") before?

    float skybox_height_coordinate = elevation_angle / 180.f;
    float skybox_width_coordinate  = azimuth_angle   / 360.f;

    Color skybox_color = Color(0, 0, 0, 255);

    float epsilon = ray_resolution_multiplier / 7;
    if(anti_aliasing){
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle - epsilon) / 180.f, (azimuth_angle - epsilon) / 360.f, skybox_width);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle - epsilon) / 180.f, (azimuth_angle -    0   ) / 360.f, skybox_width);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle - epsilon) / 180.f, (azimuth_angle + epsilon) / 360.f, skybox_width);
    
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle - epsilon) / 360.f, skybox_width);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle -    0   ) / 360.f, skybox_width);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle + epsilon) / 360.f, skybox_width);

        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle + epsilon) / 180.f, (azimuth_angle - epsilon) / 360.f, skybox_width);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle + epsilon) / 180.f, (azimuth_angle -    0   ) / 360.f, skybox_width);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle + epsilon) / 180.f, (azimuth_angle + epsilon) / 360.f, skybox_width);

        skybox_color.red /= 9;
        skybox_color.green /= 9;
        skybox_color.blue /= 9;
        skybox_color.alpha = 255;
    }
    else{
        skybox_color = extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle -    0   ) / 360.f, skybox_width);
    }

    // TODO: merge into single function, maybe discard use of Color class
    set_image_pixel(output_pixels, i, j, output_width, skybox_color); // TODO: shared mem
}

int
main(int argc, char * argv[]){
/************************************************************************************************************************
 * args:                                                                                                                *
 * 1 - output width                                                                                                     *
 * 2 - output height                                                                                                    *
 * 3 - field of view (degrees)                                                                                          *
 * 4 - skybox filename, optional, must be an equirectangular image (2:1 aspect ratio), default: starmap_2020_2k_gal.png *
 * 5 - anti-aliasing, optional, default = true                                                                          *
 * 6 - dynamic ray resolution - faster but lower quality, introduces artifcts when close to black hole, default = false *
 ************************************************************************************************************************/
    int output_width    = atoi(argv[1]);
    int output_height   = atoi(argv[2]);
    int field_of_view   = atoi(argv[3]);
    Vector camera_position  = Vector::Zero();
    Vector camera_direction = Vector::South();
    Vector camera_up        = Vector::Up();
    constexpr float front_plane_distance   = 0;
    constexpr float back_plane_distance    = 1000;
    Camera camera(  output_width, output_height, field_of_view,
                    camera_position, camera_direction, camera_up,
                    front_plane_distance, back_plane_distance);
    constexpr float default_movement_speed = 0.2f;
    constexpr float look_speed = 0.2f;    

    unsigned char * d_pixels;
    unsigned char * h_pixels = (unsigned char *)malloc(output_width * output_height * 3 * sizeof(unsigned char));
    std::vector<unsigned char> h_skybox;
    unsigned int skybox_width   = UINT32_MAX;
    unsigned int skybox_height  = UINT32_MAX;
    std::string skybox_filename = ".\\textures\\";
    BlackHole * d_black_holes;
    std::vector<BlackHole> h_black_holes;
    if(argc > 4){
        skybox_filename += argv[4];
    }
    else{
        skybox_filename += "starmap_2020_2k_gal.png";
    }

    bool anti_aliasing;
    if(argc > 5){
        if(strcmp(argv[5], "true") == 0){
            anti_aliasing = true;
        }
        else{
            anti_aliasing = false;
        }
    }
    else{
        anti_aliasing = true;
    }

    bool dynamic_ray_resolution;
    if(argc > 6){
        if(strcmp(argv[6], "true") == 0){
            dynamic_ray_resolution = true;
        }
        else{
            dynamic_ray_resolution = false;
        }
    }
    else{
        dynamic_ray_resolution = false;
    }

    cv::Mat skybox_matrix = cv::imread(skybox_filename);
    if(skybox_matrix.empty()){
        fprintf(stderr, "Error opening skybox file %s\n", skybox_filename.c_str());
        exit(1);
    }
    if(skybox_matrix.channels() == 3){
        cv::cvtColor(skybox_matrix, skybox_matrix, cv::COLOR_BGR2RGBA);
    }
    skybox_matrix.convertTo(skybox_matrix, CV_32FC4);
    printf("Skybox matrix info:\n");
    printf("Datatype code: %d\n", skybox_matrix.type());
    printf("Skybox width: %d\n", skybox_matrix.cols);
    printf("Skybox height: %d\n", skybox_matrix.rows);
    printf("Skybox channels: %d\n", skybox_matrix.channels());

    skybox_width = skybox_matrix.cols;
    skybox_height = skybox_matrix.rows;

    // skybox texture memory
    cudaChannelFormatDesc skybox_channel_desc = cudaCreateChannelDesc<float4>();
    cudaArray_t skybox_array;
    cudaHandleError(cudaMallocArray(&skybox_array, &skybox_channel_desc, skybox_width, skybox_height));

    const size_t skybox_source_pitch = skybox_width * sizeof(float4);
    cudaHandleError(cudaMemcpy2DToArray(skybox_array, 0, 0, skybox_matrix.data, skybox_source_pitch, skybox_width * sizeof(float4), skybox_height, cudaMemcpyHostToDevice));

    cudaResourceDesc skybox_resource_desc;
    std::memset(&skybox_resource_desc, 0, sizeof(cudaResourceDesc));
    skybox_resource_desc.resType = cudaResourceTypeArray;
    skybox_resource_desc.res.array.array = skybox_array;

    cudaTextureDesc skybox_texture_desc;
    std::memset(&skybox_texture_desc, 0, sizeof(cudaTextureDesc));
    skybox_texture_desc.addressMode[0] = cudaAddressModeClamp;
    skybox_texture_desc.addressMode[1] = cudaAddressModeClamp;
    skybox_texture_desc.filterMode = cudaFilterModeLinear;
    skybox_texture_desc.readMode = cudaReadModeElementType;
    skybox_texture_desc.normalizedCoords = true;

    cudaTextureObject_t skybox_texture_object = 0;
    cudaHandleError(cudaCreateTextureObject(&skybox_texture_object, &skybox_resource_desc, &skybox_texture_desc, nullptr));

    // declaring black holes
    h_black_holes.push_back(BlackHole(500, Vector(0, 0, 0)));
    h_black_holes.push_back(BlackHole(500, Vector(0, 0, 40)));
    int num_black_holes = (int)h_black_holes.size();

    // allocating device memory
    cudaHandleError(cudaMalloc((void **)&d_pixels, output_width * output_height * 3 * sizeof(unsigned char))); // TODO: mallocManaged?
    cudaHandleError(cudaMalloc((void **)&d_black_holes, num_black_holes * sizeof(BlackHole))); // TODO: mallocManaged?
    cudaHandleError(cudaMemcpy(d_black_holes, h_black_holes.data(), num_black_holes * sizeof(BlackHole), cudaMemcpyHostToDevice));

    // initialize GLFW
    GLFWwindow* main_window;

    if(!glfwInit()){
        return -1;
    }

    main_window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if(!main_window){
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(main_window);
    glfwSetInputMode(main_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    initOpenGL(output_width, output_height);

    GLuint screen_texture_id;
    glGenTextures(1, &screen_texture_id);
    glBindTexture(GL_TEXTURE_2D, screen_texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, output_width, output_height, 0, GL_RGB, GL_UNSIGNED_BYTE, h_pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    float angle = 0;
    float progress_percent = -1;
    auto t_start = std::chrono::high_resolution_clock::now();
    camera.position = Vector(60, 0, 0);

    // float black_hole_distance = 40;

    // Get the primary monitor
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);
    glfwSetWindowMonitor(main_window, primaryMonitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
    glfwSetCursorPos(main_window, mode->width/2, mode->height/2);

    printf("Screen width: %d\n", mode->width);
    printf("Screen height: %d\n", mode->height);
    printf("Screen refresh rate: %d\n", mode->refreshRate);

    int frames_processed = 0;
    double mouse_x;
    double mouse_y;
    float delta_mouse_x;
    float delta_mouse_y;
    while(!glfwWindowShouldClose(main_window)){ // TODO?: call multiple kernels from multiple threads
        // SPIN IN THE CENTER:
        // camera.direction = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle)));
        // ORBIT AROUND CENTER:
        // camera.position = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle))) * camera_distance_from_center;
        // camera.direction = (Vector::Zero() - camera.position).normalize();
        // MOVE INTO CENTER:
        // TODO: issue noticed when camera is at "perfect" integer coordinates, got CUDA error: an illegal memory access was encountered - NaN values?
        // camera.position = camera.position + camera.direction / 2;

        int remaining_width = output_width;
        int grid_index = 0;
        constexpr int threads_per_block = 1024; // TO STUDY: declared new variables in kernel, tried rendering 1080p
                                     // with 1024 threads per block (what was by defualt) and got:
                                     // "CUDA error: too many resources requested for launch"
                                     // (works with 896 on my GPU, but might differ on others)
                                     // SOLUTION: switching from double precision to single precision

        auto frame_start = std::chrono::high_resolution_clock::now();
        while(remaining_width > 0){
            if(remaining_width >= threads_per_block){
                ray<<<output_height, threads_per_block>>>(camera, skybox_texture_object, d_pixels, output_width, output_height, skybox_width, skybox_height, grid_index, threads_per_block, d_black_holes, num_black_holes, anti_aliasing, dynamic_ray_resolution);
            }
            else{ // remainder kernel call
                ray<<<output_height, remaining_width>>>(camera, skybox_texture_object, d_pixels, output_width, output_height, skybox_width, skybox_height, grid_index, threads_per_block, d_black_holes, num_black_holes, anti_aliasing, dynamic_ray_resolution);                
            }
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error line 468: %s: %s\n", cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
                exit(1);
            }
            remaining_width -= threads_per_block;
            grid_index++;
        }
        cudaDeviceSynchronize();
        auto frame_end = std::chrono::high_resolution_clock::now();
        float frame_duration = std::chrono::duration<float, std::milli>(frame_end - frame_start).count();
        printf("frame%03d time: %f\n", (int)(angle*2.), frame_duration);

        auto other_start = std::chrono::high_resolution_clock::now();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            fprintf(stderr, "CUDA error line 477: %s: %s\n", cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
            exit(1);
        }

        cudaHandleError(cudaMemcpy(h_pixels, d_pixels, output_width * output_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        // TODO: refactor to controls method
        float movement_speed = default_movement_speed;
        if (glfwGetKey(main_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(main_window, GLFW_TRUE);
        }
        if (glfwGetKey(main_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            movement_speed *= 2;
        }
        if(glfwGetKey(main_window, GLFW_KEY_W) == GLFW_PRESS){
            camera.position += camera.direction * movement_speed;
        }
        if(glfwGetKey(main_window, GLFW_KEY_S) == GLFW_PRESS){
            camera.position -= camera.direction * movement_speed;
        }
        if(glfwGetKey(main_window, GLFW_KEY_A) == GLFW_PRESS){
            camera.position += (camera.up ^ camera.direction) * movement_speed;
        }
        if(glfwGetKey(main_window, GLFW_KEY_D) == GLFW_PRESS){
            camera.position -= (camera.up ^ camera.direction) * movement_speed;
        }
        if(glfwGetKey(main_window, GLFW_KEY_SPACE) == GLFW_PRESS){
            camera.position += Vector::Up() * movement_speed;
        }
        if(glfwGetKey(main_window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){
            camera.position += Vector::Down() * movement_speed;
        }
        glfwGetCursorPos(main_window, &mouse_x, &mouse_y);
        glfwSetCursorPos(main_window, mode->width/2, mode->height/2);
        delta_mouse_x = (float)mouse_x - mode->width/2;
        delta_mouse_y = (float)mouse_y - mode->height/2;
        if(delta_mouse_x != 0 || delta_mouse_y != 0){
            camera.update_direction(-look_speed * delta_mouse_x, look_speed * delta_mouse_y);
        }
        // if(delta_mouse_x != 0){
        //     camera.increment_azimuth(-look_speed * delta_mouse_x);
        // }

        glBindTexture(GL_TEXTURE_2D, screen_texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, output_width, output_height, GL_RGB, GL_UNSIGNED_BYTE, h_pixels);

        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_QUADS);
        glTexCoord2f(0, 0); glVertex2f(0, 0);
        glTexCoord2f(1, 0); glVertex2f((GLfloat)output_width, 0);
        glTexCoord2f(1, 1); glVertex2f((GLfloat)output_width, (GLfloat)output_height);
        glTexCoord2f(0, 1); glVertex2f(0, (GLfloat)output_height);
        glEnd();

        glfwSwapBuffers(main_window);

        glfwPollEvents();

        // // save to output
        // char * output_path = (char*)malloc(25 * sizeof(char)); //TODO: free
        // sprintf(output_path, "output\\frame%03d.png", (int)(angle*2.));
        // bool imwrite_success = cv::imwrite(output_path, cv::Mat(cv::Size(output_width, output_height), CV_8UC3, h_pixels));
        // if(!imwrite_success){
        //     fprintf(stderr, "Failed to imwrite output frame\n");
        //     exit(1);
        // }

        float current_progress_percent = angle / 360.0f * 100;

        if (current_progress_percent - progress_percent > 1) {
            progress_percent = current_progress_percent;
            // printf("%d%% Done\n", (int)progress_percent);
        }
        
        angle += 0.5;

        // // collision
        // black_hole_distance-=0.25;
        // if(black_hole_distance < 0){
        //     break;
        // }
        h_black_holes.clear();
        h_black_holes.push_back(BlackHole(500, Vector(0, 0, 0)));
        h_black_holes.push_back(BlackHole(500, Vector(0, 0, clip(40.f - angle/64.f, 0, 40))));
        
        cudaHandleError(cudaMemcpy(d_black_holes, h_black_holes.data(), h_black_holes.size() * sizeof(BlackHole), cudaMemcpyHostToDevice));
        auto other_end = std::chrono::high_resolution_clock::now();
        float other_duration = std::chrono::duration<float, std::milli>(other_end - other_start).count();

        frames_processed++;
        // printf("frame%03d other time: %f\n", (int)(angle*2.), other_duration);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    printf("Time: %f ms\n", duration);
    printf("Estimated fps: %f\n", frames_processed / (duration / 1000));

    printf("Done\n");

    glfwTerminate();

    //TODO: free memory
    
    return 0;
}
