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
// #include <cuda_gl_interop.h>

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

#define RAY_MAX_ITERATIONS 50 // maximum amount of ray positions computed per pixel
#define MAX_VECTOR_KERNEL_SIZE 127

#define MIN_ACCRETION_DISK_DISTANCE 30
#define MAX_ACCRETION_DISK_DISTANCE 70

__constant__ float vector_conv_kernel[MAX_VECTOR_KERNEL_SIZE * MAX_VECTOR_KERNEL_SIZE];

void cudaHandleError(cudaError_t cudaResult){ // TODO: (low priority) convert to macro
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
Vector{

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
            return std::sqrtf(length_2());
        }

        __device__ __host__ Vector
        normalize() const{
            float norm = length();
            if(norm > 0){
                return Vector(x / norm, y / norm, z / norm);
            }
            return Vector(x, y, z);
        }

        // projection of vector a onto vector b
        __device__ __host__ static Vector
        projection(Vector a, Vector b){
            return b * ((a * b) / b.length_2()); 
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
        azimuth_angle_on_xOz() const{ // TODO: (low priority) implement azimuth for any plane
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
Color{
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

        __device__ Color
        operator*(const float& c) const{
            return Color(red * c, green * c, blue * c, alpha * c);
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

__device__ float
sinf_kernel(float x){
    return std::sinf(x);
}

__device__ Color
extract_texture_color(cudaTextureObject_t texture_object, float i, float j){
    return Color(
        tex2D<float4>(texture_object, j, i).x,
        tex2D<float4>(texture_object, j, i).y,
        tex2D<float4>(texture_object, j, i).z,
        0
    );
}

__device__ float
cauchy_bell(float x, float c){
    return 1.f / ((x * c) * (x * c) + 1.f); 
}

__global__ void // TODO: recycle variables, use scope operators to conserve register space
ray(
Camera camera,
cudaTextureObject_t accretion_disk_texture_object,
int output_width, int output_height,
int grid_index, int block_size, // currently, even if ray is called as a "remainder" kernel with eg. <<<1920, 128>>>,
                                // block_size is still passed as the block size of a "non-remainder" kernel, eg. 896
                                // in order for the image coordinate arithmetic to be correct
BlackHole * black_holes, int num_black_holes,
bool anti_aliasing, bool dynamic_ray_resolution,
int frames_processed, // for experimental effects based on frame number
float * d_ray_direction_vector_map,
float * d_ray_resolution_multiplier_map,
unsigned char * d_accretion_disk_intersection
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
    Vector old_velocity = ray_direction.normalize();
    Vector new_position = Vector::Zero();
    Vector new_velocity = Vector::Zero();
    constexpr float gravitational_constant = GRAVITATIONAL_CONSTANT;
                                                                // to paint a pixel black, the ray has to be stuck
    int escape_sphere_radius = ESCAPE_SPHERE_RADIUS;            // in a sphere of radius = escape_sphere_radius
    Vector escape_sphere_center = old_position;                 // centered in escape_sphere_center
    int escape_sphere_iterations = ESCAPE_SPHERE_ITERATIONS;    // for escape_sphere_iterations iterations

    bool ray_going_opposite = true;
    float min_impact_parameter = (black_holes[0].position - camera.position).length();
    float ray_travel_distance;
    float closest_black_hole_distance = (black_holes[0].position - camera.position).length();
    float closest_black_hole_mass = black_holes[0].mass;
    for(int black_hole_index = 0; black_hole_index<num_black_holes; black_hole_index++){
        Vector camera_to_black_hole = (black_holes[black_hole_index].position - camera.position);
        Vector ray_to_black_hole_projection = Vector::projection(ray_direction, camera_to_black_hole);
        float impact_parameter;
        if(!((camera_to_black_hole + ray_to_black_hole_projection).length() < camera_to_black_hole.length())){
            ray_going_opposite = false;
        }
        impact_parameter = (ray_direction - ray_to_black_hole_projection).length();
        if(impact_parameter < min_impact_parameter){
            min_impact_parameter = impact_parameter;
        }
        if(camera_to_black_hole.length() < closest_black_hole_distance){
            closest_black_hole_distance = camera_to_black_hole.length();
            closest_black_hole_mass = black_holes[black_hole_index].mass;
        }
    }
    
    float distance_mass_cauchy_dropoff_rate = closest_black_hole_distance / (closest_black_hole_mass / 5);
    float ray_resolution_multiplier = cauchy_bell(min_impact_parameter, (distance_mass_cauchy_dropoff_rate > 1.f) ? 1.f : distance_mass_cauchy_dropoff_rate);    
    if(ray_going_opposite && distance_mass_cauchy_dropoff_rate > 1.f){ // TODO: find better solution
        d_ray_resolution_multiplier_map[i * output_width + j] = 1.f / distance_mass_cauchy_dropoff_rate; // so that anti-aliasing doesn't blur the skybox opposite to black holes 
    }
    else{
        d_ray_resolution_multiplier_map[i * output_width + j] = ray_resolution_multiplier;
    }
    float ray_time_resolution;
    int ray_iterations;
    
    if(dynamic_ray_resolution){  // rays that are passing closer to black holes will be computed with a higher resolution (steps)
        ray_time_resolution = 1.f / ray_resolution_multiplier;
        ray_iterations = ((int)(ray_resolution_multiplier * (float)RAY_MAX_ITERATIONS) > 1) ?
                         ((int)(ray_resolution_multiplier * (float)RAY_MAX_ITERATIONS)) :
                         (1);
    }
    else{
        ray_time_resolution = RAY_TIME_RESOLUTION;
        ray_iterations = RAY_MAX_ITERATIONS;
    }

    // escape_sphere_radius *= ray_time_resolution;
    // ray_travel_distance *= ray_time_resolution;
    bool intersected_accretion_disk = false;
    if(!intersected_accretion_disk){
        set_image_pixel(d_accretion_disk_intersection, i, j, output_width, Color::Black());
    }
    

    for(int iter=0; iter<ray_iterations; iter++){
        Vector resultant_force = Vector::Zero();
        float closest_black_hole_to_ray_distance = (old_position - black_holes[0].position).length();
        float closest_black_hole_to_ray_mass = black_holes[0].mass;
        for(int black_hole_index=0; black_hole_index<num_black_holes; black_hole_index++){ // TO STUDY?: extract to separate kernel
            Vector black_hole_position = black_holes[black_hole_index].position;
            float black_hole_mass = black_holes[black_hole_index].mass;
            float r_squared = (black_hole_position - old_position).length_2();
            Vector r_hat = (black_hole_position - old_position).normalize();
            resultant_force += r_hat * gravitational_constant * black_hole_mass * 4 / r_squared;

            float r = std::sqrtf(r_squared);
            if(r < closest_black_hole_to_ray_distance){
                closest_black_hole_to_ray_distance = r;
                closest_black_hole_to_ray_mass = black_hole_mass;
            }
            // spin experiment
            // resultant_force += ((old_position - black_holes[black_hole_index].position).normalize() ^ Vector::North()) / (old_position - black_holes[black_hole_index].position).length_2();
        }
        // TODO: try making the ray_travel_distance in terms of the magnitude of resultant_force, clipping it to the distance to the closest black hole
        // ray_travel_distance /= 2; // binary search-like ray
        ray_travel_distance = (1.f - cauchy_bell(closest_black_hole_to_ray_distance, 1.f / std::powf(closest_black_hole_to_ray_mass * GRAVITATIONAL_CONSTANT, 3))) * closest_black_hole_to_ray_distance;
        // ray_travel_distance *= ray_time_resolution;
        ray_travel_distance = ray_travel_distance > 1.f ? ray_travel_distance : 1.f;
        new_velocity = (old_velocity + resultant_force * ray_travel_distance).normalize() * old_velocity.length();
        new_position = old_position + new_velocity * ray_travel_distance;

        if((new_position - escape_sphere_center).length() > escape_sphere_radius){
            // the ray has escaped the sphere, reset the iteration counter and set
            // sphere center to new_position
            escape_sphere_center = new_position;
            escape_sphere_iterations = ESCAPE_SPHERE_ITERATIONS;
        }
        else{
            escape_sphere_iterations--;
            if(escape_sphere_iterations < 0){
                d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 0] = 0.f;
                d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 1] = 0.f;
                d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 2] = 0.f;            
                return;
            }
        }

        if(old_position.y * new_position.y < 0 && !intersected_accretion_disk){
            float accretion_disk_intersection_t = (-old_position.y)/(new_position.y - old_position.y);
            Vector accretion_disk_intersection_position = Vector(
                old_position.x + accretion_disk_intersection_t * (new_position.x - old_position.x),
                old_position.y + accretion_disk_intersection_t * (new_position.y - old_position.y),
                old_position.z + accretion_disk_intersection_t * (new_position.z - old_position.z)
            );
            if(accretion_disk_intersection_position.length() > MIN_ACCRETION_DISK_DISTANCE && accretion_disk_intersection_position.length() < MAX_ACCRETION_DISK_DISTANCE){
                float accretion_disk_texture_position_y = (accretion_disk_intersection_position.length() - MIN_ACCRETION_DISK_DISTANCE)/(MAX_ACCRETION_DISK_DISTANCE - MIN_ACCRETION_DISK_DISTANCE);
                float accretion_disk_texture_position_x = (int)(accretion_disk_intersection_position.azimuth_angle_on_xOz() + frames_processed) % 360 / 360.f;

                Color accretion_disk_color = extract_texture_color(accretion_disk_texture_object, accretion_disk_texture_position_x, accretion_disk_texture_position_y);
                set_image_pixel(d_accretion_disk_intersection, i, j, output_width, accretion_disk_color);
                intersected_accretion_disk = true;
            }
        }

        old_velocity = new_velocity;
        old_position = new_position;
    }
    ray_direction = new_velocity;

    d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 0] = ray_direction.x;
    d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 1] = ray_direction.y;
    d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 2] = ray_direction.z;
}

__global__ void
ray_post_processing(
float * d_ray_direction_vector_map, float * d_output_ray_direction_vector_map,
int grid_index,
int gaussian_kernel_size,
int block_size, int output_width, int output_height
){
    int i = blockIdx.x;
    int j = threadIdx.x + grid_index * block_size;

    if(d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 0] == 0.f &&
       d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 1] == 0.f &&
       d_ray_direction_vector_map[i * output_width * 3 + j * 3 + 2] == 0.f){
        d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 0] = 0.f;
        d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 1] = 0.f;
        d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 2] = 0.f;
        return;    
    }

    Vector sum = Vector::Zero();
    for(int n = 0; n < gaussian_kernel_size; n++){
        for(int m = 0; m < gaussian_kernel_size; m++){
            sum += Vector(
                d_ray_direction_vector_map[(int)clip(i - gaussian_kernel_size/2 + n, 0, output_height-1) * output_width * 3 + (int)clip(j - gaussian_kernel_size/2 + m, 0, output_width-1) * 3 + 0],
                d_ray_direction_vector_map[(int)clip(i - gaussian_kernel_size/2 + n, 0, output_height-1) * output_width * 3 + (int)clip(j - gaussian_kernel_size/2 + m, 0, output_width-1) * 3 + 1],
                d_ray_direction_vector_map[(int)clip(i - gaussian_kernel_size/2 + n, 0, output_height-1) * output_width * 3 + (int)clip(j - gaussian_kernel_size/2 + m, 0, output_width-1) * 3 + 2]
            ) * vector_conv_kernel[n * gaussian_kernel_size + m];
        }
    }

    d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 0] = sum.x;
    d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 1] = sum.y;
    d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 2] = sum.z;
}

__global__ void
skybox_texture_extraction(
cudaTextureObject_t skybox_texture_object,
float * d_output_ray_direction_vector_map,
float * d_ray_resolution_multiplier_map,
unsigned char * output_pixels,
int grid_index, int block_size,
bool anti_aliasing,
int output_width,
unsigned char * d_accretion_disk_intersection
){
    int i = blockIdx.x;
    int j = threadIdx.x + grid_index * block_size;

    Color accretion_disk_intersection_color = Color(
        d_accretion_disk_intersection[i * output_width * 3 + j * 3 + 0],
        d_accretion_disk_intersection[i * output_width * 3 + j * 3 + 1],
        d_accretion_disk_intersection[i * output_width * 3 + j * 3 + 2],
        0
    );
    bool return_if_ray_did_not_escape = true;

    if( accretion_disk_intersection_color.red != 0 ||
        accretion_disk_intersection_color.green != 0 ||
        accretion_disk_intersection_color.blue != 0){ // ray intersected accretion disk
        
        // set_image_pixel(output_pixels, i, j, output_width, accretion_disk_intersection_color);
        return_if_ray_did_not_escape = false;
    }

    Vector ray_direction = Vector(
        d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 0],
        d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 1],
        d_output_ray_direction_vector_map[i * output_width * 3 + j * 3 + 2]
    );
    if(ray_direction.x == 0.f && ray_direction.y == 0.f && ray_direction.z == 0.f){ // ray did not escape black hole
        set_image_pixel(output_pixels, i, j, output_width, Color::Black());
        if(return_if_ray_did_not_escape) return;    
    }

    float ray_resolution_multiplier = d_ray_resolution_multiplier_map[i * output_width + j];

    // skybox rendering
    float elevation_angle = Vector::Up().angle(ray_direction); // 0 degrees <=> straight up, 180 degress <=> straight down
    float azimuth_angle = ray_direction.azimuth_angle_on_xOz();

    // float skybox_height_coordinate = elevation_angle / 180.f;
    // float skybox_width_coordinate  = azimuth_angle   / 360.f;

    Color skybox_color = Color::Black();

    // TODO: weigh the perturbed rays based on a 3x3 gaussian kernel
    float epsilon = (1 - cauchy_bell(ray_resolution_multiplier, 1.f/2.f)) * ray_resolution_multiplier;
    if(anti_aliasing){ // TODO: (low priority) try perturbing the ray only on the plane with the black hole with the minimum impact parameter
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle - epsilon) / 180.f, (azimuth_angle - epsilon) / 360.f);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle - epsilon) / 180.f, (azimuth_angle -    0   ) / 360.f);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle - epsilon) / 180.f, (azimuth_angle + epsilon) / 360.f);
    
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle - epsilon) / 360.f);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle -    0   ) / 360.f);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle + epsilon) / 360.f);

        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle + epsilon) / 180.f, (azimuth_angle - epsilon) / 360.f);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle + epsilon) / 180.f, (azimuth_angle -    0   ) / 360.f);
        skybox_color = skybox_color + extract_texture_color(skybox_texture_object, (elevation_angle + epsilon) / 180.f, (azimuth_angle + epsilon) / 360.f);

        skybox_color.red /= 9;
        skybox_color.green /= 9;
        skybox_color.blue /= 9;
        skybox_color.alpha = 255;
    }
    else{
        skybox_color = extract_texture_color(skybox_texture_object, (elevation_angle -    0   ) / 180.f, (azimuth_angle -    0   ) / 360.f);
    }

    float accretion_disk_intersection_color_alpha = 
        (accretion_disk_intersection_color.red + accretion_disk_intersection_color.green + accretion_disk_intersection_color.blue) / 3.f / 255;
    
    skybox_color = accretion_disk_intersection_color * accretion_disk_intersection_color_alpha + skybox_color * (1 - accretion_disk_intersection_color_alpha);
    set_image_pixel(output_pixels, i, j, output_width, skybox_color); // TODO: shared mem?
}

std::vector<std::vector<float>> generate_gaussian_kernel(int size, float sigma) {
    std::vector<std::vector<float>> kernel(size, std::vector<float>(size));
    float sum = 0.0;

    // Calculate normalization factor
    float normFactor = 1.0f / (2 * PI * sigma * sigma);

    // Generate kernel values
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int x = i - size / 2;
            int y = j - size / 2;
            kernel[i][j] = normFactor * exp(-(x * x + y * y) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }

    // Normalize kernel
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

int
main(int argc, char * argv[]){
/**************************************************************************************************************************************
 * args:                                                                                                                              *
 * 1 - output width                                                                                                                   *
 * 2 - output height                                                                                                                  *
 * 3 - field of view (degrees)                                                                                                        *
 * 4 - skybox filename, optional, must be an equirectangular image (2:1 aspect ratio), default: starmap_2020_2k_gal.png               *
 * 5 - anti-aliasing, optional, default = true                                                                                        *
 * 6 - dynamic ray resolution - wether rays will have a variable number of iterations, based on the impact parameter, default = false *
 **************************************************************************************************************************************/
    int output_width    = atoi(argv[1]); // TODO: check argc
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
    unsigned char * d_accretion_disk_intersection;
    float * d_ray_direction_vector_map;
    float * d_output_ray_direction_vector_map;
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

    int gaussian_kernel_size;
    if(argc > 7){
        if(atoi(argv[7]) != 0){
            gaussian_kernel_size = atoi(argv[7]);
        }
        else{
            gaussian_kernel_size = 5;
            printf("Using default gaussian kernel size 5\n");
        }

        if(gaussian_kernel_size % 2 == 0){
            gaussian_kernel_size++;
        }
    }
    else{
        gaussian_kernel_size = 5;
        printf("Using default gaussian kernel size 5\n");
    }

    float gaussian_kernel_sigma;
    if(argc > 8){
        if(atof(argv[8]) != 0){
            gaussian_kernel_sigma = (float)atof(argv[8]);
        }
        else{
            gaussian_kernel_sigma = 1.f;
            printf("Using default gaussian kernel sigma 1.0\n");
        }
    }
    else{
        gaussian_kernel_sigma = 1.f;
        printf("Using default gaussian kernel sigma 1.0\n");
    }

    float * h_gaussian_kernel = (float *)malloc(gaussian_kernel_size * gaussian_kernel_size * sizeof(float));
    std::vector<std::vector<float>> gaussian_kernel_matrix = generate_gaussian_kernel(gaussian_kernel_size, gaussian_kernel_sigma);
    for(int i = 0; i < gaussian_kernel_size; i++){
        memcpy((h_gaussian_kernel + i * gaussian_kernel_size), gaussian_kernel_matrix[i].data(), gaussian_kernel_size * sizeof(float));
    }
    // float * d_gaussian_kernel; // TODO compare performance with constant memory
    // cudaHandleError(cudaMalloc((void **)&d_gaussian_kernel, gaussian_kernel_size * gaussian_kernel_size * sizeof(float)));
    cudaHandleError(cudaMemcpyToSymbol(vector_conv_kernel, h_gaussian_kernel, gaussian_kernel_size * gaussian_kernel_size * sizeof(float)));

    float * d_ray_resolution_multiplier_map;
    cudaHandleError(cudaMalloc((void **)&d_ray_resolution_multiplier_map, output_width * output_height * sizeof(float)));

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

    // accretion disk texture memory
    cv::Mat accretion_disk_matrix = cv::imread(".\\textures\\accretion_disk.png");
    if(skybox_matrix.empty()){
        fprintf(stderr, "Error opening accretion disk file %s\n", ".\\textures\\accretion_disk.png");
        exit(1);
    }
    if(accretion_disk_matrix.channels() == 3){
        cv::cvtColor(accretion_disk_matrix, accretion_disk_matrix, cv::COLOR_BGR2RGBA);
    }
    accretion_disk_matrix.convertTo(accretion_disk_matrix, CV_32FC4);
    printf("Accretion disk matrix info:\n");
    printf("Datatype code: %d\n", accretion_disk_matrix.type());
    printf("Accretion disk width: %d\n", accretion_disk_matrix.cols);
    printf("Accretion disk height: %d\n", accretion_disk_matrix.rows);
    printf("Accretion disk channels: %d\n", accretion_disk_matrix.channels());

    unsigned int accretion_disk_width = accretion_disk_matrix.cols;
    unsigned int accretion_disk_height = accretion_disk_matrix.rows;

    cudaChannelFormatDesc accretion_disk_channel_desc = cudaCreateChannelDesc<float4>();
    cudaArray_t accretion_disk_array;
    cudaHandleError(cudaMallocArray(&accretion_disk_array, &accretion_disk_channel_desc, accretion_disk_width, accretion_disk_height));

    const size_t accretion_disk_source_pitch = accretion_disk_width * sizeof(float4);
    cudaHandleError(cudaMemcpy2DToArray(accretion_disk_array, 0, 0, accretion_disk_matrix.data, accretion_disk_source_pitch, accretion_disk_width * sizeof(float4), accretion_disk_height, cudaMemcpyHostToDevice));

    cudaResourceDesc accretion_disk_resource_desc;
    std::memset(&accretion_disk_resource_desc, 0, sizeof(cudaResourceDesc));
    accretion_disk_resource_desc.resType = cudaResourceTypeArray;
    accretion_disk_resource_desc.res.array.array = accretion_disk_array;

    cudaTextureDesc accretion_disk_texture_desc;
    std::memset(&accretion_disk_texture_desc, 0, sizeof(cudaTextureDesc));
    accretion_disk_texture_desc.addressMode[0] = cudaAddressModeClamp;
    accretion_disk_texture_desc.addressMode[1] = cudaAddressModeClamp;
    accretion_disk_texture_desc.filterMode = cudaFilterModeLinear;
    accretion_disk_texture_desc.readMode = cudaReadModeElementType;
    accretion_disk_texture_desc.normalizedCoords = true;

    cudaTextureObject_t accretion_disk_texture_object = 0;
    cudaHandleError(cudaCreateTextureObject(&accretion_disk_texture_object, &accretion_disk_resource_desc, &accretion_disk_texture_desc, nullptr));

    // declaring black holes
    h_black_holes.push_back(BlackHole(500, Vector(0, 0, 0)));
    // h_black_holes.push_back(BlackHole(500, Vector(0, 0, 500)));
    // h_black_holes.push_back(BlackHole(500, Vector(0, 0, 1000)));
    int num_black_holes = (int)h_black_holes.size();

    // allocating device memory
    cudaHandleError(cudaMalloc((void **)&d_pixels, output_width * output_height * 3 * sizeof(unsigned char))); // TODO: mallocManaged?
    cudaHandleError(cudaMalloc((void **)&d_black_holes, num_black_holes * sizeof(BlackHole))); // TODO: mallocManaged?
    cudaHandleError(cudaMemcpy(d_black_holes, h_black_holes.data(), num_black_holes * sizeof(BlackHole), cudaMemcpyHostToDevice));
    cudaHandleError(cudaMalloc((void **)&d_ray_direction_vector_map, output_width * output_height * 3 * sizeof(float)));
    cudaHandleError(cudaMalloc((void **)&d_output_ray_direction_vector_map, output_width * output_height * 3 * sizeof(float)));
    cudaHandleError(cudaMalloc((void **)&d_accretion_disk_intersection, output_width * output_height * 3 * sizeof(unsigned char)));

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
    camera.position = Vector(200, 30, 0);

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
    while(!glfwWindowShouldClose(main_window)){ // TODO: pipelining
        // SPIN IN THE CENTER:
        // camera.direction = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle)));
        // ORBIT AROUND CENTER:
        // camera.position = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle))) * camera_distance_from_center;
        // camera.direction = (Vector::Zero() - camera.position).normalize();
        // MOVE INTO CENTER:
        // camera.position = camera.position + camera.direction / 2;

        int remaining_width = output_width;
        int grid_index = 0;
        constexpr int threads_per_block = 1024; // TO STUDY: declared new variables in kernel, tried rendering 1080p
                                     // with 1024 threads per block (what was by defualt) and got:
                                     // "CUDA error: too many resources requested for launch"
                                     // (works with 896 on my GPU, but might differ on others)
                                     // SOLUTION: switching from double precision to single precision

        static int rotation_frames_processed = 0;
        if(camera.position.length() > 30 && rotation_frames_processed == 0){
            camera.position.x = (100 - frames_processed / 10.f);
        }
        else{
            camera.position = Vector(
                cosf(rotation_frames_processed / 100.f + PI / 2),
                sinf(rotation_frames_processed / 100.f + PI / 2),
                0
            )*(30 - rotation_frames_processed / 100.f);
            camera.up = camera.position.normalize();
            camera.direction = camera.up ^ Vector::West();
            rotation_frames_processed++;
        }
        
        auto frame_start = std::chrono::high_resolution_clock::now();
        while(remaining_width > 0){
            if(remaining_width >= threads_per_block){
                ray<<<output_height, threads_per_block>>>(camera, accretion_disk_texture_object, output_width, output_height, grid_index, threads_per_block, d_black_holes, num_black_holes, anti_aliasing, dynamic_ray_resolution, frames_processed, d_ray_direction_vector_map, d_ray_resolution_multiplier_map, d_accretion_disk_intersection);
            }
            else{ // remainder kernel call
                ray<<<output_height, remaining_width>>>(camera, accretion_disk_texture_object, output_width, output_height, grid_index, threads_per_block, d_black_holes, num_black_holes, anti_aliasing, dynamic_ray_resolution, frames_processed, d_ray_direction_vector_map, d_ray_resolution_multiplier_map, d_accretion_disk_intersection);                
            }
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error line %d: %s: %s\n", __LINE__, cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
                exit(1);
            }
            remaining_width -= threads_per_block;
            grid_index++;
        }
        cudaDeviceSynchronize();
        
        if(dynamic_ray_resolution){
            remaining_width = output_width;
            grid_index = 0;
            while(remaining_width > 0){
                if(remaining_width >= threads_per_block){
                    ray_post_processing<<<output_height, threads_per_block>>>(d_ray_direction_vector_map, d_output_ray_direction_vector_map, grid_index, gaussian_kernel_size, threads_per_block, output_width, output_height);
                }
                else{
                    ray_post_processing<<<output_height, remaining_width>>>(d_ray_direction_vector_map, d_output_ray_direction_vector_map, grid_index, gaussian_kernel_size, threads_per_block, output_width, output_height);
                }
                cudaError_t cudaError = cudaGetLastError();
                if (cudaError != cudaSuccess) {
                    fprintf(stderr, "CUDA error line %d: %s: %s\n", __LINE__, cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
                    exit(1);
                }
                remaining_width -= threads_per_block;
                grid_index++;
            }
        }
        else{
            cudaHandleError(cudaMemcpy(d_output_ray_direction_vector_map, d_ray_direction_vector_map, output_width * output_height * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        cudaDeviceSynchronize();

        remaining_width = output_width;
        grid_index = 0;
        while(remaining_width > 0){
            if(remaining_width >= threads_per_block){
                skybox_texture_extraction<<<output_height, threads_per_block>>>(skybox_texture_object, d_output_ray_direction_vector_map, d_ray_resolution_multiplier_map, d_pixels, grid_index, threads_per_block, anti_aliasing, output_width, d_accretion_disk_intersection);
            }
            else{
                skybox_texture_extraction<<<output_height, remaining_width>>>(skybox_texture_object, d_output_ray_direction_vector_map, d_ray_resolution_multiplier_map, d_pixels, grid_index, threads_per_block, anti_aliasing, output_width, d_accretion_disk_intersection); 
            }
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error line %d: %s: %s\n", __LINE__, cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
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
            fprintf(stderr, "CUDA error line %d: %s: %s\n", __LINE__, cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
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
        // h_black_holes.clear();
        // h_black_holes.push_back(BlackHole(500, Vector(0, 0, 0)));
        // h_black_holes.push_back(BlackHole(500, Vector(0, 0, clip(500.f - angle, 0, 1000))));
        // h_black_holes.push_back(BlackHole(500, Vector(0, 0, clip(1000.f - 2*angle, 0, 1000))));
        
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
