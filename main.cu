#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <ctime>

#include "lodepng.h"

#define GRAVITATIONAL_CONSTANT 6.674e-3 // real gravitational constant is 6.674e-11
#define RAY_TIME_RESOLUTION 1 // time between two ray positions when computing gravitational lensing
#define RAY_MAX_ITERATIONS 300 // maximum amount of ray positions computed per pixel

__device__ __host__ double
radians_to_degrees(double radians){
    return radians * 180.0 / 3.1415;
}

__device__ __host__ double
degrees_to_radians(double degrees){
    return degrees * 3.1415 / 180.0;
}

class 
Vector{

    public:
        double x;
        double y;
        double z;

        __device__ __host__
        Vector(double x, double y, double z):
            x(x),
            y(y),
            z(z)
        {}

        __device__ __host__ double
        length_2() const{
            return x * x + y * y + z * z;
        }

        __device__ __host__ double
        length() const{
            return std::sqrt(length_2());
        }

        __device__ __host__ Vector
        normalize() const{
            double norm = length();
            if(norm > 0){
                return Vector(x / norm, y / norm, z / norm);
            }
            return Vector(x, y, z);
        }

        __device__ __host__ Vector
        operator+(const Vector& b) const{
            return Vector(x + b.x, y + b.y, z + b.z);
        }

        __device__ __host__ Vector
        operator-(const Vector& b) const{
            return Vector(x - b.x, y - b.y, z - b.z);
        }

        __device__ __host__ double
        operator*(const Vector& b) const{
            return x * b.x + y * b.y + z * b.z;
        }

        __device__ __host__ Vector
        operator^(const Vector& b) const{
            return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
        }

        __device__ __host__ Vector
        operator*(double k) const{
            return Vector(x * k, y * k, z * k);
        }

        __device__ __host__ Vector
        operator/(double k) const{
            return Vector(x / k, y / k, z / k);
        }

        __device__ __host__ Vector
        operator+=(const Vector& b){
            *this = *this + b;
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
        unsigned char red;
        unsigned char green;
        unsigned char blue;
        unsigned char alpha;

        __device__
        Color(  unsigned char red, unsigned char green,
                unsigned char blue, unsigned char alpha):
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
};

class
Camera{

    public:
        Vector position;
        Vector direction;
        Vector up;

        double view_plane_distance;
        double view_plane_width;
        double view_plane_height;

        double front_plane_distance;
        double back_plane_distance;

        Camera( Vector position, Vector direction, Vector up,
                double view_plane_distance, double view_plane_width, double view_plane_height,
                double front_plane_distance, double back_plane_distance):

            position(position), direction(direction), up(up),
            view_plane_distance(view_plane_distance), view_plane_width(view_plane_width), view_plane_height(view_plane_height),
            front_plane_distance(front_plane_distance), back_plane_distance(back_plane_distance)
        {}

        Camera( int output_width, int output_height, int field_of_view,
                Vector position, Vector direction, Vector up,
                double front_plane_distance, double back_plane_distance):
            
            position(position), direction(direction), up(up),
            front_plane_distance(front_plane_distance), back_plane_distance(back_plane_distance)
        {
            view_plane_distance = 1;
            int theta = 90 - field_of_view / 2;
            view_plane_width = 2 * view_plane_distance / std::tan(degrees_to_radians(theta));
            double aspect_ratio = (double)output_width / (double)output_height;
            view_plane_height = view_plane_width / aspect_ratio;
        }
};

class
BlackHole {

    public:
        double mass;
        Vector position;

        BlackHole(double mass, Vector position):
            mass(mass),
            position(position)
        {}
};

__device__ void
set_image_pixel(unsigned char * pixels, int i, int j, int output_width, Color color){
    pixels[i * output_width * 4 + j * 4 + 0] = color.red;
    pixels[i * output_width * 4 + j * 4 + 1] = color.green;
    pixels[i * output_width * 4 + j * 4 + 2] = color.blue;
    pixels[i * output_width * 4 + j * 4 + 3] = color.alpha;
}

__device__ double
image_to_view_plane(int n, int img_size, double view_plane_size){
    return - n * view_plane_size / img_size + view_plane_size / 2;
}

__device__ __host__ double
angle_between_vectors(Vector a, Vector b){ // result between 0 and 180
    double dot_product = a * b;
    double length_a = a.length();
    double length_b = b.length();

    return radians_to_degrees(std::acos(dot_product / (length_a * length_b)));
}

__device__ Color
extract_skybox_color(unsigned char * pixels, int i, int j, int skybox_width){
    return Color(
        pixels[i * skybox_width * 4 + j * 4 + 0],
        pixels[i * skybox_width * 4 + j * 4 + 1],
        pixels[i * skybox_width * 4 + j * 4 + 2],
        pixels[i * skybox_width * 4 + j * 4 + 3]
    );
}

__global__ void
ray(Camera camera,
    unsigned char * skybox, unsigned char * pixels,
    int output_width, int output_height,
    int skybox_width, int skybox_height,
    int grid_index, int block_size, // currently, even if ray is called as a "remainder" kernel with eg. <<<1920, 128>>>,
                                    // block_size is still passed as the block size of a non-"remainder" kernel, eg. 896
                                    // in order for the image coordinate arithmetic to be correct
    BlackHole * black_holes, int num_black_holes){

    int i = blockIdx.x;
    int j = threadIdx.x + grid_index * block_size;
    
    // initial ray direction
    Vector view_parallel = (camera.up ^ camera.direction).normalize();
    Vector camera_to_view_plane = camera.direction * camera.view_plane_distance;
    double image_to_view_plane_width    = image_to_view_plane(j, output_width , camera.view_plane_width);
    double image_to_view_plane_height   = image_to_view_plane(i, output_height, camera.view_plane_height);
    Vector ray_direction = camera_to_view_plane + view_parallel * image_to_view_plane_width + camera.up * image_to_view_plane_height;

    // gravitational lensing computation
    Vector old_position = camera.position;
    Vector old_velocity = ray_direction; // TODO?: try normalize
    Vector new_position = Vector::Zero();
    Vector new_velocity = Vector::Zero();
    double gravitational_constant = GRAVITATIONAL_CONSTANT;
    double ray_time_resolution = RAY_TIME_RESOLUTION;
                                                /* to paint a pixel black, the ray has to be stuck */
    int escape_sphere_radius = 5;               /* in a sphere of radius = escape_radius           */
    Vector escape_sphere_center = old_position; /* centered in escape_sphere_center                */
    int escape_sphere_iterations = 10;          /* for escape_sphere_iterations iterations         */                                         
    for(int iter=0; iter<RAY_MAX_ITERATIONS; iter++){
        Vector resultant_force = Vector::Zero();
        for(int black_hole_index=0; black_hole_index<num_black_holes; black_hole_index++){ // TO STUDY?: extract to separate kernel
            Vector black_hole_position = black_holes[black_hole_index].position;
            double black_hole_mass = black_holes[black_hole_index].mass;
            double r_squared = (black_hole_position - old_position).length_2();
            Vector r_hat = (black_hole_position - old_position).normalize();
            resultant_force += r_hat * gravitational_constant * black_hole_mass / r_squared;
        }
        new_velocity = (old_velocity + resultant_force * ray_time_resolution).normalize() * old_velocity.length();
        new_position = old_position + new_velocity * ray_time_resolution;

        if((new_position - escape_sphere_center).length() > escape_sphere_radius){
            // the ray has escaped the sphere, reset the iteration counter and set
            // sphere center to new_position
            escape_sphere_center = new_position;
            escape_sphere_iterations = 10;
        }
        else{
            escape_sphere_iterations--;
            if(escape_sphere_iterations < 0){
                set_image_pixel(pixels, i, j, output_width, Color::Black());
                return;
            }
        }

        old_velocity = new_velocity;
        old_position = new_position;
    }
    ray_direction = new_velocity;
    
    // skybox rendering
    double elevation_angle = angle_between_vectors(Vector::Up(), ray_direction); // 0 degrees <=> straight up, 180 degress <=> straight down
    // project ray_direction on xOz to calculate azimuth
    Vector ray_direction_projection_on_xOz = Vector(ray_direction.x, 0, ray_direction.z);
    double projection_north_angle = angle_between_vectors(ray_direction_projection_on_xOz, Vector::North());
    double azimuth_angle = ray_direction_projection_on_xOz.z > 0 ?
                           projection_north_angle :
                    (360 - projection_north_angle); // if z component is negative => azimuth angle > 180 degrees

    // TO STUDY: why did i need printf("") before?

    int skybox_height_coordinate    = skybox_height * elevation_angle / 180;
    int skybox_width_coordinate     = skybox_width  * azimuth_angle   / 360;

    Color skybox_color = extract_skybox_color(skybox, skybox_height_coordinate, skybox_width_coordinate, skybox_width);

    set_image_pixel(pixels, i, j, output_width, skybox_color);
}

void
saveImage(std::string filename, const unsigned char * pixels, int width, int height) {
    if (0 != lodepng::encode(filename, pixels, width, height)) {
        fprintf(stderr, "Error when saving PNG file: %s\n", filename.c_str());
        exit(1);
    }
}

int
main(int argc, char ** argv){
/*
args:
1 - output width
2 - output height
3 - field of view (degrees)
4 - skybox filename, optional, must be an equirectangular image (2:1 aspect ratio), default: starmap_2020_2k_gal.png
*/
    int output_width    = atoi(argv[1]);
    int output_height   = atoi(argv[2]);
    int field_of_view   = atoi(argv[3]);
    Vector camera_position  = Vector::Zero();
    Vector camera_direction = Vector::South();
    Vector camera_up        = Vector::Up();
    const double front_plane_distance   = 0;
    const double back_plane_distance    = 1000;
    const double camera_distance_from_center = 60;
    Camera camera(  output_width, output_height, field_of_view,
                    camera_position, camera_direction, camera_up,
                    front_plane_distance, back_plane_distance);

    unsigned char * d_pixels;
    unsigned char * d_skybox;
    unsigned char * h_pixels = (unsigned char *)malloc(output_width * output_height * 4 * sizeof(unsigned char));
    std::vector<unsigned char> h_skybox;
    unsigned int skybox_width   = UINT32_MAX;
    unsigned int skybox_height  = UINT32_MAX;
    std::string skybox_filename;
    BlackHole * d_black_holes;
    std::vector<BlackHole> h_black_holes;
    if(argc > 4){
        skybox_filename = argv[4];
    }
    else{
        skybox_filename = "starmap_2020_2k_gal.png";
    }
    if(0 != lodepng::decode(h_skybox, skybox_width, skybox_height, skybox_filename)){
        fprintf(stderr, "Error opening skybox file %s\n", skybox_filename.c_str());
        exit(1);
    };

    // assuming skybox image is equirectangular
    int num_pixels  = h_skybox.size() / 4;
    skybox_height   = std::sqrt(num_pixels / 2);
    skybox_width    = 2 * skybox_height;

    // declaring black holes
    h_black_holes.push_back(BlackHole(500, Vector(-20, 0, -20)));
    h_black_holes.push_back(BlackHole(500, Vector( 20, 0,  20)));
    int num_black_holes = h_black_holes.size();

    // allocating device memory
    cudaMalloc((void **)&d_pixels, output_width * output_height * 4 * sizeof(unsigned char));
    cudaMalloc((void **)&d_skybox, skybox_width * skybox_height * 4 * sizeof(unsigned char));
    cudaMalloc((void **)&d_black_holes, num_black_holes * sizeof(BlackHole));
    cudaMemcpy(d_skybox, h_skybox.data(), skybox_width * skybox_height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_black_holes, h_black_holes.data(), num_black_holes * sizeof(BlackHole), cudaMemcpyHostToDevice);
    
    double angle = 0;
    double progress_percent = -1;
    auto t_start = std::chrono::high_resolution_clock::now();
    while(angle < 360){ // TODO?: call multiple kernels from multiple threads
        // ORIGINAL:
        // camera.direction = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle)));
        camera.position = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle))) * camera_distance_from_center;
        camera.direction = (Vector::Zero() - camera.position).normalize();
        
        int remaining_width = output_width;
        int grid_index = 0;
        int threads_per_block = 896; // TO STUDY: declared new variables in kernel, tried rendering 1080p
                                     // with 1024 threads per block (what was by defualt) and got:
                                     // "CUDA error: too many resources requested for launch"
                                     // (works with 896 on my GPU, but might differ on others)
        while(remaining_width > 0){
            if(remaining_width >= threads_per_block){
                ray<<<output_height, threads_per_block>>>(camera, d_skybox, d_pixels, output_width, output_height, skybox_width, skybox_height, grid_index, threads_per_block, d_black_holes, num_black_holes);
            }
            else{ // remainder kernel call
                ray<<<output_height, remaining_width>>>(camera, d_skybox, d_pixels, output_width, output_height, skybox_width, skybox_height, grid_index, threads_per_block, d_black_holes, num_black_holes);                
            }
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
                exit(1);
            }
            remaining_width -= threads_per_block;
            grid_index++;
        }
        cudaDeviceSynchronize();

        char * output_path = (char*)malloc(25 * sizeof(char));
        sprintf(output_path, "output\\frame%03d.png", (int)angle);
        cudaMemcpy(h_pixels, d_pixels, output_width * output_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        saveImage(output_path, h_pixels, output_width, output_height);

        double current_progress_percent = angle / 360.0 * 100;

        if (current_progress_percent - progress_percent > 1) {
            progress_percent = current_progress_percent;
            printf("%d%% Done\n", (int)progress_percent);
        }
        
        angle++;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("Time: %f ms\n", duration);
    printf("Estimated fps: %f\n", 360 / (duration / 1000));

    printf("Done\n");
    
    return 0;
}