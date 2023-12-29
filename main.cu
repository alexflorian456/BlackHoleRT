#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <string>

#include "lodepng.h"

#define WIDTH 800
#define HEIGHT 450

#define SKYBOX_WIDTH 2048
#define SKYBOX_HEIGHT 1024

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

        __device__ Vector
        normalize() const{
            double norm = length();
            if(norm > 0){
                return Vector(x / norm, y / norm, z / norm);
            }
            return Vector(x, y, z);
        }

        __device__ Vector
        operator+(const Vector& b) const{
            return Vector(x + b.x, y + b.y, z + b.z);
        }

        __device__ Vector
        operator-(const Vector& b) const{
            return Vector(x - b.x, y - b.y, z - b.z);
        }

        __device__ __host__ double
        operator*(const Vector& b) const{
            return x * b.x + y * b.y + z * b.z;
        }

        __device__ Vector
        operator^(const Vector& b) const{
            return Vector(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
        }

        __device__ Vector
        operator*(double& k) const{
            return Vector(x * k, y * k, z * k);
        }

        __device__ Vector
        operator/(double& k) const{
            return Vector(x / k, y / k, z / k);
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

};

__device__ void
set_image_pixel(unsigned char * pixels, int i, int j, Color color){
    pixels[i * WIDTH * 4 + j * 4 + 0] = color.red;
    pixels[i * WIDTH * 4 + j * 4 + 1] = color.green;
    pixels[i * WIDTH * 4 + j * 4 + 2] = color.blue;
    pixels[i * WIDTH * 4 + j * 4 + 3] = color.alpha;
}

__device__ double
image_to_view_plane(int n, int img_size, double view_plane_size){
    return - n * view_plane_size / img_size + view_plane_size / 2;
}

__device__ __host__ double
radians_to_degrees(double radians){
    return radians * 180.0 / 3.1415;
}

__device__ __host__ double
degrees_to_radians(double degrees){
    return degrees * 3.1415 / 180.0;
}

__device__ __host__ double
angle_between_vectors(Vector a, Vector b){ // result between 0 and 180
    double dot_product = a * b;
    double length_a = a.length();
    double length_b = b.length();

    return radians_to_degrees(std::acos(dot_product / (length_a * length_b)));
}

__device__ Color
extract_skybox_color(unsigned char * pixels, int i, int j){
    return Color(
        pixels[i * SKYBOX_WIDTH * 4 + j * 4 + 0],
        pixels[i * SKYBOX_WIDTH * 4 + j * 4 + 1],
        pixels[i * SKYBOX_WIDTH * 4 + j * 4 + 2],
        pixels[i * SKYBOX_WIDTH * 4 + j * 4 + 3]
    );
}

__global__ void
ray(Camera camera, unsigned char * skybox, unsigned char * pixels){
    int i = blockIdx.x;
    int j = threadIdx.x;

    // initial ray direction
    Vector view_parallel = (camera.up ^ camera.direction).normalize();
    Vector camera_to_view_plane = camera.direction * camera.view_plane_distance;
    double image_to_view_plane_width = image_to_view_plane(j, WIDTH, camera.view_plane_width);
    double image_to_view_plane_height = image_to_view_plane(i, HEIGHT, camera.view_plane_height);
    Vector ray_direction = camera_to_view_plane + view_parallel * image_to_view_plane_width + camera.up * image_to_view_plane_height;
    
    // skybox rendering
    double elevation_angle = angle_between_vectors(Vector::Up(), ray_direction); // 0 degrees <=> straight up, 180 degress <=> straight down
    // project ray_direction on xOz to calculate azimuth
    Vector ray_direction_projection_on_xOz = Vector(ray_direction.x, 0, ray_direction.z);
    double projection_north_angle = angle_between_vectors(ray_direction_projection_on_xOz, Vector::North());
    double azimuth_angle = ray_direction_projection_on_xOz.z > 0 ?
                           projection_north_angle :
                      (360-projection_north_angle); // if z component is negative => azimuth angle > 180 degrees

    // TODO: why did i need printf("") before?

    int skybox_height_coordinate = SKYBOX_HEIGHT * elevation_angle / 180;
    int skybox_width_coordinate = SKYBOX_WIDTH * azimuth_angle / 360;

    // printf("%d %d\n", skybox_height_coordinate, skybox_width_coordinate);

    Color skybox_color = extract_skybox_color(skybox, skybox_height_coordinate, skybox_width_coordinate);

    set_image_pixel(pixels, i, j, skybox_color);
}

void
saveImage(std::string filename, const unsigned char * pixels, int width, int height) {
    if (lodepng::encode(filename, pixels, width, height) != 0) {
        std::cerr << "Error while saving PNG file: " << filename << std::endl;
    }
}

int
main(int argc, char ** argv){

/*
args:
1 - skybox filename, default: starmap_2020_2k_gal.png
*/
    
    Vector camera_position  = Vector::Zero();
    Vector camera_direction = Vector::South();
    Vector camera_up        = Vector::Up();
    const double view_plane_distance    = 20;
    const double view_plane_width       = 160;
    const double view_plane_height      = 90;
    const double front_plane_distance   = 0;
    const double back_plane_distance    = 1000;
    Camera camera(  camera_position, camera_direction, camera_up,
                    view_plane_distance, view_plane_width, view_plane_height,
                    front_plane_distance, back_plane_distance);

    unsigned char * d_pixels;
    unsigned char * d_skybox;
    unsigned char * h_pixels = new unsigned char[WIDTH * HEIGHT * 4];
    std::vector<unsigned char> h_skybox_vector;
    
    cudaMalloc((void **)&d_pixels, WIDTH * HEIGHT * 4 * sizeof(unsigned char));
    cudaMalloc((void **)&d_skybox, SKYBOX_WIDTH * SKYBOX_HEIGHT * 4 * sizeof(unsigned char));

    unsigned int unsigned_skybox_width = SKYBOX_WIDTH;
    unsigned int unsigned_skybox_height = SKYBOX_HEIGHT;
    std::string skybox_filename;
    if(argc > 1){
        skybox_filename = argv[1];
    }
    else{
        skybox_filename = "starmap_2020_2k_gal.png";
    }
    if(0 != lodepng::decode(h_skybox_vector, unsigned_skybox_width, unsigned_skybox_height, skybox_filename)){
        std::cerr << "Error opening skybox file " << skybox_filename << std::endl;
        return 1;
    };
    cudaMemcpy(d_skybox, h_skybox_vector.data(), SKYBOX_WIDTH * SKYBOX_HEIGHT * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    double angle = 0;
    double progress_percent = -1;
    while(angle < 360){
        camera.direction = Vector(std::cos(degrees_to_radians(angle)), 0, std::sin(degrees_to_radians(angle)));
        double current_angle = angle_between_vectors(Vector::North(), camera.direction);
        ray<<<HEIGHT, WIDTH>>>(camera, d_skybox, d_pixels);
        cudaDeviceSynchronize();

        char * output_path = (char *)malloc(25 * sizeof(char));
        sprintf(output_path, "output\\frame%03d.png", (int)angle);
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        saveImage(output_path, h_pixels, WIDTH, HEIGHT);

        double current_progress_percent = angle / 360.0 * 100;

        if (current_progress_percent - progress_percent > 1) {
            progress_percent = current_progress_percent;
            printf("%d%% Done\n", (int)progress_percent);
        }
        
        angle++;
    }

    printf("Done\n");
    
    return 0;
}