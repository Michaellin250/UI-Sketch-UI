from __future__ import print_function
import torch
import math
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import renderer
import time
import sys, os
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
#from sketch import *

Tensor = torch.cuda.FloatTensor 

def grid_construction_sphere(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1

    return grid.cuda()

def grid_construction_cube(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a cube with size 2
    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    # cube_left_bound_index = float(grid_res - 1) / 4;
    # cube_right_bound_index = float(grid_res - 1) / 4 * 3;
    cube_left_bound_index = float(grid_res - 1) / 8 * 3;
    cube_right_bound_index = float(grid_res - 1) / 8 * 5;
    cube_center = float(grid_res - 1) / 2;

    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                if (i >= cube_left_bound_index and i <= cube_right_bound_index and
                    j >= cube_left_bound_index and j <= cube_right_bound_index and
                    k >= cube_left_bound_index and k <= cube_right_bound_index):
                    grid[i,j,k] = voxel_size * max(abs(i - cube_center), abs(j - cube_center), abs(k - cube_center)) - 1;
                else:
                    grid[i,j,k] = math.sqrt(pow(voxel_size * (max(i - cube_right_bound_index, cube_left_bound_index - i, 0)), 2) +
                                    pow(voxel_size * (max(j - cube_right_bound_index, cube_left_bound_index - j, 0)), 2) +
                                    pow(voxel_size * (max(k - cube_right_bound_index, cube_left_bound_index - k, 0)), 2));
    return grid


def generate_mask(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid, camera, back): #, camera_list

    # Generate rays
    e = camera
    
    w_h_3 = torch.zeros(width, height, 3).cuda()
    w_h = torch.zeros(width, height).cuda()
    eye_x = e[0]
    eye_y = e[1]
    eye_z = e[2]

    # Do ray tracing in cpp
    outputs = renderer.ray_matching(w_h_3, w_h, grid, width, height, \
     bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                            grid_res_x, grid_res_y, grid_res_z, \
                            eye_x, \
                            eye_y, \
                            eye_z
                            )

    # {intersection_pos, voxel_position, directions}
    intersection_pos_rough = outputs[0]
    voxel_min_point_index = outputs[1]
    ray_direction = outputs[2]

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:,:,0] != -1).type(Tensor)

    return mask



def calculate_distance(sdf_x, sdf_y):

    # compute the magnitude of the distance
    distance = torch.sqrt(torch.pow(sdf_x, 2) \
                         + torch.pow(sdf_y, 2))

    return distance


# use the distance function that compute the magnitude of the distance with offset
def compare_offset_distance(sdf_2d_x, sdf_2d_y, sdf_x, sdf_y, offset_x, offset_y):

    # compute the magnitude of the distance with offset
    distance_with_offset = calculate_distance(sdf_x + offset_x, sdf_y + offset_y)

    # compute the current sdf values
    current_sdf_value = calculate_distance(sdf_2d_x[1:-1, 1:-1], sdf_2d_y[1:-1, 1:-1])

    # update the sdf with smaller distance
    sdf_2d_x[1:-1, 1:-1] = (distance_with_offset < current_sdf_value).type(Tensor) \
                        * (sdf_2d_x[1+offset_x:-1+offset_x+sdf_2d_x.shape[0], \
                                    1+offset_y:-1+offset_y+sdf_2d_x.shape[1]] + offset_x) \
                        + (distance_with_offset >= current_sdf_value).type(Tensor) \
                        * sdf_2d_x[1:-1, 1:-1]
    sdf_2d_y[1:-1, 1:-1] = (distance_with_offset < current_sdf_value).type(Tensor) \
                        * (sdf_2d_y[1+offset_x:-1+offset_x+sdf_2d_y.shape[0], \
                                    1+offset_y:-1+offset_y+sdf_2d_y.shape[1]] + offset_y) \
                        + (distance_with_offset >= current_sdf_value).type(Tensor) \
                        * sdf_2d_y[1:-1, 1:-1]


    return sdf_2d_x, sdf_2d_y


# update sdf by 8-points Signed Sequential Euclidean Distance Transform
def update_sdf_2d_by_8_values(sdf_2d_x, sdf_2d_y):

    # update sdf values by comparing the 8 values around
    sdf_2d_x_prev = torch.zeros_like(sdf_2d_x)
    sdf_2d_y_prev = torch.zeros_like(sdf_2d_y)
    while not torch.all(torch.eq(sdf_2d_x_prev, sdf_2d_x)) or \
             not torch.all(torch.eq(sdf_2d_y_prev, sdf_2d_y)):
        sdf_2d_x_prev = sdf_2d_x.clone()
        sdf_2d_y_prev = sdf_2d_y.clone()
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[:-2, :-2], sdf_2d_y[:-2, :-2], -1, -1)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[1:-1, :-2], sdf_2d_y[1:-1, :-2], 0, -1)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[2:, :-2], sdf_2d_y[2:, :-2], 1, -1)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[:-2, 1:-1], sdf_2d_y[:-2, 1:-1], -1, 0)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[2:, 1:-1], sdf_2d_y[2:, 1:-1], 1, 0)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[:-2, 2:], sdf_2d_y[:-2, 2:], -1, 1)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[1:-1, 2:], sdf_2d_y[1:-1, 2:], 0, 1)
        sdf_2d_x, sdf_2d_y = compare_offset_distance(sdf_2d_x, \
                     sdf_2d_y, sdf_2d_x[2:, 2:], sdf_2d_y[2:, 2:], 1, 1)

    # compute final sdf values
    sdf_value = calculate_distance(sdf_2d_x[1:-1, 1:-1], sdf_2d_y[1:-1, 1:-1])

    return sdf_value


def compute_sdf_2d(mask):
     
    # compute the distance outside the object
    # we create sdf values with padded to avoid boundary problems
    sdf_2d_x_positive = Tensor(mask.shape[0] + 2, mask.shape[1] + 2)
    sdf_2d_y_positive = Tensor(mask.shape[0] + 2, mask.shape[1] + 2)
    sdf_2d_x_negative = Tensor(mask.shape[0] + 2, mask.shape[1] + 2)
    sdf_2d_y_negative = Tensor(mask.shape[0] + 2, mask.shape[1] + 2)

    # initialize the grids to either (0,0) if the pixel is 'inside', 
    # or (+INF,+INF) if it's outside.
    x = mask.clone()
    y = mask.clone()
    x[mask == 0] = 999
    y[mask == 0] = 999
    x[mask == 1] = 0
    y[mask == 1] = 0
    sdf_2d_x_positive[0, :] = 999
    sdf_2d_x_positive[-1, :] = 999
    sdf_2d_x_positive[:, 0] = 999
    sdf_2d_x_positive[:, -1] = 999
    sdf_2d_y_positive[0, :] = 999
    sdf_2d_y_positive[-1, :] = 999
    sdf_2d_y_positive[:, 0] = 999
    sdf_2d_y_positive[:, -1] = 999
    sdf_2d_x_positive[1:-1, 1:-1] = x
    sdf_2d_y_positive[1:-1, 1:-1] = y

    # compute sdf value for background
    positive_sdf_value = update_sdf_2d_by_8_values(sdf_2d_x_positive, sdf_2d_y_positive)

    # initialize the grids to either (+INF,+INF) if the pixel is 'inside', 
    # or (0, 0) if it's outside.
    x = mask.clone()
    y = mask.clone()
    x[mask == 0] = 0
    y[mask == 0] = 0
    x[mask == 1] = 999
    y[mask == 1] = 999
    sdf_2d_x_negative[0, :] = 0
    sdf_2d_x_negative[-1, :] = 0
    sdf_2d_x_negative[:, 0] = 0
    sdf_2d_x_negative[:, -1] = 0
    sdf_2d_y_negative[0, :] = 0
    sdf_2d_y_negative[-1, :] = 0
    sdf_2d_y_negative[:, 0] = 0
    sdf_2d_y_negative[:, -1] = 0
    sdf_2d_x_negative[1:-1, 1:-1] = x
    sdf_2d_y_negative[1:-1, 1:-1] = y

    # compute sdf value for foreground
    negative_sdf_value = update_sdf_2d_by_8_values(sdf_2d_x_negative, sdf_2d_y_negative)

    # final sdf value is positive minus negative
    sdf_value = positive_sdf_value - negative_sdf_value

    return sdf_value

# 2D interpolation on the image plane
# given a point position on the image, get the sdf value
def compute_sdf_3d_from_2d(sdf_value_2d, pixel_x, pixel_y, width, height):

    # get the min pixel of the point
    pixel_x_min = torch.floor(pixel_x - 0.5).type(torch.cuda.LongTensor)
    pixel_y_min = torch.floor(pixel_y - 0.5).type(torch.cuda.LongTensor)

    #print(sdf_value_2d.shape)
    #print(sdf_value_2d.view(-1).shape)
    #print(pixel_y_min.reshape(262144).shape)
    #pixel_y_min = pixel_y_min.reshape(262144)
    #pixel_x_min = pixel_y_min.reshape(262144)
    #tp = torch.index_select(sdf_value_2d.view(-1), 0, \
    #                 pixel_y_min + width * pixel_x_min.view(-1)
    #print(tp.shape)
    # get the four sdf values in the pixels around the point we want
    sdf_value_3d_1 = torch.index_select(sdf_value_2d.view(-1), 0, \
                     pixel_y_min.reshape(262144)+ width * pixel_y_min.reshape(262144)).view(pixel_x.shape)
    sdf_value_3d_2 = torch.index_select(sdf_value_2d.view(-1), 0, \
                     (pixel_y_min + 1).reshape(262144)\
                                    + width * pixel_x_min.reshape(262144)).view(pixel_x.shape)
    sdf_value_3d_3 = torch.index_select(sdf_value_2d.view(-1), 0, \
                     pixel_y_min.reshape(262144) + width * (pixel_x_min + 1).reshape(262144)).view(pixel_x.shape)
    sdf_value_3d_4 = torch.index_select(sdf_value_2d.view(-1), 0, \
                     (pixel_y_min + 1).reshape(262144) + width \
                        * (pixel_x_min + 1).reshape(262144)).view(pixel_x.shape)

    # linear interpolation along x-axis
    tx = pixel_x - pixel_x_min.type(Tensor)
    c0 = (1 - tx) * sdf_value_3d_1 + tx * sdf_value_3d_3
    c1 = (1 - tx) * sdf_value_3d_2 + tx * sdf_value_3d_4

    # linear interpolation along y-axis
    ty = pixel_y - pixel_y_min.type(Tensor)
    sdf_value_3d = (1 - ty) * c0 + ty * c1

    return sdf_value_3d

# map sdf values on image to sdf values on sdf grid
def map_sdf_2d_to_3d(camera, bounding_box_min_x, bounding_box_max_x, \
                        grid_res_x, width, height, mask):

    # get the positions of grid points
    grid_base = torch.linspace(bounding_box_min_x, bounding_box_max_x, \
                                 grid_res_x).repeat(grid_res_x, grid_res_x, 1).cuda()
    grid_x_pos = grid_base.transpose(0, 2)
    grid_y_pos = grid_base.transpose(1, 2)
    grid_z_pos = grid_base

    # distance between camera and plane is 1 
    # by line-plane intersection, image plane is the set of p
    # where (p - p0) dot n = 0
    # where n is a normal vector to the plane and p0 is a point on the plane.
    n = (Tensor([0, 0, 0]) - camera) / torch.norm((Tensor([0, 0, 0]) - camera), p=2) 
    p0 = camera + n 
    
    # the vector equation for the line is p = I0 + Id
    # where I is a vector in the direction of the line, I0 is a point on the line, 
    # and d is a scalar in the real number domain, I0 = camera
    I_x = grid_x_pos - camera[0]
    I_y = grid_y_pos - camera[1]
    I_z = grid_z_pos - camera[2]
    I_length = torch.sqrt(I_x ** 2 + I_y ** 2 + I_z ** 2)
    I_x /= I_length
    I_y /= I_length
    I_z /= I_length

    # we finally get d = ((p0 - I0) dot n) / (I dot n) = n dot n / I dot n
    d = torch.dot(n, n) / (I_x * n[0] + I_y * n[1] + I_z * n[2])

    # the point of line-plane intersection is I0 + Id
    grid_to_image_intersection_x = camera[0] + I_x * d
    grid_to_image_intersection_y = camera[1] + I_y * d
    grid_to_image_intersection_z = camera[2] + I_z * d

    # image boundary is tan(30) = sqrt(3) / 3
    image_boundary = Tensor([math.sqrt(3) / 3])
    top = right = image_boundary
    bottom = left = -image_boundary

    # compute camera vectors
    at = Tensor([0, 0, 0])
    up = Tensor([0, 1, 0])
    w = (camera - at) / torch.norm((camera - at), p=2) 
    v = torch.cross(up, w) / torch.norm(torch.cross(up, w), p=2)
    u = torch.cross(w, v)

    # compute image origin in 3D
    image_pixel_distance_x = 2 * image_boundary / width
    image_pixel_distance_y = 2 * image_boundary / height
    origin_x = left
    origin_y = bottom 
    image_origin_3d = camera + origin_x * u + origin_y * v - w
    # print(image_origin_3d)
    # exit

    # compute the distance between intersection points and origin
    distance_to_origin_x = grid_to_image_intersection_x - image_origin_3d[0]
    distance_to_origin_y = grid_to_image_intersection_y - image_origin_3d[1]
    distance_to_origin_z = grid_to_image_intersection_z - image_origin_3d[2]

    # we have au + bv = distance_to_origin, solve it
    a = (v[0] * distance_to_origin_y - v[1] * distance_to_origin_x) \
                                        / (v[0] * u[1] - u[0] * v[1])
    b = (u[0] * distance_to_origin_y - u[1] * distance_to_origin_x) \
                                        / (u[0] * v[1] - v[0] * u[1])
    if torch.isnan(a).any():
        a = (v[0] * distance_to_origin_z - v[2] * distance_to_origin_x) \
                                        / (v[0] * u[2] - u[0] * v[2])
        b = (u[0] * distance_to_origin_z - u[2] * distance_to_origin_x) \
                                        / (u[0] * v[2] - v[0] * u[2])
    if torch.isnan(a).any():
        a = (v[1] * distance_to_origin_z - v[2] * distance_to_origin_y) \
                                        / (v[1] * u[2] - u[1] * v[2])
        b = (u[1] * distance_to_origin_z - u[2] * distance_to_origin_y) \
                                        / (u[1] * v[2] - v[1] * u[2])

    # get which pixel the intersection locates
    pixel_x = width - a / image_pixel_distance_x
    pixel_y = b / image_pixel_distance_y
    pixel_x += width
    pixel_y += height

    # create a larger mask to avoid clamp
    large_mask = torch.zeros(3 * width, 3 * height).cuda()
    large_mask[width:2*width,height:2*height] += mask

    # compute sdf values of the larger image
    sdf_value_2d = compute_sdf_2d(large_mask)

    # compute sdf values for 3d grid
    sdf_value_3d = compute_sdf_3d_from_2d(sdf_value_2d, pixel_x, pixel_y, 3 * width, 3 * height) 

    # translate from pixel number distance to real distance and also scale 
    # according to projection camera2gridcenter_dist / camera2image_dist(=1)
    sdf_value_3d *= image_pixel_distance_x *  ((grid_x_pos - camera[0]) * n[0] \
                    + (grid_y_pos - camera[1]) * n[1] + (grid_z_pos - camera[2]) * n[2])
    

    return sdf_value_3d


# compute visual hull from mask
def compute_visual_hull(mask, camera, bounding_box_min_x, bounding_box_max_x, \
                                            grid_res_x, width, height):

    # map sdf values on image to sdf values on sdf grid
    sdf_value_3d = map_sdf_2d_to_3d(camera, \
                     bounding_box_min_x, bounding_box_max_x, \
                      grid_res_x, width, height, mask)

    return sdf_value_3d

if __name__ == "__main__":

    # define the folder name for results
    dir_name = "comb/"
    os.makedirs("./" + dir_name, exist_ok=True)

    camera_list = [Tensor([0,0,5]), # 0
                   Tensor([0.1,5,0]),
                   Tensor([5,0,0]), 
                   Tensor([0,0,-5]), 
                   Tensor([0.1,-5,0]), 
                   Tensor([-5,0,0])] # 5

    # bounding box
    bounding_box_min_x = bounding_box_min_y = bounding_box_min_z = -2.
    bounding_box_max_x = bounding_box_max_y = bounding_box_max_z = 2. 

    # size of the image
    width = height = 64
    grid_res_x = grid_res_y = grid_res_z = 17

    # Construct the sdf grid
    grid = grid_construction_cube(grid_res_x, bounding_box_min_x, bounding_box_max_x) 
    # grid = torch.cat((grid[:,:,10:], grid[:,:,:10]), dim=2)
    # grid = torch.cat((grid[:,57:,:], grid[:,:57,:]), dim=1)
    grid = torch.cat((grid[:,:,3:], grid[:,:,:3]), dim=2)
    # grid = torch.cat((grid[:,57:,:], grid[:,:57,:]), dim=1)
  

    # generate visual hull initialization
    for i in range(len(camera_list)):
        camera = camera_list[i]
        mask = generate_mask(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                            4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width,\
                            height, grid, camera, 1, camera_list)

        torchvision.utils.save_image(mask, "./" + dir_name + "mask" + str(i) + ".png")
        visual_hull = compute_visual_hull(mask, camera, bounding_box_min_x, \
                                            bounding_box_max_x, grid_res_x, width, height)

        for cam in range(len(camera_list)):             
            image_initial = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width, height, visual_hull, camera_list[cam], 1, camera_list)
            torchvision.utils.save_image(image_initial, "./" + dir_name + str(i) + "grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png")


        # merge visual hull results
        if i == 0:
            visual_hull_initialization = visual_hull
        else:
            visual_hull_initialization = torch.max(visual_hull, visual_hull_initialization)


        for cam in range(len(camera_list)):
            image = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
                                bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                                4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width,\
                                height, visual_hull_initialization, camera_list[cam], 1, camera_list)
            torchvision.utils.save_image(image, "./" + dir_name + "mask" + str(i) + "_" + str(cam) + ".png")


    print("----- END -----")
