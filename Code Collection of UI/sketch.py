from __future__ import print_function
import torch
import math
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import renderer
import time
import sys, os
#import sketch as sk
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from visual_hull import *

cuda = True if torch.cuda.is_available() else False
#width = height = 64
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# construct a rounded cylinder
def grid_construction_rounded_cylinder(grid_res, bounding_box_min, bounding_box_max, rounded_radius, radius, height):

    # Construct the sdf grid for a rounded cylinder with size 2
    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    center = float(grid_res - 1) / 2;

    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                # distance between xz on the circle and the radius of the circle (negative inside)
                d_xz = voxel_size * math.sqrt((i - center) ** 2 + (k - center) ** 2) - radius

                # distance between y and the height
                d_y = voxel_size * abs(j - center) - height
                # exit()

                # calculate SDF value
                grid[i,j,k] = min(max(d_xz, d_y), 0.0) + math.sqrt(max(d_xz, 0) ** 2 + max(d_y, 0) ** 2)
                # exit()
    return grid

# construct a rounded cylinder
def grid_construction_cylinder_horizontal(grid_res, bounding_box_min, bounding_box_max, rounded_radius, radius, height):

    # Construct the sdf grid for a rounded cylinder with size 2
    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    center = float(grid_res - 1) / 2;

    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                # distance between xz on the circle and the radius of the circle (negative inside)
                d_xz = voxel_size * math.sqrt((i - center) ** 2 + (j - center) ** 2) - radius

                # distance between y and the height
                d_y = voxel_size * abs(k - center) - height
                # exit()

                # calculate SDF value
                grid[i,j,k] = min(max(d_xz, d_y), 0.0) + math.sqrt(max(d_xz, 0) ** 2 + max(d_y, 0) ** 2)
                # exit()
    return grid

def read_txt(file_path, grid_res_x, grid_res_y, grid_res_z):
    with open(file_path) as file:
        grid = Tensor(grid_res_x, grid_res_y, grid_res_z)
        for i in range(grid_res_x):
            for j in range(grid_res_y):
                for k in range(grid_res_z):
                    grid[i][j][k] = float(file.readline())
    print (grid)
    
    return grid

# Read a file and create a sdf grid with target_grid_res
def read_sdf(file_path, target_grid_res, target_bounding_box_min, target_bounding_box_max, target_voxel_size):

    with open(file_path) as file:  
        line = file.readline()

        # Get grid resolutions
        grid_res = line.split()
        grid_res_x = int(grid_res[0])
        grid_res_y = int(grid_res[1])
        grid_res_z = int(grid_res[2])

        # Get bounding box min
        line = file.readline()
        bounding_box_min = line.split()
        bounding_box_min_x = float(bounding_box_min[0]) 
        bounding_box_min_y = float(bounding_box_min[1])
        bounding_box_min_z = float(bounding_box_min[2]) 

        line = file.readline()
        voxel_size = float(line)

        # max bounding box (we need to plus 0.0001 to avoid round error)
        bounding_box_max_x = bounding_box_min_x + voxel_size * (grid_res_x - 1)
        bounding_box_max_y = bounding_box_min_y + voxel_size * (grid_res_y - 1) 
        bounding_box_max_z = bounding_box_min_z + voxel_size * (grid_res_z - 1) 

        min_bounding_box_min = min(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z) 
        # print(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z)
        max_bounding_box_max = max(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z) 
        # print(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z)
        max_dist = max(bounding_box_max_x - bounding_box_min_x, bounding_box_max_y - bounding_box_min_y, bounding_box_max_z - bounding_box_min_z)

        # max_dist += 0.1
        max_grid_res = max(grid_res_x, grid_res_y, grid_res_z)

        grid = []
        for i in range(grid_res_x):
            grid.append([])
            for j in range(grid_res_y):
                grid[i].append([])
                for k in range(grid_res_z):
                    # grid_value = float(file.readline())
                    grid[i][j].append(2)
                    # lst.append(grid_value)

        for i in range(grid_res_z):
            for j in range(grid_res_y):
                for k in range(grid_res_x):
                    grid_value = float(file.readline())
                    grid[k][j][i] = grid_value

        grid = Tensor(grid)
        target_grid = Tensor(target_grid_res, target_grid_res, target_grid_res)

        linear_space_x = torch.linspace(0, target_grid_res-1, target_grid_res)
        linear_space_y = torch.linspace(0, target_grid_res-1, target_grid_res)
        linear_space_z = torch.linspace(0, target_grid_res-1, target_grid_res)
        first_loop = linear_space_x.repeat(target_grid_res * target_grid_res, 1).t().contiguous().view(-1).unsqueeze_(1)
        second_loop = linear_space_y.repeat(target_grid_res, target_grid_res).t().contiguous().view(-1).unsqueeze_(1)
        third_loop = linear_space_z.repeat(target_grid_res * target_grid_res).unsqueeze_(1)
        loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()

        min_x = Tensor([bounding_box_min_x]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        min_y = Tensor([bounding_box_min_y]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        min_z = Tensor([bounding_box_min_z]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1) 

        move_to_center_x = Tensor([(max_dist - (bounding_box_max_x - bounding_box_min_x)) / 2]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        move_to_center_y = Tensor([(max_dist - (bounding_box_max_y - bounding_box_min_y)) / 2]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        move_to_center_z = Tensor([(max_dist - (bounding_box_max_z - bounding_box_min_z)) / 2]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        move_to_center_matrix = torch.cat((move_to_center_x, move_to_center_y, move_to_center_z), 1)
        
        # Get the position of the grid points in the refined grid
        points = bounding_min_matrix + target_voxel_size * max_dist / (target_bounding_box_max - target_bounding_box_min) * loop - move_to_center_matrix 
        if points[(points[:, 0] < bounding_box_min_x)].shape[0] != 0:
            points[(points[:, 0] < bounding_box_min_x)] = Tensor([bounding_box_max_x, bounding_box_max_y, bounding_box_max_z]).view(1,3)
        if points[(points[:, 1] < bounding_box_min_y)].shape[0] != 0:
            points[(points[:, 1] < bounding_box_min_y)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 2] < bounding_box_min_z)].shape[0] != 0:
            points[(points[:, 2] < bounding_box_min_z)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 0] > bounding_box_max_x)].shape[0] != 0:
            points[(points[:, 0] > bounding_box_max_x)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 1] > bounding_box_max_y)].shape[0] != 0:
            points[(points[:, 1] > bounding_box_max_y)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 2] > bounding_box_max_z)].shape[0] != 0:
            points[(points[:, 2] > bounding_box_max_z)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        voxel_min_point_index_x = torch.floor((points[:,0].unsqueeze_(1) - min_x) / voxel_size).clamp(max=grid_res_x-2)
        voxel_min_point_index_y = torch.floor((points[:,1].unsqueeze_(1) - min_y) / voxel_size).clamp(max=grid_res_y-2)
        voxel_min_point_index_z = torch.floor((points[:,2].unsqueeze_(1) - min_z) / voxel_size).clamp(max=grid_res_z-2)
        voxel_min_point_index = torch.cat((voxel_min_point_index_x, voxel_min_point_index_y, voxel_min_point_index_z), 1)
        voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

        # Compute the sdf value of the grid points in the refined grid
        target_grid = calculate_sdf_value(grid, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x, grid_res_y, grid_res_z).view(target_grid_res, target_grid_res, target_grid_res)

        # "shortest path" algorithm to fill the values (for changing from "cuboid" SDF to "cube" SDF)
        # min of the SDF values of the closest points + the distance to these points
        # calculate the max resolution get which areas we need to compute the shortest path
        max_res = max(grid_res_x, grid_res_y, grid_res_z)
        if grid_res_x == max_res:
            min_x = 0
            max_x = target_grid_res - 1
            min_y = math.ceil((target_grid_res - target_grid_res / float(grid_res_x) * grid_res_y) / 2)
            max_y = target_grid_res - min_y - 1
            min_z = math.ceil((target_grid_res - target_grid_res / float(grid_res_x) * grid_res_z) / 2)
            max_z = target_grid_res - min_z - 1
        if grid_res_y == max_res:
            min_x = math.ceil((target_grid_res - target_grid_res / float(grid_res_y) * grid_res_x) / 2)
            max_x = target_grid_res - min_x - 1
            min_y = 0
            max_y = target_grid_res - 1
            min_z = math.ceil((target_grid_res - target_grid_res / float(grid_res_y) * grid_res_z) / 2)
            max_z = target_grid_res - min_z - 1
        if grid_res_z == max_res:
            min_x = math.ceil((target_grid_res - target_grid_res / float(grid_res_z) * grid_res_x) / 2)
            max_x = target_grid_res - min_x - 1
            min_y = math.ceil((target_grid_res - target_grid_res / float(grid_res_z) * grid_res_y) / 2)
            max_y = target_grid_res - min_y - 1
            min_z = 0
            max_z = target_grid_res - 1
        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)
        min_z = int(min_z)
        max_z = int(max_z)
       
        # fill the values
        res = target_grid.shape[0]
        for i in range(res):
            for j in range(res):
                for k in range(res):

                    # fill the values outside both x-axis and y-axis
                    if k < min_x and j < min_y:
                        target_grid[k][j][i] = target_grid[min_x][min_y][i] + math.sqrt((min_x - k) ** 2 + (min_y - j) ** 2) * voxel_size
                    elif k < min_x and j > max_y:
                        target_grid[k][j][i] = target_grid[min_x][max_y][i] + math.sqrt((min_x - k) ** 2 + (max_y - j) ** 2) * voxel_size
                    elif k > max_x and j < min_y:
                        target_grid[k][j][i] = target_grid[max_x][min_y][i] + math.sqrt((max_x - k) ** 2 + (min_y - j) ** 2) * voxel_size
                    elif k > max_x and j > max_y:
                        target_grid[k][j][i] = target_grid[max_x][max_y][i] + math.sqrt((max_x - k) ** 2 + (max_y - j) ** 2) * voxel_size
                    
                    # fill the values outside both x-axis and z-axis
                    elif k < min_x and i < min_z:
                        target_grid[k][j][i] = target_grid[min_x][j][min_z] + math.sqrt((min_x - k) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif k < min_x and i > max_z:
                        target_grid[k][j][i] = target_grid[min_x][j][max_z] + math.sqrt((min_x - k) ** 2 + (max_z - i) ** 2) * voxel_size
                    elif k > max_x and i < min_z:
                        target_grid[k][j][i] = target_grid[max_x][j][min_z] + math.sqrt((max_x - k) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif k > max_x and i > max_z:
                        target_grid[k][j][i] = target_grid[max_x][j][max_z] + math.sqrt((max_x - k) ** 2 + (max_z - i) ** 2) * voxel_size

                    # fill the values outside both y-axis and z-axis
                    elif j < min_y and i < min_z:
                        target_grid[k][j][i] = target_grid[k][min_y][min_z] + math.sqrt((min_y - j) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif j < min_y and i > max_z:
                        target_grid[k][j][i] = target_grid[k][min_y][max_z] + math.sqrt((min_y - j) ** 2 + (max_z - i) ** 2) * voxel_size
                    elif j > max_y and i < min_z:
                        target_grid[k][j][i] = target_grid[k][max_y][min_z] + math.sqrt((max_y - j) ** 2 + (min_z - i) ** 2) * voxel_size
                    elif j > max_y and i > max_z:
                        target_grid[k][j][i] = target_grid[k][max_y][max_z] + math.sqrt((max_y - j) ** 2 + (max_z - i) ** 2) * voxel_size

                    # fill the values outside x-axis
                    elif k < min_x:
                        target_grid[k][j][i] = target_grid[min_x][j][i] + math.sqrt((min_x - k) ** 2) * voxel_size
                    elif k > max_x:
                        target_grid[k][j][i] = target_grid[max_x][j][i] + math.sqrt((max_x - k) ** 2) * voxel_size

                    # fill the values outside y-axis
                    elif j < min_y:
                        target_grid[k][j][i] = target_grid[k][min_y][i] + math.sqrt((min_y - j) ** 2) * voxel_size
                    elif j > max_y:
                        target_grid[k][j][i] = target_grid[k][max_y][i] + math.sqrt((max_y - j) ** 2) * voxel_size

                    # fill the values outside z-axis
                    elif i < min_z:
                        target_grid[k][j][i] = target_grid[k][j][min_z] + math.sqrt((min_z - i) ** 2) * voxel_size
                    elif i > max_z:
                        target_grid[k][j][i] = target_grid[k][j][max_z] + math.sqrt((max_z - i) ** 2) * voxel_size

        return target_grid

        
def grid_construction_cube(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a cube with size 2
    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    cube_left_bound_index = float(grid_res - 1) / 4;
    cube_right_bound_index = float(grid_res - 1) / 4 * 3;
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

def grid_construction_torus(grid_res, bounding_box_min, bounding_box_max):
    
    # radius of the circle between the two circles
    radius_big = 1.5

    # radius of the small circle
    radius_small = 0.5

    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                x = bounding_box_min + voxel_size * i
                y = bounding_box_min + voxel_size * j
                z = bounding_box_min + voxel_size * k

                grid[i,j,k] = math.sqrt(math.pow((math.sqrt(math.pow(y, 2) + math.pow(z, 2)) - radius_big), 2)
                              + math.pow(x, 2)) - radius_small;

    return grid



def grid_construction_sphere_big(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1.6
    if cuda:
        return grid.cuda()
    else:
        return grid

def grid_construction_sphere_small(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1.5
    if cuda:
        return grid.cuda()
    else:
        return grid


def get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z):
    
    # largest index
    n_x = grid_res_x - 1
    n_y = grid_res_y - 1
    n_z = grid_res_z - 1

    # x-axis normal vectors
    X_1 = torch.cat((grid[1:,:,:], (3 * grid[n_x,:,:] - 3 * grid[n_x-1,:,:] + grid[n_x-2,:,:]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid[1,:,:] + 3 * grid[0,:,:] + grid[2,:,:]).unsqueeze_(0), grid[:n_x,:,:]), 0)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

    # y-axis normal vectors
    Y_1 = torch.cat((grid[:,1:,:], (3 * grid[:,n_y,:] - 3 * grid[:,n_y-1,:] + grid[:,n_y-2,:]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid[:,1,:] + 3 * grid[:,0,:] + grid[:,2,:]).unsqueeze_(1), grid[:,:n_y,:]), 1)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)

    # z-axis normal vectors
    Z_1 = torch.cat((grid[:,:,1:], (3 * grid[:,:,n_z] - 3 * grid[:,:,n_z-1] + grid[:,:,n_z-2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid[:,:,1] + 3 * grid[:,:,0] + grid[:,:,2]).unsqueeze_(2), grid[:,:,:n_z]), 2)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)


    return [grid_normal_x, grid_normal_y, grid_normal_z]


def get_intersection_normal(intersection_grid_normal, intersection_pos, voxel_min_point, voxel_size):

    # Compute parameters
    tx = (intersection_pos[:,:,0] - voxel_min_point[:,:,0]) / voxel_size
    ty = (intersection_pos[:,:,1] - voxel_min_point[:,:,1]) / voxel_size
    tz = (intersection_pos[:,:,2] - voxel_min_point[:,:,2]) / voxel_size

    intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:,:,0] \
                        + tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:,:,1] \
                        + (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:,:,2] \
                        + tz * ty * (1 - tx) * intersection_grid_normal[:,:,3] \
                        + (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:,:,4] \
                        + tz * (1 - ty) * tx * intersection_grid_normal[:,:,5] \
                        + (1 - tz) * ty * tx * intersection_grid_normal[:,:,6] \
                        + tz * ty * tx * intersection_grid_normal[:,:,7]

    return intersection_normal


#for initialization part
def initialization(mask_list, camera_list, bounding_box_min_x, \
                    bounding_box_max_x, grid_res_x, width, height):
    for i in range(len(camera_list)):
        # compute visual hull
        visual_hull = compute_visual_hull(mask_list[i], camera_list[i], bounding_box_min_x, \
                                        bounding_box_max_x, grid_res_x, width, height)

        # # merge visual hull results
        if i == 0:
            grid_initial = visual_hull
        else:
            grid_initial = torch.max(visual_hull, grid_initial)

    return grid_initial


# need to develop further here
def update_mask():
    ss = 1

# Do one more step for ray matching
def calculate_sdf_value(grid, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x, grid_res_y, grid_res_z):

    string = ""

    # Linear interpolate along x axis the eight values
    tx = (points[:,0] - voxel_min_point[:,0]) / voxel_size;
    string = string + "\n\nvoxel_size: \n" + str(voxel_size)
    string = string + "\n\ntx: \n" + str(tx)

    if cuda:
        tx = tx.cuda()
        x = voxel_min_point_index.long()[:,0]
        y = voxel_min_point_index.long()[:,1]
        z = voxel_min_point_index.long()[:,2]

        string = string + "\n\nx: \n" + str(x)
        string = string + "\n\ny: \n" + str(y)
        string = string + "\n\nz: \n" + str(z)

        c01 = (1 - tx) * grid[x,y,z] + tx * grid[x+1,y,z];
        c23 = (1 - tx) * grid[x,y+1,z] + tx * grid[x+1,y+1,z];
        c45 = (1 - tx) * grid[x,y,z+1] + tx * grid[x+1,y,z+1];
        c67 = (1 - tx) * grid[x,y+1,z+1] + tx * grid[x+1,y+1,z+1];

        string = string + "\n\n(1 - tx): \n" + str((1 - tx))
        string = string + "\n\ngrid[x,y,z]: \n" + str(grid[x,y,z])
        string = string + "\n\ngrid[x+1,y,z]: \n" + str(grid[x+1,y,z])
        string = string + "\n\nc01: \n" + str(c01)
        string = string + "\n\nc23: \n" + str(c23)
        string = string + "\n\nc45: \n" + str(c45)
        string = string + "\n\nc67: \n" + str(c67)

        # Linear interpolate along the y axis
        ty = (points[:,1] - voxel_min_point[:,1]) / voxel_size;
        ty = ty.cuda()
        c0 = (1 - ty) * c01 + ty * c23;
        c1 = (1 - ty) * c45 + ty * c67;

        string = string + "\n\nty: \n" + str(ty)

        string = string + "\n\nc0: \n" + str(c0)
        string = string + "\n\nc1: \n" + str(c1)

        # Return final value interpolated along z
        tz = (points[:,2] - voxel_min_point[:,2]) / voxel_size;
        tz = tz.cuda()
        string = string + "\n\ntz: \n" + str(tz)
        
    else:
        x = voxel_min_point_index.numpy()[:,0]
        y = voxel_min_point_index.numpy()[:,1]
        z = voxel_min_point_index.numpy()[:,2]

        c01 = (1 - tx) * grid[x,y,z] + tx * grid[x+1,y,z];
        c23 = (1 - tx) * grid[x,y+1,z] + tx * grid[x+1,y+1,z];
        c45 = (1 - tx) * grid[x,y,z+1] + tx * grid[x+1,y,z+1];
        c67 = (1 - tx) * grid[x,y+1,z+1] + tx * grid[x+1,y+1,z+1];

        # Linear interpolate along the y axis
        ty = (points[:,1] - voxel_min_point[:,1]) / voxel_size;
        c0 = (1 - ty) * c01 + ty * c23;
        c1 = (1 - ty) * c45 + ty * c67;

        # Return final value interpolated along z
        tz = (points[:,2] - voxel_min_point[:,2]) / voxel_size;

    result = (1 - tz) * c0 + tz * c1;

    return result


def compute_intersection_pos(grid, intersection_pos_rough, voxel_min_point, voxel_min_point_index, ray_direction, voxel_size, mask,width,height):
    
    # Linear interpolate along x axis the eight values
    tx = (intersection_pos_rough[:,:,0] - voxel_min_point[:,:,0]) / voxel_size;

    if cuda:

        x = voxel_min_point_index.long()[:,:,0]
        y = voxel_min_point_index.long()[:,:,1]
        z = voxel_min_point_index.long()[:,:,2]

        c01 = (1 - tx) * grid[x,y,z].cuda() + tx * grid[x+1,y,z].cuda();
        c23 = (1 - tx) * grid[x,y+1,z].cuda() + tx * grid[x+1,y+1,z].cuda();
        c45 = (1 - tx) * grid[x,y,z+1].cuda() + tx * grid[x+1,y,z+1].cuda();
        c67 = (1 - tx) * grid[x,y+1,z+1].cuda() + tx * grid[x+1,y+1,z+1].cuda();

    else:
        x = voxel_min_point_index.numpy()[:,:,0]
        y = voxel_min_point_index.numpy()[:,:,1]
        z = voxel_min_point_index.numpy()[:,:,2]

        c01 = (1 - tx) * grid[x,y,z] + tx * grid[x+1,y,z];
        c23 = (1 - tx) * grid[x,y+1,z] + tx * grid[x+1,y+1,z];
        c45 = (1 - tx) * grid[x,y,z+1] + tx * grid[x+1,y,z+1];
        c67 = (1 - tx) * grid[x,y+1,z+1] + tx * grid[x+1,y+1,z+1];     
           
    # Linear interpolate along the y axis
    ty = (intersection_pos_rough[:,:,1] - voxel_min_point[:,:,1]) / voxel_size;
    c0 = (1 - ty) * c01 + ty * c23;
    c1 = (1 - ty) * c45 + ty * c67;

    # Return final value interpolated along z
    tz = (intersection_pos_rough[:,:,2] - voxel_min_point[:,:,2]) / voxel_size;

    sdf_value = (1 - tz) * c0 + tz * c1;
    #print("sdf_value: ",sdf_value.shape)
    #print("width: ",width)
    #print("height: ",height)
    #print(intersection_pos_rough.shape)
    #print(ray_direction.shape) 
   # print(sdf_value.view(width,height,1).shape)
    return (intersection_pos_rough + ray_direction * sdf_value.view(width,height,1).repeat(1,1,3))\
                            + (1 - mask.view(width,height,1).repeat(1,1,3))

def generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid, camera, back, camera_list):

    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z)

    # Generate rays
    e = camera
    
    w_h_3 = torch.zeros(width, height, 3).cuda()
    w_h = torch.zeros(width, height).cuda()
    eye_x = e[0]
    eye_y = e[1]
    eye_z = e[2]

    # Do ray tracing in cpp
    outputs = renderer.ray_matching(w_h_3, w_h, grid, width, height, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
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

    # Initialize grid values and normals for intersection voxels
    intersection_grid_normal_x = Tensor(width, height, 8)
    intersection_grid_normal_y = Tensor(width, height, 8)
    intersection_grid_normal_z = Tensor(width, height, 8)
    intersection_grid = Tensor(width, height, 8)

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:,:,0] != -1).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:,:,0].type(torch.cuda.LongTensor)
    y = voxel_min_point_index[:,:,1].type(torch.cuda.LongTensor)
    z = voxel_min_point_index[:,:,2].type(torch.cuda.LongTensor)
    x[x == -1] = 0
    y[y == -1] = 0
    z[z == -1] = 0

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x2 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x3 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x5 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y2 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y3 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y4 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y5 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y6 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y7 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y8 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 2) + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z2 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z3 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z4 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z5 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z6 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z7 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z8 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 2) + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor([bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(grid, intersection_pos_rough,\
                                                voxel_min_point, voxel_min_point_index,\
                                                ray_direction, voxel_size, mask,width,height)

    intersection_pos = intersection_pos * mask.repeat(3,1,1).permute(1,2,0)
    shading = Tensor(width, height).fill_(0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x, intersection_pos, voxel_min_point, voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y, intersection_pos, voxel_min_point, voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z, intersection_pos, voxel_min_point, voxel_size)
    
    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
    intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
    intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
    intersection_normal = torch.cat((intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 2)
    intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=2), 2).repeat(1, 1, 3)

    # Create the point light
    light_position = camera.repeat(width, height, 1)
    light_norm = torch.unsqueeze(torch.norm(light_position - intersection_pos, p=2, dim=2), 2).repeat(1, 1, 3)
    light_direction_point = (light_position - intersection_pos) / light_norm

    # Create the directional light
    shading = 0
    light_direction = (camera / torch.norm(camera, p=2)).repeat(width, height, 1)
    l_dot_n = torch.sum(light_direction * intersection_normal, 2).unsqueeze_(2)
    shading += 10 * torch.max(l_dot_n, Tensor(width, height, 1).fill_(0))[:,:,0] / torch.pow(torch.sum((light_position - intersection_pos) * light_direction_point, dim=2), 2) 

    # Get the final image 
    image = shading * mask  
    image[mask == 0] = 0

    return image

# The energy E captures the difference between a rendered image and
# a desired target image, and the rendered image is a function of the
# SDF values. You could write E(SDF) = ||rendering(SDF)-target_image||^2.
# In addition, there is a second term in the energy as you observed that
# constrains the length of the normal of the SDF to 1. This is a regularization
# term to make sure the output is still a valid SDF.
def loss_fn(output, target, grid, narrow_band, voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height):
    
    image_loss = torch.sum(torch.abs(target - output)) #/ (width * height)

    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z)
    sdf_loss = torch.sum(narrow_band[1:-1,1:-1,1:-1] * torch.abs(torch.pow(grid_normal_x[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                 + torch.pow(grid_normal_y[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                 + torch.pow(grid_normal_z[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2) - 1))\
                                  # / (grid_res_x - 1) #* (grid_res_y - 1) * (grid_res_z - 1))


    print("\n\nimage loss: ", image_loss)
    print("sdf loss: ", sdf_loss)

    return image_loss, sdf_loss

def sdf_diff(sdf1, sdf2):
    return torch.sum(torch.abs(sdf1 - sdf2)).item()


def get_8points_values(grid, width, height, grid_res_x, x, y, z, mask):

    # Get grid value for the 8 points around the intersecting voxel
    # (x, y, z) are intersection points
    # This line is equivalent to grid[x,y,z]
    x1 = torch.index_select(grid.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x2 = torch.index_select(grid.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x3 = torch.index_select(grid.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x5 = torch.index_select(grid.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(grid.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    grid_8points = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) #+ (1 - mask.view(width, height, 1).repeat(1,1,8))

    return grid_8points


# Compute the shape operator matrix of level set surface according to Appendix B of
# https://www.cs.drexel.edu/~david/Papers/kmu_siggraph02.pdf
def compute_shape_operator_matrix(grid_normal_x, grid_normal_y, grid_normal_z, intersection_normal, intersection_pos, voxel_min_point,\
                                 voxel_size, x, y, z, width, height, grid_res_x, grid_res_y, grid_res_z, mask):

    # largest index
    n_x = grid_res_x - 1
    n_y = grid_res_y - 1
    n_z = grid_res_z - 1

    # print(intersection_pos[128,128])
    # exit()
    # print(x[128,128], y[128,128], z[128,128]) ### 15,15,27
    # exit()

    # Compute the derivative of normal vectors
    # derivative of grid_normal_x along x-axis 
    X_1 = torch.cat((grid_normal_x[1:,:,:], (2 * grid_normal_x[n_x,:,:] - 3 * grid_normal_x[n_x-1,:,:] + grid_normal_x[n_x-2,:,:]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid_normal_x[1,:,:] + 2 * grid_normal_x[0,:,:] + grid_normal_x[2,:,:]).unsqueeze_(0), grid_normal_x[:n_x,:,:]), 0)
    grid_normal_x_dx = (X_1 - X_2) / 2
    # print(grid_normal_x_dx[15,15,27], X_1[15,15,27], X_2[15,15,27])
    # exit()

    # derivative of grid_normal_x along y-axis 
    Y_1 = torch.cat((grid_normal_x[:,1:,:], (2 * grid_normal_x[:,n_y,:] - 3 * grid_normal_x[:,n_y-1,:] + grid_normal_x[:,n_y-2,:]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid_normal_x[:,1,:] + 2 * grid_normal_x[:,0,:] + grid_normal_x[:,2,:]).unsqueeze_(1), grid_normal_x[:,:n_y,:]), 1)
    grid_normal_x_dy = (Y_1 - Y_2) / 2 

    # derivative of grid_normal_x along z-axis 
    Z_1 = torch.cat((grid_normal_x[:,:,1:], (2 * grid_normal_x[:,:,n_z] - 3 * grid_normal_x[:,:,n_z-1] + grid_normal_x[:,:,n_z-2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid_normal_x[:,:,1] + 2 * grid_normal_x[:,:,0] + grid_normal_x[:,:,2]).unsqueeze_(2), grid_normal_x[:,:,:n_z]), 2)
    grid_normal_x_dz = (Z_1 - Z_2) / 2

    # derivative of grid_normal_y along x-axis 
    X_1 = torch.cat((grid_normal_y[1:,:,:], (2 * grid_normal_y[n_x,:,:] - 3 * grid_normal_y[n_x-1,:,:] + grid_normal_y[n_x-2,:,:]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid_normal_y[1,:,:] + 2 * grid_normal_y[0,:,:] + grid_normal_y[2,:,:]).unsqueeze_(0), grid_normal_y[:n_x,:,:]), 0)
    grid_normal_y_dx = (X_1 - X_2) / 2

    # derivative of grid_normal_y along y-axis 
    Y_1 = torch.cat((grid_normal_y[:,1:,:], (2 * grid_normal_y[:,n_y,:] - 3 * grid_normal_y[:,n_y-1,:] + grid_normal_y[:,n_y-2,:]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid_normal_y[:,1,:] + 2 * grid_normal_y[:,0,:] + grid_normal_y[:,2,:]).unsqueeze_(1), grid_normal_y[:,:n_y,:]), 1)
    grid_normal_y_dy = (Y_1 - Y_2) / 2

    # derivative of grid_normal_y along z-axis 
    Z_1 = torch.cat((grid_normal_y[:,:,1:], (2 * grid_normal_y[:,:,n_z] - 3 * grid_normal_y[:,:,n_z-1] + grid_normal_y[:,:,n_z-2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid_normal_y[:,:,1] + 2 * grid_normal_y[:,:,0] + grid_normal_y[:,:,2]).unsqueeze_(2), grid_normal_y[:,:,:n_z]), 2)
    grid_normal_y_dz = (Z_1 - Z_2) / 2

    # derivative of grid_normal_z along x-axis 
    X_1 = torch.cat((grid_normal_z[1:,:,:], (2 * grid_normal_z[n_x,:,:] - 3 * grid_normal_z[n_x-1,:,:] + grid_normal_z[n_x-2,:,:]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid_normal_z[1,:,:] + 2 * grid_normal_z[0,:,:] + grid_normal_z[2,:,:]).unsqueeze_(0), grid_normal_z[:n_x,:,:]), 0)
    grid_normal_z_dx = (X_1 - X_2) / 2

    # derivative of grid_normal_z along y-axis 
    Y_1 = torch.cat((grid_normal_z[:,1:,:], (2 * grid_normal_z[:,n_y,:] - 3 * grid_normal_z[:,n_y-1,:] + grid_normal_z[:,n_y-2,:]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid_normal_z[:,1,:] + 2 * grid_normal_z[:,0,:] + grid_normal_z[:,2,:]).unsqueeze_(1), grid_normal_z[:,:n_y,:]), 1)
    grid_normal_z_dy = (Y_1 - Y_2) / 2

    # derivative of grid_normal_z along z-axis 
    Z_1 = torch.cat((grid_normal_z[:,:,1:], (2 * grid_normal_z[:,:,n_z] - 3 * grid_normal_z[:,:,n_z-1] + grid_normal_z[:,:,n_z-2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid_normal_z[:,:,1] + 2 * grid_normal_z[:,:,0] + grid_normal_z[:,:,2]).unsqueeze_(2), grid_normal_z[:,:,:n_z]), 2)
    grid_normal_z_dz = (Z_1 - Z_2) / 2

    # get all the elements of matrix N
    N11 = grid_normal_x_dx
    N12 = grid_normal_y_dx
    N13 = grid_normal_z_dx
    N21 = grid_normal_x_dy
    N22 = grid_normal_y_dy
    N23 = grid_normal_z_dy
    N31 = grid_normal_x_dz
    N32 = grid_normal_y_dz
    N33 = grid_normal_z_dz

    # a = 15
    # b = 15
    # c = 27
    # N = Tensor([[N11[a,b,c], N12[a,b,c], N13[a,b,c]],
    #     [N21[a,b,c], N22[a,b,c], N23[a,b,c]],
    #     [N31[a,b,c], N32[a,b,c], N33[a,b,c]],
    #     ])

    # n = Tensor([grid_normal_x[a,b,c], grid_normal_y[a,b,c], grid_normal_z[a,b,c]])
    # print("yueyueyueyue=============")
    # print(n)


    # print("NNNNNNNNNNNNNNN")
    # print(N)


    # M = torch.eye(3).cuda() - Tensor([[n[0] * n[0], n[0] * n[1], n[0] * n[2]],
    #                                    [n[1] * n[0], n[1] * n[1], n[1] * n[2]],
    #                                    [n[2] * n[0], n[2] * n[1], n[2] * n[2]]])
    # B = torch.mm(N , M)
    # print("MMMMMMMMMMMMMM")
    # print(M)
    # print("BBBBBBBBBBBBB")
    # print(B)
    # print(torch.eig(B, True))
    # exit()


    # N11 = -grid_normal_x_dx
    # N12 = -grid_normal_x_dy
    # N13 = -grid_normal_x_dz
    # N21 = -grid_normal_y_dx
    # N22 = -grid_normal_y_dy
    # N23 = -grid_normal_y_dz
    # N31 = -grid_normal_z_dx
    # N32 = -grid_normal_z_dy
    # N33 = -grid_normal_z_dz

    # Get N for the 8 points around the intersecting voxel
    N11_8points = get_8points_values(N11, width, height, grid_res_x, x, y, z, mask)
    N12_8points = get_8points_values(N12, width, height, grid_res_x, x, y, z, mask)
    N13_8points = get_8points_values(N13, width, height, grid_res_x, x, y, z, mask)
    N21_8points = get_8points_values(N21, width, height, grid_res_x, x, y, z, mask)
    N22_8points = get_8points_values(N22, width, height, grid_res_x, x, y, z, mask)
    N23_8points = get_8points_values(N23, width, height, grid_res_x, x, y, z, mask)
    N31_8points = get_8points_values(N31, width, height, grid_res_x, x, y, z, mask)
    N32_8points = get_8points_values(N32, width, height, grid_res_x, x, y, z, mask)
    N33_8points = get_8points_values(N33, width, height, grid_res_x, x, y, z, mask)

    # Compute N for intersection points
    N11_intersection = get_intersection_normal(N11_8points, intersection_pos, voxel_min_point, voxel_size)
    N12_intersection = get_intersection_normal(N12_8points, intersection_pos, voxel_min_point, voxel_size)
    N13_intersection = get_intersection_normal(N13_8points, intersection_pos, voxel_min_point, voxel_size)
    N21_intersection = get_intersection_normal(N21_8points, intersection_pos, voxel_min_point, voxel_size)
    N22_intersection = get_intersection_normal(N22_8points, intersection_pos, voxel_min_point, voxel_size)
    N23_intersection = get_intersection_normal(N23_8points, intersection_pos, voxel_min_point, voxel_size)
    N31_intersection = get_intersection_normal(N31_8points, intersection_pos, voxel_min_point, voxel_size)
    N32_intersection = get_intersection_normal(N32_8points, intersection_pos, voxel_min_point, voxel_size)
    N33_intersection = get_intersection_normal(N33_8points, intersection_pos, voxel_min_point, voxel_size)

    # print(N11_intersection[128,128], N12_intersection[128,128], N13_intersection[128,128])
    # print(N21_intersection[128,128], N22_intersection[128,128], N23_intersection[128,128])
    # print(N31_intersection[128,128], N32_intersection[128,128], N33_intersection[128,128])
    # print("++++++")

    # get the intersection normals for each axis
    intersection_normal_x = intersection_normal[:,:,0]
    intersection_normal_y = intersection_normal[:,:,1]
    intersection_normal_z = intersection_normal[:,:,2]

    # print(intersection_normal[128,128])


    # compute M = I - n tensor product n
    M11 = 1 - intersection_normal_x * intersection_normal_x
    M12 = -intersection_normal_x * intersection_normal_y
    M13 = -intersection_normal_x * intersection_normal_z
    M21 = -intersection_normal_y * intersection_normal_x
    M22 = 1 - intersection_normal_y * intersection_normal_y
    M23 = -intersection_normal_y * intersection_normal_z
    M31 = -intersection_normal_z * intersection_normal_x
    M32 = -intersection_normal_z * intersection_normal_y
    M33 = 1 - intersection_normal_z * intersection_normal_z

    # print("--------")
    # print(M11[128,128], M12[128,128], M13[128,128])
    # print(M21[128,128], M22[128,128], M23[128,128])
    # print(M31[128,128], M32[128,128], M33[128,128])

    # compute the projection of the drivative matrix N onto the tangent
    # which is B = NM
    B11 = N11_intersection * M11 + N12_intersection * M21 + N13_intersection * M31
    B12 = N11_intersection * M12 + N12_intersection * M22 + N13_intersection * M32
    B13 = N11_intersection * M13 + N12_intersection * M23 + N13_intersection * M33
    B21 = N21_intersection * M11 + N22_intersection * M21 + N23_intersection * M31
    B22 = N21_intersection * M12 + N22_intersection * M22 + N23_intersection * M32
    B23 = N21_intersection * M13 + N22_intersection * M23 + N23_intersection * M33
    B31 = N31_intersection * M11 + N32_intersection * M21 + N33_intersection * M31
    B32 = N31_intersection * M12 + N32_intersection * M22 + N33_intersection * M32
    B33 = N31_intersection * M13 + N32_intersection * M23 + N33_intersection * M33


    # # compute M = I - n tensor product n
    # M11 = 1 - grid_normal_x * grid_normal_x
    # M12 = -grid_normal_x * grid_normal_y
    # M13 = -grid_normal_x * grid_normal_z
    # M21 = -grid_normal_y * grid_normal_x
    # M22 = 1 - grid_normal_y * grid_normal_y
    # M23 = -grid_normal_y * grid_normal_z
    # M31 = -grid_normal_z * grid_normal_x
    # M32 = -grid_normal_z * grid_normal_y
    # M33 = 1 - grid_normal_z * grid_normal_z

    # # print("--------")
    # # print(M11[128,128], M12[128,128], M13[128,128])
    # # print(M21[128,128], M22[128,128], M23[128,128])
    # # print(M31[128,128], M32[128,128], M33[128,128])

    # # compute the projection of the drivative matrix N onto the tangent
    # # which is B = NM
    # B11 = N11 * M11 + N12 * M21 + N13 * M31
    # B12 = N11 * M12 + N12 * M22 + N13 * M32
    # B13 = N11 * M13 + N12 * M23 + N13 * M33
    # B21 = N21 * M11 + N22 * M21 + N23 * M31
    # B22 = N21 * M12 + N22 * M22 + N23 * M32
    # B23 = N21 * M13 + N22 * M23 + N23 * M33
    # B31 = N31 * M11 + N32 * M21 + N33 * M31
    # B32 = N31 * M12 + N32 * M22 + N33 * M32
    # B33 = N31 * M13 + N32 * M23 + N33 * M33


    # print("--------")
    # # print(B11[128,128], B12[128,128], B13[128,128])
    # # print(B21[128,128], B22[128,128], B23[128,128])
    # # print(B31[128,128], B32[128,128], B33[128,128])


    # a = 15
    # b = 15
    # c = 27
    # x = a
    # y = b
    # print(grid_normal_x[a,b,c], grid_normal_y[a,b,c], grid_normal_z[a,b,c])
    # print("yueyueyueyueyueyueyue")


    # N = Tensor([[N11[a,b,c], N12[a,b,c], N13[a,b,c]],
    #             [N21[a,b,c], N22[a,b,c], N23[a,b,c]],
    #             [N31[a,b,c], N32[a,b,c], N33[a,b,c]],
    #     ])

    # print("NNNNNNNNNNNNNNN")
    # print(N)

    # M = Tensor([[M11[a,b,c], M12[a,b,c], M13[a,b,c]],
    #             [M21[a,b,c], M22[a,b,c], M23[a,b,c]],
    #             [M31[a,b,c], M32[a,b,c], M33[a,b,c]],
    #     ])

    # print("MMMMMMMMMMMMMM")
    # print(M)
    # B = Tensor([[B11[a,b,c], B12[a,b,c], B13[a,b,c]],
    #             [B21[a,b,c], B22[a,b,c], B23[a,b,c]],
    #             [B31[a,b,c], B32[a,b,c], B33[a,b,c]],
    #     ])

    # print("BBBBBBBBBBBBB")
    # print(B)
    # print(torch.eig(B, True))
    # exit()


    # a = 128
    # b = 150
    # x = a
    # y = b
    # print(intersection_normal[a,b])
    #  # print("yueyueyueyueyueyueyue")


    # N = Tensor([[N11_intersection[a,b], N12_intersection[a,b], N13_intersection[a,b]],
    #             [N21_intersection[a,b], N22_intersection[a,b], N23_intersection[a,b]],
    #             [N31_intersection[a,b], N32_intersection[a,b], N33_intersection[a,b]],
    #     ])

    # print("NNNNNNNNNNNNNNN")
    # print(N)

    # M = Tensor([[M11[a,b], M12[a,b], M13[a,b]],
    #             [M21[a,b], M22[a,b], M23[a,b]],
    #             [M31[a,b], M32[a,b], M33[a,b]],
    #     ])

    # print("MMMMMMMMMMMMMM")
    # print(M)
    # B = Tensor([[B11[a,b], B12[a,b], B13[a,b]],
    #             [B21[a,b], B22[a,b], B23[a,b]],
    #             [B31[a,b], B32[a,b], B33[a,b]],
    #     ])

    # print("BBBBBBBBBBBBB")
    # print(B)
    # # B = Tensor([[B11[a,b], B12[a,b], B13[a,b]],
    # #             [B21[a,b], B22[a,b], B23[a,b]],
    # #             [B31[a,b], B32[a,b], B33[a,b]],
    # #     ])
    # print(torch.eig(B, True))
    # exit()

    # compute BtB
    BtB11 = B11 * B11 + B21 * B21 + B31 * B31
    BtB12 = B11 * B12 + B21 * B22 + B31 * B32
    BtB13 = B11 * B13 + B21 * B23 + B31 * B33
    BtB21 = B12 * B11 + B22 * B21 + B32 * B31
    BtB22 = B12 * B12 + B22 * B22 + B32 * B32
    BtB23 = B12 * B13 + B22 * B23 + B32 * B33
    BtB31 = B13 * B11 + B23 * B21 + B33 * B31
    BtB32 = B13 * B12 + B23 * B22 + B33 * B32
    BtB33 = B13 * B13 + B23 * B23 + B33 * B33

    # compute the eigenvalue of BtB
    p1 = BtB12 ** 2 + BtB13 ** 2 + BtB23 ** 2
    q = (BtB11 + BtB22 + BtB33) / 3
    p2 = (BtB11 - q) ** 2 + (BtB22 - q) ** 2 + (BtB33 - q) ** 2 + 2 * p1
    p = torch.sqrt(p2 / 6)
    a = (1 / p) * (BtB11 - q)
    b = (1 / p) * (BtB12)
    c = (1 / p) * (BtB13)
    d = (1 / p) * (BtB21)
    e = (1 / p) * (BtB22 - q)
    f = (1 / p) * (BtB23)
    g = (1 / p) * (BtB31)
    h = (1 / p) * (BtB32)
    k = (1 / p) * (BtB33 - q)
    detBB = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
    r = detBB / 2
    phi = torch.acos(r) / 3
    eigenvalue = (q + 2 * p * torch.cos(phi)) #* mask * 0.5
    DD = torch.sqrt(eigenvalue)
    D = torch.sqrt(torch.pow(B11, 2) + torch.pow(B12, 2) + torch.pow(B13, 2) + \
                    torch.pow(B21, 2) + torch.pow(B22, 2) + torch.pow(B23, 2) + \
                    torch.pow(B31, 2) + torch.pow(B32, 2) + torch.pow(B33, 2))
    H = (B11 + B22 + B33) / 2
    curvature = H + torch.sqrt(D ** 2 / 2 - H ** 2)

    # print(curvature[15,15,27])
    # # exit()

    # # print(D[120:130, 120:130])
    # print(curvature[120:130, 120:130])

    # # D[120:130, 120:130] = 0
    # curvature[x,140:150] = 1

    # torchvision.utils.save_image(curvature, "../chair/ccc_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

    
    # exit()

    return B11, B12, B13, B21, B22, B23, B31, B32, B33


# Compute projection matrix P for apperent ridges
# transform shape operator matrix from object space to image space
def compute_transformation_matrix(camera, ray_direction, intersection_pos, intersection_normal, width, height):

    # calculate basis vectors for camera
    w = camera / torch.norm(camera, p=2)
    u = torch.cross(Tensor([0,1,0]), w)
    u = u / torch.norm(u, p=2)
    v = torch.cross(w, u)
    # print(torch.norm(u), torch.norm(v))
    # exit()

    # move camera along the basis vectors
    o1 = camera + u
    o2 = camera + v 
    o1 = o1.unsqueeze(0).unsqueeze(0).repeat(width, height, 1)
    o2 = o2.unsqueeze(0).unsqueeze(0).repeat(width, height, 1)

    # tangent plane is (p - intersection_pos) dot intersection_normal = 0
    # new rays are p = o + t * ray_direction
    # we need to calculate the intersection according to 
    # https://en.m.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    t1 = (torch.sum((intersection_pos - o1) * intersection_normal, dim=2) /\
         torch.sum(ray_direction * intersection_normal, dim=2)).unsqueeze(2).repeat(1,1,3)
    t2 = (torch.sum((intersection_pos - o2) * intersection_normal, dim=2) /\
         torch.sum(ray_direction * intersection_normal, dim=2)).unsqueeze(2).repeat(1,1,3)
    
    # compute basis vectors in object space
    basis1 = o1 + t1 * ray_direction - intersection_pos
    basis2 = o2 + t2 * ray_direction - intersection_pos
    # basis1 = basis1 / torch.unsqueeze(torch.norm(basis1, p=2, dim=2), 2).repeat(1, 1, 3)
    # basis2 = basis2 / torch.unsqueeze(torch.norm(basis2, p=2, dim=2), 2).repeat(1, 1, 3)
    # print(basis1[128,228])
    # print(basis2[128,228])
    # exit()
    # print(w, u, v)
    # # exit()
    # print((o1 + t1 * ray_direction)[128, 128], intersection_pos[128, 128])
    # print("128", torch.norm(basis1[128, 128]), torch.norm(basis2[128, 128]))
    # print("150", torch.norm(basis1[128, 150]), torch.norm(basis2[128, 150]))
    # print("180", torch.norm(basis1[128, 180]), torch.norm(basis2[128, 180]))

    # the cancatenation of basis1 and basis2 is P
    P11 = basis1[:,:,0]
    P12 = basis2[:,:,0]
    P21 = basis1[:,:,1]
    P22 = basis2[:,:,1]
    P31 = basis1[:,:,2]
    P32 = basis2[:,:,2]

    return P11, P12, P21, P22, P31, P32


def generate_image_sketch(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid, camera, back, curvature_threshold):   #0.55

    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z)

    # normalize grid normals
    grid_normal = torch.cat((grid_normal_x.unsqueeze(3), grid_normal_y.unsqueeze(3), grid_normal_z.unsqueeze(3)), 3)
    grid_normal = grid_normal / (torch.unsqueeze(torch.norm(grid_normal, p=2, dim=3), 3).repeat(1, 1, 1, 3) + 1e-4) 
    grid_normal_x = grid_normal[:,:,:,0]
    grid_normal_y = grid_normal[:,:,:,1]
    grid_normal_z = grid_normal[:,:,:,2]

    # Generate rays
    e = camera
    
    w_h_3 = torch.zeros(width, height, 3).cuda()
    w_h = torch.zeros(width, height).cuda()
    eye_x = e[0]
    eye_y = e[1]
    eye_z = e[2]

    # Do ray tracing in cpp
    outputs = renderer.ray_matching(w_h_3, w_h, grid, width, height, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
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

    # Initialize grid values and normals for intersection voxels
    intersection_grid_normal_x = Tensor(width, height, 8)
    intersection_grid_normal_y = Tensor(width, height, 8)
    intersection_grid_normal_z = Tensor(width, height, 8)
    intersection_grid = Tensor(width, height, 8)

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:,:,0] != -1).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:,:,0].type(torch.cuda.LongTensor)
    y = voxel_min_point_index[:,:,1].type(torch.cuda.LongTensor)
    z = voxel_min_point_index[:,:,2].type(torch.cuda.LongTensor)
    x[x == -1] = 0
    y[y == -1] = 0
    z[z == -1] = 0

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x2 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x3 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x5 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) #+ (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y2 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y3 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y4 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y5 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y6 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y7 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y8 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 2)# + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z2 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z3 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z4 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z5 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z6 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z7 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z8 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 2) #+ (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor([bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(grid, intersection_pos_rough,\
                                                voxel_min_point, voxel_min_point_index,\
                                                ray_direction, voxel_size, mask,width,height)

    intersection_pos = intersection_pos * mask.repeat(3,1,1).permute(1,2,0)
    shading = Tensor(width, height).fill_(0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x, intersection_pos, voxel_min_point, voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y, intersection_pos, voxel_min_point, voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z, intersection_pos, voxel_min_point, voxel_size)

    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
    intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
    intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
    intersection_normal = torch.cat((intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 2)
    intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=2), 2).repeat(1, 1, 3)

    # Normalize ray direction
    ray_direction = ray_direction / torch.unsqueeze(torch.norm(ray_direction, p=2, dim=2), 2).repeat(1, 1, 3)
    torch.autograd.set_detect_anomaly(True)

    # set all the normals of background to be the the same as ray directions
    intersection_normal[torch.unsqueeze(mask, 2).repeat(1, 1, 3) == 0] = ray_direction[torch.unsqueeze(mask, 2).repeat(1, 1, 3) == 0]

    # compute shape operator matrix (the directional derivative of the normal along vector
    # r in the tangent plane) for intersection points in object space 
    B11, B12, B13, B21, B22, B23, B31, B32, B33 = compute_shape_operator_matrix(grid_normal_x, grid_normal_y,\
                                 grid_normal_z, intersection_normal, intersection_pos, voxel_min_point,\
                                 voxel_size, x, y, z, width, height, grid_res_x, grid_res_y, grid_res_z, mask)

    # Compute transformation matrix for apperent ridges
    # transform shape operator matrix from object space to image space
    P11, P12, P21, P22, P31, P32 = compute_transformation_matrix(camera, ray_direction, intersection_pos, intersection_normal, width, height)

    # Compute shape operator matrix for intersection points in image space Q = BP
    Q11 = B11 * P11 + B12 * P21 + B13 * P31
    Q12 = B11 * P12 + B12 * P22 + B13 * P32
    Q21 = B21 * P11 + B22 * P21 + B23 * P31
    Q22 = B21 * P12 + B22 * P22 + B23 * P32
    Q31 = B31 * P11 + B32 * P21 + B33 * P31
    Q32 = B31 * P12 + B32 * P22 + B33 * P32

    # print(Q11.shape)
    a = 8
    # a = 160
    # b = 110
    b = 14
    x = a
    y = b
    # print(P11[a,b])
    # print(P12[a,b])
    # print(P21[a,b])
    # print(P22[a,b])
    # print(P31[a,b])
    # print(P32[a,b])

    # Q128 = Tensor([[P11[a,b], P12[a,b]], 
    #               [P21[a,b], P22[a,b]], 
    #               [P31[a,b], P32[a,b]], 
    #   ])
    # print(Q128)
    # exit()
    # Q11 = P11
    # Q12 = P12
    # Q21 = P21
    # Q22 = P22
    # Q31 = P31
    # Q32 = P32
    
    # Compute Q^T*Q for calculating singular values and vectors 
    # http://www.iust.ac.ir/files/mech/madoliat_bcc09/pdf/SVD.pdf
    QtQ11 = Q11 * Q11 + Q21 * Q21 + Q31 * Q31
    QtQ12 = Q11 * Q12 + Q21 * Q22 + Q31 * Q32
    QtQ21 = Q12 * Q11 + Q22 * Q21 + Q32 * Q31
    QtQ22 = Q12 * Q12 + Q22 * Q22 + Q32 * Q32
 
    # calculate max singular vectors
    a = QtQ11
    b = QtQ12
    c = QtQ21
    d = QtQ22
    S1 = torch.pow(a, 2) + torch.pow(b, 2) + torch.pow(c, 2) + torch.pow(d, 2)
    S2 = torch.sqrt(torch.pow(torch.pow(a, 2) + torch.pow(b, 2) - torch.pow(c, 2) - torch.pow(d, 2), 2)\
         + 4 * torch.pow(a * c + b * d, 2))
    theta = 0.5 * torch.atan2(2 * a * c + 2 * b * d + 1e-8, torch.pow(a, 2) + torch.pow(b, 2) - torch.pow(c, 2) - torch.pow(d, 2))
    phi = 0.5 * torch.atan2(2 * a * b + 2 * c * d + 1e-8, torch.pow(a, 2) - torch.pow(b, 2) + torch.pow(c, 2) - torch.pow(d, 2))
    s11 = (a * torch.cos(theta) + c * torch.sin(theta)) * torch.cos(phi) + (b * torch.cos(theta) + d * torch.sin(theta)) * torch.sin(phi)
    s22 = (a * torch.sin(theta) + c * torch.cos(theta)) * torch.sin(phi) + (-b * torch.sin(theta) + d * torch.cos(theta)) * torch.cos(phi)
    singularvector1 = s11 / (torch.abs(s11) + 1e-8) * torch.cos(phi)
    singularvector2 = s11 / (torch.abs(s11) + 1e-8) * torch.sin(phi) ####???

    # print(singularvector1[x, y])
    # print(singularvector2[x, y])
    # exit()
    singularvector = torch.cat((singularvector1.unsqueeze(2), singularvector2.unsqueeze(2)), 2)
    # print(singularvector.shape)
    singularvector_length = torch.unsqueeze(torch.norm(singularvector, p=2, dim=2), 2).repeat(1, 1, 2)
    # print(torch.sum(torch.norm(singularvector, p=2, dim=2)==0))
    # print(singularvector_length.shape)
    # singularvector /= singularvector_length
    # singularvector1 /= torch.norm(singularvector, p=2, dim=2)
    # singularvector2 /= torch.norm(singularvector, p=2, dim=2)
    # exit()

    # return singularvector1

    # Check the angle between singular vector v (directional gradient) and Bv


    # print(singularvector[:,:,0].shape)
    # exit()

    # Say the principal direction (found using SVD of Q^T Q) is t (singularvector).
    # To determine whether the curvature is positive or negative, we first take t,
    # project it back onto the tangent plane, and evaluate the shape operator,
    # which is BPt. 
    Pv1 = P11 * (singularvector1 + 1e-8) + P12 * (singularvector2 + 1e-8)
    Pv2 = P21 * (singularvector1 + 1e-8) + P22 * (singularvector2 + 1e-8)
    Pv3 = P31 * (singularvector1 + 1e-8) + P32 * (singularvector2 + 1e-8)
    Qv1 = Q11 * (singularvector1 + 1e-8) + Q12 * (singularvector2 + 1e-8)
    Qv2 = Q21 * (singularvector1 + 1e-8) + Q22 * (singularvector2 + 1e-8)
    Qv3 = Q31 * (singularvector1 + 1e-8) + Q32 * (singularvector2 + 1e-8)

    # If the dot product of Pt and BPt is positive, the curvature
    # is positive, otherwise negative. 
    curvature_sign = ((Pv1 * Qv1 + Pv2 * Qv2 + Pv3 * Qv3 > 0).type(Tensor) \
             - (Pv1 * Qv1 + Pv2 * Qv2 + Pv3 * Qv3 < 0).type(Tensor)) * mask
    curvature_sign = (torch.sigmoid((Pv1 * Qv1 + Pv2 * Qv2 + Pv3 * Qv3) * 10) - 0.5) * mask * 2


    # torchvision.utils.save_image(curvature_sign, "../chair/sign_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # # torchvision.utils.save_image(-singularvector[:,:,1], "../chair/222-_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # exit()

    # a = Q11
    # b = Q12
    # c = Q21
    # d = Q22
    # eigenvalue = torch.clamp((a + d + torch.sqrt(1e-8 + torch.clamp(torch.pow(a + d, 2) - 4 * (a * d - b * c), min=0))) / 2., min=0)
    # print(torch.min(eigenvalue))
    # exit()
    # # print(torch.sum())
    # eigenvector1 = torch.ones(width, height).cuda()
    # eigenvector2 = (eigenvalue - a) / b
    # eigenvector2 = torch.ones(width, height).cuda()
    # eigenvector1 = (eigenvalue - d) / (c + 0.0001)
    # # eigenvector1[c == 0] = 0

    # print(torch.norm(singularvector, dim=2).shape)
    # print(torch.max(torch.norm(singularvector, dim=2)))

    # P_col1 = torch.norm(torch.cat((P11.unsqueeze(2), P21.unsqueeze(2), P31.unsqueeze(2)), 2), dim=2) * 0.2
    # P_col2 = torch.norm(torch.cat((P12.unsqueeze(2), P22.unsqueeze(2), P32.unsqueeze(2)), 2), dim=2) * 0.2


    # torchvision.utils.save_image(P_col1 * mask, "../chair/Pcollength1_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(P_col2 * mask, "../chair/Pcollength2_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(torch.norm(singularvector, dim=2) * mask, "../chair/eigenlength_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(P_col2 * mask, "../chair/Pcollength2_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)


    # print(singularvector[x,y])
    # exit()


    # eigenvector = torch.cat((eigenvector1.unsqueeze_(2), eigenvector2.unsqueeze_(2)), 2) 

    # eigenvector = eigenvector / (torch.unsqueeze(torch.norm(eigenvector, p=2, dim=2), 2).repeat(1, 1, 2) + 0.0001)


    # calculate the corresponding singular value ##########3
    Qv1 = Q11 * singularvector1 + Q12 * singularvector2
    Qv2 = Q21 * singularvector1 + Q22 * singularvector2
    Qv3 = Q31 * singularvector1 + Q32 * singularvector2
    singularvalue = torch.sqrt(torch.pow(Qv1, 2) + torch.pow(Qv2, 2) + torch.pow(Qv3, 2) + 1e-8)
    # singularvalue[torch.isnan(singularvalue)] = 0

    # print(torch.min(singularvalue))
    # exit()




    # print("128", singularvalue[128, 128])
    # print("150", singularvalue[128, 150])
    # print("180", singularvalue[128, 180])


    # print("150", singularvalue[150, 150])
    # exit()
    # singularvalue[128, 180] = 1

    # singularvalue =  torch.sqrt(torch.sqrt((S1 + S2) / 2))
    # print("128", singularvalue[128, 128])
    # print("150", singularvalue[128, 150])
    # print("180", singularvalue[128, 180])


    # singularvalue = torch.clamp((a + d + torch.sqrt(1e-8 + torch.clamp(torch.pow(a + d, 2) - 4 * (a * d - b * c), min=0))) / 2., min=0)
    # # print(torch.sum())
    # # eigenvector1 = torch.ones(width, height).cuda()
    # # eigenvector2 = (eigenvalue - a) / b
    # singularvector2 = torch.ones(width, height).cuda()
    # singularvector1 = (singularvalue - d) / (c + 0.0001)
    # singularvector = torch.cat((singularvector1.unsqueeze(2), singularvector2.unsqueeze(2)), 2)

    # return singularvalue * mask 
    # print(torch.max(singularvalue), torch.min(singularvalue))

    # exit()

    # return torch.clamp(singularvalue, min=0, max = 1)

    # eigenvector = torch.cat((eigenvector1.unsqueeze_(2), eigenvector2.unsqueeze_(2)), 2) 

    # eigenvector = eigenvector / (torch.unsqueeze(torch.norm(eigenvector, p=2, dim=2), 2).repeat(1, 1, 2) + 0.0001)

    # get a filter such that the final values are close to 0 if curvature is close to 0
    # eigenvalue[mask == 0] = 0
    # singularvalue_filter = (torch.sigmoid(singularvalue) - 0.65) * 20 #/ (torch.max(torch.sigmoid(singularvalue)) - 0.5)
    # return singularvalue_filter * mask
    # exit()

    # singularvalue = torch.abs(singularvalue) * mask
    # torchvision.utils.save_image(eigenvalue, "../chair/iii_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(eigenvector[:,:,0]*mask, "../chair/eigen1_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(eigenvector[:,:,1]*mask, "../chair/eigen2_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # x = np.linspace(1,width,height)
    # y = np.linspace(1,width,height)
    # x, y = np.meshgrid(x,y)
    # u = (eigenvector[:,:,0] * mask ).detach().type(torch.FloatTensor).numpy()
    # v = (eigenvector[:,:,1] * mask ).detach().type(torch.FloatTensor).numpy()


    # singularvalue[x:x+10,y] = 0
    singularvalue = torch.abs(torch.sqrt(singularvalue)) * mask 

    # singularvalue[35:45, 25:35] = 0
    # torchvision.utils.save_image(singularvalue, "../chair/iii_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(eigenvector[:,:,0]*mask, "../chair/eigen1_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(eigenvector[:,:,1]*mask, "../chair/eigen2_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # x = np.linspace(1,width,width)
    # y = np.linspace(1,height,height)
    # # # print(x)
    # # # exit()
    # x, y = np.meshgrid(x,y)
    # u = (singularvector[:,:,0] * mask).detach().type(torch.FloatTensor).numpy()
    # v = (singularvector[:,:,1] * mask).detach().type(torch.FloatTensor).numpy()

    # print(singularvector[40,30])

    # # print(torch.max(singularvector2 * mask))
    # # exit()
    # # print(x)

    # # x = np.array((5))
    # # y= np.array((5))
    # # u = np.array((2))
    # # v = np.array((-2))
    # # print(x, y)

    # # widths = np.linspace(0, 1, x.size)
    # plt.clf()
    # plt.quiver(y, x, u, v, scale=100)
    # # plt.show()
    # plt.savefig("../chair/plot_" + str(cam) + ".png", bbox_inches='tight')
    # exit()
    # exit()

    # print(torch.min(singularvalue), torch.max(singularvalue))
    # exit()


    # # # calculate max eigenvalues and eigenvectors
    # a = Q11
    # b = Q12
    # c = Q21
    # d = Q22
    # eigenvalue = (a + d + torch.sqrt(torch.clamp(torch.pow(a + d, 2) - 4 * (a * d - b * c) + 1e-8, min=1e-8))) / 2.
    # # print(torch.min(1e-8 + (torch.pow(a + d, 2) - 4 * (a * d - b * c))))
    # # print(torch.min(torch.sqrt(1e-8 + (torch.pow(a + d, 2) - 4 * (a * d - b * c)))))

    # # eigenvalue[torch.isnan(eigenvalue)] = 0
    # # print(torch.sum())
    # # eigenvector1 = torch.ones(width, height).cuda()
    # # eigenvector2 = (eigen value - a) / b
    # eigenvector2 = torch.ones(width, height).cuda()
    # eigenvector1 = (eigenvalue - d) / (c + 0.0001)
    # # eigenvector1[c == 0] = 0


    # eigenvector = torch.cat((eigenvector1.unsqueeze_(2), eigenvector2.unsqueeze_(2)), 2) 

    # eigenvector = eigenvector / (torch.unsqueeze(torch.norm(eigenvector, p=2, dim=2), 2).repeat(1, 1, 2) + 0.0001)
    # print(torch.min(eigenvalue), torch.max(eigenvalue))
    # exit()

    # print(torch.min(eigenvalue), torch.max(eigenvalue))


    cos_between_ray_dir_and_neg_camera = torch.sum(ray_direction * (-camera.repeat(width, height, 1)), 2) / torch.norm(ray_direction, p=2, dim=2) / torch.norm((-camera.repeat(width, height, 1)), p=2, dim=2)
    depth_image = torch.norm(camera.repeat(width, height, 1) - intersection_pos, p=2, dim=2) * cos_between_ray_dir_and_neg_camera 
    depth_image = (torch.max(depth_image) - depth_image) / 3
    # return depth_image * mask

    # # ##################
    # return (torch.sigmoid(singularvalue) - 0.5) * 5 * mask * curvature_sign, depth_image * mask


    # modified singularvalue filter
    singularvalue_filter = (torch.sigmoid(singularvalue) - 0.64) * 5 #* mask
    # return singularvalue_filter
    # singularvalue_filter[110:130, 110:160] = 1
    # singularvalue_filter[112:128, 112:158] = 0

    # loss = torch.sum(singularvalue_filter)
    # loss.backward()
    # print("yueyueyue")
    # singularvalue_filter[100:110, 152:158] = 1
    # torchvision.utils.save_image(singularvalue_filter, "../chair/fff_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # exit()

    # return singularvector[:,:,1] *30 * mask * singularvalue_filter
    # * singularvalue_filter * 10

    # eigenvector[mask == 0] = 0
    # return eigenvector[:,:,0]
    
  
    # calculate curvature gradient which is the finite differencing of max singularvalues(curvatures)
    # curvature along the x-axis
    n_x = singularvalue.shape[0] - 1
    CX_1 = torch.cat((singularvalue[1:,:], (2 * singularvalue[n_x,:] - 3 * singularvalue[n_x-1,:] + singularvalue[n_x-2,:]).unsqueeze_(0)), 0)
    CX_2 = torch.cat(((-3 * singularvalue[1,:] + 2 * singularvalue[0,:] + singularvalue[2,:]).unsqueeze(0), singularvalue[:n_x,:]), 0)
    curvature_gradient1 = (CX_1 - CX_2) / 2

    # return curvature_gradient1 * 5

    # print(CX_1[35:45,30], CX_2[35:45,30], curvature_gradient1[40,30])

    # print(torch.max(curvature_gradient1))

    # curvature along the y-axis
    n_y = singularvalue.shape[1] - 1
    CY_1 = torch.cat((singularvalue[:,1:], (2 * singularvalue[:,n_y] - 3 * singularvalue[:,n_y-1] + singularvalue[:,n_y-2]).unsqueeze_(1)), 1)
    CY_2 = torch.cat(((-3 * singularvalue[:,1] + 2 * singularvalue[:,0] + singularvalue[:,2]).unsqueeze(1), singularvalue[:,:n_y]), 1)
    curvature_gradient2 = (CY_1 - CY_2) / 2

    # torchvision.utils.save_image(curvature_gradient1, "../chair/111_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(curvature_gradient2, "../chair/222_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # exit()



    # print(CY_1[40,25:35], CY_2[40,25:35], curvature_gradient2[40,30])

    # print(torch.max(curvature_gradient2))

    # get the curvature which is the directional gradient
    # set the gradient to be 0 if the gradient is too small
    curvature_gradient = torch.cat((curvature_gradient1.unsqueeze(2), curvature_gradient2.unsqueeze(2)), 2) 

    # print(curvature_gradient.shape, singularvector.shape)
    # exit()


    
    # curvature_length = torch.unsqueeze(torch.norm(curvature_gradient, p=2, dim=2), 2).repeat(1, 1, 2)
    # curvature_gradient = curvature_gradient / curvature_length
    # curvature_gradient[torch.isnan(curvature_length)] = 0
    # curvature_length[torch.isnan(curvature_length)] = 0
    # curvature_gradient[curvature_length < 0.05] = 0


    # # calculate the directional derivatives along principle direction 
    # dir_derivatives = torch.sum(curvature_gradient * singularvector, dim=2)# * 1000# * singularvector



    


    # print(curvature_gradient1.shape)
    # print((curvature_gradient1 / torch.norm(curvature_gradient, p=2, dim=2)).shape)
    # curvature_gradient1 /= torch.norm(curvature_gradient, p=2, dim=2)
    # curvature_gradient1[torch.isnan(curvature_gradient1)] = 1e-8

    # return curvature_gradient1 * 5
    # torchvision.utils.save_image((curvature_gradient[:,:,0]), "../chair/pp111_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image((curvature_gradient[:,:,1]), "../chair/pp222_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image((-curvature_gradient[:,:,0]), "../chair/nn111_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image((-curvature_gradient[:,:,1]), "../chair/nn222_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

    # return torch.sum(curvature_gradient, 2) 
    # print(torch.min(curvature_gradient))
    
    # exit()
    # print(torch.min(curvature_gradient))
    # exit()

    # x = np.linspace(1,width,width)
    # y = np.linspace(1,height,height)
    # # print(x)
    # # exit()
    # x, y = np.meshgrid(x,y)
    # u = (curvature_gradient[:,:,0] * mask).detach().type(torch.FloatTensor).numpy()
    # v = (curvature_gradient[:,:,1] * mask).detach().type(torch.FloatTensor).numpy()
    
    
    # # exit()
    # plt.clf()
    # plt.quiver(y, x, u, v, scale=100)
    # # plt.show()
    # plt.savefig("../chair/curvature_gradient_" + str(cam) + ".png", dpi=300)#, bbox_inches='tight')

    # return dir_derivatives, dir_derivatives, dir_derivatives


    # exit()
    # return singularvalue_filter
    # return curvature[:,:,1] *  singularvalue_filter * 10 * mask
    # return (torch.abs(torch.sum(curvature * singularvector, dim=2)) < 0.2).type(Tensor)

    # flip the singularvector if dot_product(gradient, singularvector) < 0
    flip = (torch.sum(curvature_gradient * singularvector, dim=2) < 0).type(Tensor).unsqueeze_(2).repeat(1, 1, 2)
    singularvector += (-2) * flip * singularvector

    # return eigenvector[:,:,1]
    # exit()
    # return singularvector[:,:,1] *  singularvalue_filter * 10 * mask

    # If two neighboring eigenvectors have > 180 degree angle, then mark it as 1
    image = torch.zeros(width, height).cuda()
    # image[1:,:] = torch.sigmoid(torch.sum(singularvector[1:,:] * singularvector[:width-1,:], dim=2)) - 0.5 + image[1:,:]
    # image[:width-1,:] = torch.sigmoid(-torch.sum(singularvector[1:,:] * singularvector[:width-1,:], dim=2)) - 0.5  + image[:width-1,:]
    # image[:,1:] = torch.sigmoid(-torch.sum(singularvector[:,1:] * singularvector[:,:height-1], dim=2)) - 0.5 + image[:,1:]
    # image[:,:height-1] = torch.sigmoid(-torch.sum(singularvector[:,1:] * singularvector[:,:height-1], dim=2)) - 0.5 + image[:,:height-1]
                    
    image[:width-1,:] = torch.clamp((torch.sigmoid(torch.clamp(-torch.sum(singularvector[1:,:] * singularvector[:width-1,:], dim=2), \
                                    min=0)) - 0.5) + image[:width-1,:], min=0, max=1)
    image[1:,:] = torch.clamp((torch.sigmoid(torch.clamp(-torch.sum(singularvector[1:,:] * singularvector[:width-1,:]), \
                                    min=0)) - 0.5) + image[1:,:], min=0, max=1)
    image[:,:height-1] = torch.clamp((torch.sigmoid(torch.clamp(-torch.sum(singularvector[:,1:] * singularvector[:,:height-1], dim=2), \
                                    min=0)) - 0.5) + image[:,:height-1], min=0, max=1)
    image[:,1:] = torch.clamp((torch.sigmoid(torch.clamp(-torch.sum(singularvector[:,1:] * singularvector[:,:height-1], dim=2), \
                                    min=0)) - 0.5) + image[:,1:], min=0, max=1) 

    #################### temporary ########################

    # # set all the normals of background to be the the same as ray directions
    # intersection_normal[torch.unsqueeze(mask, 2).repeat(1, 1, 3) == 0] = ray_direction[torch.unsqueeze(mask, 2).repeat(1, 1, 3) == 0]

    # # calculate the gradient of normal vectors along the x-axis using finite differencing
    # n_x = intersection_normal.shape[0] - 1
    # X_1 = torch.cat((intersection_normal[1:,:], (2 * intersection_normal[n_x,:] - 3 * intersection_normal[n_x-1,:] + intersection_normal[n_x-2,:]).unsqueeze_(0)), 0)
    # X_2 = torch.cat(((-3 * intersection_normal[1,:] + 2 * intersection_normal[0,:] + intersection_normal[2,:]).unsqueeze_(0), intersection_normal[:n_x,:]), 0)
    # diff_normal_x = (X_1 - X_2) / 2
    # gradient_normal_x = torch.norm(diff_normal_x, dim=2)

    # # calculate the gradient of normal vectors along the y-axis using finite differencing
    # n_y = intersection_normal.shape[1] - 1
    # Y_1 = torch.cat((intersection_normal[:,1:], (2 * intersection_normal[:,n_y] - 3 * intersection_normal[:,n_y-1] + intersection_normal[:,n_y-2]).unsqueeze_(1)), 1)
    # Y_2 = torch.cat(((-3 * intersection_normal[:,1] + 2 * intersection_normal[:,0] + intersection_normal[:,2]).unsqueeze_(1), intersection_normal[:,:n_y]), 1)
    # diff_normal_y = (Y_1 - Y_2) / 2
    # gradient_normal_y = torch.norm(diff_normal_y, dim=2)

    # # concatenate gradient of normals to have 2 elements for each pixel
    # gradient_normal = torch.cat((gradient_normal_x.unsqueeze(2), gradient_normal_y.unsqueeze(2)), 2)

    # # calculate the first column of hessian matrix of normal vectors along the x-axis using finite differencing
    # n_x = gradient_normal.shape[0] - 1
    # HX_1 = torch.cat((gradient_normal[1:,:], (2 * gradient_normal[n_x,:] - 3 * gradient_normal[n_x-1,:] + gradient_normal[n_x-2,:]).unsqueeze_(0)), 0)
    # HX_2 = torch.cat(((-3 * gradient_normal[1,:] + 2 * gradient_normal[0,:] + gradient_normal[2,:]).unsqueeze_(0), gradient_normal[:n_x,:]), 0)
    # hessian_1 = (HX_1 - HX_2) / 2

    # # calculate the second column of hessian matrix of normal vectors along the y-axis using finite differencing
    # n_y = gradient_normal.shape[1] - 1
    # HY_1 = torch.cat((gradient_normal[:,1:], (2 * gradient_normal[:,n_y] - 3 * gradient_normal[:,n_y-1] + gradient_normal[:,n_y-2]).unsqueeze_(1)), 1)
    # HY_2 = torch.cat(((-3 * gradient_normal[:,1] + 2 * gradient_normal[:,0] + gradient_normal[:,2]).unsqueeze_(1), gradient_normal[:,:n_y]), 1)
    # hessian_2 = (HY_1 - HY_2) / 2

    # # concatenate hessian of normals to have 4 elements for each pixel
    # hessian = torch.cat((hessian_1, hessian_2), 2)

    # # calculate max eigenvalues and eigenvectors
    # a = hessian[:,:,0] * 100
    # b = hessian[:,:,1] * 100
    # c = hessian[:,:,2] * 100
    # d = hessian[:,:,3] * 100
    # torchvision.utils.save_image(a, "../chair/a_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(b, "../chair/b_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(c, "../chair/c_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # torchvision.utils.save_image(d, "../chair/d_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    # # exit()

    ######################

    # print(image[100:110, 152:158])

    # print('-------------')
    # print(singularvalue_filter[100:110, 152:158])

    # print('-------------')
    # print(mask[100:110, 152:158])

    # print("++++")
    # print((image * singularvalue_filter * 20)[100:110, 152:158])



    # torchvision.utils.save_image(image *10 * mask, "../chair/extrema_grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

    # print("===")
    # exit()ss
    # image[1:,:] += (torch.sum(singularvector[1:,:] * singularvector[:width-1,:], dim=2) < 0).type(Tensor)
    # image[:,1:] += (torch.sum(singularvector[:,1:] * singularvector[:,:height-1], dim=2) < 0).type(Tensor)
    # exit()
    # print(torch.max(singularvalue_filter))
    # # exit()
    # print("000000000")

    
    # calculate the directional derivatives along principle direction 
    dir_derivatives = torch.sum(curvature_gradient * singularvector, dim=2) # * singularvector
    
    # return image
    image = image * singularvalue_filter * 10
    image = torch.clamp(image, min=0, max=1)
    # exit()
    # image[mask == 0] = 0
    # return image
    # * curvature_sign
    return depth_image * mask, (torch.sigmoid(singularvalue) - curvature_threshold) * 10 * mask * curvature_sign, dir_derivatives * mask, image * mask


# for update sketch file after confirmation
def update_image(newpose, grid_new, objname):  #objname is sketch I Guess here
    # define the folder name for results
    dir_name = "results/"
    os.makedirs("./" + dir_name, exist_ok=True)
     # check cuda
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # image size
    width = 256
    height = 256

     # camere poses
    camm = Tensor(newpose)   #list to tensor
    #print(camm)
    # bounding box
    bounding_box_min_x = bounding_box_min_y = bounding_box_min_z = -2.
    bounding_box_max_x = bounding_box_max_y = bounding_box_max_z = 2.


if __name__ == "__main__":

    # define the folder name for results
    dir_name = "comb/"
    os.makedirs("./" + dir_name, exist_ok=True)

    # get input
    image_scale = sys.argv[1]
    sdf_scale = sys.argv[2]
    lp_scale = sys.argv[3]
    iter_threshold = sys.argv[4]
    curvature_threshold = sys.argv[5]

    # Speed up
    torch.backends.cudnn.benchmark = True

    cuda = True if torch.cuda.is_available() else False
    print(cuda)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    width = 256
    height = 256

    camera_list = [Tensor([0,0,5]), # 0
                   Tensor([0.1,5,0]), 
                   Tensor([5,0,0]), 
                   Tensor([0,0,-5]), 
                   Tensor([0.1,-5,0]), 
                   Tensor([-5,0,0]), # 5

                   Tensor([5/math.sqrt(2),0,5/math.sqrt(2)]),
                   Tensor([5/math.sqrt(2),5/math.sqrt(2),0]),
                   Tensor([0,5/math.sqrt(2),5/math.sqrt(2)]),

                   Tensor([-5/math.sqrt(2),0,-5/math.sqrt(2)]),
                   Tensor([-5/math.sqrt(2),-5/math.sqrt(2),0]), #10
                   Tensor([0,-5/math.sqrt(2),-5/math.sqrt(2)]),

                   Tensor([-5/math.sqrt(2),0,5/math.sqrt(2)]),
                   Tensor([-5/math.sqrt(2),5/math.sqrt(2),0]),
                   Tensor([0,-5/math.sqrt(2),5/math.sqrt(2)]),

                   Tensor([5/math.sqrt(2),0,-5/math.sqrt(2)]),
                   Tensor([5/math.sqrt(2),-5/math.sqrt(2),0]),
                   Tensor([0,5/math.sqrt(2),-5/math.sqrt(2)]),

                   Tensor([5/math.sqrt(3),5/math.sqrt(3),5/math.sqrt(3)]),
                   Tensor([5/math.sqrt(3),5/math.sqrt(3),-5/math.sqrt(3)]),
                   Tensor([5/math.sqrt(3),-5/math.sqrt(3),5/math.sqrt(3)]),
                   Tensor([-5/math.sqrt(3),5/math.sqrt(3),5/math.sqrt(3)]),
                   Tensor([-5/math.sqrt(3),-5/math.sqrt(3),5/math.sqrt(3)]),
                   Tensor([-5/math.sqrt(3),5/math.sqrt(3),-5/math.sqrt(3)]),
                   Tensor([5/math.sqrt(3),-5/math.sqrt(3),-5/math.sqrt(3)]),
                   Tensor([-5/math.sqrt(3),-5/math.sqrt(3),-5/math.sqrt(3)])]

    camera_list_simple = [camera_list[0]]#, camera_list[1]] #, camera_list[4], camera_list[5]]
    camera_list_simple = [camera_list[0], camera_list[2], camera_list[5]]

    # bounding box  +- 2
    bounding_box_min_x = -2.
    bounding_box_min_y = -2.
    bounding_box_min_z = -2.
    bounding_box_max_x = 2.
    bounding_box_max_y = 2.
    bounding_box_max_z = 2.
    

    # size of the image
    width = 256
    height = 256

    loss = 500

    image_loss_list = []
    sdf_loss_list = []
    e = camera_list[0]

    # Find proper grid resolution
    pixel_distance = torch.tan(Tensor([math.pi/6])) * 2 / height

    # Compute largest distance between the grid and the camera
    largest_distance_camera_grid = torch.sqrt(torch.pow(max(torch.abs(e[0] - bounding_box_max_x), torch.abs(e[0] - bounding_box_min_x)), 2)
                                            + torch.pow(max(torch.abs(e[1] - bounding_box_max_y), torch.abs(e[1] - bounding_box_min_y)), 2)
                                            + torch.pow(max(torch.abs(e[2] - bounding_box_max_z), torch.abs(e[2] - bounding_box_min_z)), 2))
    grid_res_x = 8
    grid_res_y = 8
    grid_res_z = 8

    # define the resolutions of the multi-resolution part
    voxel_res_list = []
    # for i in range(16, 65, 10):
    #     voxel_res_list.append(i)
    voxel_res_list = [64]  
    grid_res_x = grid_res_y = grid_res_z = voxel_res_list.pop(0)

    grid_final_res = 64
    voxel_size_final = Tensor([4. / (grid_final_res-1)])

    # Construct the sdf grid
    # grid_initial = read_sdf("./face.sdf", grid_res_x, bounding_box_min_x, bounding_box_max_x, 4. / (grid_res_x-1))

    grid_initial = grid_construction_sphere_small(grid_res_x, bounding_box_min_x, bounding_box_max_x) ####
    # grid_initial_inside = grid_construction_rounded_cylinder(grid_res_x, bounding_box_min_x, bounding_box_max_x, 0, 0.8, 1)
    # grid_initial_inside_result = torch.cat((grid_initial_inside[:,60:,:], grid_initial_inside[:,:60,:]), dim=1)
    
    # grid_initial = torch.max(grid_initial, -grid_initial_inside_result)


    # create target object
    # grid_target = grid_construction_rounded_cylinder(grid_final_res, bounding_box_min_x, bounding_box_max_x, 0, 1, 1)

    # grid_target_handle = grid_construction_rounded_cylinder(grid_final_res, bounding_box_min_x, bounding_box_max_x, 0, 0.2, 0.6)
    # grid_target_handle_result = torch.cat((grid_target_handle[:,:,22:], grid_target_handle[:,:,:22]), dim=2)

    # grid_target_handle_upper = grid_construction_cylinder_horizontal(grid_final_res, bounding_box_min_x, bounding_box_max_x, 0, 0.2, 0.3)
    # grid_target_handle_upper_result = torch.cat((grid_target_handle_upper[:,:,17:], grid_target_handle_upper[:,:,:17]), dim=2)
    # grid_target_handle_upper_result = torch.cat((grid_target_handle_upper_result[:,7:,:], grid_target_handle_upper_result[:,:7,:]), dim=1)

    # grid_target_handle_lower = grid_construction_cylinder_horizontal(grid_final_res, bounding_box_min_x, bounding_box_max_x, 0, 0.2, 0.3)
    # grid_target_handle_lower_result = torch.cat((grid_target_handle_lower[:,:,17:], grid_target_handle_upper[:,:,:17]), dim=2)
    # grid_target_handle_lower_result = torch.cat((grid_target_handle_lower_result[:,57:,:], grid_target_handle_lower_result[:,:57,:]), dim=1)
    
    # grid_target_add_model = torch.min(torch.min(grid_target_handle_result, grid_target_handle_upper_result), grid_target_handle_lower_result)
    # grid_target = torch.min(grid_target, grid_target_add_model)

    # create initial object
    # grid_initial = grid_construction_rounded_cylinder(grid_res_x, bounding_box_min_x, bounding_box_max_x, 0, 1, 1)

    # for i in range(len(camera_list_simple)):

    #     # grid = grid_construction_rounded_cylinder(grid_res_x, bounding_box_min_x, bounding_box_max_x, 0, 1, 1)
    #     grid = grid_target_handle_lower
    #     # generate image mask from the model
    #     mask = generate_mask(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    #                     bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    #                     4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width,\
    #                     height, grid_target_add_model, camera_list_simple[i], 1, camera_list)
    #     torchvision.utils.save_image(mask, "./" + dir_name + "mask" + str(i) + ".png")
        
    #     # compute visual hull
    #     visual_hull = compute_visual_hull(mask, camera_list_simple[i], bounding_box_min_x, \
    #                                     bounding_box_max_x, grid_res_x, width, height)

    #     # merge visual hull results
    #     if i == 0:
    #         visual_hull_initialization = visual_hull
    #     else:
    #         visual_hull_initialization = torch.max(visual_hull, visual_hull_initialization)

        
    # # exit()

    # # merge initial object and visual_hull_initialization
    # grid_initial = torch.min(grid_initial, visual_hull_initialization)
# 
    # for cam in range(len(camera_list_simple)):
    #     image = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    #                         bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    #                         4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width,\
    #                         height, grid_initial, camera_list_simple[cam], 1, camera_list)
    #     torchvision.utils.save_image(image, "./" + dir_name + "mask" + str(i) + "_" + str(cam) + ".png")




    # output images
    image_target = []
    for cam in range(len(camera_list_simple)):             
        face_grid = read_sdf("./face.sdf", grid_res_x, bounding_box_min_x, bounding_box_max_x, 4. / (grid_res_x-1))

        image = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
        4. / (grid_final_res-1), grid_final_res, grid_final_res, grid_final_res, width, height, grid_initial, camera_list_simple[cam]+ torch.randn_like(camera_list[0]) * 0.015, 1, camera_list)
        face = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
        4. / (grid_final_res-1), grid_final_res, grid_final_res, grid_final_res, width, height, face_grid, camera_list_simple[cam]+ torch.randn_like(camera_list[0]) * 0.015, 1, camera_list)

        if cam == 0:
            image[80:200, 50:256-50] = face[80:200, 50:256-50]
        elif cam == 1:
            image[100:200, 50:120] = face[100:200, 50:120]
            image[50:200, 120:220] = 0
        else:
            image[100:200, 256-120:256-50] = face[100:200, 256-120:256-50]
            image[50:200, 256-220:256-120] = 0
        # torchvision.utils.save_image(image, "./" + dir_name + "grid_res_" + str(grid_res_x) + "_target_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

        image_target.append(image)
    # exit()

    visual_hull_list = []
    for i in range(len(camera_list_simple)):

        # generate image mask from the model
        mask = (image_target[i] != 0).type(Tensor)
        torchvision.utils.save_image(mask, "./" + dir_name + "mask" + str(i) + ".png")
        
        # compute visual hull
        visual_hull = compute_visual_hull(mask, camera_list_simple[i], bounding_box_min_x, \
                                        bounding_box_max_x, grid_res_x, width, height)
        visual_hull_list.append(visual_hull)

    # merge initial object and visual_hull_initialization
    grid_initial = torch.min(torch.max(grid_initial, visual_hull_list[0]), \
                        torch.max(visual_hull_list[0], torch.max(visual_hull_list[1], visual_hull_list[2])))

    # set parameters
    sdf_diff_list = []
    time_list = []
    image_loss = [1000] * len(camera_list)
    sdf_loss = [1000] * len(camera_list)
    iterations = 0
    scale = 1 
    start_time = time.time()
    learning_rate = 0.0001
    tolerance = 8 / 10

    # image size
    width = 256
    height = 256

    start_time = time.time()
    while (grid_res_x <= 64):
        tolerance *= 1.05
        image_target = []
        grid_initial.requires_grad = True

        grid_target = grid_construction_sphere_small(grid_res_x, bounding_box_min_x, bounding_box_max_x)
        face_grid = read_sdf("./face.sdf", grid_res_x, bounding_box_min_x, bounding_box_max_x, 4. / (grid_res_x-1))
        
        optimizer = torch.optim.Adam([grid_initial], lr = learning_rate, eps=1e-2)

        # voxel_size = Tensor([4. / (grid_final_res-1)])

        # output images
        curvature_target_list = []
        for cam in range(len(camera_list_simple)):             

            image_initial = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list_simple[cam], 1, camera_list)
            image = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
            4. / (grid_final_res-1), grid_final_res, grid_final_res, grid_final_res, width, height, grid_target, camera_list_simple[cam]+ torch.randn_like(camera_list[0]) * 0.015, 1, camera_list)
            face = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
            4. / (grid_final_res-1), grid_final_res, grid_final_res, grid_final_res, width, height, face_grid, camera_list_simple[cam]+ torch.randn_like(camera_list[0]) * 0.015, 1, camera_list)
            # if cam == 0:
            #     for i in range(128 - 40, 128 + 20):
            #         for j in range(128 - 40, 128 + 40):
            #             if (i - 128) ** 2 + (j - 128) ** 2 - 14 * 14 < -10 \
            #                 or (i - (128-15)) ** 2 + (j - (128-15)) ** 2 - 7 * 7 < -5 \
            #                 or (i - (128-15)) ** 2 + (j - (128+15)) ** 2 - 7 * 7 < -5:
            #                 image[i, j] = 0
            #             elif (i - 128) ** 2 + (j - 128) ** 2 - 14 * 14 < 10 \
            #                 or (i - (128-15)) ** 2 + (j - (128-15)) ** 2 - 7 * 7 < 5 \
            #                 or (i - (128-15)) ** 2 + (j - (128+15)) ** 2 - 7 * 7 < 5:
            #                 image[i, j] = 1

            if cam == 0:
                image[80:200, 50:256-50] = face[80:200, 50:256-50]
            elif cam == 1:
                image[100:200, 50:120] = face[100:200, 50:120]
                image[50:200, 120:220] = 0
            else:
                image[100:200, 256-120:256-50] = face[100:200, 256-120:256-50]
                image[50:200, 256-220:256-120] = 0
            # image_initial[100:200, 50:120] = image[100:200, 50:120] ### 2
            # image_initial[100:200, 256-120:256-50] = image[100:200, 256-120:256-50] ### 5
            # image_initial[80:200, 50:256-50] = image[80:200, 50:256-50] ### 0
            torchvision.utils.save_image(image_initial, "./" + dir_name + "grid_res_" + str(grid_res_x) + "_start_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
            torchvision.utils.save_image(image, "./" + dir_name + "grid_res_" + str(grid_res_x) + "_target_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
            image_target.append(image)
            depth_initial, curvature_initial, dir_derivatives_initial, sketch_initial = generate_image_sketch(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
            4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list_simple[cam], 1, curvature_threshold)
            depth_face, curvature_face, dir_derivatives_face, sketch_face = generate_image_sketch(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
            4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width, height, face_grid, camera_list_simple[cam], 1, curvature_threshold)
            depth_target, curvature_target, dir_derivatives_target, sketch_target = generate_image_sketch(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
            4. / (grid_final_res-1), grid_final_res, grid_final_res, grid_final_res, width, height, grid_target, camera_list_simple[cam], 1, curvature_threshold)
         
            # if cam == 0:
            #     for i in range(128 - 40, 128 + 20):
            #         for j in range(128 - 40, 128 + 40):
            #             if (i - 128) ** 2 + (j - 128) ** 2 - 14 * 14 < -10 \
            #                 or (i - (128-15)) ** 2 + (j - (128-15)) ** 2 - 7 * 7 < -5 \
            #                 or (i - (128-15)) ** 2 + (j - (128+15)) ** 2 - 7 * 7 < -5:
            #                 curvature_target[i, j] = 0
            #             elif (i - 128) ** 2 + (j - 128) ** 2 - 14 * 14 < 10 \
            #                 or (i - (128-15)) ** 2 + (j - (128-15)) ** 2 - 7 * 7 < 5 \
            #                 or (i - (128-15)) ** 2 + (j - (128+15)) ** 2 - 7 * 7 < 5:
            #                 curvature_target[i, j] = 1
            # curvature_target[40:60, 80:145] = 0
            # curvature_target[60, 80:145] = 1
            # curvature_target_list.append(curvature_target)

            if cam == 0:
                curvature_target[80:200, 50:256-50] = curvature_face[80:200, 50:256-50]
                dir_derivatives_target[80:200, 50:256-50] = dir_derivatives_face[80:200, 50:256-50]
            elif cam == 1:
                curvature_target[100:200, 50:120] = curvature_face[100:200, 50:120]
                curvature_target[50:200, 120:220] = 0
            else:
                curvature_target[100:200, 256-120:256-50] = curvature_face[100:200, 256-120:256-50]
                curvature_target[50:200, 256-220:256-120] = 0
            curvature_target_list.append(curvature_target)
            torchvision.utils.save_image(curvature_target, "./" + dir_name + "grid_res_" + str(grid_res_x) + "_curvature_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
            torchvision.utils.save_image(sketch_target, "./" + dir_name + "grid_res_" + str(grid_res_x) + "_sketch_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
        # grid_res_x = grid_res_y = grid_res_z = voxel_res_list.pop(0)
        # exit()
        # visual_hull_list = []
        # for i in range(len(camera_list_simple)):

        #     # grid = grid_construction_rounded_cylinder(grid_res_x, bounding_box_min_x, bounding_box_max_x, 0, 1, 1)
        #     # grid = grid_target_handle_lower
        #     # generate image mask from the model
        #     mask = (image_target[i] != 0).type(Tensor)
        #     # mask = generate_mask(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
        #     #                 bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
        #     #                 4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width,\
        #     #                 height, grid_target_add_model, camera_list_simple[i], 1, camera_list)
        #     torchvision.utils.save_image(mask, "./" + dir_name + "mask" + str(i) + ".png")

            
        #     # compute visual hull
        #     visual_hull = compute_visual_hull(mask, camera_list_simple[i], bounding_box_min_x, \
        #                                     bounding_box_max_x, grid_res_x, width, height)
        #     visual_hull_list.append(visual_hull)

        #     # # merge visual hull results
            # if i == 0:
            #     visual_hull_initialization = visual_hull
            # else:
            #     visual_hull_initialization = torch.max(visual_hull, visual_hull_initialization)

        
        # exit()

        # merge initial object and visual_hull_initialization
        # grid_initial = torch.min(torch.max(grid_initial, visual_hull_list[0]), \
        #                     torch.max(visual_hull_list[0], torch.max(visual_hull_list[1], visual_hull_list[2])))
        # # grid_initial = visual_hull_initialization

        # for cam in range(len(camera_list_simple)):
        #     image = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
        #                         bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
        #                         4. / (grid_res_x-1), grid_res_x, grid_res_y, grid_res_z, width,\
        #                         height, grid_initial, camera_list_simple[cam], 1, camera_list)
        #     torchvision.utils.save_image(image, "./" + dir_name + "mask" + str(i) + "_" + str(cam) + ".png")

        # Save the final SDF result
        # with open("./" + dir_name + str(grid_res_x) + "_face.pt", 'wb') as f:
        #     torch.save(grid_initial, f) 

        # # exit()
        # print("========== Start ==========")
        # continue
        voxel_size = 4. / (grid_res_x - 1)

        image_initial = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
            bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
            voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list[2],0, camera_list)

        torchvision.utils.save_image(image_initial, "./" + dir_name + "2final_cam_" + str(grid_res_x) + "_" + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
        

        # exit()
        # deform initial SDf to target SDF
        i = 0
        loss_list = [10000000] * len(camera_list_simple)
        sum_loss = sum(loss_list) + 1
        while sum(loss_list) / len(loss_list) < sum_loss:# - tolerance / 2:
            # break
            sum_loss = sum(loss_list) / len(loss_list)

            # loss_list = []
            for cam in range(1):#range(len(camera_list_simple)):
                loss = 10000000000
                prev_loss = loss + 1
                num = 0
                while((num < iter_threshold) or loss < prev_loss): 
                    num += 1;
                    prev_loss = loss
                    iterations += 1

                    optimizer.zero_grad()

                    # Generate images
                    image_initial = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list_simple[cam], 1, camera_list)    
                    depth_initial, curvature_initial, dir_derivatives_initial, sketch_initial = generate_image_sketch(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list_simple[cam], 1, camera_list, curvature_threshold)

                    # narrow band
                    narrow_band = torch.abs(grid_initial) < 0.01
                    narrow_band = narrow_band.float()
                    # print(torch.sum(narrow_band))
                    # exit()

                    if cam == 1:
                        curvature_initial[:, 120:] = 0
                        image_initial[:, :256-120] = 0
                    elif cam == 2:
                        curvature_initial[:, :256-120] = 0
                        image_initial[:, :256-120] = 0

                    # Perform backprobagation
                    image_loss[cam], sdf_loss[cam] = loss_fn(curvature_initial, curvature_target_list[cam], grid_initial, narrow_band, voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height)
                    # background_loss = torch.sum(torch.clamp((1 - curvature_target_list[cam]) * 1000 - (1 - curvature_initial) * 1000, min=0, max=1) ** 2) #* 1 + torch.sum(curvature_target_list[cam][curvature_initial == 0]**2) * 1
                    # background_loss = torch.sum(torch.clamp((1 - curvature_target_list[cam]) * 1000 - (1 - curvature_initial) * 1000, min=0, max=1) ** 2) #* 1 + torch.sum(curvature_target_list[cam][curvature_initial == 0]**2) * 1
                    background_loss = torch.sum(image_initial[image_target[cam] == 0]**2) 
                    print("back", background_loss)
                    # torchvision.utils.save_image(torch.clamp((1 - curvature_target_list[cam]) * 1000 - (1 - curvature_initial) * 1000, min=0, max=1), "./" + dir_name + "back_" + str(cam) + "_" + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                    # torchvision.utils.save_image(torch.clamp(image_initial[image_target[cam] == 0], min=0, max=1), "./" + dir_name + "back_" + str(cam) + "_" + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

                    

                    loss = image_loss[cam] * 0.1 + sdf_loss[cam] * 0.001 + background_loss * 0.2
                    loss = background_loss * 0.2 + sdf_loss[cam] * 0.0001+ image_loss[cam] * 1
                    loss = image_loss[cam] * image_scale + sdf_loss[cam] * sdf_scale #+ background_loss * 0.2
                    conv_input = (grid_initial * narrow_band).unsqueeze(0).unsqueeze(0)
                    conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])
                    Lp_loss = torch.sum(F.conv3d(conv_input, conv_filter) ** 2)# / (64**3)
                    loss += Lp_loss * lp_scale
                    image_loss[cam] = image_loss[cam] / len(camera_list)
                    sdf_loss[cam] = sdf_loss[cam] / len(camera_list)
                    # loss_camera[cam] = image_loss[cam] + sdf_loss[cam]       
                    loss_list[cam] = prev_loss   
                    
                    print("grid res:", grid_res_x, "iteration:", i, "num:", num, "loss:", loss, "\ncamera:", camera_list[cam])
                    loss.backward()
                    optimizer.step()      

            # print(loss_list, sum(loss_list), sum_loss)
            # print(sum(loss_list) / len(loss_list), sum_loss - tolerance / 2)
            # print(sum(loss_list) / len(loss_list) < sum_loss - tolerance / 2, i)
            # exit()          

            i += 1
            if i % 1 == 0:
                for cam in range(len(camera_list_simple)):
                    image_initial = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
        voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list_simple[cam], 1, camera_list)
                    torchvision.utils.save_image(image_initial, "./" + dir_name + str(cam) + "final_cam_" + str(grid_res_x) + "_" + str(i) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
                
        for cam in range(len(camera_list)):
            image_initial = generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
        bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
        voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid_initial, camera_list[cam],0, camera_list)

            torchvision.utils.save_image(image_initial, "./" + dir_name + "final_cam_" + str(grid_res_x) + "_" + str(cam) + ".png", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

        # Save the final SDF result
        with open("./" + dir_name + str(grid_res_x) + "_best_sdf_face.pt", 'wb') as f:
            torch.save(grid_initial, f) 

        # moves on to the next resolution stage 
        grid_res_update_x = grid_res_update_y = grid_res_update_z = voxel_res_list.pop(0)
        voxel_size_update = (bounding_box_max_x - bounding_box_min_x) / (grid_res_update_x - 1)
        grid_initial_update = Tensor(grid_res_update_x, grid_res_update_y, grid_res_update_z)
        linear_space_x = torch.linspace(0, grid_res_update_x-1, grid_res_update_x)
        linear_space_y = torch.linspace(0, grid_res_update_y-1, grid_res_update_y)
        linear_space_z = torch.linspace(0, grid_res_update_z-1, grid_res_update_z)
        first_loop = linear_space_x.repeat(grid_res_update_y * grid_res_update_z, 1).t().contiguous().view(-1).unsqueeze_(1)
        second_loop = linear_space_y.repeat(grid_res_update_z, grid_res_update_x).t().contiguous().view(-1).unsqueeze_(1)
        third_loop = linear_space_z.repeat(grid_res_update_x * grid_res_update_y).unsqueeze_(1)
        loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()
        min_x = Tensor([bounding_box_min_x]).repeat(grid_res_update_x*grid_res_update_y*grid_res_update_z, 1)
        min_y = Tensor([bounding_box_min_y]).repeat(grid_res_update_x*grid_res_update_y*grid_res_update_z, 1)
        min_z = Tensor([bounding_box_min_z]).repeat(grid_res_update_x*grid_res_update_y*grid_res_update_z, 1)
        bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1)

        # Get the position of the grid points in the refined grid
        points = bounding_min_matrix + voxel_size_update * loop
        voxel_min_point_index_x = torch.floor((points[:,0].unsqueeze_(1) - min_x) / voxel_size).clamp(max=grid_res_x-2)
        voxel_min_point_index_y = torch.floor((points[:,1].unsqueeze_(1) - min_y) / voxel_size).clamp(max=grid_res_y-2)
        voxel_min_point_index_z = torch.floor((points[:,2].unsqueeze_(1) - min_z) / voxel_size).clamp(max=grid_res_z-2)
        voxel_min_point_index = torch.cat((voxel_min_point_index_x, voxel_min_point_index_y, voxel_min_point_index_z), 1)
        voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

        # Compute the sdf value of the grid points in the refined grid
        grid_initial_update = calculate_sdf_value(grid_initial, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x, grid_res_y, grid_res_z).view(grid_res_update_x, grid_res_update_y, grid_res_update_z)

      
        # Update the grid resolution for the refined sdf grid
        grid_res_x = grid_res_update_x
        grid_res_y = grid_res_update_y
        grid_res_z = grid_res_update_z

        # Update the voxel size for the refined sdf grid
        voxel_size = voxel_size_update

        # Update the sdf grid
        grid_initial = grid_initial_update.data

        # Double the size of the image
        if width < 256:
            width = int(width * 2)
            height = int(height * 2)
        learning_rate /= 1.03

    print("Time:", time.time() - start_time)

    print("----- END -----")
    